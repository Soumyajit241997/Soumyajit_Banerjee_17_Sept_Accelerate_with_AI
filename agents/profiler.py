"""Profiler Agent.

Inspects the raw landing files (any mix of CSV / JSON / XLSX) and produces a
combined profile: per-file shape, dtypes, null/unique stats, and an
LLM-authored semantic analysis (column meanings, join-key candidates,
data-quality notes relevant to a returns/revenue analysis).
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from langchain_core.tools import tool

from core.agent_factory import build_agent, run_agent
from core.audit import log_event
from core.config import PROFILE_DIR
from core.json_utils import extract_json
from core.loaders import describe_format, load_any
from core.observability import AgentTrace

SYSTEM_PROMPT = """You are the Profiler Agent in RADAR, an agentic pipeline that identifies \
which e-commerce product categories have return rates quietly eroding net revenue.

You will be given raw landing files that may be CSV, JSON, or Excel. Your job:
1. Call inspect_files_tool to preview each file's shape, columns, and sample values.
2. Call profiler_tool to compute full column statistics.
3. Reason about what each file/column means for an orders-and-returns revenue analysis:
   which columns identify a product/category, which identify an order, which identify a
   return, which columns are join keys across files, and any data-quality issues
   (nulls, inconsistent formats, suspicious values) that later layers should handle.
4. Respond with ONLY a JSON object (no prose) of the form:
   {
     "semantic_analysis": {
       "<filename>": {
         "likely_role": "orders | returns | products | other",
         "column_meanings": {"<column>": "<meaning>"},
         "join_key_candidates": ["<column>", ...],
         "quality_notes": ["<note>", ...]
       }
     }
   }
"""


def _inspect_files(file_paths: list[str]) -> dict:
    preview = {}
    for path in file_paths:
        df = load_any(path)
        preview[Path(path).name] = {
            "format": describe_format(path),
            "shape": list(df.shape),
            "columns": list(df.columns),
            "dtypes": {c: str(t) for c, t in df.dtypes.items()},
            "sample": json.loads(df.head(3).to_json(orient="records", date_format="iso")),
        }
    return preview


def _profile_files(file_paths: list[str]) -> dict:
    stats = {}
    for path in file_paths:
        df = load_any(path)
        col_stats = {}
        for col in df.columns:
            series = df[col]
            entry = {
                "dtype": str(series.dtype),
                "null_count": int(series.isna().sum()),
                "null_pct": round(float(series.isna().mean()) * 100, 2),
                "unique_count": int(series.nunique(dropna=True)),
            }
            if pd_is_numeric(series):
                entry.update(
                    {
                        "min": _safe_float(series.min()),
                        "max": _safe_float(series.max()),
                        "mean": _safe_float(series.mean()),
                    }
                )
            col_stats[col] = entry
        stats[Path(path).name] = {
            "format": describe_format(path),
            "row_count": int(len(df)),
            "columns": col_stats,
        }
    return stats


def pd_is_numeric(series) -> bool:
    import pandas as pd

    return pd.api.types.is_numeric_dtype(series)


def _safe_float(value):
    try:
        f = float(value)
        return None if f != f else round(f, 4)  # NaN check
    except (TypeError, ValueError):
        return None


def _make_tools(file_paths: list[str], scratchpad: dict):
    @tool
    def inspect_files_tool() -> str:
        """Preview the shape, columns, dtypes, and 3 sample rows of every landing file."""
        return json.dumps(_inspect_files(file_paths), default=str)

    @tool
    def profiler_tool() -> str:
        """Compute full column statistics (nulls, uniques, min/max/mean) for every landing file."""
        stats = _profile_files(file_paths)
        scratchpad["stats"] = stats
        return json.dumps(stats, default=str)

    return [inspect_files_tool, profiler_tool]


def run_profiler_agent(file_paths: list[str], run_id: str) -> str:
    """Runs the Profiler agent end-to-end and returns the saved profile path."""
    trace = AgentTrace("profiler", run_id)
    scratchpad: dict = {}
    tools = _make_tools(file_paths, scratchpad)
    agent = build_agent(SYSTEM_PROMPT, tools)

    goal = (
        f"Profile these landing files: {file_paths}. "
        "Call inspect_files_tool then profiler_tool, then return the semantic_analysis JSON."
    )
    messages = run_agent(agent, goal)
    trace.from_messages(messages)

    final_text = messages[-1].content
    try:
        semantic = extract_json(final_text).get("semantic_analysis", {})
    except Exception:
        semantic = {}

    stats = scratchpad.get("stats") or _profile_files(file_paths)

    combined = {
        "run_id": run_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "datasets": stats,
        "semantic_analysis": semantic,
    }

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    out_path = PROFILE_DIR / f"profile_combined_{timestamp}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(combined, f, indent=2, default=str)

    log_event(run_id, "profiler", "profile_generated", profile_path=str(out_path), files=file_paths)
    trace.finish("success")
    return str(out_path)
