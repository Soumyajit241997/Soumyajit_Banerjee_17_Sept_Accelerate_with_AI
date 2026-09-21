"""Bronze Layer Ingestion Agent.

Applies the approved Bronze STTM to the raw landing files (CSV, JSON, or
XLSX — format is irrelevant here since core.loaders.load_any normalises
everything to a DataFrame first) and writes one Parquet file per source
table with renamed/cast columns plus _load_timestamp/_source_file metadata.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from langchain_core.tools import tool

from core.agent_factory import build_agent, run_agent
from core.audit import log_event
from core.config import BRONZE_DIR
from core.fuzzy import best_match
from core.loaders import describe_format, load_any
from core.observability import AgentTrace
from core.transforms import cast_series, preserve_missing_id_columns

SYSTEM_PROMPT = """You are the Bronze Agent in RADAR. Your job is mechanical: apply the
approved Bronze STTM rules to ingest raw landing files into clean, typed Parquet tables.
Call inspect_task_tool to see the files and rules, then call bronze_ingestion_tool to
execute. Report the output Parquet paths."""


def _table_name(path: str) -> str:
    return Path(path).stem.lower()


def _apply_bronze_rules(file_paths: list[str], sttm_path: str, run_id: str) -> list[str]:
    sttm = pd.read_csv(sttm_path)
    if "approved" in sttm.columns:
        sttm = sttm[sttm["approved"].astype(str).str.lower() != "false"]

    file_map = {_table_name(p): p for p in file_paths}
    output_paths = []

    for table_name, group in sttm.groupby("target_table"):
        matched_name = table_name if table_name in file_map else best_match(table_name, list(file_map.keys()))
        if matched_name is None:
            log_event(run_id, "bronze_agent", "table_skipped_no_match", target_table=table_name,
                       available_files=list(file_map.keys()))
            continue
        source_path = file_map[matched_name]
        df = load_any(source_path)
        out_df = pd.DataFrame(index=df.index)

        for _, rule in group.iterrows():
            ttype = str(rule.get("transformation_type", "")).lower()
            tcol = rule.get("target_column")
            scol = rule.get("source_column")

            if ttype == "metadata":
                if tcol == "_load_timestamp":
                    out_df[tcol] = datetime.now(timezone.utc).isoformat()
                elif tcol == "_source_file":
                    out_df[tcol] = str(source_path)
                continue

            if not isinstance(scol, str) or scol not in df.columns:
                continue

            series = df[scol]
            if ttype == "indirect":
                series = cast_series(series, rule.get("transformation_logic", ""))
            out_df[tcol] = series

        out_df, recovered = preserve_missing_id_columns(df, out_df)
        if recovered:
            log_event(run_id, "bronze_agent", "id_columns_auto_recovered", table=matched_name, columns=recovered)

        out_path = BRONZE_DIR / f"{matched_name}_bronze.parquet"
        out_df.to_parquet(out_path, index=False)
        output_paths.append(str(out_path))
        log_event(
            run_id, "bronze_agent", "table_ingested",
            table=matched_name, source_format=describe_format(source_path),
            rows=len(out_df), output_path=str(out_path),
        )

    if not output_paths and len(sttm) > 0:
        raise ValueError(
            f"Bronze STTM has {len(sttm)} approved row(s) but none of their target_table "
            f"values matched an uploaded file. STTM target tables: {sorted(sttm['target_table'].unique())}; "
            f"uploaded files resolve to: {list(file_map.keys())}. Fix the target_table column in "
            "the Bronze STTM above and try again."
        )

    return output_paths


def _make_tools(file_paths: list[str], sttm_path: str, run_id: str, scratchpad: dict):
    @tool
    def inspect_task_tool() -> str:
        """Preview the landing files and the Bronze STTM rules that will be applied."""
        sttm = pd.read_csv(sttm_path)
        preview = {
            "files": [{"path": p, "format": describe_format(p)} for p in file_paths],
            "rule_count": len(sttm),
            "rules_by_table": sttm.groupby("target_table").size().to_dict(),
        }
        return json.dumps(preview, default=str)

    @tool
    def bronze_ingestion_tool(confirmation: str = "execute") -> str:
        """Execute Bronze ingestion: apply STTM rules and write Parquet outputs."""
        paths = _apply_bronze_rules(file_paths, sttm_path, run_id)
        scratchpad["output_paths"] = paths
        return json.dumps(paths)

    return [inspect_task_tool, bronze_ingestion_tool]


def run_bronze_agent(file_paths: list[str], sttm_path: str, run_id: str) -> list[str]:
    trace = AgentTrace("bronze_agent", run_id)
    scratchpad: dict = {}
    tools = _make_tools(file_paths, sttm_path, run_id, scratchpad)
    agent = build_agent(SYSTEM_PROMPT, tools)

    messages = run_agent(agent, "Ingest the approved Bronze STTM rules into Parquet.")
    trace.from_messages(messages)

    paths = scratchpad.get("output_paths") or _apply_bronze_rules(file_paths, sttm_path, run_id)
    trace.finish("success")
    return paths
