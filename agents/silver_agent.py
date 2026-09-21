"""Silver Layer Cleansing Agent.

Applies the approved Silver STTM to Bronze Parquet: null handling,
deduplication, type casting, date standardisation (-> YYYY-MM-DD), text
normalisation, and a pk_<table>_silver_id surrogate key.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from langchain_core.tools import tool

from core.agent_factory import build_agent, run_agent
from core.audit import log_event
from core.config import SILVER_DIR
from core.fuzzy import best_match
from core.observability import AgentTrace
from core.transforms import (
    add_surrogate_key, cast_series, fill_nulls, normalize_text,
    preserve_missing_id_columns, standardize_date,
)

SYSTEM_PROMPT = """You are the Silver Agent in RADAR. Apply the approved Silver STTM rules
to cleanse Bronze Parquet tables: handle nulls, deduplicate, cast types, standardise dates,
normalise text, and inject a surrogate key. Call inspect_task_tool to review the plan, then
silver_ingestion_tool to execute."""


def _bronze_table_name(path: str) -> str:
    return Path(path).stem.replace("_bronze", "").lower()


def _apply_silver_rules(bronze_paths: list[str], sttm_path: str, run_id: str) -> list[str]:
    sttm = pd.read_csv(sttm_path)
    if "approved" in sttm.columns:
        sttm = sttm[sttm["approved"].astype(str).str.lower() != "false"]

    bronze_map = {_bronze_table_name(p): p for p in bronze_paths}
    output_paths = []

    for table_name, group in sttm.groupby("target_table"):
        matched_name = table_name if table_name in bronze_map else best_match(table_name, list(bronze_map.keys()))
        if matched_name is None:
            log_event(run_id, "silver_agent", "table_skipped_no_match", target_table=table_name,
                       available_bronze_tables=list(bronze_map.keys()))
            continue
        df = pd.read_parquet(bronze_map[matched_name])
        out_df = pd.DataFrame(index=df.index)

        drop_null_cols = []
        surrogate_key_col = None
        dedup_rules = []

        for _, rule in group.iterrows():
            ttype = str(rule.get("transformation_type", "")).lower()
            tcol = rule.get("target_column")
            scol = rule.get("source_column")
            logic = str(rule.get("transformation_logic", ""))

            if ttype == "surrogate_key":
                surrogate_key_col = tcol
                continue
            if ttype == "dedup":
                dedup_rules.append(logic)
                # NOTE: "dedup" marks this column as a dedup key, it does NOT mean
                # "don't materialise it" -- fall through and still copy the data
                # (this is exactly the bug that dropped order_id/product_id and
                # broke every downstream join: the row was skipped entirely).

            if not isinstance(scol, str) or scol not in df.columns:
                continue
            series = df[scol]

            if ttype == "type_cast":
                series = cast_series(series, logic)
            elif ttype == "date_standardize":
                series = standardize_date(series)
            elif ttype == "text_normalize":
                series = normalize_text(series, logic)
            elif ttype == "null_handling":
                if logic.strip().lower() == "drop":
                    drop_null_cols.append(tcol)
                else:
                    series = fill_nulls(series, logic)
            # "passthrough" and anything else: copy as-is

            out_df[tcol] = series

        out_df, recovered = preserve_missing_id_columns(df, out_df)
        if recovered:
            log_event(run_id, "silver_agent", "id_columns_auto_recovered", table=matched_name, columns=recovered)

        if drop_null_cols:
            out_df = out_df.dropna(subset=[c for c in drop_null_cols if c in out_df.columns])

        if dedup_rules:
            out_df = out_df.drop_duplicates()

        out_df = add_surrogate_key(out_df, surrogate_key_col or f"pk_{matched_name}_silver_id")

        out_path = SILVER_DIR / f"{matched_name}_silver.parquet"
        out_df.to_parquet(out_path, index=False)
        output_paths.append(str(out_path))
        log_event(
            run_id, "silver_agent", "table_cleansed",
            table=matched_name, rows=len(out_df), output_path=str(out_path),
        )

    if not output_paths and len(sttm) > 0:
        raise ValueError(
            f"Silver STTM has {len(sttm)} approved row(s) but none of their target_table "
            f"values matched a Bronze table. STTM target tables: {sorted(sttm['target_table'].unique())}; "
            f"Bronze tables available: {list(bronze_map.keys())}. Fix the target_table column in "
            "the Silver STTM above and try again."
        )

    return output_paths


def _make_tools(bronze_paths: list[str], sttm_path: str, run_id: str, scratchpad: dict):
    @tool
    def inspect_task_tool() -> str:
        """Preview Bronze Parquet schemas and the Silver STTM cleansing rules."""
        sttm = pd.read_csv(sttm_path)
        preview = {}
        for p in bronze_paths:
            df = pd.read_parquet(p)
            preview[Path(p).name] = {"columns": list(df.columns), "row_count": len(df)}
        return json.dumps({"schemas": preview, "rule_count": len(sttm)}, default=str)

    @tool
    def silver_ingestion_tool(confirmation: str = "execute") -> str:
        """Execute Silver cleansing: apply STTM rules and write Parquet outputs."""
        paths = _apply_silver_rules(bronze_paths, sttm_path, run_id)
        scratchpad["output_paths"] = paths
        return json.dumps(paths)

    return [inspect_task_tool, silver_ingestion_tool]


def run_silver_agent(bronze_paths: list[str], sttm_path: str, run_id: str) -> list[str]:
    trace = AgentTrace("silver_agent", run_id)
    scratchpad: dict = {}
    tools = _make_tools(bronze_paths, sttm_path, run_id, scratchpad)
    agent = build_agent(SYSTEM_PROMPT, tools)

    messages = run_agent(agent, "Cleanse the Bronze tables per the approved Silver STTM.")
    trace.from_messages(messages)

    paths = scratchpad.get("output_paths") or _apply_silver_rules(bronze_paths, sttm_path, run_id)
    trace.finish("success")
    return paths
