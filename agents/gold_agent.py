"""Gold Layer Materialisation Agent.

Aggregates the Signal layer's order_line_signals table into the two
analytics-ready tables the Gold STTM calls for: a per-category return/
revenue scorecard, and a per-category monthly trend used later by the
Reporter agent's forecast.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from langchain_core.tools import tool

from core.agent_factory import build_agent, run_agent
from core.audit import log_event
from core.config import GOLD_DIR
from core.observability import AgentTrace
from core.transforms import add_surrogate_key

SYSTEM_PROMPT = """You are the Gold Agent in RADAR. Aggregate the Signal layer's
order_line_signals table into category-level return/revenue tables per the approved Gold
STTM. Call inspect_task_tool to review the schema and rules, then gold_ingestion_tool to
execute."""


def _build_category_performance(df: pd.DataFrame) -> pd.DataFrame:
    grouped = df.groupby("category", dropna=False).agg(
        total_orders=("order_id", "count"),
        total_returns=("is_returned", "sum"),
        gross_revenue=("order_amount", "sum"),
        total_refunds=("revenue_erosion", "sum"),
        net_revenue=("net_revenue", "sum"),
        avg_days_to_return=("days_to_return", "mean"),
    ).reset_index()
    grouped["return_rate"] = grouped["total_returns"] / grouped["total_orders"]
    grouped["revenue_erosion_pct"] = grouped["total_refunds"] / grouped["gross_revenue"].replace(0, pd.NA)
    return add_surrogate_key(grouped, "pk_gold_id")


def _build_monthly_trend(df: pd.DataFrame) -> pd.DataFrame:
    grouped = df.groupby(["category", "order_month"], dropna=False).agg(
        orders=("order_id", "count"),
        returns=("is_returned", "sum"),
        net_revenue=("net_revenue", "sum"),
    ).reset_index()
    grouped["return_rate"] = grouped["returns"] / grouped["orders"]
    grouped = grouped.sort_values(["category", "order_month"])
    return add_surrogate_key(grouped, "pk_gold_id")


def _apply_gold_rules(signal_paths: list[str], sttm_path: str, run_id: str) -> list[str]:
    if not signal_paths:
        raise ValueError("Gold layer requires the Signal layer's order_line_signals table, but none was provided.")
    df = pd.read_parquet(signal_paths[0])

    tables = {
        "category_return_performance": _build_category_performance(df),
        "monthly_category_trend": _build_monthly_trend(df),
    }

    output_paths = []
    for name, table in tables.items():
        out_path = GOLD_DIR / f"{name}.parquet"
        table.to_parquet(out_path, index=False)
        output_paths.append(str(out_path))
        log_event(run_id, "gold_agent", "table_materialised", table=name, rows=len(table), output_path=str(out_path))

    return output_paths


def _make_tools(signal_paths: list[str], sttm_path: str, run_id: str, scratchpad: dict):
    @tool
    def inspect_task_tool() -> str:
        """Preview the Signal layer schema and the approved Gold STTM rules."""
        df = pd.read_parquet(signal_paths[0])
        sttm = pd.read_csv(sttm_path)
        preview = {
            "signal_columns": list(df.columns),
            "signal_row_count": len(df),
            "gold_target_tables": sttm["target_table"].unique().tolist() if "target_table" in sttm.columns else [],
        }
        return json.dumps(preview, default=str)

    @tool
    def gold_ingestion_tool(confirmation: str = "execute") -> str:
        """Execute Gold materialisation: aggregate Signal data into category/trend tables."""
        paths = _apply_gold_rules(signal_paths, sttm_path, run_id)
        scratchpad["output_paths"] = paths
        return json.dumps(paths)

    return [inspect_task_tool, gold_ingestion_tool]


def run_gold_agent(signal_paths: list[str], sttm_path: str, run_id: str) -> list[str]:
    trace = AgentTrace("gold_agent", run_id)
    scratchpad: dict = {}
    tools = _make_tools(signal_paths, sttm_path, run_id, scratchpad)
    agent = build_agent(SYSTEM_PROMPT, tools)

    messages = run_agent(agent, "Materialise the Gold tables per the approved Gold STTM.")
    trace.from_messages(messages)

    paths = scratchpad.get("output_paths") or _apply_gold_rules(signal_paths, sttm_path, run_id)
    trace.finish("success")
    return paths
