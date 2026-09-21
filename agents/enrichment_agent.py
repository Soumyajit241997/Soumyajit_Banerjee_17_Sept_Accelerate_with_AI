"""Signal Agent — the layer that does not exist in IDAMP.

IDAMP goes straight from Silver to Gold. RADAR inserts a Signal layer here
because "which categories are quietly eroding net revenue" cannot be
answered from cleansed-but-unjoined orders/returns/products tables alone —
it needs order-line-grain features first: whether a line was returned, how
long it took, and how much revenue it actually erased. Computing those
features once here (rather than repeating join/derive logic in every Gold
aggregate or report query) is the "different transformational logic in
between" this build adds to the medallion pattern.

Column lookups are done by keyword/substring rather than exact name so the
agent tolerates minor naming variance from the LLM-authored Silver STTM
(e.g. "order_amt" vs "order_amount") while still being fully deterministic
and unit-testable.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from langchain_core.tools import tool

from core.agent_factory import build_agent, run_agent
from core.audit import log_event
from core.config import SIGNAL_DIR
from core.fuzzy import best_match
from core.observability import AgentTrace

SYSTEM_PROMPT = """You are the Signal Agent in RADAR. You turn cleansed Silver orders,
returns, and products tables into ONE unified order-line-grain feature table
(order_line_signals) that flags whether each order line was returned and how much net
revenue and revenue erosion it produced. Call inspect_task_tool to review the Silver
schemas and the approved Signal STTM, then signal_ingestion_tool to execute."""


def _silver_table_name(path: str) -> str:
    return Path(path).stem.replace("_silver", "").lower()


def _find_col(df: pd.DataFrame, *keywords: str) -> str | None:
    for col in df.columns:
        low = col.lower()
        if any(kw in low for kw in keywords):
            return col
    return None


def _find_join_key(df_a: pd.DataFrame, df_b: pd.DataFrame) -> str | None:
    """Exact-match join key: a business (non-pk_) column ending in _id present
    in both frames under the SAME name."""
    common = set(df_a.columns) & set(df_b.columns)
    candidates = [c for c in common if c.endswith("_id") and not c.startswith("pk_")]
    return candidates[0] if candidates else None


def _resolve_join_columns(df_a: pd.DataFrame, df_b: pd.DataFrame, *keywords: str) -> tuple[str, str]:
    """Resolve the join column independently on each side by keyword, for the case
    where Silver renamed it asymmetrically (or an exact-name match failed). Raises a
    clear error naming what's actually available, instead of a bare pandas KeyError
    from merging on a column that doesn't exist on one side."""
    exact = _find_join_key(df_a, df_b)
    if exact:
        return exact, exact

    col_a = _find_col(df_a, *keywords)
    col_b = _find_col(df_b, *keywords)
    if col_a and col_b:
        return col_a, col_b

    raise ValueError(
        f"Could not resolve a join column matching {keywords} between the two Silver "
        f"tables. Columns available: left={list(df_a.columns)}, right={list(df_b.columns)}. "
        "This usually means the Silver STTM dropped a business/join-key column "
        "(e.g. order_id or product_id) — check that it was kept as a passthrough row."
    )


def _get_table(tables: dict, *keywords: str):
    """Fuzzy-resolve a Silver table by role (e.g. "orders") rather than requiring an
    exact name match — tolerates upstream naming drift like "order_transactions"."""
    names = list(tables.keys())
    for keyword in keywords:
        match = best_match(keyword, names, cutoff=0.4)
        if match:
            return tables[match]
    return None


def _apply_signal_rules(silver_paths: list[str], sttm_path: str, run_id: str) -> str:
    tables = {_silver_table_name(p): pd.read_parquet(p) for p in silver_paths}
    orders = _get_table(tables, "orders", "order")
    returns = _get_table(tables, "returns", "return")
    products = _get_table(tables, "products", "product")

    if orders is None:
        raise ValueError(
            f"Signal layer requires a Silver 'orders' table; found tables: {list(tables.keys())}"
        )

    merged = orders.copy()

    if returns is not None:
        left_key, right_key = _resolve_join_columns(orders, returns, "order_id")
        merged = merged.merge(returns, left_on=left_key, right_on=right_key, how="left", suffixes=("", "_ret"))

    if products is not None:
        left_key, right_key = _resolve_join_columns(orders, products, "product_id")
        merged = merged.merge(products, left_on=left_key, right_on=right_key, how="left", suffixes=("", "_prod"))

    return_indicator_col = _find_col(returns, "return_id") if returns is not None else None
    return_date_col = _find_col(returns, "return_date") if returns is not None else None
    order_date_col = _find_col(orders, "order_date")
    refund_col = _find_col(returns, "refund", "refund_amount") if returns is not None else None
    amount_col = _find_col(orders, "order_amount", "amount")
    reason_col = _find_col(returns, "reason") if returns is not None else None
    category_col = _find_col(products, "category") if products is not None else _find_col(merged, "category")
    subcategory_col = _find_col(products, "sub_category", "subcategory") if products is not None else None

    signals = pd.DataFrame(index=merged.index)
    signals["order_id"] = merged.get(_find_col(orders, "order_id") or "order_id")
    signals["product_id"] = merged.get(_find_col(orders, "product_id") or "product_id")
    signals["category"] = merged.get(category_col) if category_col else "unknown"
    signals["sub_category"] = merged.get(subcategory_col) if subcategory_col else "unknown"
    signals["quantity"] = merged.get(_find_col(orders, "quantity"))
    signals["order_amount"] = merged.get(amount_col)

    is_returned = merged[return_indicator_col].notna().astype(int) if return_indicator_col else 0
    signals["is_returned"] = is_returned
    signals["return_reason"] = merged.get(reason_col) if reason_col else pd.NA

    if order_date_col and return_date_col:
        order_dt = pd.to_datetime(merged[order_date_col], errors="coerce")
        return_dt = pd.to_datetime(merged[return_date_col], errors="coerce")
        signals["days_to_return"] = (return_dt - order_dt).dt.days
    else:
        signals["days_to_return"] = pd.NA

    refund_amount = pd.to_numeric(merged[refund_col], errors="coerce").fillna(0) if refund_col else 0
    order_amount_numeric = pd.to_numeric(signals["order_amount"], errors="coerce").fillna(0)
    signals["revenue_erosion"] = refund_amount
    signals["net_revenue"] = order_amount_numeric - refund_amount

    if order_date_col:
        order_dt = pd.to_datetime(merged[order_date_col], errors="coerce")
        signals["order_month"] = order_dt.dt.strftime("%Y-%m-01")
    else:
        signals["order_month"] = pd.NA

    out_path = SIGNAL_DIR / "order_line_signals.parquet"
    signals.to_parquet(out_path, index=False)
    log_event(
        run_id, "signal_agent", "signals_derived",
        rows=len(signals), return_rate=round(float(signals["is_returned"].mean()), 4),
        output_path=str(out_path),
    )
    return str(out_path)


def _make_tools(silver_paths: list[str], sttm_path: str, run_id: str, scratchpad: dict):
    @tool
    def inspect_task_tool() -> str:
        """Preview Silver Parquet schemas and the approved Signal STTM rules."""
        sttm = pd.read_csv(sttm_path)
        preview = {}
        for p in silver_paths:
            df = pd.read_parquet(p)
            preview[Path(p).name] = {"columns": list(df.columns), "row_count": len(df)}
        return json.dumps({"schemas": preview, "rule_count": len(sttm)}, default=str)

    @tool
    def signal_ingestion_tool(confirmation: str = "execute") -> str:
        """Execute Signal derivation: join Silver tables and compute return/revenue features."""
        path = _apply_signal_rules(silver_paths, sttm_path, run_id)
        scratchpad["output_path"] = path
        return json.dumps({"output_path": path})

    return [inspect_task_tool, signal_ingestion_tool]


def run_signal_agent(silver_paths: list[str], sttm_path: str, run_id: str) -> list[str]:
    trace = AgentTrace("signal_agent", run_id)
    scratchpad: dict = {}
    tools = _make_tools(silver_paths, sttm_path, run_id, scratchpad)
    agent = build_agent(SYSTEM_PROMPT, tools)

    messages = run_agent(agent, "Derive the order_line_signals table per the approved Signal STTM.")
    trace.from_messages(messages)

    path = scratchpad.get("output_path") or _apply_signal_rules(silver_paths, sttm_path, run_id)
    trace.finish("success")
    return [path]
