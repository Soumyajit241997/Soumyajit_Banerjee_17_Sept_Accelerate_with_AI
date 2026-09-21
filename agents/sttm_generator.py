"""STTM (Source-to-Target Mapping) Generator Agent.

Generates the transformation rule CSV for each Medallion layer. RADAR has
one more layer than IDAMP's Bronze/Silver/Gold: a Signal layer sits between
Silver and Gold and is responsible for the returns-specific feature
engineering (join orders+returns+products, derive is_returned,
days_to_return, net_revenue, revenue_erosion, order_month, ...). Bronze and
Silver stay intent-agnostic; Signal and Gold are intent-driven.

Every STTM row has the same shape:
    source_schema | source_table | source_column | target_schema |
    target_table | target_column | transformation_type | transformation_logic
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from langchain_core.tools import tool

from core.agent_factory import build_agent, run_agent
from core.audit import log_event
from core.config import STTM_DIR, get_llm
from core.fuzzy import best_match
from core.json_utils import extract_json
from core.observability import AgentTrace

STTM_COLUMNS = [
    "source_schema", "source_table", "source_column",
    "target_schema", "target_table", "target_column",
    "transformation_type", "transformation_logic",
]


def _call_llm_for_rows(instructions: str, context: str) -> list[dict]:
    llm = get_llm()
    prompt = (
        f"{instructions}\n\n"
        f"CONTEXT:\n{context}\n\n"
        "Respond with ONLY a JSON array of STTM row objects. Each object must have exactly "
        f"these keys: {STTM_COLUMNS}. No prose, no markdown fences."
    )
    response = llm.invoke(prompt)
    rows = extract_json(response.content)
    if not isinstance(rows, list):
        raise ValueError("STTM generator did not return a JSON array")
    return rows


def save_sttm(rows: list[dict], out_path: str | Path) -> str:
    df = pd.DataFrame(rows)
    for col in STTM_COLUMNS:
        if col not in df.columns:
            df[col] = ""
    df = df[STTM_COLUMNS]
    df.insert(0, "approved", True)
    df.to_csv(out_path, index=False)
    return str(out_path)


# ---------------------------------------------------------------------------
# Bronze STTM (intent-agnostic: mechanical column mapping + metadata)
# ---------------------------------------------------------------------------
BRONZE_INSTRUCTIONS = """You generate Bronze-layer STTM rows for RADAR, an e-commerce
orders-and-returns analytics pipeline. Bronze is intent-agnostic: map every source column
to a same-named (or cleaned-up snake_case) target column with an appropriate type cast
(text/integer/float/datetime). transformation_type is one of "direct" (straight copy/rename),
"indirect" (copy + type cast), or "metadata". Also emit two metadata rows per source table
with transformation_type="metadata": one for target_column "_load_timestamp"
(transformation_logic="current UTC timestamp") and one for "_source_file"
(transformation_logic="original file path").

target_schema="bronze" for every row. CRITICAL: source_table and target_table MUST be set to
EXACTLY the canonical table name listed below for that file — do not invent, pluralise,
abbreviate, or otherwise alter it, even if a different name feels more natural:
{table_map}"""


def _canonical_tables_from_profile(profile_context: str) -> dict[str, str]:
    """Extract {filename: canonical_table_name} from the profiler's output JSON.
    canonical_table_name is the file's stem, lowercased — the same name
    core.loaders/_apply_bronze_rules use to look the file back up."""
    try:
        profile = json.loads(profile_context)
        filenames = list(profile.get("datasets", {}).keys())
    except (json.JSONDecodeError, AttributeError, TypeError):
        filenames = []
    return {name: Path(name).stem.lower() for name in filenames}


def _normalize_table_names(rows: list[dict], canonical_tables: list[str]) -> list[dict]:
    """Snap each row's source_table/target_table to the nearest canonical name, so a
    row that survives (e.g. an LLM writing "order_transactions" instead of "orders")
    doesn't silently vanish later when bronze_agent looks up the file by exact name."""
    if not canonical_tables:
        return rows
    for row in rows:
        for key in ("source_table", "target_table"):
            value = row.get(key)
            if isinstance(value, str) and value:
                match = best_match(value, canonical_tables)
                if match:
                    row[key] = match
    return rows


def build_bronze_sttm_tools(profile_context: str):
    canonical = _canonical_tables_from_profile(profile_context)
    table_map = "\n".join(f'  - {fname} -> "{table}"' for fname, table in canonical.items()) or "  (none found)"
    instructions = BRONZE_INSTRUCTIONS.format(table_map=table_map)

    @tool
    def inspect_context_tool() -> str:
        """Preview the profiler's output (per-file schema, stats, semantic analysis)."""
        return profile_context

    @tool
    def generate_bronze_sttm_tool() -> str:
        """Generate Bronze STTM rows (mechanical ingestion mapping) as a JSON array."""
        rows = _call_llm_for_rows(instructions, profile_context)
        rows = _normalize_table_names(rows, list(canonical.values()))
        return json.dumps(rows, default=str)

    return inspect_context_tool, generate_bronze_sttm_tool


# ---------------------------------------------------------------------------
# Silver STTM (intent-agnostic: cleansing)
# ---------------------------------------------------------------------------
SILVER_INSTRUCTIONS = """You generate Silver-layer STTM rows for RADAR. Silver is
intent-agnostic cleansing applied uniformly to every Bronze table. For each Bronze table,
first emit ONE surrogate-key row: target_column = "pk_<table>_silver_id",
transformation_type="surrogate_key", transformation_logic="sequential integer id". Then for
every business column emit a row with transformation_type one of: "null_handling"
(logic e.g. "drop" or "fill:mean/median/mode/0/unknown"), "dedup" (logic describing the
dedup key), "type_cast" (logic: text/integer/float/datetime), "date_standardize"
(logic: "parse to YYYY-MM-DD"), "text_normalize" (logic: lowercase/strip), or "passthrough".
Drop internal Bronze metadata columns (_load_timestamp, _source_file) — do not carry them
into Silver.

CRITICAL: adding the pk_<table>_silver_id surrogate key does NOT make any business/join-key
column (e.g. order_id, product_id, customer_id — anything ending in "_id") redundant. You
MUST still emit a row for every such column (transformation_type="passthrough" or
"type_cast") — later layers join tables together using these exact columns, so dropping one
silently breaks every downstream join. Never omit a *_id column just because a surrogate key
was added for that table.

target_schema="silver" for every row. CRITICAL: source_table and target_table MUST be set to
EXACTLY one of these canonical table names (the actual Bronze tables below) — do not invent,
pluralise, abbreviate, or otherwise alter them: {table_list}"""


def _canonical_tables_from_parquet_context(context: str, strip_suffix: str) -> list[str]:
    """Extract canonical table names from a preview_parquet_files() JSON string, e.g.
    {"orders_bronze.parquet": {...}} -> ["orders"]."""
    try:
        preview = json.loads(context)
        filenames = list(preview.keys())
    except (json.JSONDecodeError, AttributeError, TypeError):
        filenames = []
    return [Path(name).stem.lower().replace(strip_suffix, "") for name in filenames]


def _columns_by_table_from_parquet_context(context: str, strip_suffix: str) -> dict[str, list[str]]:
    """Extract {table_name: [business columns]} from a preview_parquet_files() JSON
    string, excluding Bronze's internal _load_timestamp/_source_file metadata."""
    try:
        preview = json.loads(context)
    except (json.JSONDecodeError, AttributeError, TypeError):
        return {}
    result = {}
    for filename, info in preview.items():
        table = Path(filename).stem.lower().replace(strip_suffix, "")
        result[table] = [c for c in info.get("columns", []) if not c.startswith("_")]
    return result


def _ensure_id_columns_preserved(rows: list[dict], columns_by_table: dict[str, list[str]]) -> list[dict]:
    """Safety net for exactly the failure this is guarding against: the LLM decided a
    business/join-key column (order_id, product_id, ...) was redundant once a surrogate
    key existed and dropped it, silently breaking every downstream join. Any *_id source
    column not already covered by some row for its table gets a default passthrough row
    appended."""
    covered = {(r.get("source_table"), r.get("source_column")) for r in rows}
    for table, columns in columns_by_table.items():
        for col in columns:
            if col.endswith("_id") and (table, col) not in covered:
                rows.append({
                    "source_schema": "bronze", "source_table": table, "source_column": col,
                    "target_schema": "silver", "target_table": table, "target_column": col,
                    "transformation_type": "passthrough",
                    "transformation_logic": "auto-preserved business/join key",
                })
    return rows


def build_silver_sttm_tools(bronze_context: str):
    canonical = _canonical_tables_from_parquet_context(bronze_context, "_bronze")
    columns_by_table = _columns_by_table_from_parquet_context(bronze_context, "_bronze")
    instructions = SILVER_INSTRUCTIONS.format(table_list=canonical or "(none found)")

    @tool
    def inspect_context_tool() -> str:
        """Preview Bronze Parquet schemas and row samples."""
        return bronze_context

    @tool
    def generate_silver_sttm_tool() -> str:
        """Generate Silver STTM rows (cleansing rules) as a JSON array."""
        rows = _call_llm_for_rows(instructions, bronze_context)
        rows = _normalize_table_names(rows, canonical)
        rows = _ensure_id_columns_preserved(rows, columns_by_table)
        return json.dumps(rows, default=str)

    return inspect_context_tool, generate_silver_sttm_tool


# ---------------------------------------------------------------------------
# Signal STTM (NEW layer, intent-aware feature engineering)
# ---------------------------------------------------------------------------
SIGNAL_INSTRUCTIONS = """You generate Signal-layer STTM rows for RADAR. This layer is UNIQUE
to RADAR (IDAMP's original pipeline goes straight from Silver to Gold) — it sits between
Silver and Gold and turns cleansed orders/returns/products tables into ONE unified,
order-line-grain feature table that later feeds any category/revenue/return-rate analysis.

Business intent: identify which product categories have return rates that are quietly
eroding net revenue.

Emit rows targeting target_schema="signal", target_table="order_line_signals" that:
1. Join Silver orders to Silver returns on the order id (left join — most orders are NOT
   returned), transformation_type="join".
2. Join the result to Silver products on the product id to bring in category/sub_category/
   cost_price, transformation_type="join".
3. Derive these columns with transformation_type="derive" (put the exact formula in
   transformation_logic using the available column names):
   - "is_returned": 1 if a matching return exists else 0
   - "days_to_return": return_date - order_date in days (null if not returned)
   - "net_revenue": order line revenue minus any refund amount for that line
   - "revenue_erosion": the refund amount attributable to the return (0 if not returned)
   - "order_month": order_date truncated to the first of its month, format YYYY-MM-01
4. Passthrough the columns needed downstream (order_id, product_id, category,
   sub_category, quantity, order_amount, return_reason) with transformation_type="passthrough".

Use the ACTUAL column names visible in the context below (they may differ from these
examples) — do not invent columns that are not present."""


def build_signal_sttm_tools(silver_context: str):
    @tool
    def inspect_context_tool() -> str:
        """Preview Silver Parquet schemas and row samples for orders/returns/products."""
        return silver_context

    @tool
    def generate_signal_sttm_tool() -> str:
        """Generate Signal STTM rows (joins + derived return/revenue features) as a JSON array."""
        rows = _call_llm_for_rows(SIGNAL_INSTRUCTIONS, silver_context)
        return json.dumps(rows, default=str)

    return inspect_context_tool, generate_signal_sttm_tool


# ---------------------------------------------------------------------------
# Gold STTM (intent-driven aggregation)
# ---------------------------------------------------------------------------
GOLD_INSTRUCTIONS = """You generate Gold-layer STTM rows for RADAR. Gold is fully
intent-driven: shape the Signal layer's order_line_signals table into analytics-ready
tables that directly answer the business question below.

Business intent: identify which product categories have return rates that are quietly
eroding net revenue, so the business can address the underlying causes.

Emit rows for TWO target tables (target_schema="gold"):

1. target_table="category_return_performance" (one row per category), with
   transformation_type="aggregate", grouped by category, containing at least:
   total_orders (count of order lines), total_returns (sum of is_returned),
   return_rate (total_returns / total_orders), gross_revenue (sum of order_amount),
   total_refunds (sum of revenue_erosion), net_revenue (sum of net_revenue),
   revenue_erosion_pct (total_refunds / gross_revenue), avg_days_to_return.
   Put the exact aggregation formula in transformation_logic.

2. target_table="monthly_category_trend" (one row per category x order_month), with
   transformation_type="aggregate", grouped by category and order_month, containing at
   least: orders, returns, return_rate, net_revenue. This table feeds a time-series
   forecast, so order_month must be preserved as its own column.

Also emit one surrogate-key row per target table: target_column="pk_gold_id",
transformation_type="surrogate_key"."""


def build_gold_sttm_tools(signal_context: str):
    @tool
    def inspect_context_tool() -> str:
        """Preview the Signal layer's order_line_signals schema and row samples."""
        return signal_context

    @tool
    def generate_gold_sttm_tool() -> str:
        """Generate Gold STTM rows (category/trend aggregation rules) as a JSON array."""
        rows = _call_llm_for_rows(GOLD_INSTRUCTIONS, signal_context)
        return json.dumps(rows, default=str)

    return inspect_context_tool, generate_gold_sttm_tool


def log_sttm_generated(run_id: str, layer: str, path: str) -> None:
    log_event(run_id, "sttm_generator", f"{layer}_sttm_generated", sttm_path=path)


_LAYER_BUILDERS = {
    "bronze": build_bronze_sttm_tools,
    "silver": build_silver_sttm_tools,
    "signal": build_signal_sttm_tools,
    "gold": build_gold_sttm_tools,
}


def run_sttm_stage(layer: str, context: str, run_id: str) -> str:
    """Run a small ReAct agent that inspects `context` and generates the STTM
    for one Medallion layer, then saves it as data/sttm/sttm_<layer>_<run_id>.csv."""
    if layer not in _LAYER_BUILDERS:
        raise ValueError(f"Unknown STTM layer '{layer}'")

    inspect_tool, generate_tool = _LAYER_BUILDERS[layer](context)
    system_prompt = (
        f"You are the STTM Agent generating {layer}-layer mapping rules for RADAR. "
        f"Call inspect_context_tool first, then call {generate_tool.name} exactly once "
        "to produce the STTM rows, then reply with the word 'done'."
    )
    agent = build_agent(system_prompt, [inspect_tool, generate_tool])

    trace = AgentTrace(f"sttm_{layer}", run_id)
    messages = run_agent(agent, f"Generate the {layer} STTM.")
    trace.from_messages(messages)

    rows = None
    for msg in reversed(messages):
        if msg.__class__.__name__ == "ToolMessage" and getattr(msg, "name", None) == generate_tool.name:
            rows = extract_json(msg.content)
            break

    if rows is None:
        trace.finish("error", error=f"{generate_tool.name} was never called")
        raise RuntimeError(f"STTM agent did not call {generate_tool.name} for layer '{layer}'")

    out_path = STTM_DIR / f"sttm_{layer}_{run_id}.csv"
    save_sttm(rows, out_path)
    log_sttm_generated(run_id, layer, str(out_path))
    trace.finish("success")
    return str(out_path)
