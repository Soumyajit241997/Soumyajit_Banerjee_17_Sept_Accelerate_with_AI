"""Executive Report Agent.

Answers the business question ("which product categories have return rates
that are quietly eroding net revenue?") with DuckDB SQL run against the Gold
tables, renders Plotly evidence charts, and — the addition beyond IDAMP's
original Reporter — appends a Forecast & Outlook section: a per-category
linear-trend projection of return rate and net revenue (core/forecasting.py)
with an LLM-written narrative grounded in those computed numbers.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import pandas as pd
from langchain_core.tools import tool

from core.agent_factory import build_agent, run_agent
from core.audit import log_event
from core.charts import render_forecast_chart, render_query_chart
from core.config import FORECAST_HORIZON_MONTHS, REPORTS_DIR
from core.forecasting import compute_forecast, rank_at_risk_categories
from core.json_utils import extract_json, message_content_to_text
from core.observability import AgentTrace

SYSTEM_PROMPT = """You are the Reporter Agent in RADAR. You answer this business question
using SQL evidence from the Gold tables:

"Which product categories have return rates that are quietly eroding net revenue, so the
business can address the underlying causes?"

Steps:
1. Call inspect_gold_tables_tool to see what's available.
2. Call load_gold_data_tool to register the Gold tables in DuckDB.
3. Call execute_query_tool with SQL to find the categories with the highest return rates
   AND the highest revenue erosion (they are not always the same categories — call that
   out if so). Run as many queries as you need.
4. Call forecast_tool to see the computed return-rate and net-revenue trend projections
   per category for the next few months.
5. Respond with ONLY a JSON object (no prose, no markdown fences) with these keys:
   {
     "direct_answer": "2-4 sentence direct answer naming the worst offending categories",
     "sql_used": ["<sql query 1>", "<sql query 2>", ...],
     "charts": [
       {"title": "...", "chart_type": "bar|line|pie", "sql": "<SELECT ...>", "x": "<col>", "y": "<col>"}
     ],
     "detailed_analysis": "A few paragraphs on root causes suggested by the data (e.g. slow
        returns, high refund-to-revenue ratio, specific reasons) and what to address.",
     "forecast_insight": "2-4 sentences interpreting the forecast_tool numbers: which
        categories are on a worsening trajectory and roughly how much net revenue is at
        risk over the forecast horizon if nothing changes.",
     "at_risk_categories": ["<category>", ...]
   }
Only include charts whose SQL you can actually run against the registered tables.
"""


def run_sql_query(con, sql: str) -> str:
    """Run `sql` against `con` and return up to 200 rows as a JSON string, or a
    JSON {"error": ...} string on failure — including the case where `.fetchdf()`
    runs without raising but returns None (e.g. a non-row-returning statement),
    which used to crash the whole agent with `AttributeError: 'NoneType' object
    has no attribute 'head'`."""
    try:
        df = con.execute(sql).fetchdf()
    except Exception as exc:
        return json.dumps({"error": str(exc)})
    if df is None:
        return json.dumps({"error": "Query ran but returned no result set — use a SELECT statement."})
    return df.head(200).to_json(orient="records")


def _make_tools(gold_paths: list[str], run_id: str, scratchpad: dict):
    con = duckdb.connect(database=":memory:")
    scratchpad["con"] = con
    scratchpad["gold_paths"] = gold_paths

    @tool
    def inspect_gold_tables_tool() -> str:
        """Preview each Gold table's columns, dtypes, row count, and 3 sample rows."""
        preview = {}
        for p in gold_paths:
            df = pd.read_parquet(p)
            preview[Path(p).stem] = {
                "columns": list(df.columns),
                "row_count": len(df),
                "sample": json.loads(df.head(3).to_json(orient="records")),
            }
        return json.dumps(preview, default=str)

    @tool
    def load_gold_data_tool() -> str:
        """Register every Gold Parquet file as a DuckDB table and return the catalog."""
        catalog = {}
        for p in gold_paths:
            table_name = Path(p).stem
            con.execute(f"CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM read_parquet(?)", [p])
            catalog[table_name] = con.execute(f"DESCRIBE {table_name}").fetchdf()["column_name"].tolist()
        return json.dumps(catalog, default=str)

    @tool
    def execute_query_tool(sql: str) -> str:
        """Run a SQL SELECT against the DuckDB tables registered by load_gold_data_tool."""
        return run_sql_query(con, sql)

    @tool
    def forecast_tool() -> str:
        """Compute the linear return-rate/net-revenue trend forecast per category."""
        trend_path = next((p for p in gold_paths if "monthly_category_trend" in p), None)
        if trend_path is None:
            return json.dumps({"error": "monthly_category_trend table not found"})
        trend_df = pd.read_parquet(trend_path)
        forecast = compute_forecast(trend_df, FORECAST_HORIZON_MONTHS)
        scratchpad["forecast"] = forecast
        summary = {
            name: {
                "trend_direction": entry["trend_direction"],
                "current_return_rate": round(entry["history_return_rate"][-1], 4),
                "projected_return_rate_end_of_horizon": round(entry["projected_return_rate"][-1], 4),
                "return_rate_slope_per_month": round(entry["return_rate_slope_per_month"], 5),
                "net_revenue_slope_per_month": round(entry["net_revenue_slope_per_month"], 2),
                "projected_revenue_at_risk": round(entry["projected_revenue_at_risk"], 2),
            }
            for name, entry in forecast.items()
        }
        return json.dumps(summary, default=str)

    return [inspect_gold_tables_tool, load_gold_data_tool, execute_query_tool, forecast_tool]


def _build_report_html(run_id: str, analysis: dict, con: duckdb.DuckDBPyConnection, forecast: dict) -> str:
    at_risk = analysis.get("at_risk_categories") or rank_at_risk_categories(forecast)

    chart_html_blocks = []
    for chart in analysis.get("charts", []):
        try:
            df = con.execute(chart["sql"]).fetchdf()
            chart_html_blocks.append(render_query_chart(df, chart.get("title", ""), chart.get("chart_type", "bar"), chart.get("x"), chart.get("y")))
        except Exception as exc:
            chart_html_blocks.append(f"<p><em>Chart '{chart.get('title')}' failed: {exc}</em></p>")

    forecast_chart_blocks = []
    for category in at_risk:
        entry = forecast.get(category)
        if not entry:
            continue
        forecast_chart_blocks.append(render_forecast_chart(category, entry, "return_rate", "Return rate"))
        forecast_chart_blocks.append(render_forecast_chart(category, entry, "net_revenue", "Net revenue"))

    sql_html = "".join(f"<pre>{sql}</pre>" for sql in analysis.get("sql_used", []))

    at_risk_html = "".join(f"<li>{c}</li>" for c in at_risk) or "<li>No category met the rising-return / falling-revenue criteria.</li>"

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>RADAR Executive Report — {run_id}</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
  body {{ font-family: -apple-system, Segoe UI, Roboto, Arial, sans-serif; margin: 0; background: #f7f8fa; color: #1a1a1a; }}
  .wrap {{ max-width: 960px; margin: 0 auto; padding: 32px 24px 64px; }}
  h1 {{ font-size: 1.6rem; }}
  section {{ background: #fff; border: 1px solid #e3e5e8; border-radius: 10px; padding: 24px; margin-bottom: 20px; }}
  section h2 {{ margin-top: 0; font-size: 1.15rem; color: #2b3a55; }}
  pre {{ background: #f0f2f5; padding: 10px 14px; border-radius: 6px; overflow-x: auto; font-size: 0.85rem; }}
  .badge {{ display: inline-block; background: #fde8e8; color: #b02a2a; border-radius: 999px; padding: 3px 12px; font-size: 0.8rem; margin: 2px 4px 2px 0; }}
  .meta {{ color: #6b7280; font-size: 0.85rem; }}
</style>
</head>
<body>
<div class="wrap">
  <h1>RADAR Executive Report</h1>
  <p class="meta">Run {run_id} &middot; generated {datetime.now(timezone.utc).isoformat()}</p>

  <section>
    <h2>Direct Answer</h2>
    <p>{analysis.get("direct_answer", "")}</p>
  </section>

  <section>
    <h2>Approach &amp; SQL</h2>
    {sql_html}
  </section>

  <section>
    <h2>Visual Evidence</h2>
    {"".join(chart_html_blocks)}
  </section>

  <section>
    <h2>Forecast &amp; Outlook (next {FORECAST_HORIZON_MONTHS} months)</h2>
    <p>{analysis.get("forecast_insight", "")}</p>
    <p><strong>Categories at risk:</strong></p>
    <div>{"".join(f'<span class="badge">{c}</span>' for c in at_risk)}</div>
    {"".join(forecast_chart_blocks)}
  </section>

  <section>
    <h2>Detailed Analysis</h2>
    <p>{analysis.get("detailed_analysis", "").replace(chr(10), "<br>")}</p>
  </section>
</div>
</body>
</html>"""


def run_reporter_agent(gold_paths: list[str], run_id: str) -> tuple[str, str]:
    trace = AgentTrace("reporter", run_id)
    scratchpad: dict = {}
    tools = _make_tools(gold_paths, run_id, scratchpad)
    agent = build_agent(SYSTEM_PROMPT, tools)

    messages = run_agent(agent, "Answer the business question and build the report per the system prompt.")
    trace.from_messages(messages)

    final_text = messages[-1].content
    try:
        analysis = extract_json(final_text)
    except Exception:
        analysis = {
            "direct_answer": message_content_to_text(final_text), "sql_used": [], "charts": [],
            "detailed_analysis": "", "forecast_insight": "", "at_risk_categories": [],
        }

    con = scratchpad.get("con") or duckdb.connect(database=":memory:")
    for p in gold_paths:
        con.execute(f"CREATE OR REPLACE TABLE {Path(p).stem} AS SELECT * FROM read_parquet(?)", [p])

    forecast = scratchpad.get("forecast")
    if forecast is None:
        trend_path = next((p for p in gold_paths if "monthly_category_trend" in p), None)
        forecast = compute_forecast(pd.read_parquet(trend_path)) if trend_path else {}

    html = _build_report_html(run_id, analysis, con, forecast)

    report_path = REPORTS_DIR / f"report_{run_id}.html"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html)

    json_path = REPORTS_DIR / f"report_{run_id}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"analysis": analysis, "forecast": forecast}, f, indent=2, default=str)

    log_event(run_id, "reporter", "report_generated", report_path=str(report_path))
    trace.finish("success")
    return str(report_path), str(json_path)


# ---------------------------------------------------------------------------
# Ad-hoc Q&A — lets the user ask ANY follow-up question against the Gold
# tables from the UI, instead of being limited to the one business question
# the pipeline was launched with.
# ---------------------------------------------------------------------------
QA_SYSTEM_PROMPT = """You are RADAR's ad-hoc data Q&A assistant. The user will ask a
free-form question about their e-commerce orders/returns data. Answer it using SQL against
the Gold tables (category_return_performance, monthly_category_trend):
1. Call load_gold_data_tool once to see what's registered.
2. Call execute_query_tool with whatever SQL answers the question (you may run more than one).
3. If the question is about trends, projections, or the future, also call forecast_tool.
Respond with ONLY a JSON object (no prose, no markdown fences):
{
  "answer": "direct 2-4 sentence answer to the user's question, with actual numbers",
  "sql": "the single most relevant SQL query you ran, or empty string if none",
  "chart": null or {"title": "...", "chart_type": "bar|line|pie", "sql": "<SELECT ...>", "x": "<col>", "y": "<col>"}
}
Only include a chart if a query result would genuinely help visualise the answer."""


def answer_question(gold_paths: list[str], question: str, run_id: str) -> dict:
    """Answer an arbitrary follow-up question against the already-materialised Gold
    tables. Does not touch the saved report; safe to call repeatedly."""
    trace = AgentTrace("reporter_qa", run_id)
    scratchpad: dict = {}
    tools = _make_tools(gold_paths, run_id, scratchpad)
    agent = build_agent(QA_SYSTEM_PROMPT, tools)

    messages = run_agent(agent, question)
    trace.from_messages(messages)

    try:
        result = extract_json(messages[-1].content)
    except Exception:
        result = {"answer": message_content_to_text(messages[-1].content), "sql": "", "chart": None}

    con = scratchpad.get("con") or duckdb.connect(database=":memory:")
    for p in gold_paths:
        con.execute(f"CREATE OR REPLACE TABLE {Path(p).stem} AS SELECT * FROM read_parquet(?)", [p])

    chart_html = None
    chart_spec = result.get("chart")
    if chart_spec:
        try:
            df = con.execute(chart_spec["sql"]).fetchdf()
            chart_html = render_query_chart(
                df, chart_spec.get("title", ""), chart_spec.get("chart_type", "bar"),
                chart_spec.get("x"), chart_spec.get("y"),
            )
        except Exception as exc:
            chart_html = f"<p><em>Chart failed: {exc}</em></p>"

    log_event(run_id, "reporter_qa", "question_answered", question=question, answer=result.get("answer", ""))
    trace.finish("success")
    return {"question": question, "answer": result.get("answer", ""), "sql": result.get("sql", ""), "chart_html": chart_html}
