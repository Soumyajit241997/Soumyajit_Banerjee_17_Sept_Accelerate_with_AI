# RADAR — Returns Analytics via Data-driven Agentic Reasoning

RADAR is an agentic, medallion-architecture data pipeline that answers one
question for an e-commerce business:

> **Which product categories have return rates that are quietly eroding net
> revenue, so the business can address the underlying causes?**

It follows the same intent-driven, human-in-the-loop agentic pattern as
**IDAMP** (Intent-Driven Agentic Medallion Pipeline) — upload raw files,
type a business question, let a team of LLM-powered agents profile,
transform, and analyze the data, approving at each Medallion transition —
but is built specifically for orders/returns/revenue analysis, with three
deliberate differences from that reference design:

| | IDAMP | RADAR |
|---|---|---|
| **Landing formats** | CSV only | CSV **+ JSON + Excel (XLSX)** — orders arrive as CSV, returns as a JSON export, the product master as a spreadsheet |
| **Medallion layers** | Bronze → Silver → Gold | Bronze → Silver → **Signal** → Gold — a new feature-engineering layer that joins orders/returns/products and derives `is_returned`, `days_to_return`, `net_revenue`, `revenue_erosion`, `order_month` once, so Gold and the report never repeat that join logic |
| **Reporter output** | Direct answer + SQL evidence + charts | Same, **plus a Forecast & Outlook section**: a per-category linear-trend projection of return rate and net revenue for the next N months, with an LLM narrative grounded in those computed numbers and a ranked list of at-risk categories |

---

## 1. Architecture

```
Landing (CSV/JSON/XLSX)
        │
        ▼
   Profiler Agent  ──▶  profile JSON (schema + semantic analysis)
        │
        ▼
   STTM Agent (Bronze)  ──▶  sttm_bronze.csv   ⏸ HITL approval
        │
        ▼
   Bronze Agent  ──▶  bronze_layer/*.parquet  (typed, format-agnostic ingest)
        │
        ▼
   STTM Agent (Silver)  ──▶  sttm_silver.csv   ⏸ HITL approval
        │
        ▼
   Silver Agent  ──▶  silver_layer/*.parquet  (cleansed, deduped, standardised)
        │
        ▼
   STTM Agent (Signal)  ──▶  sttm_signal.csv   ⏸ HITL approval        ◀── NEW LAYER
        │
        ▼
   Signal Agent  ──▶  signal_layer/order_line_signals.parquet
        │              (joins orders+returns+products; derives is_returned,
        │               days_to_return, net_revenue, revenue_erosion, order_month)
        ▼
   STTM Agent (Gold)  ──▶  sttm_gold.csv   ⏸ HITL approval
        │
        ▼
   Gold Agent  ──▶  gold_layer/category_return_performance.parquet
        │           gold_layer/monthly_category_trend.parquet
        ▼
   Reporter Agent  ──▶  DuckDB SQL answer + Plotly charts
                        + linear-trend forecast (core/forecasting.py)
                        + reports/report_<run_id>.html / .json
```

Every specialist agent is a small LangGraph **ReAct agent** (`core/agent_factory.py`)
built from an LLM (Anthropic Claude preferred, fallback Groq, then Gemini — `core/config.py`) and a handful of
`@tool`-decorated Python functions. A **Supervisor** orchestrator
(`agents/orchestrator.py`) runs one phase at a time, each phase dispatching the
relevant specialist agents and stopping at a human approval gate before the
next phase begins. File paths never round-trip through the LLM: each phase
uses a shared `scratchpad` dict so one tool's output becomes the next tool's
input directly.

### Why a Signal layer?

"Which categories are eroding net revenue" cannot be answered from cleansed
but *unjoined* orders/returns/products tables — it needs an
**order-line-grain** feature first: was this line returned, how long did it
take, and how much revenue did it actually erase. IDAMP's Gold agent joins
and aggregates in the same step; RADAR splits those apart so the join/derive
logic is computed once, deterministically, and can be reused by any future
Gold table or ad-hoc query without re-deriving it.

### Why STTM-guided but code-executed transformations?

Bronze and Silver STTM rows are interpreted literally (rename → cast → clean),
same as IDAMP. Signal and Gold execution is **deterministic Python guided by
the STTM's declared shape** (which tables, which columns) rather than a
literal interpreter of LLM-authored join/aggregation formulas — RADAR never
executes arbitrary code an LLM wrote. This mirrors IDAMP's own Gold agent,
which resolves joins by matching `_id`-suffixed columns rather than parsing
formulas. The STTM still serves its purpose: a human-reviewable, auditable
record of what's about to happen, approved before execution.

---

## 2. The Forecast & Outlook section

`core/forecasting.py` fits a simple linear trend (`numpy.polyfit`, degree 1)
per category on its monthly `return_rate` and `net_revenue` history from
`monthly_category_trend`, then projects it `FORECAST_HORIZON_MONTHS` (default
3) months forward. Categories with fewer than 3 months of history are
skipped rather than extrapolated from noise. A category is flagged **at
risk** when its return rate is trending up *and* its net revenue is trending
down — ranked by projected revenue at risk over the horizon.

The Reporter's LLM never does this arithmetic itself; it only narrates
numbers `compute_forecast()` already produced (via `forecast_tool`), so the
report's forecast section stays numerically trustworthy even though the
narrative is LLM-written.

---

## 3. Project structure

```
radar-returns-pipeline/
├── core/                   # infrastructure: config, state, audit, memory,
│                           # observability, loaders, transforms, forecasting,
│                           # charts, agent_factory, json_utils, preview
├── agents/
│   ├── orchestrator.py     # Supervisor — 5 phases
│   ├── profiler.py
│   ├── sttm_generator.py   # unified STTM agent: bronze/silver/signal/gold
│   ├── bronze_agent.py
│   ├── silver_agent.py
│   ├── enrichment_agent.py # Signal layer — the new one
│   ├── gold_agent.py
│   └── reporter.py         # SQL answer + charts + forecast
├── app/
│   └── streamlit_app.py    # 6-phase UI: Upload → Bronze/Silver/Signal/Gold STTM → Report
├── scripts/
│   └── generate_sample_data.py
├── sample_data/            # generated demo: orders.csv, returns.json, products.xlsx
├── data/                   # runtime artifacts (gitignored, dirs kept via .gitkeep)
├── reports/                # generated HTML/JSON reports (gitignored)
├── audit_logs/             # JSONL audit trail per run (gitignored)
└── tests/                  # pytest — deterministic core logic, no API key needed
```

---

## 4. Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
copy .env.example .env          # then fill in ANTHROPIC_API_KEY, GROQ_API_KEY, or GOOGLE_API_KEY
```

Generate the synthetic demo dataset (6 months, 3 formats, a seeded rising
return-rate trend in Apparel and Electronics):

```bash
python scripts/generate_sample_data.py
```

Run the app:

```bash
streamlit run app/streamlit_app.py
```

Upload `sample_data/orders.csv`, `sample_data/returns.json`, and
`sample_data/products.xlsx` together, keep the default business question,
and step through the four approval gates to the final report.

## 5. Tests

The full transformation core (Bronze → Silver → Signal → Gold) is unit- and
integration-tested against the real generated sample data with a
hand-authored STTM, so the suite needs **no LLM API key**:

```bash
python -m pip install -r requirements.txt
python scripts/generate_sample_data.py
pytest
```

## 6. Configuration

| Env var | Default | Purpose |
|---|---|---|
| `ANTHROPIC_API_KEY` | — | Preferred LLM provider (Claude) |
| `ANTHROPIC_MODEL` | `claude-sonnet-5` | |
| `GROQ_API_KEY` | — | Used if `ANTHROPIC_API_KEY` is not set |
| `GOOGLE_API_KEY` | — | Last-resort fallback LLM provider (Gemini) |
| `GROQ_MODEL` | `llama-3.3-70b-versatile` | |
| `GEMINI_MODEL` | `gemini-2.5-flash` | |
| `FORECAST_HORIZON_MONTHS` | `3` | Months the Reporter projects forward |
