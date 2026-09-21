"""Supervisor Orchestrator.

RADAR has 5 phases (IDAMP had 4) because of the extra Signal layer between
Silver and Gold. Each phase instantiates a fresh Supervisor ReAct agent with
phase-scoped tools built via closures, using the same scratchpad pattern as
IDAMP so file paths are never round-tripped through the LLM: a specialist
agent writes its output path into `scratchpad`, and the next tool reads it
straight back out.

Each phase's two steps are also exposed as plain idempotent functions
(`step_fns`), not just as tools. After the Supervisor agent runs, the phase
driver calls both directly as a deterministic safety net: an LLM-driven
ReAct loop can decide a task is "done" after only the first tool call (we
saw this happen — the agent ingested Bronze but never called the STTM tool
for Silver, leaving `sttm_silver_path` None downstream). Each step function
is a no-op if its scratchpad key is already set, so this costs nothing when
the agent behaved correctly and only kicks in as a repair when it didn't.

Every phase also runs through `_execute_phase`, which catches ANY exception
raised while the phase runs (LLM error, malformed STTM, a pandas join
failure deep in a specialist agent, ...) and turns it into
`state.error` + `state.status` reverted to the gate the user just approved
from, instead of an unhandled traceback crashing the whole Streamlit app.
The UI re-shows that same STTM editor with the error message on top, so the
user can just retry, or edit the STTM and retry. Phase 1 has no prior gate
to fall back to, so it reverts to "upload_error" instead.

Phase map:
  1. run_until_bronze_sttm      -> profile landing files, generate Bronze STTM   [HITL gate]
  2. run_bronze_to_silver_sttm  -> ingest Bronze, generate Silver STTM           [HITL gate]
  3. run_silver_to_signal_sttm  -> cleanse Silver, generate Signal STTM          [HITL gate]  (NEW layer)
  4. run_signal_to_gold_sttm    -> derive Signal features, generate Gold STTM    [HITL gate]
  5. run_gold_and_report        -> materialise Gold, generate the report
"""
from __future__ import annotations

import uuid

from langchain_core.tools import tool

from agents.bronze_agent import run_bronze_agent
from agents.enrichment_agent import run_signal_agent
from agents.gold_agent import run_gold_agent
from agents.profiler import run_profiler_agent
from agents.reporter import run_reporter_agent
from agents.silver_agent import run_silver_agent
from agents.sttm_generator import run_sttm_stage
from core.agent_factory import build_agent, run_agent
from core.audit import log_event
from core.memory import remember
from core.preview import preview_parquet_files
from core.state import PipelineState

SUPERVISOR_PROMPT = """You are the Supervisor in RADAR, an agentic returns/revenue analytics
pipeline. For this phase you have exactly two tools. THINK about the goal, PLAN the order
(the first tool must run before the second), ACT by calling each tool exactly once, then
VERIFY both succeeded and reply with a one-line CONFIRM summary. Do not call a tool more
than once unless it returned an error. Both tools MUST be called before you finish — do not
stop after only the first one."""


def _new_run_id() -> str:
    return f"run_{uuid.uuid4().hex[:10]}"


def _run_phase(run_id: str, scratchpad: dict, step_fns: list, tools: list, goal: str, phase_name: str) -> None:
    agent = build_agent(SUPERVISOR_PROMPT, tools)
    run_agent(agent, goal)

    for step_key, step_fn in step_fns:
        if step_key not in scratchpad:
            log_event(run_id, "supervisor", f"{phase_name}_fallback_triggered", missing_step=step_key)
            step_fn()


def _execute_phase(state: PipelineState, phase_name: str, fallback_status: str, body) -> PipelineState:
    """Run `body()` (which mutates `state` in place, ending by setting its
    success status). Any exception is caught, recorded on `state.error`, and
    `state.status` is reverted to `fallback_status` so the UI can show the
    error next to a retry point instead of crashing."""
    state.error = None
    try:
        body()
    except Exception as exc:
        state.error = str(exc)
        state.status = fallback_status
        log_event(state.run_id, "supervisor", f"{phase_name}_failed", error=str(exc))
    return state


# ---------------------------------------------------------------------------
# Phase 1: profile + Bronze STTM
# ---------------------------------------------------------------------------
def _make_phase1_tools(uploaded_files: list[str], run_id: str, scratchpad: dict):
    def do_profile() -> str:
        if "profile_path" not in scratchpad:
            scratchpad["profile_path"] = run_profiler_agent(uploaded_files, run_id)
        return scratchpad["profile_path"]

    def do_bronze_sttm() -> str:
        if "sttm_bronze_path" not in scratchpad:
            with open(do_profile(), "r", encoding="utf-8") as f:
                context = f.read()
            scratchpad["sttm_bronze_path"] = run_sttm_stage("bronze", context, run_id)
        return scratchpad["sttm_bronze_path"]

    @tool
    def profiler_agent_tool() -> str:
        """Run the Profiler agent over the uploaded landing files (CSV/JSON/XLSX)."""
        return f"Profile saved to {do_profile()}"

    @tool
    def sttm_agent_tool() -> str:
        """Run the STTM agent to generate Bronze ingestion rules from the saved profile."""
        return f"Bronze STTM saved to {do_bronze_sttm()}"

    tools = [profiler_agent_tool, sttm_agent_tool]
    step_fns = [("profile_path", do_profile), ("sttm_bronze_path", do_bronze_sttm)]
    return tools, step_fns


def run_until_bronze_sttm(uploaded_files: list[str], business_intent: str) -> PipelineState:
    run_id = _new_run_id()
    state = PipelineState(run_id=run_id, uploaded_files=uploaded_files, business_intent=business_intent, status="phase1_running")

    def body():
        remember(f"{run_id}_intent", business_intent, {"run_id": run_id, "type": "business_intent"})
        log_event(run_id, "supervisor", "phase1_start", files=uploaded_files, intent=business_intent)

        scratchpad: dict = {}
        tools, step_fns = _make_phase1_tools(uploaded_files, run_id, scratchpad)
        goal = f"Business intent: {business_intent!r}. Files: {uploaded_files}. Profile them, then generate the Bronze STTM."
        _run_phase(run_id, scratchpad, step_fns, tools, goal, "phase1")

        state.profile_path = scratchpad.get("profile_path")
        state.sttm_bronze_path = scratchpad.get("sttm_bronze_path")
        state.status = "awaiting_bronze_approval"
        log_event(run_id, "supervisor", "phase1_complete", state=state.model_dump())

    return _execute_phase(state, "phase1", "upload_error", body)


# ---------------------------------------------------------------------------
# Phase 2: Bronze execution + Silver STTM
# ---------------------------------------------------------------------------
def _make_phase2_tools(state: PipelineState, scratchpad: dict):
    def do_bronze() -> list[str]:
        if "bronze_output_paths" not in scratchpad:
            scratchpad["bronze_output_paths"] = run_bronze_agent(state.uploaded_files, state.sttm_bronze_path, state.run_id)
        return scratchpad["bronze_output_paths"]

    def do_silver_sttm() -> str:
        if "sttm_silver_path" not in scratchpad:
            context = preview_parquet_files(do_bronze())
            scratchpad["sttm_silver_path"] = run_sttm_stage("silver", context, state.run_id)
        return scratchpad["sttm_silver_path"]

    @tool
    def bronze_agent_tool() -> str:
        """Run the Bronze agent to ingest landing files into Bronze Parquet."""
        return f"Bronze Parquet written: {do_bronze()}"

    @tool
    def sttm_agent_tool() -> str:
        """Run the STTM agent to generate Silver cleansing rules from Bronze schemas."""
        return f"Silver STTM saved to {do_silver_sttm()}"

    tools = [bronze_agent_tool, sttm_agent_tool]
    step_fns = [("bronze_output_paths", do_bronze), ("sttm_silver_path", do_silver_sttm)]
    return tools, step_fns


def run_bronze_to_silver_sttm(state: PipelineState) -> PipelineState:
    def body():
        state.status = "phase2_running"
        log_event(state.run_id, "supervisor", "phase2_start")

        scratchpad: dict = {}
        tools, step_fns = _make_phase2_tools(state, scratchpad)
        goal = "Ingest the approved Bronze STTM into Parquet, then generate the Silver STTM."
        _run_phase(state.run_id, scratchpad, step_fns, tools, goal, "phase2")

        state.bronze_output_paths = scratchpad.get("bronze_output_paths", [])
        state.sttm_silver_path = scratchpad.get("sttm_silver_path")
        state.status = "awaiting_silver_approval"
        log_event(state.run_id, "supervisor", "phase2_complete", state=state.model_dump())

    return _execute_phase(state, "phase2", "awaiting_bronze_approval", body)


# ---------------------------------------------------------------------------
# Phase 3: Silver execution + Signal STTM   (NEW layer vs. IDAMP)
# ---------------------------------------------------------------------------
def _make_phase3_tools(state: PipelineState, scratchpad: dict):
    def do_silver() -> list[str]:
        if "silver_output_paths" not in scratchpad:
            scratchpad["silver_output_paths"] = run_silver_agent(state.bronze_output_paths, state.sttm_silver_path, state.run_id)
        return scratchpad["silver_output_paths"]

    def do_signal_sttm() -> str:
        if "sttm_signal_path" not in scratchpad:
            context = preview_parquet_files(do_silver())
            scratchpad["sttm_signal_path"] = run_sttm_stage("signal", context, state.run_id)
        return scratchpad["sttm_signal_path"]

    @tool
    def silver_agent_tool() -> str:
        """Run the Silver agent to cleanse Bronze Parquet into Silver Parquet."""
        return f"Silver Parquet written: {do_silver()}"

    @tool
    def sttm_agent_tool() -> str:
        """Run the STTM agent to generate Signal (feature-engineering) rules from Silver schemas."""
        return f"Signal STTM saved to {do_signal_sttm()}"

    tools = [silver_agent_tool, sttm_agent_tool]
    step_fns = [("silver_output_paths", do_silver), ("sttm_signal_path", do_signal_sttm)]
    return tools, step_fns


def run_silver_to_signal_sttm(state: PipelineState) -> PipelineState:
    def body():
        state.status = "phase3_running"
        log_event(state.run_id, "supervisor", "phase3_start")

        scratchpad: dict = {}
        tools, step_fns = _make_phase3_tools(state, scratchpad)
        goal = "Cleanse the approved Silver STTM into Parquet, then generate the Signal STTM."
        _run_phase(state.run_id, scratchpad, step_fns, tools, goal, "phase3")

        state.silver_output_paths = scratchpad.get("silver_output_paths", [])
        state.sttm_signal_path = scratchpad.get("sttm_signal_path")
        state.status = "awaiting_signal_approval"
        log_event(state.run_id, "supervisor", "phase3_complete", state=state.model_dump())

    return _execute_phase(state, "phase3", "awaiting_silver_approval", body)


# ---------------------------------------------------------------------------
# Phase 4: Signal execution + Gold STTM
# ---------------------------------------------------------------------------
def _make_phase4_tools(state: PipelineState, scratchpad: dict):
    def do_signal() -> list[str]:
        if "signal_output_paths" not in scratchpad:
            scratchpad["signal_output_paths"] = run_signal_agent(state.silver_output_paths, state.sttm_signal_path, state.run_id)
        return scratchpad["signal_output_paths"]

    def do_gold_sttm() -> str:
        if "sttm_gold_path" not in scratchpad:
            context = preview_parquet_files(do_signal())
            scratchpad["sttm_gold_path"] = run_sttm_stage("gold", context, state.run_id)
        return scratchpad["sttm_gold_path"]

    @tool
    def signal_agent_tool() -> str:
        """Run the Signal agent to derive the order_line_signals feature table."""
        return f"Signal Parquet written: {do_signal()}"

    @tool
    def sttm_agent_tool() -> str:
        """Run the STTM agent to generate Gold aggregation rules from the Signal schema."""
        return f"Gold STTM saved to {do_gold_sttm()}"

    tools = [signal_agent_tool, sttm_agent_tool]
    step_fns = [("signal_output_paths", do_signal), ("sttm_gold_path", do_gold_sttm)]
    return tools, step_fns


def run_signal_to_gold_sttm(state: PipelineState) -> PipelineState:
    def body():
        state.status = "phase4_running"
        log_event(state.run_id, "supervisor", "phase4_start")

        scratchpad: dict = {}
        tools, step_fns = _make_phase4_tools(state, scratchpad)
        goal = "Derive the approved Signal STTM into Parquet, then generate the Gold STTM."
        _run_phase(state.run_id, scratchpad, step_fns, tools, goal, "phase4")

        state.signal_output_paths = scratchpad.get("signal_output_paths", [])
        state.sttm_gold_path = scratchpad.get("sttm_gold_path")
        state.status = "awaiting_gold_approval"
        log_event(state.run_id, "supervisor", "phase4_complete", state=state.model_dump())

    return _execute_phase(state, "phase4", "awaiting_signal_approval", body)


# ---------------------------------------------------------------------------
# Phase 5: Gold execution + report (with forecast)
# ---------------------------------------------------------------------------
def _make_phase5_tools(state: PipelineState, scratchpad: dict):
    def do_gold() -> list[str]:
        if "gold_output_paths" not in scratchpad:
            scratchpad["gold_output_paths"] = run_gold_agent(state.signal_output_paths, state.sttm_gold_path, state.run_id)
        return scratchpad["gold_output_paths"]

    def do_report():
        if "report_path" not in scratchpad:
            report_path, report_json_path = run_reporter_agent(do_gold(), state.run_id)
            scratchpad["report_path"] = report_path
            scratchpad["report_json_path"] = report_json_path
        return scratchpad["report_path"]

    @tool
    def gold_agent_tool() -> str:
        """Run the Gold agent to materialise category/trend analytics tables."""
        return f"Gold Parquet written: {do_gold()}"

    @tool
    def reporter_agent_tool() -> str:
        """Run the Reporter agent to answer the business question and build the HTML report."""
        return f"Report saved to {do_report()}"

    tools = [gold_agent_tool, reporter_agent_tool]
    step_fns = [("gold_output_paths", do_gold), ("report_path", do_report)]
    return tools, step_fns


def run_gold_and_report(state: PipelineState) -> PipelineState:
    def body():
        state.status = "phase5_running"
        log_event(state.run_id, "supervisor", "phase5_start")

        scratchpad: dict = {}
        tools, step_fns = _make_phase5_tools(state, scratchpad)
        goal = "Materialise the approved Gold STTM into Parquet, then generate the executive report."
        _run_phase(state.run_id, scratchpad, step_fns, tools, goal, "phase5")

        state.gold_output_paths = scratchpad.get("gold_output_paths", [])
        state.report_path = scratchpad.get("report_path")
        state.report_json_path = scratchpad.get("report_json_path")
        state.status = "complete"
        remember(f"{state.run_id}_report", state.report_path or "", {"run_id": state.run_id, "type": "report"})
        log_event(state.run_id, "supervisor", "phase5_complete", state=state.model_dump())

    return _execute_phase(state, "phase5", "awaiting_gold_approval", body)
