"""Regression coverage for end-to-end error handling: previously, any
exception raised deep inside a specialist agent (a bad join, a malformed
STTM, an LLM hiccup) propagated all the way up through Streamlit and crashed
the whole app with a raw traceback. agents/orchestrator.py's _execute_phase
now catches it, records it on PipelineState.error, and reverts
PipelineState.status to the gate the user just approved from, so the UI can
show the error and let them retry instead of crashing.
"""
from __future__ import annotations

import agents.orchestrator as orchestrator
from core.state import PipelineState


def test_execute_phase_catches_exception_and_reverts_status():
    state = PipelineState(run_id="run_err", status="phase2_running")

    def failing_body():
        state.status = "phase2_running"
        raise ValueError("boom: Silver STTM dropped order_id")

    result = orchestrator._execute_phase(state, "phase2", "awaiting_bronze_approval", failing_body)

    assert result is state
    assert result.status == "awaiting_bronze_approval"
    assert "boom: Silver STTM dropped order_id" in result.error


def test_execute_phase_clears_previous_error_on_new_attempt():
    state = PipelineState(run_id="run_err2", status="awaiting_bronze_approval", error="stale error from last attempt")

    def succeeding_body():
        state.status = "awaiting_silver_approval"

    result = orchestrator._execute_phase(state, "phase2", "awaiting_bronze_approval", succeeding_body)

    assert result.error is None
    assert result.status == "awaiting_silver_approval"


def test_execute_phase_does_not_raise_to_caller():
    state = PipelineState(run_id="run_err3", status="phase3_running")

    def failing_body():
        raise RuntimeError("some deep pandas failure")

    # Must not raise -- this is the whole point of the fix.
    result = orchestrator._execute_phase(state, "phase3", "awaiting_silver_approval", failing_body)
    assert result.status == "awaiting_silver_approval"
    assert "some deep pandas failure" in result.error
