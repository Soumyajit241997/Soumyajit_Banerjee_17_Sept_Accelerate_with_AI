"""Regression test for the bug reported in production: the Supervisor's ReAct
agent called only the first tool of a phase (e.g. bronze_agent_tool) and
stopped, leaving the second step's scratchpad key (e.g. sttm_silver_path)
unset -- which crashed the Streamlit UI with `pd.read_csv(None)`.

agents/orchestrator.py now exposes each phase's two steps as idempotent
functions (`step_fns`) that the phase driver calls directly as a
deterministic safety net after the agent runs, regardless of whether the
agent's tool calls actually populated the scratchpad. This test drives those
step functions directly -- no LLM involved -- and checks they are correct
and idempotent (calling twice does not re-run the underlying work).
"""
from __future__ import annotations

import agents.orchestrator as orchestrator


def test_phase1_step_fns_populate_scratchpad_and_are_idempotent(tmp_path, monkeypatch):
    calls = {"profile": 0, "sttm": 0}
    profile_path = tmp_path / "profile.json"
    profile_path.write_text('{"datasets": {}}', encoding="utf-8")

    def fake_profiler(files, run_id):
        calls["profile"] += 1
        return str(profile_path)

    def fake_sttm_stage(layer, context, run_id):
        calls["sttm"] += 1
        assert layer == "bronze"
        assert context == '{"datasets": {}}'
        return str(tmp_path / "sttm_bronze.csv")

    monkeypatch.setattr(orchestrator, "run_profiler_agent", fake_profiler)
    monkeypatch.setattr(orchestrator, "run_sttm_stage", fake_sttm_stage)

    scratchpad: dict = {}
    _, step_fns = orchestrator._make_phase1_tools(["orders.csv"], "run_x", scratchpad)

    # Simulate the agent never calling sttm_agent_tool -- only the first step ran.
    step_fns[0][1]()
    assert scratchpad["profile_path"] == str(profile_path)
    assert "sttm_bronze_path" not in scratchpad

    # The deterministic fallback loop (mirroring _run_phase) must still complete it.
    for step_key, step_fn in step_fns:
        if step_key not in scratchpad:
            step_fn()

    assert scratchpad["sttm_bronze_path"] == str(tmp_path / "sttm_bronze.csv")
    assert calls == {"profile": 1, "sttm": 1}

    # Idempotency: calling both again must not re-invoke the underlying agents.
    for _, step_fn in step_fns:
        step_fn()
    assert calls == {"profile": 1, "sttm": 1}
