"""RADAR Streamlit UI.

Same shape as IDAMP's UI (upload -> STTM review gates -> report) with one
extra gate for the Signal layer, so there are 6 phases instead of 5:
Upload -> Bronze STTM -> Silver STTM -> Signal STTM -> Gold STTM -> Report.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import streamlit as st

from agents.orchestrator import (
    run_bronze_to_silver_sttm,
    run_gold_and_report,
    run_signal_to_gold_sttm,
    run_silver_to_signal_sttm,
    run_until_bronze_sttm,
)
from agents.reporter import answer_question
from core.audit import get_logs
from core.config import LANDING_DIR

st.set_page_config(page_title="RADAR — Returns & Revenue Analytics", layout="wide")

STEPS = ["Upload", "Bronze STTM", "Silver STTM", "Signal STTM", "Gold STTM", "Report"]
STATUS_TO_STEP = {
    "created": 0, "phase1_running": 0, "awaiting_bronze_approval": 1,
    "phase2_running": 1, "awaiting_silver_approval": 2,
    "phase3_running": 2, "awaiting_signal_approval": 3,
    "phase4_running": 3, "awaiting_gold_approval": 4,
    "phase5_running": 4, "complete": 5,
}

EXAMPLE_INTENT = (
    "e.g. Identify which product categories have return rates that are quietly eroding "
    "net revenue, so the business can address the underlying causes."
)
DEFAULT_INTENT = (
    "Identify which product categories have return rates that are quietly eroding net "
    "revenue, so the business can address the underlying causes."
)


def render_stepper(current_index: int) -> None:
    cols = st.columns(len(STEPS))
    for i, (col, label) in enumerate(zip(cols, STEPS)):
        marker = "✅" if i < current_index else ("🟡" if i == current_index else "⚪")
        col.markdown(f"**{marker} {label}**")


def save_uploads(uploaded) -> list[str]:
    paths = []
    for file in uploaded:
        dest = LANDING_DIR / file.name
        with open(dest, "wb") as f:
            f.write(file.getbuffer())
        paths.append(str(dest))
    return paths


def sttm_gate(title: str, sttm_path: str, session_key: str, on_approve, error: str | None = None):
    st.subheader(title)
    if error:
        st.error(f"The last attempt failed: {error}\n\nYou can edit the rules below and try "
                  "again, or just click Approve again to retry as-is.")
    try:
        df = pd.read_csv(sttm_path)
    except Exception as exc:
        st.error(f"Could not read the {title} file at {sttm_path}: {exc}")
        if st.button("Start a new analysis", key=f"reset_{session_key}"):
            st.session_state.state = None
            st.rerun()
        return

    edited = st.data_editor(df, num_rows="dynamic", use_container_width=True, key=f"editor_{session_key}")
    if st.button(f"Approve {title} & Continue", type="primary", key=f"approve_{session_key}"):
        try:
            edited.to_csv(sttm_path, index=False)
            with st.spinner(f"Running agents past {title}..."):
                new_state = on_approve()
            st.session_state.state = new_state
        except Exception as exc:
            # Last-resort safety net: agents/orchestrator.py already catches phase
            # failures and returns them via state.error, but this guards against
            # anything raised outside that (e.g. a bug in the UI glue itself).
            st.session_state.state.error = str(exc)
        st.rerun()


PLOTLY_CDN = '<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>'


def render_qa_section(state) -> None:
    st.subheader("Ask a follow-up question")
    st.caption(
        "Not limited to the original business question — ask anything about the "
        "category/return/revenue data (e.g. \"Which channel has the worst return rate for "
        "Electronics?\" or \"How does Apparel's return rate compare to last month?\")."
    )
    question = st.text_input("Your question", key="qa_input", placeholder="Ask anything about this data...")
    if st.button("Ask", key="qa_button") and question.strip():
        try:
            with st.spinner("Thinking..."):
                result = answer_question(state.gold_output_paths, question.strip(), state.run_id)
            st.session_state.setdefault("qa_history", []).append(result)
        except Exception as exc:
            st.error(f"Could not answer that question: {exc}")

    for entry in reversed(st.session_state.get("qa_history", [])):
        st.markdown(f"**Q: {entry['question']}**")
        st.write(entry["answer"])
        if entry.get("sql"):
            st.code(entry["sql"], language="sql")
        if entry.get("chart_html"):
            st.components.v1.html(PLOTLY_CDN + entry["chart_html"], height=420)
        st.markdown("---")


def render_audit_trail(run_id: str) -> None:
    st.subheader("Audit Trail")
    logs = get_logs(run_id)
    if not logs:
        st.caption("No events yet.")
        return
    for entry in reversed(logs[-50:]):
        st.markdown(f"**{entry['agent']}** · `{entry['action']}` · {entry['timestamp']}")


def main():
    st.title("RADAR — Returns Analytics via Data-driven Agentic Reasoning")
    st.caption(
        "Upload orders / returns / product files in any mix of CSV, JSON, and Excel. "
        "RADAR's agents profile, cleanse, derive return/revenue signals, aggregate, and "
        "report — with human approval at every Medallion transition."
    )

    if "state" not in st.session_state:
        st.session_state.state = None

    state = st.session_state.state
    current_step = STATUS_TO_STEP.get(state.status, 0) if state else 0
    render_stepper(current_step)
    st.divider()

    main_col, audit_col = st.columns([3, 1])

    with main_col:
        if state is None or state.status == "upload_error":
            st.subheader("1. Upload data & describe the business question")
            if state is not None and state.status == "upload_error":
                st.error(f"The last attempt failed before Bronze STTM could be generated: {state.error}\n\n"
                          "Check your files and try again.")
            uploaded = st.file_uploader(
                "Landing files (CSV, JSON, XLSX — mix and match)",
                type=["csv", "json", "xlsx", "xls"],
                accept_multiple_files=True,
            )
            intent = st.text_area(
                "Business question — ask anything about the data, this isn't fixed to one question",
                value="", height=100, placeholder=EXAMPLE_INTENT,
            )
            if st.button("Start Analysis", type="primary", disabled=not uploaded):
                try:
                    paths = save_uploads(uploaded)
                    with st.spinner("Profiling files and generating Bronze STTM..."):
                        new_state = run_until_bronze_sttm(paths, intent.strip() or DEFAULT_INTENT)
                    st.session_state.state = new_state
                except Exception as exc:
                    st.session_state.state = None
                    st.error(f"Could not start the analysis: {exc}")
                st.rerun()

        elif state.status == "awaiting_bronze_approval":
            sttm_gate("Bronze STTM", state.sttm_bronze_path, "bronze",
                      lambda: run_bronze_to_silver_sttm(state), error=state.error)

        elif state.status == "awaiting_silver_approval":
            sttm_gate("Silver STTM", state.sttm_silver_path, "silver",
                      lambda: run_silver_to_signal_sttm(state), error=state.error)

        elif state.status == "awaiting_signal_approval":
            st.info("Signal layer: joins orders + returns + products and derives is_returned, "
                    "days_to_return, net_revenue, and revenue_erosion. This layer doesn't exist "
                    "in a plain Bronze/Silver/Gold pipeline — it's RADAR's answer to needing "
                    "order-line-grain features before category aggregation.")
            sttm_gate("Signal STTM", state.sttm_signal_path, "signal",
                      lambda: run_signal_to_gold_sttm(state), error=state.error)

        elif state.status == "awaiting_gold_approval":
            sttm_gate("Gold STTM", state.sttm_gold_path, "gold",
                      lambda: run_gold_and_report(state), error=state.error)

        elif state.status == "complete":
            st.success("Pipeline complete.")
            with open(state.report_path, "r", encoding="utf-8") as f:
                html = f.read()
            st.components.v1.html(html, height=1400, scrolling=True)

            col_a, col_b = st.columns(2)
            with col_a:
                st.download_button("Download report (HTML)", data=html, file_name=Path(state.report_path).name)
            with col_b:
                with open(state.report_json_path, "r", encoding="utf-8") as f:
                    st.download_button("Download analysis (JSON)", data=f.read(), file_name=Path(state.report_json_path).name)

            st.divider()
            render_qa_section(state)

            if st.button("Start a new analysis"):
                st.session_state.state = None
                st.session_state.qa_history = []
                st.rerun()

    with audit_col:
        if state:
            if state.status != "complete" and st.button("Start over", key="reset_sidebar"):
                st.session_state.state = None
                st.session_state.qa_history = []
                st.rerun()
            render_audit_trail(state.run_id)


if __name__ == "__main__":
    main()
