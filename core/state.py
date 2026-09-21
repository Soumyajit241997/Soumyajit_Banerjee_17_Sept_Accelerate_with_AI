"""Pipeline state model shared between the orchestrator and the Streamlit UI.

One structural change from IDAMP: an extra STTM/output pair for the Signal
layer (sttm_signal_path / signal_output_paths) that sits between Silver and
Gold.
"""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class PipelineState(BaseModel):
    run_id: str
    status: str = "created"
    error: Optional[str] = None

    # Inputs
    uploaded_files: list[str] = Field(default_factory=list)
    business_intent: str = ""

    # Phase 1: profile + bronze STTM
    profile_path: Optional[str] = None
    sttm_bronze_path: Optional[str] = None

    # Phase 2: bronze execution + silver STTM
    bronze_output_paths: list[str] = Field(default_factory=list)
    sttm_silver_path: Optional[str] = None

    # Phase 3: silver execution + signal STTM  (NEW layer vs. IDAMP)
    silver_output_paths: list[str] = Field(default_factory=list)
    sttm_signal_path: Optional[str] = None

    # Phase 4: signal execution + gold STTM
    signal_output_paths: list[str] = Field(default_factory=list)
    sttm_gold_path: Optional[str] = None

    # Phase 5: gold execution + report
    gold_output_paths: list[str] = Field(default_factory=list)
    report_path: Optional[str] = None
    report_json_path: Optional[str] = None
