"""Regression tests for the "Signal layer requires a Silver 'orders' table"
bug: an LLM-authored STTM used a target_table name (e.g. "order_transactions")
that didn't exactly match the uploaded file's stem ("orders"), so Bronze
silently produced zero rows for that table and it never existed downstream.

Two layers of fix are tested here:
1. sttm_generator normalizes LLM-chosen table names to the canonical,
   file-derived name before the STTM is even saved for human approval.
2. bronze_agent / silver_agent / enrichment_agent fall back to fuzzy matching
   at execution time, in case a hand-edited STTM still doesn't match exactly.
"""
from __future__ import annotations

import json

import pandas as pd
import pytest

from agents.bronze_agent import _apply_bronze_rules
from agents.enrichment_agent import _get_table
from agents.sttm_generator import (
    STTM_COLUMNS,
    _canonical_tables_from_profile,
    _normalize_table_names,
)
from core.config import SAMPLE_DATA_DIR


def test_canonical_tables_from_profile_extracts_stems():
    profile_context = json.dumps({"datasets": {"orders.csv": {}, "returns.json": {}, "products.xlsx": {}}})
    assert _canonical_tables_from_profile(profile_context) == {
        "orders.csv": "orders", "returns.json": "returns", "products.xlsx": "products",
    }


def test_normalize_table_names_snaps_llm_invented_name_to_canonical():
    rows = [{"source_table": "order_transactions", "target_table": "order_transactions", "target_column": "order_id"}]
    normalized = _normalize_table_names(rows, ["orders", "returns", "products"])
    assert normalized[0]["source_table"] == "orders"
    assert normalized[0]["target_table"] == "orders"


def test_bronze_agent_tolerates_mismatched_target_table_name(tmp_path):
    """Even if the (hand-edited) STTM's target_table doesn't exactly match the
    uploaded file's name, Bronze should still find and ingest it via fuzzy match
    instead of silently producing zero rows."""
    rows = [
        {"source_schema": "landing", "source_table": "orders", "source_column": "order_id",
         "target_schema": "bronze", "target_table": "order_transactions",  # <- mismatched on purpose
         "target_column": "order_id", "transformation_type": "indirect", "transformation_logic": "text"},
    ]
    df = pd.DataFrame(rows)
    for col in STTM_COLUMNS:
        if col not in df.columns:
            df[col] = ""
    df = df[STTM_COLUMNS]
    df.insert(0, "approved", True)
    sttm_path = tmp_path / "sttm_bronze.csv"
    df.to_csv(sttm_path, index=False)

    output_paths = _apply_bronze_rules([str(SAMPLE_DATA_DIR / "orders.csv")], str(sttm_path), "test_run_naming")
    assert len(output_paths) == 1
    result = pd.read_parquet(output_paths[0])
    assert len(result) > 0
    assert "order_id" in result.columns


def test_signal_layer_get_table_resolves_renamed_silver_table():
    tables = {
        "order_transactions": pd.DataFrame({"order_id": [1, 2]}),
        "return_events": pd.DataFrame({"return_id": [1]}),
        "product_catalog": pd.DataFrame({"product_id": [1]}),
    }
    assert _get_table(tables, "orders", "order") is tables["order_transactions"]
    assert _get_table(tables, "returns", "return") is tables["return_events"]
    assert _get_table(tables, "products", "product") is tables["product_catalog"]


def test_signal_layer_get_table_returns_none_when_nothing_matches():
    tables = {"inventory": pd.DataFrame({"x": [1]})}
    assert _get_table(tables, "orders", "order") is None
