"""Regression test for the "KeyError: 'order_id'" bug: the Silver STTM the LLM
generated dropped the order_id/product_id passthrough rows entirely, reasoning
(incorrectly) that the new pk_orders_silver_id surrogate key made them
redundant. Confirmed directly from the production run's Silver Parquet output
-- orders_silver.parquet had no order_id column at all -- which then crashed
the Signal layer's join with a bare pandas KeyError.

Two layers of fix:
1. sttm_generator explicitly instructs against dropping *_id columns, and
   auto-re-adds any that are missing before the STTM is saved.
2. enrichment_agent resolves join columns independently per side and raises a
   clear, actionable error if a *_id column is genuinely still missing on
   either side, instead of merge() raising a bare KeyError.
"""
from __future__ import annotations

import json

import pandas as pd
import pytest

from agents.enrichment_agent import _resolve_join_columns
from agents.sttm_generator import _columns_by_table_from_parquet_context, _ensure_id_columns_preserved


def test_columns_by_table_excludes_bronze_metadata():
    context = json.dumps({
        "orders_bronze.parquet": {"columns": ["order_id", "order_date", "_load_timestamp", "_source_file"]},
    })
    result = _columns_by_table_from_parquet_context(context, "_bronze")
    assert result == {"orders": ["order_id", "order_date"]}


def test_ensure_id_columns_preserved_adds_missing_order_id():
    # The LLM's rows never mention order_id -- exactly what happened in production.
    rows = [
        {"source_table": "orders", "source_column": "order_date", "target_table": "orders",
         "target_column": "order_date", "transformation_type": "date_standardize"},
    ]
    columns_by_table = {"orders": ["order_id", "order_date", "customer_id"]}

    result = _ensure_id_columns_preserved(rows, columns_by_table)

    order_id_rows = [r for r in result if r.get("source_column") == "order_id"]
    assert len(order_id_rows) == 1
    assert order_id_rows[0]["target_table"] == "orders"
    assert order_id_rows[0]["target_column"] == "order_id"
    assert order_id_rows[0]["transformation_type"] == "passthrough"
    # customer_id was never referenced either -- also must be auto-added.
    assert any(r.get("source_column") == "customer_id" for r in result)


def test_ensure_id_columns_preserved_does_not_duplicate_existing_row():
    rows = [
        {"source_table": "orders", "source_column": "order_id", "target_table": "orders",
         "target_column": "order_id", "transformation_type": "passthrough"},
    ]
    columns_by_table = {"orders": ["order_id"]}
    result = _ensure_id_columns_preserved(rows, columns_by_table)
    assert len(result) == 1


def test_resolve_join_columns_finds_exact_match():
    orders = pd.DataFrame({"order_id": [1], "order_date": ["2025-01-01"]})
    returns = pd.DataFrame({"order_id": [1], "return_date": ["2025-01-05"]})
    left, right = _resolve_join_columns(orders, returns, "order_id")
    assert (left, right) == ("order_id", "order_id")


def test_resolve_join_columns_raises_clear_error_when_missing_on_one_side():
    # Reproduces the exact production bug: orders_silver has no order_id-like column.
    orders = pd.DataFrame({"pk_orders_silver_id": [1], "order_date": ["2025-01-01"], "order_amount": [10.0]})
    returns = pd.DataFrame({"order_id": [1], "return_date": ["2025-01-05"]})
    with pytest.raises(ValueError, match="Silver STTM dropped"):
        _resolve_join_columns(orders, returns, "order_id")
