"""Regression test for the exact production bug: the Silver STTM used
transformation_type="dedup" directly on the order_id/product_id rows (meaning
"this is the dedup key"), but _apply_silver_rules treated any "dedup" row as
pure metadata and skipped copying that column into the output entirely --
silently dropping order_id and breaking the Signal layer's join two steps
later with a `KeyError`/`ValueError`.
"""
from __future__ import annotations

import pandas as pd

from agents.silver_agent import _apply_silver_rules
from agents.sttm_generator import STTM_COLUMNS


def _write_sttm(tmp_path, rows):
    df = pd.DataFrame(rows)
    for col in STTM_COLUMNS:
        if col not in df.columns:
            df[col] = ""
    df = df[STTM_COLUMNS]
    df.insert(0, "approved", True)
    path = tmp_path / "sttm_silver.csv"
    df.to_csv(path, index=False)
    return str(path)


def test_dedup_transformation_type_still_copies_the_column(tmp_path):
    bronze_df = pd.DataFrame({
        "order_id": ["O1", "O2", "O2"],
        "order_date": ["2025-01-01", "2025-01-02", "2025-01-02"],
    })
    bronze_path = tmp_path / "orders_bronze.parquet"
    bronze_df.to_parquet(bronze_path, index=False)

    rows = [
        {"source_schema": "bronze", "source_table": "orders", "source_column": "",
         "target_schema": "silver", "target_table": "orders", "target_column": "pk_orders_silver_id",
         "transformation_type": "surrogate_key", "transformation_logic": "sequential integer id"},
        # Exactly what production generated: dedup as the transformation_type for order_id itself.
        {"source_schema": "bronze", "source_table": "orders", "source_column": "order_id",
         "target_schema": "silver", "target_table": "orders", "target_column": "order_id",
         "transformation_type": "dedup", "transformation_logic": "dedup key: order_id"},
        {"source_schema": "bronze", "source_table": "orders", "source_column": "order_date",
         "target_schema": "silver", "target_table": "orders", "target_column": "order_date",
         "transformation_type": "date_standardize", "transformation_logic": "parse to YYYY-MM-DD"},
    ]
    sttm_path = _write_sttm(tmp_path, rows)

    output_paths = _apply_silver_rules([str(bronze_path)], sttm_path, "test_run_dedup")
    result = pd.read_parquet(output_paths[0])

    assert "order_id" in result.columns
    assert set(result["order_id"]) == {"O1", "O2"}  # dedup key present AND duplicates removed
    assert len(result) == 2
