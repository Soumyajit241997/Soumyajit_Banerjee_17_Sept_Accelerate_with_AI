"""End-to-end test of the deterministic transformation core (Bronze -> Silver
-> Signal -> Gold) against the real generated sample data, using a hand-authored
STTM instead of an LLM so this suite runs without any API key.

This exercises the actual differentiator of this build: the Signal layer join
+ derive logic, and the two Gold aggregates the Reporter's forecast depends on.
"""
from __future__ import annotations

import pandas as pd
import pytest

from agents.bronze_agent import _apply_bronze_rules
from agents.enrichment_agent import _apply_signal_rules
from agents.gold_agent import _apply_gold_rules
from agents.silver_agent import _apply_silver_rules
from agents.sttm_generator import STTM_COLUMNS
from core.config import SAMPLE_DATA_DIR

RUN_ID = "test_run_integration"


def _write_sttm(tmp_path, rows, name):
    df = pd.DataFrame(rows)
    for col in STTM_COLUMNS:
        if col not in df.columns:
            df[col] = ""
    df.insert(0, "approved", True)
    path = tmp_path / name
    df.to_csv(path, index=False)
    return str(path)


def _bronze_row(table, source_col, target_col, ttype="indirect", logic="text"):
    return {
        "source_schema": "landing", "source_table": table, "source_column": source_col,
        "target_schema": "bronze", "target_table": table, "target_column": target_col,
        "transformation_type": ttype, "transformation_logic": logic,
    }


ORDERS_COLS = {
    "order_id": "text", "order_date": "datetime", "customer_id": "text", "product_id": "text",
    "quantity": "integer", "unit_price": "float", "order_amount": "float", "channel": "text",
}
RETURNS_COLS = {
    "return_id": "text", "order_id": "text", "return_date": "datetime", "return_reason": "text",
    "refund_amount": "float", "condition": "text",
}
PRODUCTS_COLS = {
    "product_id": "text", "category": "text", "sub_category": "text", "product_name": "text",
    "cost_price": "float", "supplier": "text", "launch_date": "datetime",
}


@pytest.fixture(scope="module")
def bronze_paths(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("bronze_sttm")
    rows = []
    for table, cols in (("orders", ORDERS_COLS), ("returns", RETURNS_COLS), ("products", PRODUCTS_COLS)):
        for col, cast in cols.items():
            rows.append(_bronze_row(table, col, col, "indirect", cast))
        rows.append({"source_schema": "landing", "source_table": table, "source_column": "",
                     "target_schema": "bronze", "target_table": table, "target_column": "_load_timestamp",
                     "transformation_type": "metadata", "transformation_logic": "current UTC timestamp"})
        rows.append({"source_schema": "landing", "source_table": table, "source_column": "",
                     "target_schema": "bronze", "target_table": table, "target_column": "_source_file",
                     "transformation_type": "metadata", "transformation_logic": "original file path"})

    sttm_path = _write_sttm(tmp_path, rows, "sttm_bronze.csv")
    file_paths = [
        str(SAMPLE_DATA_DIR / "orders.csv"),
        str(SAMPLE_DATA_DIR / "returns.json"),
        str(SAMPLE_DATA_DIR / "products.xlsx"),
    ]
    return _apply_bronze_rules(file_paths, sttm_path, RUN_ID)


def test_bronze_produces_three_tables_with_metadata(bronze_paths):
    assert len(bronze_paths) == 3
    for p in bronze_paths:
        df = pd.read_parquet(p)
        assert "_load_timestamp" in df.columns
        assert "_source_file" in df.columns
        assert len(df) > 0


@pytest.fixture(scope="module")
def silver_paths(tmp_path_factory, bronze_paths):
    tmp_path = tmp_path_factory.mktemp("silver_sttm")
    rows = []
    for table, cols in (("orders", ORDERS_COLS), ("returns", RETURNS_COLS), ("products", PRODUCTS_COLS)):
        rows.append({"source_schema": "bronze", "source_table": table, "source_column": "",
                     "target_schema": "silver", "target_table": table,
                     "target_column": f"pk_{table}_silver_id", "transformation_type": "surrogate_key",
                     "transformation_logic": "sequential integer id"})
        for col, cast in cols.items():
            ttype = "date_standardize" if cast == "datetime" else "type_cast"
            rows.append({
                "source_schema": "bronze", "source_table": table, "source_column": col,
                "target_schema": "silver", "target_table": table, "target_column": col,
                "transformation_type": ttype, "transformation_logic": cast,
            })

    sttm_path = _write_sttm(tmp_path, rows, "sttm_silver.csv")
    return _apply_silver_rules(bronze_paths, sttm_path, RUN_ID)


def test_silver_adds_surrogate_keys_and_standardises_dates(silver_paths):
    assert len(silver_paths) == 3
    for p in silver_paths:
        df = pd.read_parquet(p)
        assert list(df.columns)[0].startswith("pk_")
        assert len(df) > 0

    orders_df = next(pd.read_parquet(p) for p in silver_paths if "orders" in p)
    # date_standardize should produce plain YYYY-MM-DD strings
    assert orders_df["order_date"].iloc[0].count("-") == 2
    assert len(orders_df["order_date"].iloc[0]) == 10


def test_signal_layer_joins_and_derives_return_features(silver_paths, tmp_path_factory):
    dummy_sttm = tmp_path_factory.mktemp("signal_sttm2") / "sttm_signal.csv"
    pd.DataFrame({"target_table": ["order_line_signals"]}).to_csv(dummy_sttm, index=False)

    out_path = _apply_signal_rules(silver_paths, str(dummy_sttm), RUN_ID)
    df = pd.read_parquet(out_path)

    for col in ("category", "is_returned", "days_to_return", "net_revenue", "revenue_erosion", "order_month"):
        assert col in df.columns

    assert set(df["is_returned"].unique()) <= {0, 1}
    assert df["is_returned"].sum() > 0  # sample data has real returns
    # net_revenue should never exceed the original order_amount
    assert (df["net_revenue"] <= df["order_amount"] + 1e-6).all()
    # returned lines must have a non-null days_to_return
    returned = df[df["is_returned"] == 1]
    assert returned["days_to_return"].notna().all()
    assert (returned["days_to_return"] >= 0).all()
    # Apparel was seeded with the highest return rate in generate_sample_data.py
    return_rate_by_cat = df.groupby("category")["is_returned"].mean()
    assert return_rate_by_cat["Apparel"] > return_rate_by_cat["Books"]


def test_gold_layer_aggregates_match_signal_totals(silver_paths, tmp_path_factory):
    dummy_sttm = tmp_path_factory.mktemp("signal_sttm3") / "sttm_signal.csv"
    pd.DataFrame({"target_table": ["order_line_signals"]}).to_csv(dummy_sttm, index=False)
    signal_path = _apply_signal_rules(silver_paths, str(dummy_sttm), RUN_ID)
    signal_df = pd.read_parquet(signal_path)

    dummy_gold_sttm = tmp_path_factory.mktemp("gold_sttm") / "sttm_gold.csv"
    pd.DataFrame({"target_table": ["category_return_performance", "monthly_category_trend"]}).to_csv(
        dummy_gold_sttm, index=False
    )
    gold_paths = _apply_gold_rules([signal_path], str(dummy_gold_sttm), RUN_ID)
    assert len(gold_paths) == 2

    perf = pd.read_parquet(next(p for p in gold_paths if "category_return_performance" in p))
    trend = pd.read_parquet(next(p for p in gold_paths if "monthly_category_trend" in p))

    assert perf["total_orders"].sum() == len(signal_df)
    assert perf["total_returns"].sum() == signal_df["is_returned"].sum()
    assert trend["orders"].sum() == len(signal_df)
    assert (perf["return_rate"] >= 0).all() and (perf["return_rate"] <= 1).all()

    apparel_row = perf[perf["category"] == "Apparel"].iloc[0]
    books_row = perf[perf["category"] == "Books"].iloc[0]
    assert apparel_row["return_rate"] > books_row["return_rate"]
