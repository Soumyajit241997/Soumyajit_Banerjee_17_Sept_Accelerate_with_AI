import pandas as pd

from core.transforms import (
    add_surrogate_key, cast_series, fill_nulls, normalize_text,
    preserve_missing_id_columns, standardize_date,
)


def test_cast_series_integer():
    out = cast_series(pd.Series(["1", "2", None]), "integer")
    assert out.tolist() == [1, 2, pd.NA]


def test_cast_series_float():
    out = cast_series(pd.Series(["1.5", "2.25"]), "float")
    assert out.tolist() == [1.5, 2.25]


def test_cast_series_datetime():
    out = cast_series(pd.Series(["2025-01-15"]), "datetime")
    assert out.iloc[0] == pd.Timestamp("2025-01-15")


def test_standardize_date_multiple_formats():
    out = standardize_date(pd.Series(["01/15/2025", "2025-02-20"]))
    assert out.iloc[0] == "2025-01-15"
    assert out.iloc[1] == "2025-02-20"


def test_normalize_text_lowercase_strip():
    out = normalize_text(pd.Series([" Widget A "]), "lowercase, strip")
    assert out.iloc[0] == "widget a"


def test_fill_nulls_mean():
    out = fill_nulls(pd.Series([1.0, 2.0, None]), "fill:mean")
    assert out.iloc[2] == 1.5


def test_fill_nulls_unknown_default():
    out = fill_nulls(pd.Series(["a", None]), "")
    assert out.iloc[1] == "unknown"


def test_add_surrogate_key():
    df = add_surrogate_key(pd.DataFrame({"x": [10, 20, 30]}), "pk_test_id")
    assert list(df.columns)[0] == "pk_test_id"
    assert df["pk_test_id"].tolist() == [1, 2, 3]


def test_preserve_missing_id_columns_recovers_dropped_business_key():
    source = pd.DataFrame({"order_id": [1, 2], "order_date": ["2025-01-01", "2025-01-02"]})
    out = pd.DataFrame({"order_date": ["2025-01-01", "2025-01-02"]})  # order_id was dropped
    result, recovered = preserve_missing_id_columns(source, out)
    assert recovered == ["order_id"]
    assert result["order_id"].tolist() == [1, 2]


def test_preserve_missing_id_columns_ignores_surrogate_keys():
    source = pd.DataFrame({"pk_orders_silver_id": [1, 2], "order_date": ["a", "b"]})
    out = pd.DataFrame({"order_date": ["a", "b"]})
    result, recovered = preserve_missing_id_columns(source, out)
    assert recovered == []
    assert "pk_orders_silver_id" not in result.columns


def test_preserve_missing_id_columns_noop_when_already_present():
    source = pd.DataFrame({"order_id": [1, 2]})
    out = pd.DataFrame({"order_id": [1, 2]})
    result, recovered = preserve_missing_id_columns(source, out)
    assert recovered == []
    assert result["order_id"].tolist() == [1, 2]
