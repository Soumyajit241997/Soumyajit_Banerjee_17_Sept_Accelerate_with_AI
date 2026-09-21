from core.config import SAMPLE_DATA_DIR
from core.loaders import describe_format, load_any


def test_load_csv():
    df = load_any(SAMPLE_DATA_DIR / "orders.csv")
    assert "order_id" in df.columns
    assert len(df) > 0


def test_load_json():
    df = load_any(SAMPLE_DATA_DIR / "returns.json")
    assert "return_id" in df.columns
    assert len(df) > 0


def test_load_xlsx():
    df = load_any(SAMPLE_DATA_DIR / "products.xlsx")
    assert "category" in df.columns
    assert len(df) > 0


def test_describe_format():
    assert describe_format("foo.CSV") == "csv"
    assert describe_format("bar.json") == "json"
    assert describe_format("baz.xlsx") == "xlsx"
