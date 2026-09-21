import pandas as pd

from core.forecasting import compute_forecast, rank_at_risk_categories


def _trend_df():
    rows = []
    months = ["2025-01-01", "2025-02-01", "2025-03-01", "2025-04-01", "2025-05-01", "2025-06-01"]
    # Rising return rate / falling net revenue -> should be flagged at-risk
    for i, m in enumerate(months):
        rows.append({"category": "Apparel", "order_month": m, "orders": 100, "returns": 10 + i * 5,
                     "return_rate": (10 + i * 5) / 100, "net_revenue": 5000 - i * 400})
    # Flat, healthy category -> should NOT be flagged
    for i, m in enumerate(months):
        rows.append({"category": "Books", "order_month": m, "orders": 100, "returns": 2,
                     "return_rate": 0.02, "net_revenue": 3000 + i * 10})
    return pd.DataFrame(rows)


def test_compute_forecast_produces_entry_per_category_with_enough_history():
    forecast = compute_forecast(_trend_df(), horizon_months=3)
    assert set(forecast.keys()) == {"Apparel", "Books"}
    assert len(forecast["Apparel"]["future_months"]) == 3
    assert forecast["Apparel"]["future_months"][0] == "2025-07-01"


def test_compute_forecast_direction():
    forecast = compute_forecast(_trend_df(), horizon_months=3)
    assert forecast["Apparel"]["trend_direction"] == "rising"
    assert forecast["Books"]["trend_direction"] == "flat"


def test_compute_forecast_skips_short_history():
    short_df = _trend_df().groupby("category").head(2)
    forecast = compute_forecast(short_df, horizon_months=3)
    assert forecast == {}


def test_rank_at_risk_categories_flags_rising_return_falling_revenue():
    forecast = compute_forecast(_trend_df(), horizon_months=3)
    at_risk = rank_at_risk_categories(forecast)
    assert at_risk == ["Apparel"]
