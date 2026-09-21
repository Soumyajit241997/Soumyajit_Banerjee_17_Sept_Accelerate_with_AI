"""Lightweight per-category trend forecasting used by the Reporter agent.

This is the "extra information related to forecasting" the Reporter produces
beyond IDAMP's original scope. It deliberately avoids relying on the LLM for
arithmetic: a simple linear trend (numpy.polyfit) is fit per category on its
monthly return_rate and net_revenue history, then projected forward. The LLM
only ever narrates numbers this module already computed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from core.config import FORECAST_HORIZON_MONTHS, FORECAST_MIN_HISTORY_MONTHS


def _next_month_labels(last_month: str, n: int) -> list[str]:
    period = pd.Period(last_month, freq="M")
    return [(period + i).strftime("%Y-%m") + "-01" for i in range(1, n + 1)]


def compute_forecast(trend_df: pd.DataFrame, horizon_months: int = FORECAST_HORIZON_MONTHS) -> dict:
    """Fit a linear trend per category on (order_month -> return_rate, net_revenue)
    and project it `horizon_months` forward. Categories with fewer than
    FORECAST_MIN_HISTORY_MONTHS data points are skipped (not enough signal to trust
    a trend line)."""
    forecast: dict[str, dict] = {}

    for category, group in trend_df.groupby("category"):
        group = group.sort_values("order_month")
        months = group["order_month"].astype(str).tolist()
        if len(months) < FORECAST_MIN_HISTORY_MONTHS:
            continue

        t = np.arange(len(months), dtype=float)
        return_rate = group["return_rate"].to_numpy(dtype=float)
        net_revenue = group["net_revenue"].to_numpy(dtype=float)

        rr_slope, rr_intercept = np.polyfit(t, return_rate, 1)
        nr_slope, nr_intercept = np.polyfit(t, net_revenue, 1)

        future_t = np.arange(len(months), len(months) + horizon_months, dtype=float)
        projected_return_rate = np.clip(rr_slope * future_t + rr_intercept, 0.0, 1.0).tolist()
        projected_net_revenue = (nr_slope * future_t + nr_intercept).tolist()

        forecast[str(category)] = {
            "history_months": months,
            "history_return_rate": return_rate.tolist(),
            "history_net_revenue": net_revenue.tolist(),
            "future_months": _next_month_labels(months[-1], horizon_months),
            "projected_return_rate": projected_return_rate,
            "projected_net_revenue": projected_net_revenue,
            "return_rate_slope_per_month": float(rr_slope),
            "net_revenue_slope_per_month": float(nr_slope),
            "trend_direction": "rising" if rr_slope > 0.001 else ("falling" if rr_slope < -0.001 else "flat"),
            "projected_revenue_at_risk": float(sum(
                p for p in (net_revenue[-1] - v for v in projected_net_revenue) if p > 0
            )),
        }

    return forecast


def rank_at_risk_categories(forecast: dict, top_n: int = 3) -> list[str]:
    """Categories whose return rate is trending up while net revenue trends down,
    ranked by how much projected net revenue is at risk."""
    candidates = [
        (name, entry["projected_revenue_at_risk"])
        for name, entry in forecast.items()
        if entry["return_rate_slope_per_month"] > 0 and entry["net_revenue_slope_per_month"] < 0
    ]
    candidates.sort(key=lambda pair: pair[1], reverse=True)
    return [name for name, _ in candidates[:top_n]]
