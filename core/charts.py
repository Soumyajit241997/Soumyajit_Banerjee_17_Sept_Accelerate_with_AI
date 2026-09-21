"""Plotly rendering helpers used by the Reporter agent.

Charts render to HTML `<div>` fragments (Plotly.js loaded once by the report
template) rather than static images, so the final report stays interactive.
"""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def render_query_chart(df: pd.DataFrame, title: str, chart_type: str, x: str, y: str) -> str:
    chart_type = (chart_type or "bar").lower()
    if x not in df.columns or y not in df.columns:
        return f"<p><em>Chart '{title}' skipped: columns {x!r}/{y!r} not in query result.</em></p>"

    if chart_type == "line":
        fig = px.line(df, x=x, y=y, title=title, markers=True)
    elif chart_type == "pie":
        fig = px.pie(df, names=x, values=y, title=title)
    else:
        fig = px.bar(df, x=x, y=y, title=title)

    fig.update_layout(template="plotly_white", margin=dict(l=40, r=20, t=50, b=40))
    return fig.to_html(full_html=False, include_plotlyjs=False)


def render_forecast_chart(category: str, forecast_entry: dict, metric: str, y_label: str) -> str:
    history_key = f"history_{metric}"
    projected_key = f"projected_{metric}"

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=forecast_entry["history_months"], y=forecast_entry[history_key],
        mode="lines+markers", name="Actual", line=dict(color="#1f77b4"),
    ))

    bridge_x = [forecast_entry["history_months"][-1]] + forecast_entry["future_months"]
    bridge_y = [forecast_entry[history_key][-1]] + forecast_entry[projected_key]
    fig.add_trace(go.Scatter(
        x=bridge_x, y=bridge_y,
        mode="lines+markers", name="Forecast", line=dict(color="#d62728", dash="dash"),
    ))

    fig.update_layout(
        title=f"{category}: {y_label} — actual vs. forecast",
        template="plotly_white", margin=dict(l=40, r=20, t=50, b=40),
        xaxis_title="Month", yaxis_title=y_label,
    )
    return fig.to_html(full_html=False, include_plotlyjs=False)
