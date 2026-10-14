"""Chart-ready transformations shared by dashboard pages."""

from __future__ import annotations

import pandas as pd


def top_rows(data: pd.DataFrame, value: str, label: str, limit: int = 10) -> pd.DataFrame:
    """Return the highest-value rows with readable chart column names."""
    return (
        data.nlargest(limit, value)[[label, value]]
        .set_index(label)
        .sort_values(value)
    )


def forecast_chart_data(forecast: pd.DataFrame) -> pd.DataFrame:
    """Prepare forecast and interval columns indexed by forecast date."""
    result = forecast.copy()
    result["Date"] = pd.to_datetime(result["ds"])
    return result.set_index("Date")[["yhat", "yhat_lower", "yhat_upper"]].sort_index()


def anomaly_counts(summary: pd.DataFrame) -> pd.DataFrame:
    """Select count-oriented anomaly summary rows for a chart."""
    result = summary[summary["Count"].gt(0)].copy()
    return result.set_index("Pattern")[["Count"]].sort_values("Count")
