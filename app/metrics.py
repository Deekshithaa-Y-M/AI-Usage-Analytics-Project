"""Small, testable calculations used by the Streamlit pages."""

from __future__ import annotations

import pandas as pd


def overview_metrics(usage: pd.DataFrame) -> dict[str, float]:
    """Return headline metrics in dashboard display units."""
    total_requests = len(usage)
    return {
        "requests": float(total_requests),
        "cost": float(usage["CostUSD"].sum()),
        "tokens": float(usage["TotalTokens"].sum()),
        "success_rate": float(usage["SuccessFlag"].mean() * 100)
        if total_requests
        else 0.0,
        "satisfaction": float(usage["SatisfactionScore"].mean()),
        "anomaly_rate": float(usage["IsAnomaly"].mean() * 100)
        if total_requests
        else 0.0,
    }


def monthly_metrics(usage: pd.DataFrame) -> pd.DataFrame:
    """Aggregate filtered usage into a monthly trend table."""
    result = (
        usage.assign(Month=pd.to_datetime(usage["Date"]).dt.to_period("M").astype(str))
        .groupby("Month", as_index=False)
        .agg(
            CostUSD=("CostUSD", "sum"),
            TotalTokens=("TotalTokens", "sum"),
            RequestCount=("UsageID", "count"),
            SuccessRate=("SuccessFlag", "mean"),
        )
    )
    result["SuccessRate"] *= 100
    return result


def client_health(clients: pd.DataFrame, churn: pd.DataFrame) -> pd.DataFrame:
    """Combine client usage health with the latest churn score."""
    latest = clients.sort_values("Month").groupby("ClientKey", as_index=False).tail(1)
    return latest.merge(churn, on="ClientKey", how="left")


def format_number(value: float, decimals: int = 0) -> str:
    """Format a numeric value for human-readable dashboard labels."""
    return f"{value:,.{decimals}f}"
