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
        "satisfaction_coverage": float(usage["SatisfactionScore"].notna().mean() * 100)
        if total_requests
        else 0.0,
        "anomaly_rate": float(usage["IsAnomaly"].mean() * 100)
        if total_requests
        else 0.0,
    }


def monthly_metrics(usage: pd.DataFrame) -> pd.DataFrame:
    """Aggregate filtered usage into a monthly trend table."""
    if usage.empty:
        return pd.DataFrame(
            columns=["Month", "CostUSD", "TotalTokens", "RequestCount", "SuccessRate"]
        )
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


def model_costs(usage: pd.DataFrame) -> pd.DataFrame:
    """Aggregate cost and requests by model and provider."""
    return usage.groupby(["ModelName", "Provider"], as_index=False).agg(
        TotalCost=("CostUSD", "sum"),
        Requests=("UsageID", "count"),
        AvgLatency=("LatencyMs", "mean"),
        AvgTokenEfficiency=("TokenEfficiencyRatio", "mean"),
        AvgSatisfaction=("SatisfactionScore", "mean"),
    )


def provider_costs(usage: pd.DataFrame) -> pd.DataFrame:
    """Aggregate cost and requests by provider."""
    return usage.groupby("Provider", as_index=False).agg(
        TotalCost=("CostUSD", "sum"), Requests=("UsageID", "count")
    )


def feature_adoption(usage: pd.DataFrame) -> pd.DataFrame:
    """Summarize feature usage and premium adoption from the filtered snapshot."""
    return usage.groupby("FeatureName", as_index=False).agg(
        Requests=("UsageID", "count"),
        PremiumRate=("IsPremium", "mean"),
        AvgSatisfaction=("SatisfactionScore", "mean"),
    ).assign(PremiumRate=lambda frame: frame["PremiumRate"] * 100)


def risk_distribution(health: pd.DataFrame) -> pd.DataFrame:
    """Bucket model risk scores for a compact distribution chart."""
    bins = [-1, 25, 50, 75, 100]
    labels = ["Low (0–25)", "Moderate (25–50)", "High (50–75)", "Critical (75–100)"]
    result = health.copy()
    result["RiskBand"] = pd.cut(result["ChurnRiskScore"], bins=bins, labels=labels)
    return result.groupby("RiskBand", observed=False).size().reset_index(name="Clients")


def quota_by_client(usage: pd.DataFrame) -> pd.DataFrame:
    """Return one latest quota-utilization value per client."""
    return (
        usage.sort_values("Date")
        .groupby(["ClientKey", "ClientName"], as_index=False)
        .tail(1)[["ClientName", "QuotaUtilizationPct", "IsOverQuota"]]
        .sort_values("QuotaUtilizationPct", ascending=False)
    )


def data_quality_summary(
    usage: pd.DataFrame, dimensions: dict[str, pd.DataFrame], report: dict[str, object]
) -> dict[str, object]:
    """Calculate dashboard-facing quality checks without altering source artifacts."""
    key_map = {
        "ClientKey": "client_dimension",
        "ModelKey": "model_dimension",
        "FeatureKey": "feature_dimension",
        "RegionKey": "region_dimension",
        "SubscriptionTierKey": "subscription_dimension",
    }
    fk_violations = sum(
        (~usage[key].isin(dimensions[dim][key])).sum() for key, dim in key_map.items()
    )
    numeric = usage[["CostUSD", "TotalTokens", "LatencyMs"]]
    outliers = {
        column: int((numeric[column] > numeric[column].quantile(0.75) + 1.5 * (numeric[column].quantile(0.75) - numeric[column].quantile(0.25))).sum())
        for column in numeric
    }
    return {
        "row_counts": {"fact_with_ml": len(usage), **{name: len(frame) for name, frame in dimensions.items()}},
        "missing_values": usage.isna().sum().loc[lambda values: values.gt(0)].to_dict(),
        "duplicate_count": int(usage.duplicated().sum()),
        "foreign_key_violations": int(fk_violations),
        "outliers": outliers,
        "report": report,
    }


def format_number(value: float, decimals: int = 0) -> str:
    """Format a numeric value for human-readable dashboard labels."""
    return f"{value:,.{decimals}f}"
