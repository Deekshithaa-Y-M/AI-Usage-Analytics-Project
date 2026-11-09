"""Read-only Streamlit dashboard for the checked-in AI usage snapshot."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from app.charts import bar_chart, forecast_chart, forecast_chart_data, histogram, line_chart, scatter_chart
    from app.data_loader import ArtifactMissingError, load_dashboard_data
    from app.metrics import (
        client_health,
        data_quality_summary,
        feature_adoption,
        format_number,
        model_costs,
        monthly_metrics,
        overview_metrics,
        provider_costs,
        quota_by_client,
        risk_distribution,
    )
else:
    from .charts import bar_chart, forecast_chart, forecast_chart_data, histogram, line_chart, scatter_chart
    from .data_loader import ArtifactMissingError, load_dashboard_data
    from .metrics import (
        client_health,
        data_quality_summary,
        feature_adoption,
        format_number,
        model_costs,
        monthly_metrics,
        overview_metrics,
        provider_costs,
        quota_by_client,
        risk_distribution,
    )


st.set_page_config(page_title="AI Usage Analytics", page_icon="📊", layout="wide")


def _empty_message() -> None:
    st.info("No records match the current filters. Adjust the sidebar filters to continue.")


def apply_filters(usage: pd.DataFrame) -> pd.DataFrame:
    """Render global filters and return the selected usage snapshot."""
    st.sidebar.header("Global filters")
    filtered = usage.copy()
    for column, label in (
        ("RegionName", "Region"),
        ("Provider", "Provider"),
        ("ModelName", "Model"),
        ("ClientName", "Client"),
    ):
        values = sorted(filtered[column].dropna().unique())
        selected = st.sidebar.multiselect(label, values, default=values)
        filtered = filtered[filtered[column].isin(selected)]
    return filtered


def show_overview(usage: pd.DataFrame) -> None:
    st.header("Overview")
    if usage.empty:
        _empty_message()
        return
    metrics = overview_metrics(usage)
    columns = st.columns(6)
    columns[0].metric("Total requests", format_number(metrics["requests"]))
    columns[1].metric("Total cost", f"${metrics['cost']:,.2f}")
    columns[2].metric("Total tokens", format_number(metrics["tokens"]))
    columns[3].metric("Success rate", f"{metrics['success_rate']:.2f}%")
    columns[4].metric("Average latency", f"{usage['LatencyMs'].mean():,.0f} ms")
    columns[5].metric("Satisfaction coverage", f"{metrics['satisfaction_coverage']:.2f}%")
    monthly = monthly_metrics(usage)
    monthly["Month"] = pd.to_datetime(monthly["Month"])
    st.altair_chart(
        line_chart(monthly, "Month", "RequestCount", "Monthly requests", "Requests"),
        use_container_width=True,
    )
    st.caption("Satisfaction coverage is the share of requests with an observed satisfaction label.")


def show_cost_and_models(usage: pd.DataFrame) -> None:
    st.header("Cost and Models")
    if usage.empty:
        _empty_message()
        return
    models = model_costs(usage)
    providers = provider_costs(usage)
    left, right = st.columns(2)
    with left:
        st.altair_chart(bar_chart(models, "ModelName", "TotalCost", "Cost by model", True), use_container_width=True)
        st.altair_chart(bar_chart(models, "ModelName", "Requests", "Requests by model"), use_container_width=True)
    with right:
        st.altair_chart(bar_chart(providers, "Provider", "TotalCost", "Cost by provider", True), use_container_width=True)
        st.altair_chart(
            scatter_chart(
                usage[["LatencyMs", "SatisfactionScore", "ModelName"]].dropna(),
                "LatencyMs",
                "SatisfactionScore",
                "Latency versus satisfaction",
            ),
            use_container_width=True,
        )
    st.subheader("Token efficiency")
    efficiency = models[["ModelName", "AvgTokenEfficiency"]].sort_values("AvgTokenEfficiency", ascending=False)
    st.altair_chart(bar_chart(efficiency, "ModelName", "AvgTokenEfficiency", "Average token efficiency"), use_container_width=True)
    st.dataframe(models, use_container_width=True, hide_index=True)


def show_client_health(data: dict[str, object], usage: pd.DataFrame) -> None:
    st.header("Client Health")
    if usage.empty:
        _empty_message()
        return
    health = client_health(data["clients"], data["churn"])
    health = health[health["ClientName"].isin(usage["ClientName"].unique())]
    st.altair_chart(bar_chart(risk_distribution(health), "RiskBand", "Clients", "Disengagement-risk distribution"), use_container_width=True)
    st.subheader("Risk-ranked clients")
    st.dataframe(health.sort_values("ChurnRiskScore", ascending=False), use_container_width=True, hide_index=True)
    st.subheader("Quota utilization")
    quota = quota_by_client(usage)
    st.altair_chart(bar_chart(quota.head(20), "ClientName", "QuotaUtilizationPct", "Quota utilization by client"), use_container_width=True)
    st.subheader("Feature adoption")
    adoption = feature_adoption(usage)
    st.altair_chart(bar_chart(adoption, "FeatureName", "Requests", "Feature adoption"), use_container_width=True)
    st.warning("Churn risk is a model-derived proxy for disengagement risk; it is not a confirmed customer outcome.")


def show_forecasting(data: dict[str, object], usage: pd.DataFrame) -> None:
    st.header("Forecasting")
    if usage.empty:
        _empty_message()
        return
    forecast_names = {"Cost (USD)": "cost_forecast", "Requests": "request_forecast", "Tokens": "token_forecast"}
    target = st.selectbox("Forecast measure", list(forecast_names))
    forecast = forecast_chart_data(data[forecast_names[target]])
    monthly = monthly_metrics(usage)
    history_column = {"Cost (USD)": "CostUSD", "Requests": "RequestCount", "Tokens": "TotalTokens"}[target]
    history = monthly[["Month", history_column]].rename(columns={"Month": "Date", history_column: "Value"})
    history["Date"] = pd.to_datetime(history["Date"])
    future = forecast.reset_index().rename(columns={"Date": "Date", "yhat": "Value", "yhat_lower": "Lower", "yhat_upper": "Upper"})
    st.altair_chart(
        forecast_chart(
            history[["Date", "Value"]],
            future[["Date", "Value", "Lower", "Upper"]],
            f"Historical versus forecast {target.lower()}",
            target,
        ),
        use_container_width=True,
    )
    interval = forecast.reset_index()[["Date", "yhat_lower", "yhat_upper"]]
    st.subheader("Prediction intervals")
    st.dataframe(interval, use_container_width=True, hide_index=True)
    st.subheader("Forecast model comparison")
    st.dataframe(data["forecast_comparison"], use_container_width=True, hide_index=True)
    st.caption("Forecasts are 30-day planning estimates selected from the checked-in evaluation outputs; no models are retrained.")


def show_anomalies(data: dict[str, object], usage: pd.DataFrame) -> None:
    st.header("Anomalies")
    if usage.empty:
        _empty_message()
        return
    anomaly_count = int(usage["IsAnomaly"].sum())
    columns = st.columns(2)
    columns[0].metric("Anomaly count", format_number(anomaly_count))
    columns[1].metric("Anomaly percentage", f"{usage['IsAnomaly'].mean() * 100:.2f}%")
    st.altair_chart(histogram(usage, "AnomalyScore", "Anomaly score distribution"), use_container_width=True)
    only_anomalies = st.checkbox("Show anomalies only", value=True)
    table = usage[usage["IsAnomaly"]] if only_anomalies else usage
    if table.empty:
        _empty_message()
    else:
        st.dataframe(table.sort_values("AnomalyScore"), use_container_width=True, hide_index=True)
    signals = usage[["CostUSD", "TotalTokens", "LatencyMs", "IsAnomaly"]].copy()
    st.altair_chart(scatter_chart(signals, "TotalTokens", "CostUSD", "Anomaly cost versus tokens"), use_container_width=True)
    st.altair_chart(scatter_chart(signals, "LatencyMs", "CostUSD", "Anomaly cost versus latency"), use_container_width=True)
    st.caption("Anomalies are statistical and rule-based signals, not confirmed incidents.")


def show_data_quality(data: dict[str, object], usage: pd.DataFrame) -> None:
    st.header("Data Quality")
    dimensions = {name: data[name] for name in ("client_dimension", "model_dimension", "feature_dimension", "region_dimension", "subscription_dimension")}
    summary = data_quality_summary(usage, dimensions, data["quality"])
    columns = st.columns(4)
    columns[0].metric("Rows", format_number(len(usage)))
    columns[1].metric("Missing values", format_number(sum(summary["missing_values"].values())))
    columns[2].metric("Duplicates", format_number(summary["duplicate_count"]))
    columns[3].metric("Foreign-key violations", format_number(summary["foreign_key_violations"]))
    st.subheader("Row counts")
    st.json(summary["row_counts"])
    st.subheader("Missing-value summary")
    st.dataframe(pd.DataFrame({"Column": list(summary["missing_values"]), "Missing": list(summary["missing_values"].values())}), hide_index=True)
    st.subheader("Outlier summary")
    st.dataframe(pd.DataFrame({"Measure": list(summary["outliers"]), "Outlier count": list(summary["outliers"].values())}), hide_index=True)
    st.markdown("[Read the data-quality methodology](../docs/data_dictionary.md)")


def show_methodology() -> None:
    st.header("Methodology")
    st.markdown(
        """
        The pipeline joins the raw fact table to date, client, model, feature, region,
        and subscription dimensions, then writes a checked-in analytical snapshot.
        Feature engineering adds cost, quota, latency, token-efficiency, and response
        indicators. Existing scoring artifacts provide satisfaction, churn-risk, and
        anomaly signals; this page does not retrain any model.

        **Interpretation warnings:** churn risk is a proxy for disengagement, and
        satisfaction is observed only for labeled requests. Forecasts are selected
        from chronological holdout evaluation and are limited by the fixed,
        portfolio-data snapshot, data frequency, and future behavior assumptions.
        These outputs support exploration, not production commitments or causal claims.
        """
    )


def main() -> None:
    st.title("AI Usage Analytics")
    try:
        data = load_dashboard_data()
    except ArtifactMissingError as exc:
        st.error(str(exc))
        st.stop()
    usage = apply_filters(data["usage"])
    page = st.sidebar.radio("Navigate", ["Overview", "Cost and Models", "Client Health", "Forecasting", "Anomalies", "Data Quality", "Methodology"])
    if page == "Overview":
        show_overview(usage)
    elif page == "Cost and Models":
        show_cost_and_models(usage)
    elif page == "Client Health":
        show_client_health(data, usage)
    elif page == "Forecasting":
        show_forecasting(data, usage)
    elif page == "Anomalies":
        show_anomalies(data, usage)
    elif page == "Data Quality":
        show_data_quality(data, usage)
    else:
        show_methodology()


if __name__ == "__main__":
    main()
