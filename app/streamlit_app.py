"""Read-only Streamlit application for the AI usage analytics snapshot."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from app.charts import anomaly_counts, forecast_chart_data, top_rows
    from app.data_loader import ArtifactMissingError, load_dashboard_data
    from app.metrics import client_health, format_number, monthly_metrics, overview_metrics
else:
    from .charts import anomaly_counts, forecast_chart_data, top_rows
    from .data_loader import ArtifactMissingError, load_dashboard_data
    from .metrics import client_health, format_number, monthly_metrics, overview_metrics


st.set_page_config(page_title="AI Usage Analytics", page_icon="📊", layout="wide")


def apply_filters(usage: pd.DataFrame) -> pd.DataFrame:
    """Render global sidebar filters and return the filtered usage snapshot."""
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
        if selected:
            filtered = filtered[filtered[column].isin(selected)]
    return filtered


def show_overview(data: dict[str, object], usage: pd.DataFrame) -> None:
    st.header("Overview")
    metrics = overview_metrics(usage)
    columns = st.columns(5)
    columns[0].metric("Requests", format_number(metrics["requests"]))
    columns[1].metric("API cost", f"${metrics['cost']:,.2f}")
    columns[2].metric("Tokens", format_number(metrics["tokens"]))
    columns[3].metric("Success rate", f"{metrics['success_rate']:.2f}%")
    columns[4].metric("Anomaly rate", f"{metrics['anomaly_rate']:.2f}%")
    st.subheader("Monthly activity")
    monthly = monthly_metrics(usage).set_index("Month")
    st.line_chart(monthly[["CostUSD", "RequestCount"]])
    st.caption("Cost is in USD; request count is the number of usage events.")


def show_cost_and_models(data: dict[str, object]) -> None:
    st.header("Cost and Models")
    models = data["models"]
    st.dataframe(models, use_container_width=True, hide_index=True)
    st.bar_chart(top_rows(models, "TotalCost", "ModelName"), y="TotalCost")
    st.caption("Cost is in USD. Latency is in milliseconds and success rate is a percentage.")


def show_client_health(data: dict[str, object]) -> None:
    st.header("Client Health")
    health = client_health(data["clients"], data["churn"])
    st.dataframe(
        health.sort_values("ChurnRiskScore", ascending=False),
        use_container_width=True,
        hide_index=True,
    )
    st.caption("Churn risk scores are model outputs on a 0–100 scale.")


def show_forecasting(data: dict[str, object]) -> None:
    st.header("Forecasting")
    forecast_names = {
        "Cost (USD)": "cost_forecast",
        "Requests": "request_forecast",
        "Tokens": "token_forecast",
    }
    target = st.selectbox("Forecast measure", list(forecast_names))
    forecast = data[forecast_names[target]]
    st.line_chart(forecast_chart_data(forecast)[["yhat", "yhat_lower", "yhat_upper"]])
    st.dataframe(data["forecast_comparison"], use_container_width=True, hide_index=True)
    st.caption("Forecasts cover the 30 calendar days following the observed history.")


def show_anomalies(data: dict[str, object]) -> None:
    st.header("Anomalies")
    st.bar_chart(anomaly_counts(data["anomalies"]), y="Count")
    st.dataframe(data["anomalies"], use_container_width=True, hide_index=True)
    st.caption("Anomalies are statistical and rule-based signals, not confirmed incidents.")


def show_data_quality(data: dict[str, object]) -> None:
    st.header("Data Quality")
    quality = data["quality"]
    columns = st.columns(3)
    columns[0].metric("Fact rows", format_number(quality["fact_rows"]))
    columns[1].metric("Dimension tables", format_number(quality["dimension_tables"]))
    columns[2].metric("Findings", format_number(quality["finding_count"]))
    if quality["critical_failure"]:
        st.error("The quality report contains a critical failure.")
    else:
        st.success("No critical data-quality failures were reported.")
    st.json(quality)


def show_methodology() -> None:
    st.header("Methodology")
    st.markdown(
        """
        This dashboard is a read-only view of the checked-in analytical snapshot.
        The data pipeline creates processed aggregates, the scoring pipeline writes
        model predictions and anomaly flags, and the forecasting pipeline compares
        Naive, Seasonal-naive, ARIMA, and Prophet models using a chronological
        30-day holdout. Forecasts are planning estimates, not production commitments.
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
    page = st.sidebar.radio(
        "Navigate",
        [
            "Overview",
            "Cost and Models",
            "Client Health",
            "Forecasting",
            "Anomalies",
            "Data Quality",
            "Methodology",
        ],
    )
    if page == "Overview":
        show_overview(data, usage)
    elif page == "Cost and Models":
        show_cost_and_models(data)
    elif page == "Client Health":
        show_client_health(data)
    elif page == "Forecasting":
        show_forecasting(data)
    elif page == "Anomalies":
        show_anomalies(data)
    elif page == "Data Quality":
        show_data_quality(data)
    else:
        show_methodology()


if __name__ == "__main__":
    main()
