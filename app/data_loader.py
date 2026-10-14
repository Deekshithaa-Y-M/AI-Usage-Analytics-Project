"""Cached, read-only access to dashboard artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st


class ArtifactMissingError(FileNotFoundError):
    """Raised when a dashboard artifact is not available."""


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]

ARTIFACT_PATHS = {
    "usage data": Path("data/processed/fact_with_ml.csv"),
    "client monthly aggregates": Path("data/processed/agg_client_monthly.csv"),
    "feature adoption": Path("data/processed/agg_feature_adoption.csv"),
    "model performance": Path("data/processed/agg_model_performance.csv"),
    "regional monthly aggregates": Path("data/processed/agg_region_monthly.csv"),
    "client churn scores": Path("data/processed/client_churn_scores.csv"),
    "data quality report": Path("data/processed/data_quality_report.json"),
    "forecast model comparison": Path("outputs/evaluation/forecast_model_comparison.csv"),
    "anomaly summary": Path("outputs/evaluation/anomaly_summary.csv"),
    "cost forecast": Path("outputs/forecasts/forecast_cost_30d.csv"),
    "request forecast": Path("outputs/forecasts/forecast_requests_30d.csv"),
    "token forecast": Path("outputs/forecasts/forecast_tokens_30d.csv"),
    "churn model": Path("models/churn_model.pkl"),
    "satisfaction model": Path("models/satisfaction_model.pkl"),
    "anomaly model": Path("models/anomaly_model.pkl"),
}


def artifact_path(name: str) -> Path:
    """Return an artifact path resolved from the repository root."""
    try:
        relative_path = ARTIFACT_PATHS[name]
    except KeyError as exc:
        raise KeyError(f"Unknown dashboard artifact: {name}") from exc
    return REPOSITORY_ROOT / relative_path


def require_artifact(name: str) -> Path:
    """Return an artifact path or raise an actionable missing-file error."""
    path = artifact_path(name)
    if not path.is_file():
        raise ArtifactMissingError(
            f"The {name} artifact is missing: {path}. "
            "Run the data, scoring, and forecasting pipelines before opening the dashboard."
        )
    return path


@st.cache_data(show_spinner=False)
def load_csv(name: str) -> pd.DataFrame:
    """Load a static CSV artifact into a cached DataFrame."""
    return pd.read_csv(require_artifact(name))


@st.cache_data(show_spinner=False)
def load_json(name: str) -> dict[str, Any]:
    """Load a static JSON artifact into the Streamlit cache."""
    with require_artifact(name).open(encoding="utf-8") as file:
        return json.load(file)


@st.cache_data(show_spinner=False)
def load_dashboard_data() -> dict[str, Any]:
    """Load all dashboard inputs once per Streamlit cache key."""
    for name in ARTIFACT_PATHS:
        require_artifact(name)
    return {
        "usage": load_csv("usage data"),
        "clients": load_csv("client monthly aggregates"),
        "features": load_csv("feature adoption"),
        "models": load_csv("model performance"),
        "regions": load_csv("regional monthly aggregates"),
        "churn": load_csv("client churn scores"),
        "quality": load_json("data quality report"),
        "forecast_comparison": load_csv("forecast model comparison"),
        "anomalies": load_csv("anomaly summary"),
        "cost_forecast": load_csv("cost forecast"),
        "request_forecast": load_csv("request forecast"),
        "token_forecast": load_csv("token forecast"),
    }
