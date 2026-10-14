from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")

from src.ml_scoring import (
    ANOMALY_FEATURES,
    CHURN_FEATURES,
    SATISFACTION_FEATURES,
    detect_anomalies,
    train_churn_model,
    train_satisfaction_model,
)


def _churn_features() -> pd.DataFrame:
    rows = []
    for client_key in range(1, 7):
        rows.append({
            "ClientKey": client_key,
            "DaysSinceLastRequest": 60 if client_key % 2 else 5,
            "AvgMonthlyCostTrend": float(client_key - 3),
            "SatisfactionTrend": 0.1 * client_key,
            "SuccessRateLast30Days": 0.8 + client_key / 100,
            "SuccessRateOverall": 0.85 + client_key / 100,
            "QuotaUtilizationPct": 10 if client_key % 2 else 70,
            "PremiumFeatureUsagePct": client_key * 10,
        })
    return pd.DataFrame(rows, columns=["ClientKey"] + CHURN_FEATURES)


def _satisfaction_master(rows: int = 20) -> pd.DataFrame:
    frame = pd.DataFrame({
        "ClientKey": np.arange(1, rows + 1),
        "SatisfactionScore": np.tile([1.5, 2.5, 3.5, 4.5, 5.0], rows // 5),
        "ModelKey": np.tile([1, 2], rows // 2),
        "FeatureKey": np.tile([1, 2], rows // 2),
        "RegionKey": np.tile([1, 2], rows // 2),
        "SubscriptionTierKey": np.tile([1, 2], rows // 2),
        "TotalTokens": np.arange(100, 100 + rows),
        "LatencyMs": np.arange(200, 200 + rows),
        "CostUSD": np.linspace(0.01, 0.20, rows),
        "SuccessFlag": np.ones(rows, dtype=int),
        "HourOfDay": np.arange(rows) % 24,
        "IsWeekend": np.arange(rows) % 2,
        "RequestType": np.tile(["Chat", "Completion"], rows // 2),
    })
    return frame[["ClientKey"] + SATISFACTION_FEATURES + ["SatisfactionScore"]]


def _anomaly_master(rows: int = 30) -> pd.DataFrame:
    return pd.DataFrame({
        "CostUSD": np.linspace(0.01, 2.0, rows),
        "TotalTokens": np.linspace(100, 3000, rows),
        "LatencyMs": np.linspace(100, 1500, rows),
    })


def test_churn_output_columns_and_score_range(tmp_path):
    result = train_churn_model(_churn_features(), tmp_path / "models", tmp_path / "evaluation")
    assert list(result.columns) == ["ClientKey", "ChurnRiskScore"]
    assert result["ChurnRiskScore"].between(0, 100).all()


def test_churn_model_is_reproducible(tmp_path):
    first = train_churn_model(_churn_features(), tmp_path / "one", tmp_path / "eval_one")
    second = train_churn_model(_churn_features(), tmp_path / "two", tmp_path / "eval_two")
    pd.testing.assert_frame_equal(first, second)


def test_satisfaction_predictions_are_bounded_and_evaluation_is_written(tmp_path):
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    scored, metrics = train_satisfaction_model(
        _satisfaction_master(), model_dir, tmp_path / "evaluation"
    )
    assert {"SatisfactionPredicted", "SatisfactionPredictionPopulation"} <= set(scored)
    assert scored["SatisfactionPredicted"].between(1, 5).all()
    assert {"RMSE", "MAE", "R2", "CV_RMSE"} <= set(metrics)
    assert (tmp_path / "evaluation" / "satisfaction_evaluation.csv").exists()


def test_anomaly_labels_are_boolean_and_evaluation_is_written(tmp_path):
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    scored = detect_anomalies(_anomaly_master(), model_dir, tmp_path / "evaluation")
    assert set(scored["IsStatisticalAnomaly"].unique()) <= {True, False}
    assert set(scored["IsRuleBasedOutlier"].unique()) <= {True, False}
    assert (tmp_path / "evaluation" / "anomaly_summary.csv").exists()
