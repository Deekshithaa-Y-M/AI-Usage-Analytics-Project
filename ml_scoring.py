"""Generate churn, satisfaction, and billing-anomaly signals for AI usage."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, IsolationForest, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


CHURN_FEATURES = [
    "DaysSinceLastRequest", "AvgMonthlyCostTrend", "SatisfactionTrend",
    "SuccessRateLast30Days", "SuccessRateOverall", "QuotaUtilizationPct",
    "PremiumFeatureUsagePct",
]
SATISFACTION_FEATURES = [
    "ModelKey", "FeatureKey", "RegionKey", "SubscriptionTierKey", "TotalTokens",
    "LatencyMs", "CostUSD", "SuccessFlag", "HourOfDay", "IsWeekend", "RequestType",
]
ANOMALY_FEATURES = ["CostUSD", "TotalTokens", "LatencyMs"]


def load_master(input_path: Path) -> pd.DataFrame:
    """Load the Task 1 analytical master and normalize fields needed by ML."""
    if not input_path.exists():
        raise FileNotFoundError(f"Task 1 output was not found: {input_path}")
    master = pd.read_csv(input_path, parse_dates=["Date", "OnboardingDate"])
    required_columns = set(SATISFACTION_FEATURES + ANOMALY_FEATURES)
    required_columns.update({"ClientKey", "Month", "Date", "SatisfactionScore", "IsPremium", "QuotaUtilizationPct"})
    missing_columns = sorted(required_columns.difference(master.columns))
    if missing_columns:
        raise ValueError(f"Master dataset is missing required columns: {missing_columns}")
    return master


def _linear_slope(values: pd.Series, periods: pd.Series) -> float:
    """Return a stable least-squares slope, using zero for insufficient history."""
    valid = pd.concat([periods, values], axis=1).dropna()
    if len(valid) < 2 or valid.iloc[:, 0].nunique() < 2:
        return 0.0
    return float(np.polyfit(valid.iloc[:, 0], valid.iloc[:, 1], 1)[0])


def build_churn_features(master: pd.DataFrame) -> pd.DataFrame:
    """Aggregate six-month engagement and service trends at client grain."""
    analysis_end = master["Date"].max()
    monthly_cost = master.groupby(["ClientKey", "Month"], as_index=False)["CostUSD"].sum()
    monthly_cost["MonthNumber"] = pd.PeriodIndex(monthly_cost["Month"], freq="M").asi8
    monthly_satisfaction = master.groupby(["ClientKey", "Month"], as_index=False)["SatisfactionScore"].mean()
    monthly_satisfaction["MonthNumber"] = pd.PeriodIndex(
        monthly_satisfaction["Month"], freq="M"
    ).asi8

    rows: list[dict[str, float | int]] = []
    for client_key, client_rows in master.groupby("ClientKey"):
        cost_rows = monthly_cost[monthly_cost["ClientKey"] == client_key]
        satisfaction_rows = monthly_satisfaction[monthly_satisfaction["ClientKey"] == client_key]
        last_request = client_rows["Date"].max()
        recent_rows = client_rows[client_rows["Date"] >= analysis_end - pd.Timedelta(days=30)]
        rows.append({
            "ClientKey": int(client_key),
            "DaysSinceLastRequest": int((analysis_end - last_request).days),
            "AvgMonthlyCostTrend": _linear_slope(cost_rows["CostUSD"], cost_rows["MonthNumber"]),
            "SatisfactionTrend": _linear_slope(
                satisfaction_rows["SatisfactionScore"], satisfaction_rows["MonthNumber"]
            ),
            "SuccessRateLast30Days": float(recent_rows["SuccessFlag"].mean()),
            "SuccessRateOverall": float(client_rows["SuccessFlag"].mean()),
            "QuotaUtilizationPct": float(client_rows["QuotaUtilizationPct"].max()),
            "PremiumFeatureUsagePct": float(client_rows["IsPremium"].mean() * 100),
        })
    return pd.DataFrame(rows)


def train_churn_model(churn_features: pd.DataFrame, model_dir: Path) -> pd.DataFrame:
    """Train a client churn classifier and export a normalized 0-100 risk score."""
    churn_features = churn_features.copy()
    churn_features["ChurnSignal"] = (
        (churn_features["DaysSinceLastRequest"] > 45)
        & (churn_features["QuotaUtilizationPct"] < 20)
    ).astype(int)
    x_values = churn_features[CHURN_FEATURES].fillna(0)
    y_values = churn_features["ChurnSignal"]
    model_dir.mkdir(parents=True, exist_ok=True)

    if y_values.nunique() >= 2:
        classifier: Any = Pipeline([
            ("scale", StandardScaler()),
            ("model", LogisticRegression(random_state=42, max_iter=2000)),
        ])
        classifier.fit(x_values, y_values)
        if y_values.value_counts().min() >= 2:
            folds = min(5, int(y_values.value_counts().min()))
            scores = cross_val_score(classifier, x_values, y_values, cv=folds, scoring="roc_auc")
            print(f"Churn model cross-validated ROC AUC: {scores.mean():.3f}")
        risk_scores = classifier.predict_proba(x_values)[:, 1] * 100
    else:
        classifier = GradientBoostingClassifier(random_state=42)
        synthetic_x = pd.concat([x_values, x_values.iloc[[0]]], ignore_index=True)
        synthetic_y = pd.concat([y_values, pd.Series([1 - int(y_values.iloc[0])])], ignore_index=True)
        classifier.fit(synthetic_x, synthetic_y)
        risk_scores = classifier.predict_proba(x_values)[:, 1] * 100
        print("Churn model warning: synthetic labels contain one class; added one balancing training row.")

    joblib.dump(classifier, model_dir / "churn_model.pkl")
    result = churn_features[["ClientKey"]].copy()
    result["ChurnRiskScore"] = np.round(np.clip(risk_scores, 0, 100), 2)
    return result


def _satisfaction_preprocessor() -> ColumnTransformer:
    """Create leakage-safe numeric and categorical preprocessing for satisfaction."""
    categorical = ["ModelKey", "FeatureKey", "RegionKey", "SubscriptionTierKey", "RequestType"]
    numeric = [column for column in SATISFACTION_FEATURES if column not in categorical]
    return ColumnTransformer([
        ("numeric", Pipeline([("impute", SimpleImputer(strategy="median"))]), numeric),
        ("categorical", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore")),
        ]), categorical),
    ])


def train_satisfaction_model(
    master: pd.DataFrame, model_dir: Path
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Train a random-forest regressor and predict satisfaction for every request."""
    labeled = master[master["SatisfactionScore"].notna()].copy()
    if len(labeled) < 10:
        raise ValueError("At least 10 rated requests are required for satisfaction modeling.")
    features = labeled[SATISFACTION_FEATURES]
    target = labeled["SatisfactionScore"]
    x_train, x_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=42
    )
    regressor = Pipeline([
        ("preprocess", _satisfaction_preprocessor()),
        ("model", RandomForestRegressor(
            n_estimators=250, min_samples_leaf=2, random_state=42, n_jobs=-1
        )),
    ])
    regressor.fit(x_train, y_train)
    test_predictions = regressor.predict(x_test)
    metrics = {
        "RMSE": float(np.sqrt(mean_squared_error(y_test, test_predictions))),
        "MAE": float(mean_absolute_error(y_test, test_predictions)),
        "R2": float(r2_score(y_test, test_predictions)),
    }
    folds = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_rmse = np.sqrt(-cross_val_score(
        regressor, features, target, cv=folds, scoring="neg_mean_squared_error", n_jobs=1
    ))
    metrics["CV_RMSE"] = float(cv_rmse.mean())
    print(
        f"Satisfaction model: RMSE={metrics['RMSE']:.3f}, MAE={metrics['MAE']:.3f}, "
        f"R2={metrics['R2']:.3f}, CV RMSE={metrics['CV_RMSE']:.3f}"
    )
    joblib.dump(regressor, model_dir / "satisfaction_model.pkl")

    scored = master.copy()
    scored["SatisfactionPredicted"] = regressor.predict(scored[SATISFACTION_FEATURES])
    return scored, metrics


def detect_anomalies(master: pd.DataFrame, model_dir: Path) -> pd.DataFrame:
    """Fit Isolation Forest and append anomaly flags and higher-is-more-anomalous scores."""
    anomaly_model = Pipeline([
        ("scale", StandardScaler()),
        ("model", IsolationForest(contamination=0.02, random_state=42, n_jobs=-1)),
    ])
    anomaly_model.fit(master[ANOMALY_FEATURES])
    scored = master.copy()
    scored["IsAnomaly"] = anomaly_model.predict(master[ANOMALY_FEATURES]) == -1
    scored["AnomalyScore"] = -anomaly_model.decision_function(master[ANOMALY_FEATURES])
    joblib.dump(anomaly_model, model_dir / "anomaly_model.pkl")
    print(f"Anomaly detection flagged {int(scored['IsAnomaly'].sum()):,} requests.")
    return scored


def run_ml_scoring(
    input_path: Path, output_path: Path, model_dir: Path
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Run churn scoring, satisfaction prediction, anomaly detection, and exports."""
    master = load_master(input_path)
    model_dir.mkdir(parents=True, exist_ok=True)
    churn_features = build_churn_features(master)
    churn_scores = train_churn_model(churn_features, model_dir)
    scored, metrics = train_satisfaction_model(master, model_dir)
    scored = scored.merge(churn_scores, on="ClientKey", how="left", validate="many_to_one")
    scored = detect_anomalies(scored, model_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(output_path, index=False)
    churn_scores.to_csv(output_path.parent / "client_churn_scores.csv", index=False)
    print(f"Wrote ML-scored dataset with {len(scored):,} rows to {output_path}")
    return scored, metrics


def parse_args() -> argparse.Namespace:
    """Parse input, output, and model artifact paths."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/processed/fact_master.csv"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/fact_with_ml.csv"))
    parser.add_argument("--model-dir", type=Path, default=Path("models"))
    return parser.parse_args()


def main() -> None:
    """Run the Task 2 command-line entry point."""
    arguments: Any = parse_args()
    run_ml_scoring(arguments.input, arguments.output, arguments.model_dir)


if __name__ == "__main__":
    main()
