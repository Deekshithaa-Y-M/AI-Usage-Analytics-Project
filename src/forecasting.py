"""Evaluate and create 30-day forecasts for daily AI usage metrics.

Forecasts are planning estimates only; they are not production capacity
commitments.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller


FORECAST_DAYS = 30
SEASONAL_PERIOD = 7
REQUIRED_COLUMNS = {"Date", "CostUSD", "TotalTokens", "UsageID"}
TARGETS = ("CostUSD", "TotalTokens", "RequestCount")


def load_master(input_path: Path) -> pd.DataFrame:
    """Load the processed fact table and validate forecasting fields."""
    if not input_path.exists():
        raise FileNotFoundError(f"Processed fact table was not found: {input_path}")
    master = pd.read_csv(input_path, parse_dates=["Date"])
    missing_columns = sorted(REQUIRED_COLUMNS.difference(master.columns))
    if missing_columns:
        raise ValueError(f"Master dataset is missing required columns: {missing_columns}")
    if master["Date"].isna().any():
        raise ValueError("The Date column contains missing values.")
    return master


def build_daily_series(master: pd.DataFrame) -> pd.DataFrame:
    """Aggregate requests to a complete daily calendar for modeling."""
    daily = master.groupby("Date", as_index=True).agg(
        CostUSD=("CostUSD", "sum"),
        TotalTokens=("TotalTokens", "sum"),
        RequestCount=("UsageID", "count"),
    )
    calendar = pd.date_range(daily.index.min(), daily.index.max(), freq="D")
    daily = daily.reindex(calendar, fill_value=0.0)
    daily.index.name = "Date"
    daily["IsWeekend"] = (daily.index.dayofweek >= 5).astype(int)
    return daily.reset_index()


def _holiday_regressor(dates: pd.Series | pd.DatetimeIndex) -> np.ndarray:
    """Return a US-holiday indicator for a sequence of dates."""
    try:
        import holidays
    except ImportError as error:
        raise RuntimeError("The holidays package is required for Prophet.") from error
    timestamp_dates = pd.DatetimeIndex(pd.to_datetime(dates))
    us_holidays = holidays.US(
        years=range(timestamp_dates.min().year, timestamp_dates.max().year + 1)
    )
    return np.array([int(date.date() in us_holidays) for date in timestamp_dates], dtype=int)


def _mape(actual: pd.Series, predicted: pd.Series) -> float:
    """Calculate MAPE while excluding zero-valued observations."""
    nonzero = actual.ne(0)
    if not nonzero.any():
        return float("nan")
    return float((actual[nonzero] - predicted[nonzero]).abs().div(actual[nonzero]).mean() * 100)


def _metrics(actual: pd.Series, predicted: pd.Series) -> dict[str, float]:
    """Return scale-dependent and scale-independent holdout metrics."""
    error = actual.to_numpy(dtype=float) - predicted.to_numpy(dtype=float)
    denominator = np.abs(actual.to_numpy(dtype=float)) + np.abs(predicted.to_numpy(dtype=float))
    smape = np.where(denominator == 0, 0, 2 * np.abs(error) / denominator).mean() * 100
    return {
        "MAE": float(np.abs(error).mean()),
        "RMSE": float(np.sqrt(np.mean(error**2))),
        "MAPE": _mape(actual, predicted),
        "sMAPE": float(smape),
    }


def _select_arima_order(series: pd.Series) -> tuple[tuple[int, int, int], float, float]:
    """Select a non-seasonal ARIMA order by the lowest BIC, then AIC."""
    candidates: list[tuple[tuple[int, int, int], float, float]] = []
    for p in range(3):
        for d in range(2):
            for q in range(3):
                try:
                    result = ARIMA(series, order=(p, d, q), enforce_stationarity=False).fit()
                    candidates.append(((p, d, q), float(result.aic), float(result.bic)))
                except (ValueError, np.linalg.LinAlgError):
                    continue
    if not candidates:
        raise RuntimeError("ARIMA order search did not produce a valid model.")
    return min(candidates, key=lambda item: (item[2], item[1]))


def _fit_arima(train: pd.Series, horizon: int, target: str) -> pd.DataFrame:
    """Fit ARIMA and return point forecasts with its native confidence interval."""
    order, _, _ = _select_arima_order(train)
    result = ARIMA(train, order=order, enforce_stationarity=False).fit()
    prediction = result.get_forecast(steps=horizon)
    interval = prediction.conf_int(alpha=0.05)
    return pd.DataFrame({
        "yhat": prediction.predicted_mean.to_numpy(),
        "yhat_lower": interval.iloc[:, 0].to_numpy(),
        "yhat_upper": interval.iloc[:, 1].to_numpy(),
        "order": str(order),
        "target": target,
    })


def _fit_prophet(train: pd.Series, horizon: int, target: str) -> pd.DataFrame:
    """Fit Prophet with weekly seasonality and holiday/weekend regressors."""
    try:
        from prophet import Prophet
    except ImportError as error:
        raise RuntimeError("The prophet package is required for cost and token forecasts.") from error
    training = train.rename("y").rename_axis("ds").reset_index()
    training["USHoliday"] = _holiday_regressor(training["ds"])
    training["IsWeekend"] = (training["ds"].dt.dayofweek >= 5).astype(int)
    model = Prophet(interval_width=0.95, daily_seasonality=False, weekly_seasonality=True)
    model.add_regressor("USHoliday")
    model.add_regressor("IsWeekend")
    model.fit(training)
    future = model.make_future_dataframe(periods=horizon, freq="D", include_history=False)
    future["IsWeekend"] = (future["ds"].dt.dayofweek >= 5).astype(int)
    future["USHoliday"] = _holiday_regressor(future["ds"])
    prediction = model.predict(future)
    return prediction[["yhat", "yhat_lower", "yhat_upper"]]


def _future_dates(series: pd.Series, horizon: int) -> pd.DatetimeIndex:
    """Create dates strictly after the last observed date."""
    return pd.date_range(series.index.max() + pd.Timedelta(days=1), periods=horizon, freq="D")


def _validate_forecast_dates(series: pd.Series, forecast: pd.DataFrame) -> None:
    """Reject forecasts with an incorrect horizon or dates overlapping training."""
    dates = pd.DatetimeIndex(forecast["ds"])
    if len(dates) != FORECAST_DAYS or dates.nunique() != FORECAST_DAYS:
        raise ValueError(f"Forecast for {series.name} must contain exactly {FORECAST_DAYS} unique dates.")
    if not dates[0] > series.index.max() or not dates.is_monotonic_increasing:
        raise ValueError("Forecast dates must be increasing and strictly after training dates.")


def _baseline_forecast(train: pd.Series, horizon: int, seasonal: bool) -> pd.DataFrame:
    """Create naive or weekly seasonal-naive point forecasts."""
    period = SEASONAL_PERIOD if seasonal else 1
    values = np.resize(train.iloc[-period:].to_numpy(dtype=float), horizon)
    return pd.DataFrame({"yhat": values, "yhat_lower": np.nan, "yhat_upper": np.nan})


def _plot_holdout(
    train: pd.Series,
    test: pd.Series,
    predictions: dict[str, pd.Series],
    target: str,
    output_path: Path,
) -> None:
    """Save actual-versus-predicted holdout chart for every compared model."""
    figure, axis = plt.subplots(figsize=(12, 6))
    axis.plot(train.index, train, label="Train", color="#1f4e79")
    axis.plot(test.index, test, label="Actual holdout", color="#111111")
    for model, prediction in predictions.items():
        axis.plot(test.index, prediction, label=model)
    axis.axvline(train.index.max(), color="#555555", linestyle="--", linewidth=1)
    axis.set(title=f"{target}: chronological holdout comparison", xlabel="Date", ylabel=target)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def _plot_future_interval(
    history: pd.Series, forecast: pd.DataFrame, target: str, output_path: Path
) -> None:
    """Save the selected future forecast and interval where available."""
    figure, axis = plt.subplots(figsize=(12, 6))
    axis.plot(history.index, history, label="Actual", color="#1f4e79")
    axis.plot(forecast["ds"], forecast["yhat"], label=f"{forecast['Model'].iloc[0]} forecast")
    if forecast["yhat_lower"].notna().any():
        axis.fill_between(
            forecast["ds"], forecast["yhat_lower"], forecast["yhat_upper"],
            color="#d95f02", alpha=0.2, label="95% prediction interval",
        )
    axis.axvline(history.index.max(), color="#555555", linestyle="--", linewidth=1)
    axis.set(title=f"{target}: selected {FORECAST_DAYS}-day forecast", xlabel="Date", ylabel=target)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def _plot_selected_comparison(
    daily: pd.DataFrame, forecasts: list[pd.DataFrame], output_path: Path
) -> None:
    """Save a compact chart of actual history and each selected forecast."""
    figure, axes = plt.subplots(3, 1, figsize=(13, 12), sharex=True)
    labels = [("CostUSD", "Cost (USD)"), ("TotalTokens", "Total tokens"), ("RequestCount", "Requests")]
    for axis, (metric, label) in zip(axes, labels):
        axis.plot(daily.index, daily[metric], label="Actual", color="#1f4e79")
        selected = [frame for frame in forecasts if frame["Metric"].iloc[0] == metric][0]
        axis.plot(selected["ds"], selected["yhat"], label=f"{selected['Model'].iloc[0]} forecast")
        if selected["yhat_lower"].notna().any():
            axis.fill_between(
                selected["ds"], selected["yhat_lower"], selected["yhat_upper"],
                color="#d95f02", alpha=0.2, label="95% prediction interval",
            )
        axis.set_ylabel(label)
        axis.legend()
    axes[0].set_title("30-day selected forecast comparison")
    axes[-1].set_xlabel("Date")
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def forecast_requests_arima(
    daily: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, float | str]]:
    """Backward-compatible request ARIMA forecast and summary metrics."""
    series = daily.set_index("Date")["RequestCount"].astype(float)
    train, test = series.iloc[:-FORECAST_DAYS], series.iloc[-FORECAST_DAYS:]
    order, _, _ = _select_arima_order(train)
    holdout = _fit_arima(train, FORECAST_DAYS, "RequestCount")
    full = _fit_arima(series, FORECAST_DAYS, "RequestCount")
    metrics: dict[str, float | str] = {
        "order": str(order),
        **_metrics(test, pd.Series(holdout["yhat"].to_numpy(), index=test.index)),
    }
    adf_result = adfuller(train, autolag="AIC")
    metrics.update({
        "ADFStatistic": float(adf_result[0]),
        "ADF p-value": float(adf_result[1]),
        "ADFStationaryAt5Pct": str(adf_result[1] < 0.05),
    })
    forecast = full.assign(
        ds=_future_dates(series, FORECAST_DAYS), Model="ARIMA", Metric="RequestCount"
    )
    return forecast, metrics


def _evaluate_target(series: pd.Series, target: str, output_dir: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Evaluate all available models on the same chronological holdout."""
    if len(series) <= FORECAST_DAYS * 2:
        raise ValueError("At least 61 daily observations are required for the holdout.")
    train, test = series.iloc[:-FORECAST_DAYS], series.iloc[-FORECAST_DAYS:]
    model_predictions: dict[str, pd.Series] = {}
    future_predictions: dict[str, pd.DataFrame] = {}

    for name, seasonal in (("Naive", False), ("SeasonalNaive", True)):
        if not seasonal or len(train) >= SEASONAL_PERIOD * 2:
            holdout = _baseline_forecast(train, FORECAST_DAYS, seasonal)["yhat"]
            model_predictions[name] = pd.Series(holdout.to_numpy(), index=test.index)
            future_predictions[name] = _baseline_forecast(series, FORECAST_DAYS, seasonal)

    arima_holdout = _fit_arima(train, FORECAST_DAYS, target)
    model_predictions["ARIMA"] = pd.Series(arima_holdout["yhat"].to_numpy(), index=test.index)
    future_predictions["ARIMA"] = _fit_arima(series, FORECAST_DAYS, target)

    prophet_holdout = _fit_prophet(train, FORECAST_DAYS, target)
    model_predictions["Prophet"] = pd.Series(prophet_holdout["yhat"].to_numpy(), index=test.index)
    future_predictions["Prophet"] = _fit_prophet(series, FORECAST_DAYS, target)

    rows: list[dict[str, Any]] = []
    for model, prediction in model_predictions.items():
        rows.append({"Metric": target, "Model": model, **_metrics(test, prediction)})
    comparison = pd.DataFrame(rows)
    selected = str(comparison.sort_values(["sMAPE", "RMSE"]).iloc[0]["Model"])
    _plot_holdout(train, test, model_predictions, target, output_dir / f"holdout_{target.lower()}.png")
    selected_future = future_predictions[selected].copy()
    selected_future["ds"] = _future_dates(series, FORECAST_DAYS)
    selected_future[["yhat", "yhat_lower", "yhat_upper"]] = selected_future[
        ["yhat", "yhat_lower", "yhat_upper"]
    ].clip(lower=0)
    selected_future["Model"] = selected
    selected_future["Metric"] = target
    _validate_forecast_dates(series, selected_future)
    _plot_future_interval(
        series, selected_future, target, output_dir / f"forecast_{target.lower()}_selected.png"
    )
    return comparison, {"selected": selected, "forecast": selected_future}


def run_forecasting(input_path: Path, output_dir: Path) -> dict[str, Any]:
    """Run chronological evaluation and write forecast, chart, and comparison outputs."""
    daily = build_daily_series(load_master(input_path)).set_index("Date")
    output_dir.mkdir(parents=True, exist_ok=True)
    evaluation_dir = output_dir.parent / "evaluation"
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    comparisons: list[pd.DataFrame] = []
    selected_models: dict[str, str] = {}
    selected_forecasts: list[pd.DataFrame] = []
    future_files = {
        "CostUSD": "forecast_cost_30d.csv",
        "TotalTokens": "forecast_tokens_30d.csv",
        "RequestCount": "forecast_requests_30d.csv",
    }
    for target in TARGETS:
        comparison, result = _evaluate_target(daily[target].astype(float), target, output_dir)
        comparisons.append(comparison)
        selected_models[target] = result["selected"]
        future = result["forecast"]
        selected_forecasts.append(future)
        future.to_csv(output_dir / future_files[target], index=False)
        if target in {"CostUSD", "TotalTokens"}:
            _plot_future_interval(
                daily[target], future, target,
                output_dir / f"forecast_{target.lower()}_prophet.png",
            )

    model_comparison = pd.concat(comparisons, ignore_index=True)
    model_comparison["Selected"] = model_comparison.apply(
        lambda row: row["Model"] == selected_models[row["Metric"]], axis=1
    )
    model_comparison.to_csv(evaluation_dir / "forecast_model_comparison.csv", index=False)
    pd.DataFrame(
        [{"Metric": metric, "SelectedModel": model} for metric, model in selected_models.items()]
    ).to_csv(evaluation_dir / "selected_forecast_models.csv", index=False)
    # Preserve the historical summary artifact for downstream consumers.
    request_metrics = model_comparison[model_comparison["Metric"] == "RequestCount"]
    request_metrics.to_csv(output_dir / "arima_metrics.csv", index=False)
    _plot_selected_comparison(daily, selected_forecasts, output_dir / "forecast_comparison.png")
    return {"selected_models": selected_models, "comparison": model_comparison}


def parse_args() -> argparse.Namespace:
    """Parse input and output paths."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/processed/fact_master.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/forecasts"))
    return parser.parse_args()


def main() -> None:
    """Run the forecasting entry point."""
    arguments: Any = parse_args()
    result = run_forecasting(arguments.input, arguments.output_dir)
    print(f"Selected models: {result['selected_models']}")
    print(f"Outputs written to {arguments.output_dir}")


if __name__ == "__main__":
    main()
