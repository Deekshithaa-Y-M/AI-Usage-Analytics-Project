"""Create 30-day Prophet and ARIMA forecasts from the usage fact table."""

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
REQUIRED_COLUMNS = {"Date", "CostUSD", "TotalTokens", "UsageID"}


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
    """Aggregate requests to a complete daily calendar for six-month modeling."""
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
        raise RuntimeError(
            "The holidays package is required for the Prophet holiday regressor."
        ) from error
    timestamp_dates = pd.DatetimeIndex(pd.to_datetime(dates))
    us_holidays = holidays.US(years=range(timestamp_dates.min().year, timestamp_dates.max().year + 1))
    return np.array([int(date.date() in us_holidays) for date in timestamp_dates], dtype=int)


def _prophet_forecast(daily: pd.DataFrame, target: str) -> pd.DataFrame:
    """Fit Prophet with US holidays and weekend regressors for one target."""
    try:
        from prophet import Prophet
    except ImportError as error:
        raise RuntimeError(
            "The prophet package is required for cost and token forecasts."
        ) from error

    training = daily[["Date", target, "IsWeekend"]].rename(
        columns={"Date": "ds", target: "y"}
    )
    training["USHoliday"] = _holiday_regressor(training["ds"])
    model = Prophet(interval_width=0.95, daily_seasonality=False, weekly_seasonality=True)
    model.add_regressor("USHoliday")
    model.add_regressor("IsWeekend")
    model.fit(training)

    future = model.make_future_dataframe(periods=FORECAST_DAYS, freq="D", include_history=True)
    future["IsWeekend"] = (future["ds"].dt.dayofweek >= 5).astype(int)
    future["USHoliday"] = _holiday_regressor(future["ds"])
    forecast = model.predict(future)[["ds", "yhat", "yhat_lower", "yhat_upper"]]
    forecast["yhat"] = forecast["yhat"].clip(lower=0)
    forecast["yhat_lower"] = forecast["yhat_lower"].clip(lower=0)
    forecast["yhat_upper"] = forecast["yhat_upper"].clip(lower=0)
    forecast["Model"] = "Prophet"
    forecast["Metric"] = target
    return forecast


def _plot_prophet(daily: pd.DataFrame, forecast: pd.DataFrame, target: str, output_path: Path) -> None:
    """Save an actual-versus-forecast chart with a confidence band."""
    figure, axis = plt.subplots(figsize=(12, 6))
    axis.plot(daily["Date"], daily[target], label="Actual", color="#1f4e79")
    axis.plot(forecast["ds"], forecast["yhat"], label="Prophet forecast", color="#d95f02")
    axis.fill_between(
        forecast["ds"], forecast["yhat_lower"], forecast["yhat_upper"],
        color="#d95f02", alpha=0.2, label="95% interval",
    )
    axis.axvline(daily["Date"].max(), color="#555555", linestyle="--", linewidth=1)
    axis.set(title=f"Daily {target} - Prophet 30-day forecast", xlabel="Date", ylabel=target)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


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


def _mape(actual: pd.Series, predicted: pd.Series) -> float:
    """Calculate MAPE while excluding zero-valued observations."""
    nonzero = actual.ne(0)
    if not nonzero.any():
        return float("nan")
    return float((actual[nonzero] - predicted[nonzero]).abs().div(actual[nonzero]).mean() * 100)


def forecast_requests_arima(daily: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float | str]]:
    """Evaluate ARIMA on the last 30 days and forecast the next 30 days."""
    series = daily.set_index("Date")["RequestCount"].astype(float)
    if len(series) <= FORECAST_DAYS * 2:
        raise ValueError("At least 61 daily observations are required for the ARIMA holdout.")
    train = series.iloc[:-FORECAST_DAYS]
    test = series.iloc[-FORECAST_DAYS:]
    order, _, _ = _select_arima_order(train)
    holdout_model = ARIMA(train, order=order, enforce_stationarity=False).fit()
    holdout_prediction = holdout_model.forecast(steps=FORECAST_DAYS)
    adf_result = adfuller(train, autolag="AIC")

    full_model = ARIMA(series, order=order, enforce_stationarity=False).fit()
    forecast_values = full_model.forecast(steps=FORECAST_DAYS).clip(lower=0)
    forecast_dates = pd.date_range(series.index.max() + pd.Timedelta(days=1), periods=FORECAST_DAYS)
    forecast = pd.DataFrame({
        "ds": forecast_dates,
        "yhat": forecast_values.to_numpy(),
        "Model": "ARIMA",
        "Metric": "RequestCount",
    })
    metrics: dict[str, float | str] = {
        "order": str(order),
        "AIC": float(full_model.aic),
        "BIC": float(full_model.bic),
        "MAPE": _mape(test, holdout_prediction),
        "ADFStatistic": float(adf_result[0]),
        "ADF p-value": float(adf_result[1]),
        "ADFStationaryAt5Pct": str(adf_result[1] < 0.05),
    }
    return forecast, metrics


def _plot_comparison(daily: pd.DataFrame, forecasts: list[pd.DataFrame], output_path: Path) -> None:
    """Save one comparison chart for all model forecasts."""
    figure, axes = plt.subplots(3, 1, figsize=(13, 12), sharex=True)
    configurations = [("CostUSD", "Cost (USD)"), ("TotalTokens", "Total tokens"), ("RequestCount", "Requests")]
    for axis, (metric, label) in zip(axes, configurations):
        actual = daily[["Date", metric]]
        axis.plot(actual["Date"], actual[metric], label="Actual", color="#1f4e79")
        for forecast in forecasts:
            selected = forecast[forecast["Metric"] == metric]
            if not selected.empty:
                axis.plot(selected["ds"], selected["yhat"], label=forecast["Model"].iloc[0])
        axis.set_ylabel(label)
        axis.legend()
    axes[0].set_title("30-day forecast comparison")
    axes[-1].set_xlabel("Date")
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def run_forecasting(input_path: Path, output_dir: Path) -> dict[str, float | str]:
    """Run both forecasting methods and write CSV, PNG, and metric outputs."""
    daily = build_daily_series(load_master(input_path))
    output_dir.mkdir(parents=True, exist_ok=True)
    prophet_forecasts: list[pd.DataFrame] = []
    for target, filename in [("CostUSD", "forecast_cost_30d.csv"), ("TotalTokens", "forecast_tokens_30d.csv")]:
        forecast = _prophet_forecast(daily, target)
        forecast[forecast["ds"] > daily["Date"].max()].to_csv(output_dir / filename, index=False)
        _plot_prophet(daily, forecast, target, output_dir / f"forecast_{target.lower()}_prophet.png")
        prophet_forecasts.append(forecast)

    arima_forecast, metrics = forecast_requests_arima(daily)
    arima_forecast.to_csv(output_dir / "forecast_requests_30d.csv", index=False)
    pd.DataFrame([metrics]).to_csv(output_dir / "arima_metrics.csv", index=False)
    _plot_comparison(daily, [*prophet_forecasts, arima_forecast], output_dir / "forecast_comparison.png")
    return metrics


def parse_args() -> argparse.Namespace:
    """Parse input and output paths."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/processed/fact_master.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/forecasts"))
    return parser.parse_args()


def main() -> None:
    """Run the forecasting entry point."""
    arguments: Any = parse_args()
    metrics = run_forecasting(arguments.input, arguments.output_dir)
    print(f"ARIMA {metrics['order']}: AIC={metrics['AIC']:.2f}, BIC={metrics['BIC']:.2f}, MAPE={metrics['MAPE']:.2f}%")
    print(f"ADF p-value={metrics['ADF p-value']:.4f}; outputs written to {arguments.output_dir}")


if __name__ == "__main__":
    main()