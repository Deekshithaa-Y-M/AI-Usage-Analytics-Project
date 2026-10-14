import numpy as np
import pandas as pd

from src import forecasting


def _daily_series() -> pd.Series:
    dates = pd.date_range("2025-01-01", periods=40, freq="D")
    return pd.Series(np.linspace(10, 50, len(dates)), index=dates, name="CostUSD")


def test_build_daily_series_fills_missing_calendar_dates():
    master = pd.DataFrame({
        "Date": pd.to_datetime(["2025-01-01", "2025-01-03"]),
        "CostUSD": [1.0, 3.0],
        "TotalTokens": [10, 30],
        "UsageID": [1, 2],
    })
    daily = forecasting.build_daily_series(master)
    assert len(daily) == 3
    assert daily.loc[1, "CostUSD"] == 0


def test_forecast_has_expected_columns_numeric_values_and_horizon():
    series = _daily_series()
    forecast = forecasting._baseline_forecast(series, forecasting.FORECAST_DAYS, seasonal=False)
    forecast["ds"] = forecasting._future_dates(series, forecasting.FORECAST_DAYS)
    assert {"ds", "yhat", "yhat_lower", "yhat_upper"} <= set(forecast)
    assert len(forecast) == forecasting.FORECAST_DAYS
    assert pd.api.types.is_numeric_dtype(forecast["yhat"])
    assert forecast["yhat"].notna().all()


def test_prediction_interval_is_ordered_when_model_provides_one(monkeypatch):
    series = _daily_series()
    monkeypatch.setattr(forecasting, "_select_arima_order", lambda values: ((1, 0, 0), 0.0, 0.0))
    forecast = forecasting._fit_arima(series, forecasting.FORECAST_DAYS, "CostUSD")
    assert {"yhat", "yhat_lower", "yhat_upper", "order", "target"} <= set(forecast)
    assert (forecast["yhat_lower"] <= forecast["yhat_upper"]).all()


def test_forecast_date_validation_enforces_horizon_and_future_dates():
    series = _daily_series()
    valid = pd.DataFrame({"ds": forecasting._future_dates(series, forecasting.FORECAST_DAYS)})
    forecasting._validate_forecast_dates(series, valid)
    invalid = valid.iloc[:-1].copy()
    try:
        forecasting._validate_forecast_dates(series, invalid)
    except ValueError:
        pass
    else:
        raise AssertionError("An incomplete forecast should be rejected")


def test_evaluation_metrics_include_required_outputs():
    actual = pd.Series([1.0, 2.0, 4.0])
    predicted = pd.Series([1.0, 3.0, 3.0])
    metrics = forecasting._metrics(actual, predicted)
    assert {"MAE", "RMSE", "MAPE", "sMAPE"} == set(metrics)
    assert all(np.isfinite(value) for value in metrics.values())
