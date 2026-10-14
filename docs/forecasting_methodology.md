# Forecasting methodology

## Data frequency

The processed fact table is aggregated to a complete **daily** calendar. Days
with no recorded requests are represented as zeroes. This supports the weekly
seasonality used by the seasonal-naive and Prophet models.

## Forecast horizon

The horizon is **30 calendar days**. Every forecast date is validated to be
strictly after the last observed training date, and each output contains
exactly 30 dates.

## Train/test method

Model selection uses a chronological holdout: the final 30 observed days are
the test set and all earlier days are training data. No random split is used.
After evaluation, the selected model is refit on the complete daily history
before producing the future forecast.

## Models compared

For each of request volume (`RequestCount`), cost (`CostUSD`), and token usage
(`TotalTokens`), the pipeline compares:

- **Naive**, which repeats the final observed value.
- **Seasonal-naive**, which repeats the value from the same weekday one week
  earlier (when enough daily history exists).
- **ARIMA**, with a small order grid selected by BIC, then AIC.
- **Prophet**, with weekly seasonality plus US-holiday and weekend regressors.

The prior request ARIMA output and the cost/token forecast CSV names are
preserved. New holdout charts, selected-model forecast charts, and the model
comparison table are written alongside them.

## Metrics

The holdout comparison reports:

- **MAE** (mean absolute error), in the original unit.
- **RMSE** (root mean squared error), in the original unit and more sensitive
  to large misses.
- **MAPE**, excluding zero-valued actual observations.
- **sMAPE**, used as the primary selection metric because it remains defined
  when daily series contain zeroes.

The complete table is `outputs/evaluation/forecast_model_comparison.csv`.
`Selected=True` identifies the model with the lowest sMAPE, breaking ties with
RMSE. Selected models are also listed in
`outputs/evaluation/selected_forecast_models.csv`.

## Selected models

The selected model is determined from the current data whenever the
forecasting command runs; it is not hard-coded. The selected model for each
series is recorded in the evaluation CSV and in the command output.

## Limitations

These are short-horizon statistical planning estimates, not production
capacity commitments. The data is an observational snapshot, and future
traffic can change because of product launches, customer mix, model routing,
pricing, outages, or unrecorded events. Holiday effects are US-only.
Confidence/prediction intervals inherit the assumptions of the fitted model
and are not guaranteed coverage. A single 30-day holdout is useful for a
consistent comparison but is not a substitute for rolling-origin backtesting.
Zero-filled missing days may represent missing instrumentation rather than
true zero demand. Forecasts should therefore be monitored and refreshed with
new data before operational decisions.
