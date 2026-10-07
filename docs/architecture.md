# Architecture and Data Lineage

The project is a reproducible analytical prototype built around CSV handoffs.
It is not described here as a deployed service. The core lineage is:

```text
Raw star-schema CSVs
        |
        v
ETL and data-quality checks (src/data_pipeline.py)
        |
        v
Processed master and aggregates (data/processed/)
        |
        +--------------------------+
        |                          |
        v                          v
ML scoring (src/ml_scoring.py)  Forecasting (src/forecasting.py)
        |                          |
        v                          v
fact_with_ml.csv, scores,       30-day forecast CSVs and plots
model artifacts
        \                          /
         +------------------------+
                    |
                    v
Power BI semantic/report layer (power_bi/)
                    |
                    v
Streamlit presentation layer (when used by the project UI)
```

## 1. Raw data

The source layer is a small star schema in [`data/raw/`](../data/raw/):
`Fact_AI_Usage.csv` stores one request per row, while six dimensions provide
date, client, model, feature, region, and subscription context. The fact table
retains foreign keys so joins can be checked for many-to-one cardinality and
referential integrity.

## 2. ETL and quality checks

[`src/data_pipeline.py`](../src/data_pipeline.py) loads all seven raw tables,
checks required columns and numeric types, checks null/duplicate keys and
foreign-key membership, checks negative measures, and reports unusually high
cost values. The current processed quality report records 13,500 fact rows and
one non-critical finding: 348 cost values exceed mean plus three standard
deviations.

The ETL then normalizes dates, performs validated left joins, and creates the
request-grain `fact_master.csv`. It also writes four reusable aggregate tables
for client-month, model, region-month, and feature views. This stage is the
authoritative place for quota, cost, latency, lifecycle, and satisfaction
classification calculations.

## 3. Processed data contract

The processed directory is the handoff between deterministic transformation and
modeling. `fact_master.csv` contains raw request facts, dimension attributes,
and ETL-derived fields. The aggregate CSVs reduce repeated computation for
summary visuals. The full definitions are in the
[data dictionary](data_dictionary.md).

## 4. Machine learning

[`src/ml_scoring.py`](../src/ml_scoring.py) consumes `fact_master.csv`:

- Client behavior is aggregated to client grain and scored with a churn model.
  The label is a transparent recency/quota proxy, not an observed churn event.
- Satisfaction is modeled only from rows with a non-null
  `SatisfactionScore`, then predictions are appended to all requests.
- Isolation Forest uses `CostUSD`, `TotalTokens`, and `LatencyMs` to flag
  request-level anomalies with a configured 2% contamination rate.

The stage writes `fact_with_ml.csv`, `client_churn_scores.csv`, and serialized
model artifacts under `models/`. The outputs are analytical signals and should
not be treated as automated decisions.

## 5. Forecasting

[`src/forecasting.py`](../src/forecasting.py) uses the same processed master,
aggregates requests into a complete daily calendar, and fills dates with no
recorded requests as zero. Prophet forecasts daily cost and total tokens for
the next 30 days with weekly, weekend, and US-holiday regressors and 95%
interval columns. A small non-seasonal ARIMA search forecasts daily request
count and evaluates the final 30 historical days as a holdout.

Forecast files and plots are written to `outputs/forecasts/`. They are separate
from the request-level tables and must be refreshed when the processed input
changes.

## 6. Power BI and Streamlit

Power BI consumes the processed and forecast CSV exports through the report in
[`power_bi/`](../power_bi/). DAX measures and calculated columns provide the
semantic/reporting layer, while Python visuals extend selected report views.
The expected primary analytical table is `fact_with_ml`.

Streamlit is the interactive presentation layer when the project UI is run
locally. It should consume the same processed/model/forecast artifacts rather
than independently recreating the raw joins. Power BI and Streamlit are
downstream consumers; neither changes the source data or model training logic.

## Refresh order

1. Run `python src/data_pipeline.py`.
2. Run `python src/ml_scoring.py`.
3. Run `python src/forecasting.py`.
4. Refresh the Power BI imports and/or restart the local Streamlit view.

The repository-relative scripts and CSV artifacts make the sequence reproducible
for this project snapshot, but they do not constitute scheduling, monitoring,
access control, or a production deployment.
