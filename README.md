# AI Usage Analytics Dashboard

This project uses a synthetic dataset generated for portfolio purposes. Results describe that synthetic data only and are not evidence about real users or real systems.

## Problem statement

Teams need a consistent way to explore AI request volume, cost, latency, adoption,
quality feedback, client usage patterns, statistical outliers, and short-horizon
planning estimates. This repository demonstrates that process on a fixed,
synthetic request snapshot.

**Intended users:** product, finance, platform, and customer-success teams reviewing
an analytical prototype.

**Solution:** validate a star-schema request snapshot, enrich it into reusable CSV
artifacts, score documented analytical signals, evaluate forecast alternatives, and
present the checked-in outputs in Streamlit and Power BI companion assets.

## Technology stack

- Python, pandas, NumPy, scikit-learn, Prophet, statsmodels, and DuckDB
- Streamlit and Altair for the read-only dashboard
- Power BI companion assets, DAX measures, and Python visual examples
- CSV star-schema inputs and CSV/JSON analytical outputs

## Live demo

[Open the Streamlit dashboard](https://ai-usage-analytics.streamlit.app/)

The Streamlit entry point is [`app/streamlit_app.py`](app/streamlit_app.py).

## Key features

- Validated joins from six dimensions to the request fact table
- Data-quality checks and reusable client, model, region, and feature aggregates
- Client-level churn-risk scoring using a proxy label from synthetic behaviour rules
- Satisfaction estimation trained only on rows with observed synthetic ratings
- Isolation Forest and rule-based statistical outlier signals
- Portfolio-scale comparison of daily forecast models over a 30-day horizon
- Streamlit pages for overview, cost/models, client health, forecasts, anomalies,
  data quality, and methodology

## Architecture

```text
data/raw/*.csv
    |
    v
src/data_pipeline.py
    |
    +--> data/processed/fact_master.csv and aggregates
    |
    +--> src/ml_scoring.py
    |       +--> data/processed/fact_with_ml.csv
    |       +--> data/processed/client_churn_scores.csv
    |       `--> models/*.pkl
    |
    `--> src/forecasting.py
            `--> outputs/evaluation/ and outputs/forecasts/

processed/model/forecast CSVs --> app/streamlit_app.py
processed CSVs and Power BI assets --> companion reporting views
```

## Data description and quality approach

All data in this repository is synthetic. The included snapshot contains one
synthetically generated request fact table and six supporting dimensions. The
repository shows the generated CSV snapshot and its downstream artifacts; it does
not include a separate data-generation script.

The ETL validates required columns, key uniqueness, foreign-key membership,
non-negative measures, and unusually high costs before writing processed outputs.
The current quality report records 13,500 fact rows, six dimension tables, no
critical findings, and one warning: 348 cost values exceed mean plus three
standard deviations. See [`data/processed/data_quality_report.md`](data/processed/data_quality_report.md).

## ML and forecasting methodology

The model details and interpretation boundaries are documented in the
[`docs/`](docs/) directory. In brief:

- **Churn:** `ChurnSignal` is a proxy label created from synthetic behaviour rules
  (`DaysSinceLastRequest > 45` and `QuotaUtilizationPct < 20`), not observed churn.
  The checked-in evaluation has zero positive proxy examples and therefore no
  valid two-class holdout.
- **Satisfaction:** the regressor learns from the subset of synthetic requests
  with recorded satisfaction scores. Predictions for unrated rows are estimates,
  not certain satisfaction outcomes.
- **Anomalies:** Isolation Forest and Tukey-fence outputs are statistical flags on
  synthetic request data, not confirmed incidents, failures, or root causes.
- **Forecasting:** model comparison uses a chronological 30-day holdout. Forecasts
  are portfolio-scale evaluation and planning estimates, not production forecasting
  or capacity commitments.

## Key findings

Each statement below is calculated from checked-in synthetic outputs.

1. **On synthetic data:** GPT-4o accounts for **$89.8403**, or **61.79%**, of the
   **$145.4045** total API cost across 3,727 requests.
2. **On synthetic data:** GPT-4o-mini averages **284.97 ms** latency and **4.086**
   observed satisfaction, while Claude-3.5-Sonnet averages **663.24 ms** and
   **4.124**. Overall request success is **98.43%**.
3. **On synthetic data:** **4,705 of 13,500 requests (34.85%)** have a recorded
   satisfaction score; the observed rated-request mean is **4.098** and does not
   generalize to unrated requests.
4. **On synthetic data:** model-level success rates range from **98.23% to 99.06%**,
   and observed model-level satisfaction means range from **4.076 to 4.124**.
5. **On synthetic data:** the statistical anomaly signal flags **270 requests
   (2.0%)**, and the selected ARIMA request model has **6.98% holdout MAPE**.

## Power BI companion

The repository includes Power BI companion assets for the synthetic dataset:
[`power_bi/`](power_bi/), [`power_bi/DAX measures.md`](power_bi/DAX%20measures.md),
and [`power_bi/py_visuals.py`](power_bi/py_visuals.py). No `.pbix` file is checked into this checkout, so Power BI assets are optional and
are not required to run the Streamlit app.

## Local setup

From the repository root, create an environment and install the project
requirements.

**Linux/macOS**

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

**Windows PowerShell**

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Run the checked-in Streamlit snapshot:

```bash
streamlit run app/streamlit_app.py
```

Run tests:

```bash
python -m pytest
```

The optional refresh scripts are `python src/data_pipeline.py`,
`python src/ml_scoring.py`, `python src/forecasting.py`, and
`python sql/run_analysis.py`. They write the existing processed, model, SQL, and
forecast artifacts; no Python source is modified by this documentation repair.

## Deployment

The live deployment is the [Streamlit demo](https://ai-usage-analytics.streamlit.app/).
The app entry point is `app/streamlit_app.py`, and the runtime dependency file is
[`requirements.txt`](requirements.txt). The repository documents no other
deployment service or production infrastructure.

## Limitations and ethical considerations

All figures, labels, model scores, anomaly flags, and forecasts are based on
synthetic data. Churn is a behaviour-rule proxy rather than observed cancellation
or renewal; satisfaction is incomplete and selection-biased; anomalies are
statistical signals rather than confirmed incidents; and forecasts are short-term
portfolio estimates rather than production commitments. These outputs should not
be used for consequential decisions about real people, customers, or systems.
See [`docs/limitations.md`](docs/limitations.md) and
[`docs/model_card.md`](docs/model_card.md).

## Future improvements

- Add independently observed churn outcomes and time-aware validation.
- Improve satisfaction-feedback coverage and quantify selection bias.
- Add rolling-origin forecast evaluation and drift monitoring on new data.
- Add governed ingestion, lineage, access controls, and review processes before
  any real operational use.

## Documentation

- [`docs/data_dictionary.md`](docs/data_dictionary.md): source and derived fields
- [`docs/architecture.md`](docs/architecture.md): data lineage and refresh order
- [`docs/forecasting_methodology.md`](docs/forecasting_methodology.md): forecast comparison
- [`docs/limitations.md`](docs/limitations.md): interpretation boundaries
- [`docs/model_card.md`](docs/model_card.md): model purposes, metrics, and limitations
