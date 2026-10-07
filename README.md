# AI Usage Intelligence Platform: Enterprise LLM Analytics at Scale

## Executive Summary

This project turns enterprise LLM request logs into an operational decision system for product, finance, and customer-success teams. It answers the questions that matter in a production AI business: which models and regions drive spend, whether lower-cost models preserve service quality, where latency affects satisfaction, which features are actually adopted, which clients show churn signals, and how demand and cost may evolve over the next 30 days. The workflow combines a SQL-style star schema, pandas transformations, machine-learning scores, time-series forecasts, and Power BI-ready exports in a reproducible Python pipeline.

## Tech Stack

- **Python:** reproducible data preparation, scoring, and forecasting pipelines
- **pandas / NumPy:** joins, validation, feature engineering, aggregation, and numerical computation
- **scikit-learn:** logistic regression, random forest regression, gradient boosting fallback, cross-validation, and Isolation Forest anomaly detection
- **Prophet / statsmodels:** holiday- and weekend-aware cost/token forecasts and ARIMA request forecasting
- **Power BI / DAX:** interactive KPI reporting, calculated columns, measures, and time intelligence
- **seaborn / matplotlib:** heatmaps, model comparisons, decomposition, and forecast plots
- **SQL-style star schema:** one usage fact table connected to date, client, model, feature, region, and subscription dimensions

## Architecture

```text
								 +------------------+
								 |   Dim_Date       |
								 +--------+---------+
											 |
+-------------+   +--------------+--------------+   +----------------+
| Dim_Client  |---|                              |---|  Dim_Model     |
+-------------+   |      Fact_AI_Usage           |   +----------------+
+-------------+   |   (13,500 usage events)      |   +----------------+
| Dim_Feature |---|                              |---|  Dim_Region    |
+-------------+   +--------------+--------------+   +----------------+
											 |
								 +--------+---------+
								 | Dim_Subscription |
								 +------------------+

  raw CSVs
		|
		v
  data_pipeline.py  -->  fact_master.csv + aggregate tables + quality report
		|
		+--> ml_scoring.py  -->  churn scores + satisfaction predictions + anomaly flags
		|
		+--> forecasting.py -->  Prophet cost/token forecasts + ARIMA request forecast
												  |
												  v
								  Power BI / DAX / Python visuals
```

## Key Analytical Findings

The findings below are calculated from the included processed snapshot, covering 13,500 requests and 289,706,666 tokens.

1. **GPT-4o dominates recorded spend.** GPT-4o generated **$89.8403**, or **61.79%** of the **$145.4045** total API cost, despite representing 3,727 requests. This makes model routing and substitution the clearest cost-leverage opportunity.
2. **Latency has a measurable quality tradeoff, but not a simple success-rate tradeoff.** The fastest model, GPT-4o-mini, averaged **284.97 ms** and a **4.086** satisfaction score. The slowest, Claude-3.5-Sonnet, averaged **663.24 ms** and **4.124** satisfaction. That is **2.33x** the latency for only a **0.038-point** satisfaction difference; overall request success remained high at **98.43%**.
3. **Satisfaction coverage is the main measurement gap.** Only **4,705 of 13,500 requests (34.85%)** contain a satisfaction score. Among rated requests, mean satisfaction was **4.098**, but that result cannot be generalized confidently to the unrated majority without improving feedback capture or modeling selection bias.
4. **Quality is broadly consistent across models.** Model-level success rates range from **98.23% to 99.06%**, while average satisfaction ranges from **4.076 to 4.124**. This narrow service-quality band strengthens the business case for cost-aware routing rather than treating the most expensive model as universally necessary.
5. **The monitoring layer finds targeted exceptions and supports planning.** Isolation Forest flags **270 requests (2.0%)** as anomalies, consistent with the configured contamination rate, while the ARIMA request forecast uses order **(0, 1, 2)** and achieves **6.98% holdout MAPE**. These outputs provide a starting point for spend monitoring and capacity planning, not a substitute for production backtesting.

## Project Structure

```text
.
├── pyproject.toml
├── README.md
├── data/
│   ├── raw/
│   │   ├── Dim_Client.csv
│   │   ├── Dim_Date.csv
│   │   ├── Dim_Feature.csv
│   │   ├── Dim_Model.csv
│   │   ├── Dim_Region.csv
│   │   ├── Dim_Subscription.csv
│   │   └── Fact_AI_Usage.csv
│   └── processed/
│       ├── agg_client_monthly.csv
│       ├── agg_feature_adoption.csv
│       ├── agg_model_performance.csv
│       ├── agg_region_monthly.csv
│       ├── client_churn_scores.csv
│       ├── data_quality_report.json
├── data_quality_report.md
│       ├── fact_master.csv
│       └── fact_with_ml.csv
├── models/
│   ├── anomaly_model.pkl
│   ├── churn_model.pkl
│   └── satisfaction_model.pkl
├── outputs/forecasts/
│   ├── arima_metrics.csv
│   ├── forecast_comparison.png
│   ├── forecast_cost_30d.csv
│   ├── forecast_costusd_prophet.png
│   ├── forecast_requests_30d.csv
│   ├── forecast_tokens_30d.csv
│   └── forecast_totaltokens_prophet.png
├── power_bi/
│   ├── dashboard.pbix
│   ├── dax_columns.dax
│   ├── dax_measures.dax
│   └── py_visuals.py
├── sql/analysis.sql
└── src/
	 ├── data_pipeline.py
	 ├── forecasting.py
	 └── ml_scoring.py
```

The `models/` and `outputs/` artifacts are generated locally and can be recreated from the raw snapshot. Python cache directories are intentionally omitted from the logical project tree.

## How to Run

1. **Create and activate a Python environment** from the repository root. Python **3.10 or newer** is required.

	```powershell
	python -m venv .venv
	.\.venv\Scripts\Activate.ps1
	python -m pip install --upgrade pip
	python -m pip install -e .
	```

2. **Build the analytical master and aggregates.**

	```powershell
	python src/data_pipeline.py
	```

	This validates the raw fact/dimension tables, enriches the fact table, and writes exports to
	`data/processed/`. Validation findings are written to both the human-readable
	`data_quality_report.md` and machine-readable `data_quality_report.json`. Critical
	validation failures are reported and cause the command to exit unsuccessfully without
	deleting invalid source records.

3. **Train scoring models and create ML features.**

	```powershell
	python src/ml_scoring.py
	```

	This writes `fact_with_ml.csv`, `client_churn_scores.csv`, and model artifacts to `models/`.

4. **Generate forecasts and forecast plots.**

	```powershell
	python src/forecasting.py
	```

	This writes 30-day Prophet and ARIMA outputs to `outputs/forecasts/`.

5. **Open the dashboard.** Open `power_bi/dashboard.pbix` in Power BI Desktop and refresh the imported processed tables. The DAX definitions and Python visual examples are in the same directory. For local Python visual development, install the plotting extras with `python -m pip install matplotlib seaborn` if they are not already available.

## Skills Demonstrated

| Component | Data science competency |
| --- | --- |
| Raw fact and dimensions | Data modeling, relational joins, and star-schema design |
| Data quality checks | Validation of nulls, foreign keys, numeric ranges, and outliers |
| Enrichment in `data_pipeline.py` | Feature engineering: token efficiency, quota utilization, response tiers, cohorts, and cost variance |
| Churn scoring | Client-level aggregation, behavioral labeling, logistic regression, and ROC AUC cross-validation |
| Satisfaction model | Mixed numeric/categorical preprocessing, one-hot encoding, random forest regression, and RMSE/MAE/R2 evaluation |
| Anomaly detection | Unsupervised learning with Isolation Forest and operational monitoring |
| Forecasting | Time-series aggregation, stationarity testing, ARIMA model selection, Prophet regressors, holdout evaluation, and uncertainty intervals |
| SQL analysis | Window functions, RFM segmentation, adoption funnels, regional growth, and substitution analysis |
| Power BI and DAX | KPI design, filter-context-aware measures, time intelligence, cost economics, and performance dashboards |
| Python visuals | Exploratory communication through heatmaps, decomposition plots, and normalized model comparisons |

## Future Enhancements

- Add real-time ingestion with **Azure Event Hubs** and incremental refresh for operational monitoring.
- Build an **LLM cost optimizer recommender** that proposes model substitutions subject to latency, success-rate, and satisfaction constraints.
- Add a **dbt transformation layer** with source tests, documented lineage, and warehouse-native incremental models.
- Add **MLflow experiment tracking** for model versions, parameters, metrics, artifacts, and promotion decisions.
- Expand evaluation with time-based cross-validation, calibrated churn probabilities, drift monitoring, and explicit missing-feedback bias analysis.

## Documentation

- [Data dictionary](docs/data_dictionary.md): raw-table grain, keys, columns, types, meanings, missing values, and derived fields.
- [Architecture](docs/architecture.md): data lineage from raw CSVs through ETL, ML, forecasting, Power BI, and Streamlit.
- [Limitations](docs/limitations.md): interpretation boundaries for the portfolio data, labels, models, forecasts, and prototype workflow.
