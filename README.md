# AI Usage Analytics Dashboard

An analytics workflow for AI request usage, operational performance, client engagement, and churn risk. Power BI consumes the processed CSV exports.

## Repository Layout

- `src/`: data preparation and machine-learning pipeline scripts.
- `data/raw/`: source fact and dimension CSV files.
- `data/processed/`: enriched tables, aggregates, quality reports, and ML-scored exports.
- `models/`: locally generated model binaries; ignored by Git.
- `notebooks/`: exploratory and data-quality notebooks.
- `sql/`: SQL analysis scripts.
- `power_bi/`: Power BI dashboard files.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e .
```

## Run the Pipelines

From the repository root:

```powershell
python src/data_pipeline.py
python src/ml_scoring.py
```

The data pipeline reads from `data/raw/` and writes analytical tables to `data/processed/`. The ML pipeline reads `data/processed/fact_master.csv`, writes `data/processed/fact_with_ml.csv`, and creates model files under `models/`.

## Data Notes

The processed CSV files currently provide a reproducible snapshot for the dashboard. They can be regenerated from the raw tables with the commands above. Model binaries are not committed; rerun the ML pipeline after cloning the repository.
