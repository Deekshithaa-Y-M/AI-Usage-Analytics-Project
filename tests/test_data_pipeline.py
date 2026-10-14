import json
import shutil
from pathlib import Path

import pandas as pd
import pytest

from src.data_pipeline import (
    DIMENSION_REQUIRED_COLUMNS,
    FACT_REQUIRED_COLUMNS,
    DataQualityError,
    build_aggregates,
    enrich_fact_table,
    load_tables,
    load_tables,
    run_pipeline,
    validate_data_quality,
    write_json_quality_report,
)


ROOT = Path(__file__).parents[1]


def _minimal_tables() -> dict[str, pd.DataFrame]:
    """Build one valid fact row and one row for each required dimension."""
    dimensions = {
        "date": {"DateKey": 1, "Date": "2025-01-01", "Year": 2025,
                 "Quarter": "Q1", "Month": "2025-01", "MonthName": "January",
                 "WeekOfYear": 1, "DayOfWeek": "Wednesday", "IsWeekend": False},
        "client": {"ClientKey": 1, "ClientID": "C001", "ClientName": "Client 1",
                   "Industry": "Technology", "CompanySize": "Small",
                   "OnboardingDate": "2024-12-01", "AccountManager": "A", "Status": "Active"},
        "model": {"ModelKey": 1, "ModelName": "Model 1", "Provider": "Provider",
                  "ModelFamily": "Family", "ContextWindow": 1000,
                  "CostPer1kInput": 0.01, "CostPer1kOutput": 0.02, "IsLatest": True},
        "feature": {"FeatureKey": 1, "FeatureName": "Chat", "FeatureCategory": "Core",
                    "IsPremium": False},
        "region": {"RegionKey": 1, "RegionName": "US", "Country": "US", "TimeZoneOffset": -5},
        "subscription": {"SubscriptionTierKey": 1, "TierName": "Basic",
                          "MonthlyQuotaTokens": 10000, "OverageRateMultiplier": 1.5,
                          "SupportLevel": "Standard"},
    }
    fact = {
        "UsageID": 1, "DateKey": 1, "TimeKey": 1200, "ClientKey": 1, "ModelKey": 1,
        "FeatureKey": 1, "RegionKey": 1, "SubscriptionTierKey": 1, "TokensInput": 100,
        "TokensOutput": 200, "TotalTokens": 300, "LatencyMs": 250, "CostUSD": 0.05,
        "SuccessFlag": 1, "SatisfactionScore": 4.0, "RequestType": "Chat",
    }
    return {"fact": pd.DataFrame([fact]), **{
        name: pd.DataFrame([row]) for name, row in dimensions.items()
    }}


def test_load_tables_rejects_missing_required_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="Missing required CSV files"):
        load_tables(tmp_path)


def test_validate_data_quality_reports_missing_required_columns():
    tables = _minimal_tables()
    tables["fact"] = tables["fact"].drop(columns=["CostUSD"])
    issues = validate_data_quality(tables)
    assert "missing_columns_fact" in issues


def test_validate_data_quality_reports_duplicate_primary_key():
    tables = _minimal_tables()
    tables["client"] = pd.concat([tables["client"], tables["client"]], ignore_index=True)
    issues = validate_data_quality(tables)
    assert "duplicate_primary_key_client" in issues


def test_validate_data_quality_reports_invalid_foreign_key():
    tables = _minimal_tables()
    tables["fact"].loc[0, "ClientKey"] = 999
    issues = validate_data_quality(tables)
    assert "referential_integrity_ClientKey" in issues


def test_enrich_fact_table_preserves_rows_and_creates_derived_columns():
    tables = _minimal_tables()
    enriched = enrich_fact_table(tables)
    assert len(enriched) == len(tables["fact"])
    assert {"TokenEfficiencyRatio", "Month", "QuotaUtilizationPct", "ResponseSpeedTier"} <= set(enriched)


def test_build_aggregates_creates_all_reporting_outputs():
    master = enrich_fact_table(_minimal_tables())
    aggregates = build_aggregates(master)
    assert set(aggregates) == {
        "agg_client_monthly.csv", "agg_model_performance.csv",
        "agg_region_monthly.csv", "agg_feature_adoption.csv",
    }
    assert all(not frame.empty for frame in aggregates.values())


def test_validate_data_quality_reports_integrity_and_value_failures():
    tables = load_tables(ROOT / "data" / "raw")
    fact = tables["fact"].copy()
    fact.loc[0, "ClientKey"] = -999
    fact.loc[1, "TokensInput"] = -1
    fact.loc[2] = fact.loc[3]
    tables["fact"] = fact

    issues = validate_data_quality(tables)

    assert "referential_integrity_ClientKey" in issues
    assert "negative_TokensInput" in issues
    assert "duplicate_fact_records" in issues
    assert "duplicate_fact_usage_ids" in issues


def test_validate_data_quality_reports_missing_columns_and_dimension_duplicates():
    tables = load_tables(ROOT / "data" / "raw")
    tables["fact"] = tables["fact"].drop(columns=["LatencyMs"])
    tables["model"] = pd.concat([tables["model"], tables["model"].iloc[[0]]])

    issues = validate_data_quality(tables)

    assert "missing_columns_fact" in issues
    assert "duplicate_primary_key_model" in issues
    assert "many_to_one_cardinality_model" in issues


def test_json_quality_report_is_machine_readable_and_marks_critical_failures(tmp_path):
    tables = load_tables(ROOT / "data" / "raw")
    issues = {"negative_CostUSD": ["2 negative values"], "cost_outliers": ["1 outlier"]}

    output_path = tmp_path / "data_quality_report.json"
    write_json_quality_report(issues, tables, output_path)
    report = json.loads(output_path.read_text(encoding="utf-8"))

    assert report["critical_failure"] is True
    assert report["finding_count"] == 2
    assert {finding["name"] for finding in report["findings"]} == set(issues)


def test_run_pipeline_writes_reports_and_fails_on_critical_quality_issue(tmp_path):
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    shutil.copytree(ROOT / "data" / "raw", raw_dir)
    fact_path = raw_dir / "Fact_AI_Usage.csv"
    fact = pd.read_csv(fact_path)
    fact.loc[0, "CostUSD"] = -1
    fact.to_csv(fact_path, index=False)

    with pytest.raises(DataQualityError):
        run_pipeline(raw_dir, processed_dir)

    report = json.loads(
        (processed_dir / "data_quality_report.json").read_text(encoding="utf-8")
    )
    assert report["critical_failure"] is True
    assert (processed_dir / "data_quality_report.md").exists()
