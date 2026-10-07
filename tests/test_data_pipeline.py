import json
import shutil
from pathlib import Path

import pandas as pd
import pytest

from src.data_pipeline import (
    DataQualityError,
    load_tables,
    run_pipeline,
    validate_data_quality,
    write_json_quality_report,
)


ROOT = Path(__file__).parents[1]


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
