"""Build the analytical dataset and summary tables for AI usage reporting."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DIMENSION_FILES: dict[str, str] = {
    "date": "Dim_Date.csv",
    "client": "Dim_Client.csv",
    "model": "Dim_Model.csv",
    "feature": "Dim_Feature.csv",
    "region": "Dim_Region.csv",
    "subscription": "Dim_Subscription.csv",
}
FACT_FILE = "Fact_AI_Usage.csv"
FACT_REQUIRED_COLUMNS = {
    "UsageID", "DateKey", "TimeKey", "ClientKey", "ModelKey", "FeatureKey",
    "RegionKey", "SubscriptionTierKey", "TokensInput", "TokensOutput",
    "TotalTokens", "LatencyMs", "CostUSD", "SuccessFlag",
    "SatisfactionScore", "RequestType",
}
DIMENSION_REQUIRED_COLUMNS = {
    "date": {"DateKey", "Date", "Year", "Quarter", "Month", "MonthName",
             "WeekOfYear", "DayOfWeek", "IsWeekend"},
    "client": {"ClientKey", "ClientID", "ClientName", "Industry", "CompanySize",
               "OnboardingDate", "AccountManager", "Status"},
    "model": {"ModelKey", "ModelName", "Provider", "ModelFamily", "ContextWindow",
              "CostPer1kInput", "CostPer1kOutput", "IsLatest"},
    "feature": {"FeatureKey", "FeatureName", "FeatureCategory", "IsPremium"},
    "region": {"RegionKey", "RegionName", "Country", "TimeZoneOffset"},
    "subscription": {"SubscriptionTierKey", "TierName", "MonthlyQuotaTokens",
                     "OverageRateMultiplier", "SupportLevel"},
}
DIMENSION_KEY_COLUMNS = {
    "date": "DateKey", "client": "ClientKey", "model": "ModelKey",
    "feature": "FeatureKey", "region": "RegionKey",
    "subscription": "SubscriptionTierKey",
}
FOREIGN_KEYS = {
    "DateKey": ("date", "DateKey"), "ClientKey": ("client", "ClientKey"),
    "ModelKey": ("model", "ModelKey"), "FeatureKey": ("feature", "FeatureKey"),
    "RegionKey": ("region", "RegionKey"),
    "SubscriptionTierKey": ("subscription", "SubscriptionTierKey"),
}
NUMERIC_FACT_COLUMNS = {
    "UsageID", "DateKey", "TimeKey", "ClientKey", "ModelKey", "FeatureKey",
    "RegionKey", "SubscriptionTierKey", "TokensInput", "TokensOutput",
    "TotalTokens", "LatencyMs", "CostUSD", "SuccessFlag",
}
CRITICAL_COLUMNS = [
    "UsageID", "DateKey", "ClientKey", "ModelKey", "FeatureKey",
    "RegionKey", "SubscriptionTierKey", "TokensInput", "TokensOutput",
    "TotalTokens", "LatencyMs", "CostUSD", "SuccessFlag",
]


class DataQualityError(ValueError):
    """Raised when source data cannot be safely enriched."""


def load_tables(raw_dir: Path) -> dict[str, pd.DataFrame]:
    """Load the fact table and all dimensions from a CSV directory."""
    filenames = {"fact": FACT_FILE, **DIMENSION_FILES}
    missing_files = [name for name in filenames.values() if not (raw_dir / name).exists()]
    if missing_files:
        raise FileNotFoundError(f"Missing required CSV files in {raw_dir}: {missing_files}")
    return {name: pd.read_csv(raw_dir / filename) for name, filename in filenames.items()}


def validate_data_quality(tables: dict[str, pd.DataFrame]) -> dict[str, list[str]]:
    """Inspect schema, integrity, completeness, numeric validity, and outliers."""
    fact = tables["fact"]
    issues: dict[str, list[str]] = {}

    required_columns = {"fact": FACT_REQUIRED_COLUMNS, **DIMENSION_REQUIRED_COLUMNS}
    for table_name, columns in required_columns.items():
        missing = sorted(columns - set(tables[table_name].columns))
        if missing:
            issues[f"missing_columns_{table_name}"] = [
                f"{len(missing)} required columns missing: {', '.join(missing)}"
            ]

    for column in NUMERIC_FACT_COLUMNS & set(fact.columns):
        if not pd.api.types.is_numeric_dtype(fact[column]):
            issues[f"invalid_type_fact_{column}"] = [
                f"expected numeric data, found {fact[column].dtype}"
            ]
    for table_name, key_column in DIMENSION_KEY_COLUMNS.items():
        dimension = tables[table_name]
        if key_column in dimension:
            if not pd.api.types.is_numeric_dtype(dimension[key_column]):
                issues[f"invalid_type_{table_name}_{key_column}"] = [
                    f"expected numeric data, found {dimension[key_column].dtype}"
                ]
            null_key_count = int(dimension[key_column].isna().sum())
            if null_key_count:
                issues[f"null_primary_key_{table_name}"] = [
                    f"{null_key_count} null {key_column} values"
                ]
            duplicate_count = int(dimension[key_column].duplicated(keep=False).sum())
            if duplicate_count:
                issues[f"duplicate_primary_key_{table_name}"] = [
                    f"{duplicate_count} rows have duplicate {key_column} values"
                ]
                issues[f"many_to_one_cardinality_{table_name}"] = [
                    f"{duplicate_count} rows prevent a many-to-one join on {key_column}"
                ]

    for table_name, column in (("date", "Date"), ("client", "OnboardingDate")):
        if column in tables[table_name]:
            invalid_date_count = int(
                pd.to_datetime(tables[table_name][column], errors="coerce").isna().sum()
            )
            if invalid_date_count:
                issues[f"invalid_type_{table_name}_{column}"] = [
                    f"{invalid_date_count} values are not valid dates"
                ]

    for column in CRITICAL_COLUMNS:
        if column not in fact:
            continue
        null_count = int(fact[column].isna().sum())
        if null_count:
            issues[f"null_{column}"] = [f"{null_count} null values"]

    if "UsageID" in fact:
        duplicate_count = int(fact.duplicated(keep=False).sum())
        if duplicate_count:
            issues["duplicate_fact_records"] = [
                f"{duplicate_count} rows are part of duplicate fact records"
            ]
        duplicate_id_count = int(fact["UsageID"].duplicated(keep=False).sum())
        if duplicate_id_count:
            issues["duplicate_fact_usage_ids"] = [
                f"{duplicate_id_count} rows have duplicate UsageID values"
            ]

    for fact_column, (dimension_name, dimension_column) in FOREIGN_KEYS.items():
        if fact_column not in fact or dimension_column not in tables[dimension_name]:
            continue
        valid_keys = set(tables[dimension_name][dimension_column].dropna())
        invalid_count = int((~fact[fact_column].isin(valid_keys)).sum())
        if invalid_count:
            issues[f"referential_integrity_{fact_column}"] = [
                f"{invalid_count} fact rows reference unknown dimension keys"
            ]

    for column in ["TokensInput", "TokensOutput", "TotalTokens", "LatencyMs", "CostUSD"]:
        if column not in fact:
            continue
        negative_count = int((fact[column] < 0).sum())
        if negative_count:
            issues[f"negative_{column}"] = [f"{negative_count} negative values"]

    if "CostUSD" not in fact:
        return issues
    cost_mean = fact["CostUSD"].mean()
    cost_std = fact["CostUSD"].std()
    if pd.notna(cost_std) and cost_std > 0:
        outlier_count = int((fact["CostUSD"] > cost_mean + 3 * cost_std).sum())
        if outlier_count:
            issues["cost_outliers"] = [
                f"{outlier_count} values exceed mean + 3 standard deviations"
            ]
    return issues


def _is_critical_issue(name: str) -> bool:
    """Return whether a finding means enrichment cannot be trusted."""
    return name != "cost_outliers"


def enrich_fact_table(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Join dimensions and derive operational, financial, and quota features."""
    fact = tables["fact"].copy()
    date_dimension = tables["date"].copy()
    client_dimension = tables["client"].copy()
    date_dimension["Date"] = pd.to_datetime(date_dimension["Date"])
    client_dimension["OnboardingDate"] = pd.to_datetime(client_dimension["OnboardingDate"])
    fact["DateKey"] = fact["DateKey"].astype("Int64")

    merge_specs = [
        (date_dimension, "DateKey"), (client_dimension, "ClientKey"),
        (tables["model"], "ModelKey"), (tables["feature"], "FeatureKey"),
        (tables["region"], "RegionKey"), (tables["subscription"], "SubscriptionTierKey"),
    ]
    for dimension, key in merge_specs:
        dimension_copy = dimension.copy()
        if key == "DateKey":
            dimension_copy[key] = dimension_copy[key].astype("Int64")
        fact = fact.merge(dimension_copy, on=key, how="left", validate="many_to_one")

    fact["TokenEfficiencyRatio"] = np.divide(
        fact["TokensOutput"], fact["TokensInput"],
        out=np.zeros(len(fact), dtype=float), where=fact["TokensInput"].to_numpy() != 0,
    )
    fact["CostPerSuccessfulRequest"] = fact["CostUSD"].where(fact["SuccessFlag"].eq(1), 0.0)
    fact["TheoreticalCostUSD"] = (
        fact["TokensInput"] / 1000 * fact["CostPer1kInput"]
        + fact["TokensOutput"] / 1000 * fact["CostPer1kOutput"]
    )
    fact["CostVariance"] = fact["CostUSD"] - fact["TheoreticalCostUSD"]
    fact["IsOverBudget"] = fact["CostVariance"] > 0.005
    fact["HourOfDay"] = fact["TimeKey"] // 100
    fact["PeakHour"] = fact["HourOfDay"].between(9, 17)
    fact["ResponseSpeedTier"] = np.select(
        [fact["LatencyMs"] < 400, fact["LatencyMs"] < 800],
        ["Fast", "Medium"], default="Slow",
    )
    fact["SatisfactionBucket"] = pd.cut(
        fact["SatisfactionScore"], bins=[-np.inf, 3.5, 4.5, np.inf],
        labels=["Low(<3.5)", "Mid(3.5-4.4)", "High(4.5-5.0)"], right=False,
    ).astype("string").fillna("Not Rated")
    fact["DaysSinceOnboarding"] = (fact["Date"] - fact["OnboardingDate"]).dt.days
    fact["IsFirstWeekClient"] = fact["DaysSinceOnboarding"].between(0, 7)
    fact["ClientAgeCohort"] = pd.cut(
        fact["DaysSinceOnboarding"], bins=[-np.inf, 30, 90, 180, np.inf],
        labels=["0-30 days", "31-90 days", "91-180 days", "180+ days"], right=True,
    ).astype("string")
    fact["Month"] = fact["Date"].dt.to_period("M").astype(str)
    fact = fact.sort_values(["ClientKey", "Date", "TimeKey", "UsageID"])
    fact["MonthlyTokensUsed"] = fact.groupby(["ClientKey", "Month"])["TotalTokens"].cumsum()
    fact["QuotaUtilizationPct"] = fact["MonthlyTokensUsed"] / fact["MonthlyQuotaTokens"] * 100
    fact["IsOverQuota"] = fact["QuotaUtilizationPct"] > 100
    fact["EffectiveCostWithOverage"] = fact["CostUSD"] * np.where(
        fact["IsOverQuota"], fact["OverageRateMultiplier"], 1.0
    )
    return fact.sort_values("UsageID").reset_index(drop=True)


def build_aggregates(master: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Create client, model, region, and feature reporting tables."""
    client_monthly = master.groupby(
        ["ClientKey", "ClientName", "Month"], as_index=False
    ).agg(
        TotalCost=("CostUSD", "sum"), TotalTokens=("TotalTokens", "sum"),
        RequestCount=("UsageID", "count"), AvgLatency=("LatencyMs", "mean"),
        SuccessRate=("SuccessFlag", "mean"), AvgSatisfaction=("SatisfactionScore", "mean"),
        UniqueModelsUsed=("ModelKey", "nunique"), PremiumFeatureUsagePct=("IsPremium", "mean"),
    )
    client_monthly["SuccessRate"] *= 100
    client_monthly["PremiumFeatureUsagePct"] *= 100

    model_performance = master.groupby(
        ["ModelKey", "ModelName", "Provider"], as_index=False
    ).agg(
        AvgLatency=("LatencyMs", "mean"), SuccessRate=("SuccessFlag", "mean"),
        AvgSatisfaction=("SatisfactionScore", "mean"), TotalCost=("CostUSD", "sum"),
        TotalRequests=("UsageID", "count"), AvgTokenEfficiency=("TokenEfficiencyRatio", "mean"),
    )
    model_performance["SuccessRate"] *= 100
    model_performance["CostRank"] = model_performance["TotalCost"].rank(method="min", ascending=False).astype(int)
    model_performance["LatencyRank"] = model_performance["AvgLatency"].rank(method="min").astype(int)

    region_monthly = master.groupby(
        ["RegionKey", "RegionName", "Month"], as_index=False
    ).agg(
        TotalCost=("CostUSD", "sum"), TotalTokens=("TotalTokens", "sum"),
        RequestCount=("UsageID", "count"), AvgLatency=("LatencyMs", "mean"),
        SuccessRate=("SuccessFlag", "mean"),
    )
    region_monthly["SuccessRate"] *= 100

    total_clients = master["ClientKey"].nunique()
    feature_adoption = master.groupby(
        ["FeatureKey", "FeatureName", "IsPremium"], as_index=False
    ).agg(
        UsageCount=("UsageID", "count"), PremiumAdoptionRate=("ClientKey", "nunique"),
        AvgSatisfactionByFeature=("SatisfactionScore", "mean"),
    )
    feature_adoption["PremiumAdoptionRate"] = feature_adoption["PremiumAdoptionRate"] / total_clients * 100
    return {
        "agg_client_monthly.csv": client_monthly,
        "agg_model_performance.csv": model_performance,
        "agg_region_monthly.csv": region_monthly,
        "agg_feature_adoption.csv": feature_adoption,
    }


def write_quality_report(
    issues: dict[str, list[str]], tables: dict[str, pd.DataFrame], output_path: Path
) -> None:
    """Write a Markdown report describing the source quality checks."""
    lines = [
        "# AI Usage Data Quality Report", "", "## Run Summary", "",
        f"- Fact rows: {len(tables['fact']):,}",
        f"- Dimension tables: {len(DIMENSION_FILES)}",
        f"- Quality checks with findings: {len(issues)}", "", "## Findings", "",
    ]
    lines.insert(
        7,
        f"- Critical findings: {sum(_is_critical_issue(name) for name in issues)}",
    )
    lines.extend(
        f"- **{name}**: {'; '.join(details)}" for name, details in issues.items()
    ) if issues else lines.append("- No quality issues detected.")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_json_quality_report(
    issues: dict[str, list[str]], tables: dict[str, pd.DataFrame], output_path: Path
) -> None:
    """Write a machine-readable version of the source quality checks."""
    report = {
        "fact_rows": len(tables["fact"]),
        "dimension_tables": len(DIMENSION_FILES),
        "finding_count": len(issues),
        "critical_failure": any(_is_critical_issue(name) for name in issues),
        "findings": [
            {
                "name": name,
                "severity": "critical" if _is_critical_issue(name) else "warning",
                "details": details,
            }
            for name, details in issues.items()
        ],
    }
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


def run_pipeline(raw_dir: Path, processed_dir: Path) -> pd.DataFrame:
    """Run extraction, checks, enrichment, aggregations, and file exports."""
    print(f"Loading source tables from {raw_dir}")
    tables = load_tables(raw_dir)
    issues = validate_data_quality(tables)
    print(f"Quality checks complete: {len(issues)} finding(s)")
    processed_dir.mkdir(parents=True, exist_ok=True)
    write_quality_report(issues, tables, processed_dir / "data_quality_report.md")
    write_json_quality_report(issues, tables, processed_dir / "data_quality_report.json")
    critical_issues = {
        name: details for name, details in issues.items() if _is_critical_issue(name)
    }
    if critical_issues:
        raise DataQualityError(
            "Critical data quality validation failed: "
            + "; ".join(f"{name}: {details[0]}" for name, details in critical_issues.items())
        )
    master = enrich_fact_table(tables)
    if len(master) != len(tables["fact"]):
        issues["fact_row_count_changed_after_enrichment"] = [
            f"expected {len(tables['fact'])} rows, found {len(master)}"
        ]
        write_quality_report(issues, tables, processed_dir / "data_quality_report.md")
        write_json_quality_report(issues, tables, processed_dir / "data_quality_report.json")
        raise DataQualityError(issues["fact_row_count_changed_after_enrichment"][0])
    master.to_csv(processed_dir / "fact_master.csv", index=False)
    for filename, aggregate in build_aggregates(master).items():
        aggregate.to_csv(processed_dir / filename, index=False)
    print(f"Wrote enriched master with {len(master):,} rows to {processed_dir}")
    return master


def parse_args() -> argparse.Namespace:
    """Parse paths for both the current repository and the target layout."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    return parser.parse_args()


def main() -> None:
    """Run the command-line pipeline entry point."""
    arguments: Any = parse_args()
    run_pipeline(arguments.raw_dir, arguments.processed_dir)


if __name__ == "__main__":
    main()