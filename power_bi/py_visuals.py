import matplotlib.pyplot as plt
import pandas as pd
# 'dataset' is the Power BI injected dataframe
dataset = dataset  # already injected by Power BI  # pyright: ignore[reportUndefinedVariable]

import numpy as np
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose


# Change this value when using the file in a different Power BI Python visual.
VISUAL = "heatmap"  # "heatmap", "decomposition", or "radar"


def _require_columns(frame: pd.DataFrame, columns: list[str]) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f"Power BI dataset is missing required columns: {missing}")


def render_heatmap(frame: pd.DataFrame) -> None:
    """Plot request counts by client and feature using a sequential palette."""
    _require_columns(frame, ["ClientName", "FeatureName"])
    usage = frame.pivot_table(
        index="ClientName",
        columns="FeatureName",
        values="UsageID" if "UsageID" in frame else "ClientKey",
        aggfunc="count",
        fill_value=0,
    )
    usage = usage.sort_index().sort_index(axis=1)

    figure_width = max(9, min(18, 3 + 0.45 * len(usage.columns)))
    figure_height = max(6, min(18, 2.5 + 0.24 * len(usage.index)))
    plt.figure(figsize=(figure_width, figure_height))
    sns.heatmap(
        usage,
        annot=True,
        fmt="g",
        cmap="YlGnBu",
        linewidths=0.35,
        linecolor="white",
        cbar_kws={"label": "Request count"},
    )
    plt.title("Client x Feature Usage Intensity")
    plt.xlabel("Feature")
    plt.ylabel("Client")
    plt.tight_layout()


def render_decomposition(frame: pd.DataFrame) -> None:
    """Show observed daily requests and their weekly decomposition."""
    _require_columns(frame, ["Date"])
    dates = pd.to_datetime(frame["Date"], errors="coerce")
    daily = (
        frame.assign(Date=dates)
        .dropna(subset=["Date"])
        .groupby("Date")
        .size()
        .asfreq("D", fill_value=0)
    )
    if len(daily) < 15:
        raise ValueError("At least 15 daily observations are required for decomposition.")

    decomposition = seasonal_decompose(daily, model="additive", period=7, extrapolate_trend="freq")
    figure, axes = plt.subplots(4, 1, figsize=(12, 9), sharex=True)
    series = [
        (daily, "Observed requests"),
        (decomposition.trend, "Trend"),
        (decomposition.seasonal, "Seasonality"),
        (decomposition.resid, "Residual"),
    ]
    for axis, (values, label) in zip(axes, series):
        axis.plot(values.index, values, color="#176b87", linewidth=1.5)
        axis.set_ylabel(label)
        axis.grid(alpha=0.25)
    axes[0].set_title("Daily Request Counts: Weekly Decomposition")
    axes[-1].set_xlabel("Date")
    figure.tight_layout()


def _min_max_normalize(values: pd.Series, higher_is_better: bool) -> pd.Series:
    minimum = values.min()
    maximum = values.max()
    if maximum == minimum:
        normalized = pd.Series(0.5, index=values.index)
    else:
        normalized = (values - minimum) / (maximum - minimum)
    return normalized if higher_is_better else 1 - normalized


def render_radar(frame: pd.DataFrame) -> None:
    """Compare six models across five normalized performance dimensions."""
    required = [
        "ModelName", "TotalCost", "AvgLatency", "AvgSatisfaction",
        "AvgTokenEfficiency", "SuccessRate",
    ]
    _require_columns(frame, required)
    model_metrics = frame[required].drop_duplicates("ModelName").set_index("ModelName")
    if len(model_metrics) != 6:
        raise ValueError(f"Expected 6 models for the radar chart, found {len(model_metrics)}.")

    dimensions = {
        "Cost": ("TotalCost", False),
        "Latency": ("AvgLatency", False),
        "Satisfaction": ("AvgSatisfaction", True),
        "Token Efficiency": ("AvgTokenEfficiency", True),
        "Success Rate": ("SuccessRate", True),
    }
    normalized = pd.DataFrame({
        label: _min_max_normalize(model_metrics[column].astype(float), higher_is_better)
        for label, (column, higher_is_better) in dimensions.items()
    })

    angles = np.linspace(0, 2 * np.pi, len(dimensions), endpoint=False).tolist()
    angles += angles[:1]
    figure, axis = plt.subplots(figsize=(9, 9), subplot_kw={"polar": True})
    colors = plt.cm.tab10(np.linspace(0, 1, len(normalized)))
    for (model_name, row), color in zip(normalized.iterrows(), colors):
        values = row.tolist()
        values += values[:1]
        axis.plot(angles, values, linewidth=1.8, label=model_name, color=color)
        axis.fill(angles, values, color=color, alpha=0.07)
    axis.set_thetagrids(np.degrees(angles[:-1]), list(dimensions))
    axis.set_ylim(0, 1)
    axis.set_title("Normalized Model Performance Comparison", pad=25)
    axis.legend(loc="upper left", bbox_to_anchor=(1.05, 1.05), frameon=False)
    figure.tight_layout()


if VISUAL == "heatmap":
    render_heatmap(dataset.copy())
elif VISUAL == "decomposition":
    render_decomposition(dataset.copy())
elif VISUAL == "radar":
    render_radar(dataset.copy())
else:
    raise ValueError(f"Unknown VISUAL {VISUAL!r}; choose heatmap, decomposition, or radar.")

plt.show()