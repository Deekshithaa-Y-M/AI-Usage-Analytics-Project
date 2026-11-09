"""Chart-ready transformations shared by dashboard pages."""

from __future__ import annotations

import pandas as pd


def _altair():
    import altair as alt

    return alt


def top_rows(data: pd.DataFrame, value: str, label: str, limit: int = 10) -> pd.DataFrame:
    """Return the highest-value rows with readable chart column names."""
    return (
        data.nlargest(limit, value)[[label, value]]
        .set_index(label)
        .sort_values(value)
    )


def forecast_chart_data(forecast: pd.DataFrame) -> pd.DataFrame:
    """Prepare forecast and interval columns indexed by forecast date."""
    result = forecast.copy()
    result["Date"] = pd.to_datetime(result["ds"])
    return result.set_index("Date")[["yhat", "yhat_lower", "yhat_upper"]].sort_index()


def anomaly_counts(summary: pd.DataFrame) -> pd.DataFrame:
    """Select count-oriented anomaly summary rows for a chart."""
    result = summary[summary["Count"].gt(0)].copy()
    return result.set_index("Pattern")[["Count"]].sort_values("Count")


def bar_chart(data: pd.DataFrame, x: str, y: str, title: str, currency: bool = False):
    """Build a titled, labeled bar chart."""
    alt = _altair()
    chart = alt.Chart(data).mark_bar().encode(
        x=alt.X(f"{x}:N", title=x),
        y=alt.Y(f"{y}:Q", title=y, axis=alt.Axis(format="$,.2f") if currency else None),
        tooltip=[x, y],
    ).properties(title=title)
    return chart


def line_chart(data: pd.DataFrame, x: str, y: str, title: str, y_title: str):
    """Build a titled, labeled line chart."""
    alt = _altair()
    return alt.Chart(data).mark_line(point=True).encode(
        x=alt.X(f"{x}:T", title=x),
        y=alt.Y(f"{y}:Q", title=y_title),
        tooltip=[x, y],
    ).properties(title=title)


def scatter_chart(data: pd.DataFrame, x: str, y: str, title: str):
    """Build a titled, labeled scatter chart."""
    alt = _altair()
    return alt.Chart(data).mark_circle(size=70).encode(
        x=alt.X(f"{x}:Q", title=x),
        y=alt.Y(f"{y}:Q", title=y),
        tooltip=list(data.columns),
    ).properties(title=title)


def histogram(data: pd.DataFrame, column: str, title: str):
    """Build a titled, labeled score distribution."""
    alt = _altair()
    return alt.Chart(data).mark_bar().encode(
        x=alt.X(f"{column}:Q", bin=True, title=column),
        y=alt.Y("count():Q", title="Count"),
    ).properties(title=title)


def forecast_chart(history: pd.DataFrame, forecast: pd.DataFrame, title: str, y_title: str):
    """Build a historical/forecast chart with a prediction interval band."""
    alt = _altair()
    history_line = alt.Chart(history).mark_line(color="#4c78a8").encode(
        x=alt.X("Date:T", title="Date"),
        y=alt.Y("Value:Q", title=y_title),
        tooltip=["Date:T", "Value:Q"],
    )
    forecast_line = alt.Chart(forecast).mark_line(color="#f58518").encode(
        x=alt.X("Date:T", title="Date"),
        y=alt.Y("Value:Q", title=y_title),
        tooltip=["Date:T", "Value:Q"],
    )
    interval = alt.Chart(forecast).mark_area(opacity=0.2, color="#f58518").encode(
        x=alt.X("Date:T", title="Date"),
        y=alt.Y("Lower:Q", title=y_title),
        y2="Upper:Q",
    )
    return (history_line + interval + forecast_line).properties(title=title)
