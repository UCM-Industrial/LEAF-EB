"""Diagnostic plots for historical and forecast energy series."""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd


_EXCLUDED_PREFIXES = (
    "Installed_Capacity_",
    "Available_Capacity_",
    "Refueling",
    "__LEAF_")
_EXCLUDED_COLUMNS = {"Total"}


def save_energy_series_plot(
        frame: pd.DataFrame,
        path: str | Path,
        energy_unit: str,
        date_column: str = "Date",
) -> Path:
    """Save one compact small-multiples plot for an energy dataframe.

    Hourly historical series are aggregated to daily totals. Long daily
    forecasts are aggregated to monthly totals so multi-decade outputs remain
    readable. Capacity and internal columns are intentionally excluded because
    their units differ from the energy series.
    """

    data = _prepare_plot_frame(frame, date_column)
    plot_data, period_label = _aggregate_for_plot(data, date_column)
    columns = _energy_columns(plot_data, date_column)
    if not columns:
        raise ValueError("No plottable energy columns were found.")

    ncols = 2
    nrows = math.ceil(len(columns) / ncols)
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(13, max(4.5, 2.8 * nrows)),
        squeeze=False, sharex=True)
    dates = pd.to_datetime(plot_data[date_column])

    for axis, column in zip(axes.flat, columns):
        axis.plot(dates, plot_data[column], linewidth=1.0)
        axis.set_title(str(column), fontsize=11, fontweight="bold")
        axis.set_ylabel(f"{energy_unit}/{period_label}")
        axis.grid(True, alpha=0.25)
        axis.xaxis.set_major_locator(mdates.AutoDateLocator())
        axis.xaxis.set_major_formatter(
            mdates.ConciseDateFormatter(
                axis.xaxis.get_major_locator()))

    for axis in axes.flat[len(columns):]:
        axis.set_visible(False)

    for axis in axes[-1, :]:
        if axis.get_visible():
            axis.set_xlabel("Date")

    fig.tight_layout()
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output


def _prepare_plot_frame(
        frame: pd.DataFrame,
        date_column: str,
) -> pd.DataFrame:
    """Return a sorted copy with one valid datetime column."""

    if date_column not in frame.columns:
        raise ValueError(
            f"Date column '{date_column}' was not found for plotting.")
    data = frame.copy()
    data[date_column] = pd.to_datetime(
        data[date_column], errors="coerce")
    data = data.dropna(subset=[date_column]).sort_values(date_column)
    for column in data.columns:
        if column == date_column:
            continue
        data[column] = pd.to_numeric(data[column], errors="coerce")
    return data


def _energy_columns(
        frame: pd.DataFrame,
        date_column: str,
) -> list[str]:
    """Return numeric energy columns suitable for the diagnostic plot."""

    output = []
    for column in frame.columns:
        name = str(column)
        if column == date_column or name in _EXCLUDED_COLUMNS:
            continue
        if name.startswith(_EXCLUDED_PREFIXES):
            continue
        if not pd.api.types.is_numeric_dtype(frame[column]):
            continue
        values = pd.to_numeric(frame[column], errors="coerce").fillna(0.0)
        if float(values.abs().max()) <= 1e-12:
            continue
        output.append(column)
    return output


def _aggregate_for_plot(
        frame: pd.DataFrame,
        date_column: str,
) -> tuple[pd.DataFrame, str]:
    """Aggregate only as much as needed to keep long plots readable."""

    if frame.empty:
        return frame.copy(), "period"
    dates = pd.DatetimeIndex(frame[date_column])
    differences = dates.to_series().diff().dropna()
    positive = differences[differences > pd.Timedelta(0)]
    median_step = (
        positive.median() if not positive.empty
        else pd.Timedelta(days=1))
    span_days = max(
        1.0,
        (dates.max() - dates.min()).total_seconds() / 86400.0)

    if median_step <= pd.Timedelta(hours=2):
        frequency = "D"
        label = "day"
    elif span_days > 15 * 365.25:
        frequency = "MS"
        label = "month"
    else:
        return frame.copy(), "period"

    numeric_columns = _energy_columns(frame, date_column)
    indexed = frame.set_index(date_column)[numeric_columns]
    aggregated = indexed.resample(frequency).sum(min_count=1)
    aggregated = aggregated.reset_index()
    return aggregated, label
