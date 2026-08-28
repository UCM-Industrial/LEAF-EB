"""Read and transform daily or hourly historical energy data."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.utilities.console import emit_once

from src.utilities.name_resolution import normalize_input_frame


@dataclass(frozen=True)
class HistoricalDataset:
    """Historical input in its original and requested representations."""

    raw: pd.DataFrame
    processed: pd.DataFrame
    input_resolution: str
    processing_resolution: str
    date_column: str

    def at_resolution(self, resolution: str) -> pd.DataFrame:
        """Return data at one supported resolution."""

        requested = str(resolution).strip().lower()
        if requested == self.input_resolution:
            return self.raw.copy()
        if requested == self.processing_resolution:
            return self.processed.copy()
        if self.input_resolution == "hourly" and requested == "daily":
            return aggregate_hourly_history(
                self.raw,
                self.date_column,)
        raise ValueError(
            f"Historical data cannot be converted from "
            f"{self.input_resolution} to {requested}.")


def load_historical_dataset(
        config: dict[str, Any], context: str = "historical data"
) -> HistoricalDataset:
    """Load, validate and process the configured historical data."""

    data_path = _resolve_data_path(config["historical_data_file"])
    frame = _read_table(data_path)
    frame = normalize_input_frame(frame, config, context)
    date_column = str(config.get("date_column", "Date"))

    if date_column not in frame.columns:
        raise ValueError(
            f"Date column '{date_column}' was not found in {context}.")

    frame[date_column] = pd.to_datetime(
        frame[date_column],
        errors="coerce",)
    if frame[date_column].isna().any():
        raise ValueError(
            f"Date column '{date_column}' contains invalid values.")

    frame = frame.sort_values(date_column).reset_index(drop=True)
    _validate_duplicate_timestamps(frame, date_column)
    frame = _coerce_numeric_columns(frame, date_column)
    input_resolution = detect_input_resolution(frame, date_column)
    processing_resolution = str(
        config["processing_resolution"]).strip().lower()

    if input_resolution == "daily" and processing_resolution == "hourly":
        raise ValueError(
            "Hourly processing was requested, but the historical file "
            "contains daily observations.")

    if processing_resolution == input_resolution:
        processed = frame.copy()
    else:
        processed = aggregate_hourly_history(frame, date_column)

    emit_once(
        config,
        "Historical data: "
        f"input={input_resolution}, "
        f"processing={processing_resolution}, "
        f"rows={len(frame):,}.",
        key="historical_data_summary")
    return HistoricalDataset(
        raw=frame,
        processed=processed,
        input_resolution=input_resolution,
        processing_resolution=processing_resolution,
        date_column=date_column,)


def detect_input_resolution(
        frame: pd.DataFrame, date_column: str
) -> str:
    """Detect whether timestamps represent daily or hourly observations."""

    dates = pd.DatetimeIndex(frame[date_column])
    repeated_days = dates.normalize().duplicated().any()
    differences = dates.to_series().diff().dropna()
    positive = differences[differences > pd.Timedelta(0)]

    if repeated_days:
        resolution = "hourly"
    elif positive.empty:
        resolution = "daily"
    elif positive.median() <= pd.Timedelta(hours=2):
        resolution = "hourly"
    else:
        resolution = "daily"

    _validate_temporal_spacing(dates, resolution)
    return resolution


def aggregate_hourly_history(
        frame: pd.DataFrame, date_column: str
) -> pd.DataFrame:
    """Aggregate hourly energy values to one row per calendar day."""

    dates = pd.DatetimeIndex(frame[date_column])
    data_columns = [
        column for column in frame.columns
        if column != date_column]
    numeric = frame[data_columns].copy()
    numeric.index = dates.normalize()
    daily = numeric.groupby(level=0, sort=True).sum()
    daily.insert(0, date_column, daily.index)
    return daily.reset_index(drop=True)


def build_hourly_profile(
        dataset: HistoricalDataset,
        columns: list[str],
) -> pd.DataFrame:
    """Build one normalized 8760 profile from hourly observations."""

    if dataset.input_resolution != "hourly":
        raise ValueError(
            "Hourly profiles require an hourly historical file.")

    frame = dataset.raw.copy()
    date_column = dataset.date_column
    dates = pd.DatetimeIndex(frame[date_column])
    keep = ~((dates.month == 2) & (dates.day == 29))
    frame = frame.loc[keep].copy()
    dates = pd.DatetimeIndex(frame[date_column])
    keys = pd.MultiIndex.from_arrays(
        [dates.month, dates.day, dates.hour],
        names=["month", "day", "time"],)
    reference = pd.date_range(
        "2021-01-01",
        "2021-12-31 23:00:00",
        freq="h",)
    reference_keys = pd.MultiIndex.from_arrays(
        [reference.month, reference.day, reference.hour],
        names=["month", "day", "time"],)
    output = pd.DataFrame({
        "month": reference.month,
        "day": reference.day,
        "time": reference.hour,
    })

    for column in columns:
        if column not in frame.columns:
            continue
        values = pd.Series(
            frame[column].to_numpy(dtype=float),
            index=keys,)
        profile = values.groupby(level=[0, 1, 2]).mean()
        profile = profile.reindex(reference_keys)
        if profile.isna().any():
            missing = int(profile.isna().sum())
            raise ValueError(
                f"Hourly profile for '{column}' is missing {missing} "
                "calendar-hour combinations.")
        matrix = profile.to_numpy(dtype=float, copy=True).reshape(365, 24)
        totals = matrix.sum(axis=1)
        zero_days = totals <= 1e-12
        matrix[~zero_days] /= totals[~zero_days, None]
        matrix[zero_days] = 1.0 / 24.0
        output[column] = matrix.reshape(-1)

    return output


def _coerce_numeric_columns(
        frame: pd.DataFrame, date_column: str
) -> pd.DataFrame:
    """Convert energy columns and reject missing or invalid values."""

    output = frame.copy()
    columns = [
        column for column in output.columns
        if column != date_column]
    converted = output[columns].apply(pd.to_numeric, errors="coerce")
    invalid = converted.isna() & output[columns].notna()
    if invalid.any().any():
        names = invalid.any()[invalid.any()].index.tolist()
        values = ", ".join(str(name) for name in names)
        raise ValueError(
            "Historical data contains non-numeric values in: "
            f"{values}.")
    missing = converted.isna()
    if missing.any().any():
        counts = missing.sum()
        details = ", ".join(
            f"{column}={int(count)}"
            for column, count in counts.items()
            if count > 0)
        raise ValueError(
            "Historical data contains missing numeric values: "
            f"{details}. Missing observations must be corrected or "
            "imputed before running LEAF-EB.")
    output[columns] = converted
    return output


def _validate_duplicate_timestamps(
        frame: pd.DataFrame, date_column: str
) -> None:
    """Reject repeated timestamps."""

    duplicated = frame[date_column].duplicated(keep=False)
    if duplicated.any():
        first = frame.loc[duplicated, date_column].iloc[0]
        raise ValueError(
            f"Historical data contains duplicated timestamp: {first}.")


def _validate_temporal_spacing(
        dates: pd.DatetimeIndex, resolution: str
) -> None:
    """Require a complete regular historical time axis."""

    if len(dates) < 2:
        return

    differences = dates.to_series().diff().dropna()
    if resolution == "hourly":
        valid = differences.eq(pd.Timedelta(hours=1))
        expected = "one hour"
    else:
        valid = differences.between(
            pd.Timedelta(hours=23),
            pd.Timedelta(hours=25),)
        expected = "one calendar day"

    if valid.all():
        return

    first_position = int(np.flatnonzero(~valid.to_numpy())[0])
    first_gap = differences.iloc[first_position]
    timestamp = differences.index[first_position]
    raise ValueError(
        "Historical timestamps must form a complete regular "
        f"{resolution} axis. Expected {expected}; found a gap of "
        f"{first_gap} ending at {timestamp}.")


def _resolve_data_path(file_name: object) -> Path:
    """Resolve a configured path relative to the project root."""

    path = Path(str(file_name))
    if path.is_absolute():
        return path
    root = Path(__file__).resolve().parents[2]
    return root / path


def _read_table(path: Path) -> pd.DataFrame:
    """Read a supported historical-data table."""

    if not path.is_file():
        raise FileNotFoundError(
            f"Historical data file not found: {path}")
    suffix = path.suffix.lower()
    if suffix == ".xlsx":
        return pd.read_excel(path, engine="openpyxl")
    if suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(
        "Historical data must be an XLSX or CSV file: "
        f"{path}")
