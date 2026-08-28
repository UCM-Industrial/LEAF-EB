"""Build empirical temporal patterns and stochastic residual histories.

Daily projections use a smoothed calendar cycle around a continuous local
level. Hourly projections use an empirical day-of-year/hour calendar. The
residual history is saved separately for the Monte Carlo stationary bootstrap.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.utilities.console import emit

from src.forecasting.Variability import (
    process_pre_calculated_residuals,)
from src.forecasting.historical_data import (
    HistoricalDataset,
    build_hourly_profile,
    load_historical_dataset,)
from src.forecasting.output_plots import save_energy_series_plot
from src.utilities.configuration import load_config_file



HOURLY_CALENDAR_SMOOTHING_DAYS = 15
HOURLY_RELATIVE_SIGNAL_FRACTION = 0.02
ANNUAL_LEVEL_SMOOTHING_YEARS = 3


class PatternGenerator:
    """Build projection patterns and residual histories."""

    def __init__(self, input_name: str):
        """Load one input and prepare its output directory."""

        self.input_name = input_name
        inputs_dir = Path(__file__).resolve().parents[2] / "Inputs"
        config_path = inputs_dir / f"{input_name}.yml"
        self.config: dict[str, Any] = load_config_file(config_path)
        self.output_dir = (
            Path(__file__).resolve().parents[2]
            / self.config["scenario_folder"]
            / self.config["scenario_subfolder"])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        emit(self.config, "Input loaded successfully.", "detailed")
        emit(self.config, f"Output directory: {self.output_dir}", "detailed")

    def run(self) -> None:
        """Generate pattern, residual and optional hourly-profile files."""

        dataset = load_historical_dataset(self.config)
        resolution = self.config["projection_resolution"]
        frame = dataset.at_resolution(resolution)
        date_column = dataset.date_column
        work = frame.set_index(date_column).sort_index()
        columns = self._source_columns(work)

        if self.config.get("save_projection_plots", False):
            historical_columns = [date_column, *columns]
            historical_plot = dataset.raw.loc[:, historical_columns]
            plot_path = (
                self.output_dir / "Output" / "Plots" / "Historical.png")
            save_energy_series_plot(
                historical_plot,
                plot_path,
                self.config["energy_unit"],
                date_column,)
            emit(
                self.config,
                f"Saving historical plot to: {plot_path}",
                "detailed")
        pattern_data: dict[str, np.ndarray] = {}
        residuals: dict[str, pd.Series] = {}
        local_relative: dict[str, pd.Series] = {}
        mean_values: dict[str, float] = {}

        for column in columns:
            emit(
                self.config,
                f"   Processing pattern for: {column}",
                "detailed")
            values = pd.to_numeric(
                work[column],
                errors="coerce",
            ).fillna(0.0)
            mean_values[column] = float(values.mean())

            if resolution == "daily":
                pattern, residual, relative = self._daily_pattern(
                    values, column)
                pattern_data.update(pattern)
            else:
                pattern, residual, relative = self._hourly_pattern(
                    values, column)
                pattern_data.update(pattern)

            residuals[column] = residual
            local_relative[column] = relative

        self._save_projection_pattern(pattern_data, resolution)
        self._save_hourly_profile(dataset, columns)

        if self.config.get("variability_enabled", False):
            process_pre_calculated_residuals(
                combined_resid=residuals,
                mean_values=mean_values,
                local_relative=local_relative,
                output_dir=str(self.output_dir),
                variability_mode=self.config["variability_mode"],)

    def _source_columns(self, frame: pd.DataFrame) -> list[str]:
        """Return configured source columns plus demand."""

        columns = []
        for column, source_data in self.config["sources"].items():
            model = ""
            if isinstance(source_data, dict):
                model = str(source_data.get("model", "")).strip().lower()
            custom_mode = ""
            if isinstance(source_data, dict):
                custom_mode = str(
                    source_data.get("custom_mode", "add")
                ).strip().lower()
            if model == "custom" and custom_mode != "add":
                continue
            if column not in frame.columns:
                raise ValueError(
                    f"Historical column '{column}' was not found.")
            columns.append(column)

        if "Demand" not in frame.columns:
            raise ValueError(
                "Historical column 'Demand' was not found.")
        columns.append("Demand")
        return columns

    def _daily_pattern(
            self, values: pd.Series, column: str
    ) -> tuple[dict[str, np.ndarray], pd.Series, pd.Series]:
        """Return pattern, absolute residuals and local-relative residuals."""

        keep = ~(
            (values.index.month == 2)
            & (values.index.day == 29))
        series = values.loc[keep].astype(float)
        cycle = self._empirical_annual_cycle(series)

        reference = pd.date_range(
            "2021-01-01",
            periods=365,
            freq="D",)
        output = {f"{column}_per": cycle}
        weekly = np.zeros(7, dtype=float)

        if column == "Demand":
            weekly = self._weekly_cycle(series)
            output["Demand_week_per"] = weekly[reference.weekday]

        annual_map = {
            (date.month, date.day): float(value)
            for date, value in zip(reference, cycle)}
        annual_fraction = np.asarray([
            annual_map[(date.month, date.day)]
            for date in series.index
        ], dtype=float)
        weekly_fraction = weekly[series.index.weekday]
        annual_level = self._continuous_annual_level(
            series).to_numpy(dtype=float)
        fitted = annual_level * (
            1.0 + annual_fraction + weekly_fraction)
        residual = series.to_numpy(dtype=float) - fitted
        denominator = np.where(np.abs(fitted) > 1e-12, fitted, np.nan)
        relative = residual / denominator
        relative = np.where(np.isfinite(relative), relative, 0.0)

        residual_series = pd.Series(
            residual, index=series.index, name=column)
        relative_series = pd.Series(
            relative, index=series.index, name=column)
        return output, residual_series, relative_series

    @staticmethod
    def _calendar_cycle(
            values: pd.Series, smoothing_window: int
    ) -> np.ndarray:
        """Build a circular 365-day cycle using month and day."""

        reference = pd.date_range(
            "2021-01-01",
            periods=365,
            freq="D",)
        frame = pd.DataFrame({
            "month": values.index.month,
            "day": values.index.day,
            "value": values.to_numpy(dtype=float),
        })
        grouped = frame.groupby(["month", "day"])["value"].mean()
        cycle = np.asarray([
            grouped.get((date.month, date.day), np.nan)
            for date in reference
        ], dtype=float)
        positions = np.arange(cycle.size)
        valid = np.isfinite(cycle)
        if not np.all(valid):
            cycle = np.interp(positions, positions[valid], cycle[valid])
        extended = np.concatenate([cycle, cycle, cycle])
        smoothed = pd.Series(extended).rolling(
            window=int(smoothing_window),
            center=True,
            min_periods=1,
        ).mean()
        output = smoothed.iloc[365:730].to_numpy(dtype=float, copy=True)
        output -= float(np.nanmean(output))
        return output

    @staticmethod
    def _continuous_annual_level(series: pd.Series) -> pd.Series:
        """Return a smooth continuous structural level.

        Annual means contain two different signals: long-term changes in the
        scale of a series and genuine year-to-year anomalies. Interpolating
        every annual mean exactly removes both. A centered three-year mean is
        therefore used for interior years, while the first and last annual
        means are kept as boundary anchors. This removes structural change
        without creating an artificial edge anomaly and leaves part of the
        genuine interannual variability in the residual history.

        The same rule is used for every source and for demand.
        """

        yearly = series.groupby(series.index.year).mean().sort_index()
        if yearly.empty:
            return pd.Series(0.0, index=series.index, dtype=float)

        if len(yearly) >= ANNUAL_LEVEL_SMOOTHING_YEARS:
            structural = yearly.rolling(
                window=ANNUAL_LEVEL_SMOOTHING_YEARS,
                center=True,
                min_periods=ANNUAL_LEVEL_SMOOTHING_YEARS,
            ).mean()
            structural.iloc[0] = yearly.iloc[0]
            structural.iloc[-1] = yearly.iloc[-1]
            structural = structural.interpolate(
                limit_direction="both")
        else:
            structural = yearly.copy()
        anchor_dates = pd.DatetimeIndex([
            pd.Timestamp(year=int(year), month=7, day=2)
            for year in structural.index
        ])
        anchor_values = structural.to_numpy(dtype=float)

        if anchor_values.size == 1:
            values = np.full(
                len(series), anchor_values[0], dtype=float)
        else:
            target = series.index.view("int64").astype(float)
            anchors = anchor_dates.view("int64").astype(float)
            values = np.interp(
                target, anchors, anchor_values)

        return pd.Series(values, index=series.index, dtype=float)

    @classmethod
    def _empirical_annual_cycle(cls, series: pd.Series) -> np.ndarray:
        """Estimate a normalized cycle after removing continuous trend."""

        level = cls._continuous_annual_level(series)
        denominator = level.replace(0.0, np.nan)
        relative = series / denominator - 1.0
        relative = relative.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return cls._calendar_cycle(
            relative,
            smoothing_window=31,)

    def _hourly_pattern(
            self, values: pd.Series, column: str
    ) -> tuple[dict[str, np.ndarray], pd.Series, pd.Series]:
        """Return a scale-separated hourly pattern and residual history.

        The historical series is first divided by the same continuous annual
        structural level used by the daily model. The empirical hour-of-year
        pattern is then estimated from this dimensionless shape. Residuals
        therefore describe temporal variability around the local level rather
        than changes in the historical scale of the technology or demand.
        """

        keep = ~(
            (values.index.month == 2)
            & (values.index.day == 29))
        series = values.loc[keep].astype(float)
        annual_level = self._continuous_annual_level(series)
        level_values = annual_level.to_numpy(dtype=float)

        denominator = annual_level.replace(0.0, np.nan)
        shape = series / denominator
        shape = shape.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        calendar_days = self._normalized_day_of_year(series.index)
        keys = pd.MultiIndex.from_arrays(
            [calendar_days, series.index.hour],
            names=["day_of_year", "time"],)
        calendar = self._smoothed_hourly_calendar(
            shape, HOURLY_CALENDAR_SMOOTHING_DAYS)

        calendar_mean = float(calendar.mean())
        if abs(calendar_mean) > 1e-12:
            calendar = calendar / calendar_mean

        calendar_values = calendar.reindex(keys).to_numpy(dtype=float)
        shape_values = shape.to_numpy(dtype=float)
        raw_shape_residual = shape_values - calendar_values

        weekdays = series.index.weekday.to_numpy(dtype=int)
        weekly = pd.Series(raw_shape_residual).groupby(weekdays).mean()
        weekly = weekly.reindex(range(7)).fillna(0.0)
        weekly_values = weekly.to_numpy(dtype=float, copy=True)
        weekly_values -= float(np.mean(weekly_values))

        active = self._infer_temporal_support(series)
        history_active = self._support_for_dates(active, series.index)
        fitted_shape = calendar_values + weekly_values[weekdays]
        fitted_shape = np.maximum(fitted_shape, 0.0)
        fitted_shape[~history_active] = 0.0
        fitted = level_values * fitted_shape
        residual = series.to_numpy(dtype=float) - fitted

        reconstructed = fitted + residual
        tolerance = max(
            1e-8,
            float(np.nanmax(np.abs(series.to_numpy()))) * 1e-10)
        error = float(np.nanmax(np.abs(
            reconstructed - series.to_numpy(dtype=float)
        )))
        if error > tolerance:
            raise RuntimeError(
                f"Hourly decomposition failed to reconstruct '{column}': "
                f"maximum error={error:.6g}.")

        reference = pd.date_range(
            "2021-01-01",
            "2021-12-31 23:00:00",
            freq="h",)
        reference_days = self._normalized_day_of_year(reference)
        reference_keys = pd.MultiIndex.from_arrays(
            [reference_days, reference.hour],
            names=["day_of_year", "time"],)
        reference_shape = calendar.reindex(reference_keys)
        reference_shape = reference_shape.interpolate(
            limit_direction="both",
        ).fillna(0.0)
        reference_active = self._support_for_dates(active, reference)
        reference_shape.loc[~reference_active] = 0.0

        output = {
            f"{column}_per": (
                reference_shape.to_numpy(dtype=float) - 1.0),
            f"{column}_week_per": weekly_values[reference.weekday],
            f"{column}_active": reference_active.astype(float),}

        residual_series = pd.Series(
            residual,
            index=series.index,
            name=column,)
        positive_shape = fitted_shape[
            history_active & (fitted_shape > 0.0)]
        if positive_shape.size:
            reference_signal = float(np.quantile(positive_shape, 0.95))
        else:
            reference_signal = 1.0
        signal_floor = max(
            reference_signal * HOURLY_RELATIVE_SIGNAL_FRACTION,
            1e-12,)
        relative_values = np.zeros(len(series), dtype=float)
        reliable = (
            history_active
            & (fitted_shape >= signal_floor)
            & (np.abs(fitted) > 1e-12))
        relative_values[reliable] = (
            residual[reliable] / fitted[reliable])
        relative_values = np.where(
            np.isfinite(relative_values), relative_values, 0.0)
        relative_series = pd.Series(
            relative_values,
            index=series.index,
            name=column,)
        return output, residual_series, relative_series

    @staticmethod
    def _normalized_day_of_year(
            dates: pd.DatetimeIndex
    ) -> np.ndarray:
        """Map month and day to a common non-leap calendar."""

        index = pd.DatetimeIndex(dates)
        normalized = pd.to_datetime({
            "year": np.full(len(index), 2021, dtype=int),
            "month": index.month,
            "day": index.day,})
        return normalized.dt.dayofyear.to_numpy(dtype=int)

    @classmethod
    def _smoothed_hourly_calendar(
            cls, series: pd.Series, window_days: int
    ) -> pd.Series:
        """Estimate a cyclic smooth day-of-year and hour pattern."""

        dates = pd.DatetimeIndex(series.index)
        frame = pd.DataFrame({
            "day_of_year": cls._normalized_day_of_year(dates),
            "time": dates.hour,
            "value": series.to_numpy(dtype=float),})
        pivot = frame.groupby([
            "day_of_year", "time"
        ])["value"].mean().unstack("time")
        pivot = pivot.reindex(
            index=np.arange(1, 366),
            columns=np.arange(24),)
        pivot = pivot.interpolate(
            axis=0, limit_direction="both").fillna(0.0)
        window = max(1, min(int(window_days), 365))
        if window % 2 == 0:
            window += 1
        extended = pd.concat([pivot, pivot, pivot], ignore_index=True)
        smooth = extended.rolling(
            window=window,
            center=True,
            min_periods=1,).mean().iloc[365:730].copy()
        smooth.index = np.arange(1, 366)
        smooth.index.name = "day_of_year"
        smooth.columns.name = "time"
        return smooth.stack()

    @staticmethod
    def _infer_temporal_support(series: pd.Series) -> pd.Series:
        """Infer structurally inactive month-hour slots from history."""

        values = pd.to_numeric(series, errors="coerce")
        dates = pd.DatetimeIndex(series.index)
        frame = pd.DataFrame({
            "month": dates.month,
            "hour": dates.hour,
            "value": values.to_numpy(dtype=float),
        }).dropna(subset=["value"])
        positive = frame.loc[frame["value"] > 0.0, "value"]
        if positive.empty:
            scale = 1.0
        else:
            scale = float(positive.quantile(0.95))
        zero_tolerance = max(1e-9, scale * 1e-4)
        grouped = frame.groupby(["month", "hour"])["value"]
        count = grouped.count()
        high = grouped.quantile(0.99)
        inactive = (count >= 24) & (high <= zero_tolerance)
        return (~inactive).astype(bool)

    @staticmethod
    def _support_for_dates(
            support: pd.Series, dates: pd.DatetimeIndex
    ) -> np.ndarray:
        """Map a month-hour support table to one datetime sequence."""

        index = pd.DatetimeIndex(dates)
        keys = pd.MultiIndex.from_arrays(
            [index.month, index.hour],
            names=["month", "hour"],)
        return support.reindex(keys).fillna(True).to_numpy(dtype=bool)

    @staticmethod
    def _weekly_cycle(series: pd.Series) -> np.ndarray:
        """Estimate weekday effects after removing monthly levels."""

        monthly = series.groupby([
            series.index.year,
            series.index.month,
        ]).transform("mean")
        relative = series / monthly.replace(0.0, np.nan) - 1.0
        weekday = relative.groupby(relative.index.weekday).mean()
        weekday = weekday.reindex(range(7)).fillna(0.0)
        values = weekday.to_numpy(dtype=float, copy=True)
        values -= float(np.nanmean(values))
        return values

    def _save_projection_pattern(
            self, patterns: dict[str, np.ndarray], resolution: str
    ) -> None:
        """Save the pattern table used by Predictor."""

        if resolution == "daily":
            reference = pd.date_range(
                "2021-01-01",
                periods=365,
                freq="D",)
        else:
            reference = pd.date_range(
                "2021-01-01",
                "2021-12-31 23:00:00",
                freq="h",)

        output = pd.DataFrame({
            "month": reference.month,
            "day": reference.day,
            "weekday": reference.weekday,
        })
        if resolution == "hourly":
            output["time"] = reference.hour
        for name, values in patterns.items():
            output[name] = np.asarray(values, dtype=float)

        path = self.output_dir / self.config["pattern_file"]
        emit(self.config, f"Saving patterns to: {path}", "detailed")
        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            output.to_excel(
                writer,
                index=False,
                sheet_name="Pattern",)

    def _save_hourly_profile(
            self,
            dataset: HistoricalDataset,
            columns: list[str],
    ) -> None:
        """Save normalized historical profiles when hourly data exist."""

        if dataset.input_resolution != "hourly":
            return
        if self.config.get("external_hourly_profile_file"):
            return
        profile = build_hourly_profile(dataset, columns)
        path = Path(self.config["historical_hourly_pattern_file"])
        path.parent.mkdir(parents=True, exist_ok=True)
        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            profile.to_excel(writer, index=False, sheet_name="Pattern")
        profile.to_csv(path.with_suffix(".csv"), index=False)
        emit(
            self.config,
            f"Saving historical hourly profiles to: {path}",
            "detailed")



if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python Patterns.py <input_name_no_extension>")
        sys.exit(1)
    PatternGenerator(sys.argv[1]).run()
