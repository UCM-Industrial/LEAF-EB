# -*- coding: utf-8 -*-
"""Generate deterministic demand and source projections for LEAF-EB.

The forecaster applies user-defined long-term projections to empirical
temporal patterns. It also derives installed-capacity columns and handles
custom source additions or replacements. Stochastic perturbations are applied
later by the commodities workflow, not in this module.
"""

from calendar import monthrange
from datetime import datetime
from pathlib import Path
import re
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.utilities.console import emit

from src.forecasting.anchors import get_anchor_value
from src.forecasting.custom_sources import (
    CustomSourceForecaster,
    apply_custom_replacements,
    save_replacement_diagnostics)
from src.forecasting.historical_data import load_historical_dataset
from src.forecasting.output_plots import save_energy_series_plot
from src.utilities.configuration import (
    load_config_file, write_resolved_config)
from src.utilities.name_resolution import normalize_input_frame
from src.utilities.units import (
    energy_to_mwh_factor, is_energy_unit, is_power_unit,
    power_to_mw_factor)


# ==== UTILITIES (Pure Functions) ====

def parse_objective_date(date_str: str) -> Tuple[datetime, str]:
    """
    Parses the objective date string from configuration to determine date and
    granularity.

    Parameters
    ----------
    date_str : str
        The date string from the configuration (e.g., '12/31/2030', '2030').

    Returns
    -------
    Tuple[datetime, str]
        A tuple containing the parsed datetime object and the granularity
        ('day', 'month', or 'year').
    """
    s = str(date_str).strip()
    formats = [("%m/%d/%Y", "day"), ("%m/%Y", "month"), ("%Y", "year")]
    for fmt, gran in formats:
        try:
            return datetime.strptime(s, fmt), gran
        except ValueError:
            continue
    raise ValueError(f"Unrecognized date format for 'objetive_date': {s}")

def get_period_end(dt: datetime, granularity: str) -> datetime:
    """
    Determines the end date of a given period (day, month, or year).

    Parameters
    ----------
    dt : datetime
        The date object representing the start or target.
    granularity : str
        The period granularity ('day', 'month', or 'year').

    Returns
    -------
    datetime
        The end date of the specified period.
    """
    if granularity == "day":
        return dt
    if granularity == "month":
        last_day = monthrange(dt.year, dt.month)[1]
        return datetime(dt.year, dt.month, last_day)
    if granularity == "year":
        return datetime(dt.year, 12, 31)
    raise ValueError(f"Unknown granularity: {granularity}")

def to_naive_datetime_index(s) -> pd.DatetimeIndex:
    """Converts a Series or Index to a timezone-naive DatetimeIndex (UTC
    converted to None)."""
    res = pd.to_datetime(s, errors="coerce", utc=True)
    if isinstance(res, pd.Index):
        return res.tz_convert(None)
    return res.dt.tz_convert(None)

def group_series_by(series: pd.Series, gb: str) -> pd.Series:
    """
    Groups a time series by the specified granularity (day, month, year) and
    sums the values.

    Parameters
    ----------
    series : pd.Series
        The time series data to group.
    gb : str
        The grouping granularity ('day', 'month', or 'year').

    Returns
    -------
    pd.Series
        The grouped and summed time series.
    """
    ts = to_naive_datetime_index(series.index)
    gb = str(gb).lower()
    if gb == "hour":
        return pd.Series(series.values, index=ts)
    if gb == "day":
        return pd.Series(
            series.values,
            index=ts.date,
        ).groupby(level=0).sum()
    if gb == "month":
        monthly_index = pd.MultiIndex.from_arrays(
            [ts.year, ts.month])
        return (
            pd.Series(series.values, index=monthly_index)
            .groupby(level=[0, 1])
            .sum())
    if gb == "year":
        return pd.Series(series.values, index=ts.year).groupby(level=0).sum()
    raise ValueError(f"Unrecognized groupby value: {gb}")

def parse_linear_pivots(type_str: str) -> List[Tuple[int, float]]:
    """
    Parses 'linear[ (YYYY, fraction) ]' string to extract year-fraction pivot
    points.

    Parameters
    ----------
    type_str : str
        The model type string, possibly containing pivot definitions.

    Returns
    -------
    List[Tuple[int, float]]
        A sorted list of (year, fraction) pivot points.
    """
    if not isinstance(type_str, str):
        return []
    m = re.search(
        'linear\\s*\\[\\s*(.*?)\\s*\\]$',
        type_str.strip(),
        flags=re.IGNORECASE)
    if not m:
        return []
    inside = m.group(1)
    pivots = []
    for ym in re.finditer(r'\(\s*(\d{4})\s*,\s*([0-9]*\.?[0-9]+)\s*\)', inside):
        year = int(ym.group(1)); frac = float(ym.group(2))
        pivots.append((year, frac))
    return sorted({y: f for y, f in pivots}.items(), key=lambda t: t[0])

def year_fraction_from_dates(dts) -> pd.Series:
    """
    Converts a series of datetime objects into a fractional year format (e.g.,
    2025.5).
    """
    dts = pd.Series(pd.to_datetime(dts))
    years = dts.dt.year.astype(float)
    dayofyear = dts.dt.dayofyear.astype(float)
    return years + (dayofyear - 1.0) / 365.25

def year_start_float(year: int) -> float:
    'Returns the fractional year representation of Jan 1st for a given year.'
    dt = pd.Timestamp(year=year, month=1, day=1)
    return year + (dt.dayofyear - 1.0) / 365.25

def year_end_float(year: int) -> float:
    'Returns the fractional year representation of Dec 31st for a given year.'
    last = monthrange(year, 12)[1]
    dt = pd.Timestamp(year=year, month=12, day=last)
    return year + (dt.dayofyear - 1.0) / 365.25

def build_share_with_pivots(
        dates: pd.DatetimeIndex, start_share: float,
        final_share: float,
        pivots: List[Tuple[int, float]],
        start_slope: float = 0.0) -> np.ndarray:
    """Build a continuous share path from history to explicit targets.

    Annual target values are first calculated with a historical-slope bridge
    to the first pivot and linear interpolation between later pivots. Those
    annual means are then converted into one continuous series at the
    native forecast frequency. This avoids artificial changes on January 1
    while preserving the intended annual values.
    """
    date_index = pd.DatetimeIndex(dates)
    if len(date_index) == 0:
        return np.array([], dtype=float)

    pivot_map = {
        int(year): float(fraction) * float(final_share)
        for year, fraction in (pivots or [])}
    years = date_index.year.to_numpy(dtype=int)
    first_year = int(years.min())
    last_year = int(years.max())
    annual_years = np.arange(first_year, last_year + 1, dtype=float)

    if not pivot_map:
        annual_values = np.full(
            annual_years.size, float(final_share), dtype=float)
        annual_values[0] = float(start_share)
    else:
        pivot_years = np.array(sorted(pivot_map), dtype=float)
        pivot_values = np.array(
            [pivot_map[int(year)] for year in pivot_years], dtype=float)
        annual_values = np.interp(
            annual_years, pivot_years, pivot_values,
            left=pivot_values[0], right=pivot_values[-1])

        first_pivot_year = int(pivot_years[0])
        bridge_mask = annual_years <= first_pivot_year
        bridge_years = annual_years[bridge_mask]
        duration = float(first_pivot_year - first_year)

        if duration > 0.0:
            if len(pivot_years) > 1:
                end_slope = (pivot_values[1] - pivot_values[0]) / (
                    pivot_years[1] - pivot_years[0])
            else:
                end_slope = 0.0

            t = (bridge_years - first_year) / duration
            h00 = 2.0 * t**3 - 3.0 * t**2 + 1.0
            h10 = t**3 - 2.0 * t**2 + t
            h01 = -2.0 * t**3 + 3.0 * t**2
            h11 = t**3 - t**2
            bridge = h00 * float(start_share)
            bridge += h10 * duration * float(start_slope)
            bridge += h01 * pivot_values[0]
            bridge += h11 * duration * end_slope

            lower = min(float(start_share), pivot_values[0])
            upper = max(float(start_share), pivot_values[0])
            annual_values[bridge_mask] = np.clip(bridge, lower, upper)
        else:
            annual_values[0] = pivot_values[0]

    midpoints = []
    for year in annual_years.astype(int):
        positions = np.flatnonzero(years == year)
        midpoints.append(date_index[positions[len(positions) // 2]])

    date_num = date_index.view("int64").astype(float)
    midpoint_num = pd.DatetimeIndex(midpoints).view("int64").astype(float)
    path = np.interp(
        date_num, midpoint_num, annual_values,
        left=annual_values[0], right=annual_values[-1])

    for _ in range(30):
        current = pd.Series(path, index=years).groupby(level=0).mean()
        ratios = np.array([
            annual_values[i] / max(float(current.loc[int(year)]), 1e-12)
            for i, year in enumerate(annual_years)
        ])
        correction = np.interp(
            date_num, midpoint_num, ratios,
            left=ratios[0], right=ratios[-1])
        path *= correction

    for i, year in enumerate(annual_years.astype(int)):
        if abs(float(annual_values[i])) <= 1e-12:
            path[years == year] = 0.0

    return path

# ==== GLOBAL BALANCE FUNCTION ====

def enforce_global_balance(
        saving: pd.DataFrame,
        balance: float,
        target_metric: str = "endpoint",
        demand_col: str = "Demand",
        total_col: str = "Total",
        exclude_cols: List[str] = None) -> pd.DataFrame:
    """
    Applies a final scaling factor to all included production sources to match
    the target balance (Production / Demand) for the target metric point.
    """
    if saving.empty or demand_col not in saving.columns: return saving

    prod_cols = [c for c in saving.columns if c not in (demand_col, total_col)]
    include_cols = [c for c in prod_cols if c not in (exclude_cols or [])]

    saving[total_col] = saving[prod_cols].sum(axis=1) # Initial calculation
    if not include_cols: return saving

    balance = 1.0 if balance is None else float(balance)

    if target_metric == "final_year_mean":
        index_values = pd.Index(saving.index)
        numeric_years = pd.to_numeric(index_values, errors="coerce")
        numeric_mask = np.isfinite(numeric_years)
        numeric_mask &= numeric_years >= 1800
        numeric_mask &= numeric_years <= 3000

        if bool(np.all(numeric_mask)):
            years = np.asarray(numeric_years, dtype=int)
        else:
            parsed = pd.to_datetime(index_values, errors="coerce")
            if parsed.isna().all():
                raise ValueError(
                    "Could not infer years for final-year balance.")
            years = parsed.year.to_numpy(dtype=int)

        last_yr = int(np.max(years))
        mask = years == last_yr
        dem = saving.loc[mask, demand_col].mean()
        prod = saving.loc[mask, include_cols].sum(axis=1).mean()
    else: # 'endpoint'
        dem = saving[demand_col].iloc[-1]
        prod = saving[include_cols].iloc[-1].sum()

    if dem > 0:
        scale = (dem * balance) / max(prod, 1e-12)
        saving.loc[:, include_cols] *= scale
        saving[total_col] = saving[prod_cols].sum(axis=1) # Recalculate total

    return saving


# ==== MAIN FORECASTER CLASS ====

class EnergyForecaster:
    """
    Main class for energy demand and production forecasting.
    Handles configuration loading, data processing, model selection, and results
    saving.
    """
    def __init__(self, input_name: str):
        """
        Initializes the forecaster by loading configuration and processing
        dates.

        Parameters
        ----------
        input_name : str
            The name of the YAML input configuration file (without extension).
        """
        self.input_name = input_name
        self.config: Dict[str, Any] = {}
        self.df_data: Optional[pd.DataFrame] = None
        self.df_patterns: Optional[pd.DataFrame] = None
        self.demand_series: Optional[pd.Series] = None
        self.demand_target_series: Optional[pd.Series] = None
        self._resolved_anchors: Dict[str, Dict[str, Any]] = {}

        self._load_config()
        self._process_config_dates()

    def _load_config(self):
        """Load the current-schema YAML configuration."""

        inputs_dir = Path(__file__).resolve().parents[2] / "Inputs"
        config_path = inputs_dir / f"{self.input_name}.yml"
        if not config_path.is_file():
            raise FileNotFoundError(
                f"Config file not found: {config_path}")
        self.config = load_config_file(config_path)

    def _process_config_dates(self):
        """Prepare scenario dates and demand target metadata."""

        start = pd.Timestamp(self.config["start_date"])
        end = pd.Timestamp(self.config["end_date"])
        self.config["_objetive_dt"] = end.to_pydatetime()
        self.config["_objetive_granularity"] = "year"
        self.config["_objetive_period_end"] = end.to_pydatetime()
        self.config["_forecast_start"] = start
        demand = self.config["demand"]
        self.config["_demand_target_production"] = float(
            demand["target_production"])
        self.config["_demand_balance"] = float(
            demand.get("balance", 1.0))

    def load_data(self):
        """Load historical data at the projection resolution and patterns."""

        dataset = load_historical_dataset(self.config)
        resolution = self.config["projection_resolution"]
        self.df_data = dataset.at_resolution(resolution)
        pattern_path = (
            Path(__file__).resolve().parents[2]
            / self.config["scenario_folder"]
            / self.config["scenario_subfolder"]
            / self.config["pattern_file"])
        emit(self.config, f"Loading patterns from: {pattern_path}", "detailed")
        self.df_patterns = pd.read_excel(
            pattern_path,
            sheet_name="Pattern",
            engine="openpyxl",)
        self.df_patterns = normalize_input_frame(
            self.df_patterns,
            self.config,
            "pattern data",)

    # ==== INTERNAL LOGIC HELPERS ====

    def _get_anchor_params(self, source_name: str) -> Tuple[str, int]:
        """Return an optional manual anchor or the automatic default."""
        if source_name == "Demand":
            src = self.config.get("demand", {})
        else:
            src = self.config.get("sources", {}).get(source_name, {})
        if isinstance(src, dict) and "anchor" in src:
            cfg = src["anchor"]
            return (
                str(cfg.get('method', 'auto')).lower(),
                int(cfg.get('window', 1)))
        return "auto", 1

    def _maybe_apply_pattern(
            self, df: pd.DataFrame, energy_key: str
    ) -> pd.DataFrame:
        """Apply calendar fractions and empirical temporal support."""

        if self.df_patterns is None:
            return df

        annual_col = f"{energy_key}_per"
        weekly_col = f"{energy_key}_week_per"
        active_col = f"{energy_key}_active"
        has_annual = annual_col in self.df_patterns.columns
        has_weekly = weekly_col in self.df_patterns.columns
        has_active = active_col in self.df_patterns.columns

        if not has_annual and not has_weekly and not has_active:
            return df

        output = df.copy()
        output["_pattern_fraction"] = 0.0
        output["month"] = output["ds"].dt.month
        output["day"] = output["ds"].dt.day
        keys = ["month", "day"]
        if "time" in self.df_patterns.columns:
            output["time"] = output["ds"].dt.hour
            keys.append("time")
        drop_columns = ["_pattern_fraction", *keys]

        selected = list(keys)
        for column in [annual_col, active_col]:
            if column in self.df_patterns.columns:
                selected.append(column)
        if len(selected) > len(keys):
            pattern = self.df_patterns[selected]
            output = output.merge(pattern, on=keys, how="left")

        if has_annual:
            output[annual_col] = output[annual_col].interpolate(
                limit_direction="both",
            ).fillna(0.0)
            output["_pattern_fraction"] += output[annual_col]
            drop_columns.append(annual_col)

        if has_weekly:
            weekly = self.df_patterns.copy()
            if "weekday" not in weekly.columns:
                base_days = pd.date_range(
                    "2001-01-01", periods=len(weekly), freq="D")
                weekly["weekday"] = base_days.weekday
            weekly_map = weekly.groupby("weekday")[weekly_col].mean()
            weekday = output["ds"].dt.weekday
            output["_pattern_fraction"] += weekday.map(
                weekly_map,
            ).fillna(0.0)

        output["y"] *= 1.0 + output["_pattern_fraction"]
        if has_active:
            active = pd.to_numeric(
                output[active_col],
                errors="coerce",
            ).fillna(1.0)
            output.loc[active < 0.5, "y"] = 0.0
            drop_columns.append(active_col)
        return output.drop(
            columns=list(dict.fromkeys(drop_columns)),
            errors="ignore",)

    @staticmethod
    def _window_mean(
            series: pd.Series,
            begin: pd.Timestamp,
            end: pd.Timestamp,
    ) -> float:
        """Return the mean over one physical-time window."""

        values = series.loc[(series.index >= begin) & (series.index < end)]
        if values.empty:
            return np.nan
        return float(values.mean())

    @staticmethod
    def _transition_bounds(
            history: pd.Series,
            start: pd.Timestamp,
            horizons: tuple[int, ...],
    ) -> dict[int, tuple[float, float]]:
        """Estimate empirical boundary-transition ranges by horizon."""

        bounds = {}
        for days in horizons:
            transitions = []
            delta = pd.Timedelta(days=int(days))
            for year in sorted(set(history.index.year)):
                try:
                    boundary = start.replace(year=int(year))
                except ValueError:
                    continue
                if boundary >= start:
                    continue
                before = EnergyForecaster._window_mean(
                    history, boundary - delta, boundary)
                after = EnergyForecaster._window_mean(
                    history, boundary, boundary + delta)
                if (
                        np.isfinite(before)
                        and np.isfinite(after)
                        and abs(before) > 1e-12):
                    transitions.append(after / before - 1.0)
            if len(transitions) < 4:
                continue
            lower, upper = np.percentile(
                np.asarray(transitions, dtype=float), [5.0, 95.0])
            bounds[int(days)] = (float(lower), float(upper))
        return bounds

    @staticmethod
    def _transition_distance(
            candidate: pd.Series,
            history: pd.Series,
            start: pd.Timestamp,
            bounds: dict[int, tuple[float, float]],
    ) -> tuple[bool, float]:
        """Score one boundary against all available empirical horizons."""

        total_distance = 0.0
        for days, (lower, upper) in bounds.items():
            delta = pd.Timedelta(days=int(days))
            before = EnergyForecaster._window_mean(
                history, start - delta, start)
            after = EnergyForecaster._window_mean(
                candidate, start, start + delta)
            if (
                    not np.isfinite(before)
                    or abs(before) <= 1e-12
                    or not np.isfinite(after)):
                return False, np.inf
            transition = after / before - 1.0
            if lower <= transition <= upper:
                continue
            width = max(float(upper - lower), 1e-12)
            distance = min(
                abs(transition - lower),
                abs(transition - upper))
            total_distance += distance / width
        return total_distance <= 1e-12, total_distance

    @staticmethod
    def _cosine_anchor_bridge(
            dates: pd.DatetimeIndex,
            values: np.ndarray,
            anchor_ratio: float,
            bridge_days: int,
    ) -> np.ndarray:
        """Apply a smooth anchor bridge while preserving yearly means."""

        elapsed = (
            dates - dates[0]
        ).total_seconds().to_numpy(dtype=float) / 86400.0
        phase = np.clip(
            elapsed / max(float(bridge_days), 1.0), 0.0, 1.0)
        weights = 0.5 - 0.5 * np.cos(np.pi * phase)
        scales = anchor_ratio + (1.0 - anchor_ratio) * weights
        corrected = np.asarray(values, dtype=float) * scales
        return EnergyForecaster._match_yearly_means(
            dates, corrected, np.asarray(values, dtype=float))

    @staticmethod
    def _post_pattern_anchor(
            forecast: pd.DataFrame, history: pd.DataFrame,
            method: str, window: int, bridge_days: int = 365,
            diagnostics: Optional[Dict[str, Any]] = None,
    ) -> pd.DataFrame:
        """Apply a data-driven boundary correction after patterning.

        Automatic anchoring tests the forecast boundary at 7, 14, 30, 60,
        and 90 days against empirical transitions observed at the same
        calendar boundary in prior years. If correction is needed, the
        shortest bridge that is plausible at every supported horizon is
        selected. If none is fully plausible, the candidate with the smallest
        multiscale distance is used only when it improves on the uncorrected
        forecast. A cosine transition avoids a slope break, and calendar-year
        means are restored so prescribed annual energy remains unchanged.

        Explicit manual anchor methods retain their explicit behavior because
        they represent a deliberate user instruction.
        """

        if diagnostics is not None:
            diagnostics.update({
                "method": str(method or "auto").strip().lower(),
                "window": int(window),
                "applied": False,
                "bridge_days": None})
        if forecast.empty or history.empty:
            return forecast

        normalized = str(method or "auto").strip().lower()
        if normalized != "auto":
            anchor = get_anchor_value(
                history, method=normalized, window=window)
            if not np.isfinite(anchor):
                return forecast
            count = max(1, min(int(window), len(forecast)))
            initial = pd.to_numeric(
                forecast["y"].iloc[:count], errors="coerce").dropna()
            if initial.empty:
                return forecast
            if normalized == "last":
                forecast_level = float(initial.iloc[0])
            elif normalized == "median":
                forecast_level = float(initial.median())
            else:
                forecast_level = float(initial.mean())
            if (
                    not np.isfinite(forecast_level)
                    or abs(forecast_level) <= 1e-12):
                return forecast
            output = forecast.copy()
            bridge_count = min(max(int(bridge_days), 1), len(output))
            scales = np.ones(len(output), dtype=float)
            scales[:bridge_count] = np.linspace(
                anchor / forecast_level, 1.0, bridge_count)
            ratio = anchor / forecast_level
            output["y"] = pd.to_numeric(
                output["y"], errors="coerce").fillna(0.0).to_numpy() * scales
            if diagnostics is not None:
                diagnostics.update({
                    "applied": True,
                    "anchor_value": float(anchor),
                    "forecast_boundary_level": float(forecast_level),
                    "anchor_ratio": float(ratio),
                    "bridge_days": int(bridge_count)})
            return output

        output = forecast.copy()
        output["ds"] = pd.to_datetime(output["ds"])
        hist = history.copy()
        hist["ds"] = pd.to_datetime(hist["ds"])
        hist_values = pd.to_numeric(hist["y"], errors="coerce")
        forecast_values = pd.to_numeric(output["y"], errors="coerce")
        if hist_values.notna().sum() == 0 or forecast_values.notna().sum() == 0:
            return forecast

        hist_series = pd.Series(
            hist_values.to_numpy(dtype=float),
            index=pd.DatetimeIndex(hist["ds"]),
        ).dropna().sort_index()
        forecast_series = pd.Series(
            forecast_values.to_numpy(dtype=float),
            index=pd.DatetimeIndex(output["ds"]),
        ).dropna().sort_index()
        if hist_series.empty or forecast_series.empty:
            return forecast

        start = pd.Timestamp(forecast_series.index[0])
        horizons = (7, 14, 30, 60, 90)
        bounds = EnergyForecaster._transition_bounds(
            hist_series, start, horizons)
        if diagnostics is not None:
            diagnostics["empirical_horizons_days"] = list(bounds)
        if not bounds:
            return forecast

        plausible, baseline_distance = (
            EnergyForecaster._transition_distance(
                forecast_series, hist_series, start, bounds))
        if diagnostics is not None:
            diagnostics["baseline_transition_distance"] = float(
                baseline_distance)
        if plausible:
            if diagnostics is not None:
                diagnostics["boundary_plausible_without_correction"] = True
            return forecast

        anchor_days = 30
        recent_level = EnergyForecaster._window_mean(
            hist_series,
            start - pd.Timedelta(days=anchor_days),
            start)
        initial_level = EnergyForecaster._window_mean(
            forecast_series,
            start,
            start + pd.Timedelta(days=anchor_days))
        if (
                not np.isfinite(recent_level)
                or not np.isfinite(initial_level)
                or abs(initial_level) <= 1e-12):
            return forecast

        dates = pd.DatetimeIndex(output["ds"])
        base_values = forecast_values.fillna(0.0).to_numpy(dtype=float)
        anchor_ratio = recent_level / initial_level
        bridge_candidates = (7, 14, 30, 45, 60, 90, 120, 180, 270, 365)
        best = None
        best_days = None
        best_distance = float(baseline_distance)
        for days in bridge_candidates:
            corrected = EnergyForecaster._cosine_anchor_bridge(
                dates, base_values, anchor_ratio, days)
            candidate = pd.Series(corrected, index=dates)
            plausible, distance = EnergyForecaster._transition_distance(
                candidate, hist_series, start, bounds)
            if plausible:
                best = corrected
                best_days = int(days)
                best_distance = float(distance)
                break
            if distance + 1e-12 < best_distance:
                best = corrected
                best_days = int(days)
                best_distance = float(distance)

        if best is None:
            return forecast

        output["y"] = best
        if diagnostics is not None:
            diagnostics.update({
                "applied": True,
                "anchor_days": int(anchor_days),
                "historical_boundary_level": float(recent_level),
                "forecast_boundary_level": float(initial_level),
                "anchor_ratio": float(anchor_ratio),
                "bridge_days": best_days,
                "final_transition_distance": float(best_distance)})
        return output

    # ==== CORE PROJECTION METHODS ====


    def _model_projection(
            self,
            df_history: pd.DataFrame,
            percentage: float,
            energy: str,
            model_input: dict[str, Any],
    ) -> pd.DataFrame:
        """Build one forecast from dated scaling values."""

        dates = pd.date_range(
            start=self.config["_forecast_start"],
            end=self.config["_objetive_period_end"],
            freq=self.config["projection_frequency"],)
        history = pd.DataFrame({
            "ds": pd.to_datetime(df_history["ds"], errors="coerce"),
            "y": pd.to_numeric(
                df_history["y"],
                errors="coerce",
            ).fillna(0.0),
        }).dropna(subset=["ds"])
        anchor_method, anchor_window = self._get_anchor_params(energy)
        anchor_value = get_anchor_value(
            history,
            method=anchor_method,
            window=anchor_window,)
        target_per_period = self._target_per_period()
        is_demand = energy.casefold() == "demand"

        if is_demand:
            base = np.full(
                len(dates),
                target_per_period,
                dtype=float,)
            anchor_factor = anchor_value / max(target_per_period, 1e-12)
            factors = self._factor_path(
                dates,
                model_input.get("values", {}),
                anchor_factor,
                model=str(model_input.get("model", "linear")))
            target = base * factors
        else:
            if self.demand_target_series is None:
                raise RuntimeError(
                    "Demand target path must be built before sources.")
            if abs(float(percentage)) <= 1e-12:
                target = self._zero_share_target_path(
                    dates,
                    model_input.get("values", {}),
                    anchor_value,)
            else:
                base = self.demand_target_series.reindex(dates)
                base = base.ffill().bfill().to_numpy(dtype=float)
                base *= float(percentage)
                anchor_factor = anchor_value / max(
                    float(base[0]),
                    1e-12,)
                factors = self._factor_path(
                    dates,
                    model_input.get("values", {}),
                    anchor_factor,
                    model=str(model_input.get("model", "linear")))
                target = base * factors
        result = pd.DataFrame({"ds": dates, "y": target})
        result = self._maybe_apply_pattern(result, energy)
        result["y"] = self._match_yearly_means(
            dates,
            result["y"].to_numpy(dtype=float),
            target,)
        anchor_details: Dict[str, Any] = {
            "method": anchor_method,
            "window": int(anchor_window),
            "anchor_value": float(anchor_value)}
        result = self._post_pattern_anchor(
            result, history, anchor_method, anchor_window,
            diagnostics=anchor_details)
        self._resolved_anchors[energy] = anchor_details
        if is_demand:
            self.demand_target_series = pd.Series(
                target,
                index=dates,)
        return result

    def _target_per_period(self) -> float:
        """Return demand energy already normalized to one projection row.

        ``load_config_file`` normalizes ``demand.target_production`` to the
        configured projection resolution.  No additional daily-to-hourly
        conversion belongs here; applying one would divide hourly targets by
        24 a second time.
        """

        return float(self.config["_demand_target_production"])

    @staticmethod
    def _factor_path(
            dates: pd.DatetimeIndex,
            values: dict,
            anchor_factor: float,
            model: str = "linear",
    ) -> np.ndarray:
        """Interpolate dated factors using the configured trend model."""

        points = sorted(
            (pd.Timestamp(date), float(value))
            for date, value in values.items())
        if not points:
            return np.full(len(dates), anchor_factor, dtype=float)
        first_date = dates[0]
        if points[0][0] > first_date:
            points.insert(0, (first_date, float(anchor_factor)))
        model_name = str(model).strip().lower()
        output = np.full(len(dates), points[0][1], dtype=float)
        for index in range(len(points) - 1):
            start_date, start_value = points[index]
            end_date, end_value = points[index + 1]
            mask = (dates >= start_date) & (dates <= end_date)
            if not np.any(mask):
                continue
            duration = max(
                (end_date - start_date).total_seconds(),
                1.0,)
            elapsed = (
                dates[mask] - start_date
            ).total_seconds().to_numpy(dtype=float)
            position = np.clip(elapsed / duration, 0.0, 1.0)
            output[mask] = EnergyForecaster._interpolate_factor_segment(
                position,
                start_value,
                end_value,
                model_name,)
        output[dates < points[0][0]] = points[0][1]
        output[dates > points[-1][0]] = points[-1][1]
        return output

    @staticmethod
    def _interpolate_factor_segment(
            position: np.ndarray,
            start_value: float,
            end_value: float,
            model: str,
    ) -> np.ndarray:
        """Interpolate one factor segment while preserving both endpoints.

        Exponential interpolation is used only when both endpoints are
        strictly positive. Segments that touch zero fall back to linear
        interpolation so the requested zero pivot is reached exactly.
        """

        if model == "linear":
            shape = position
        elif model == "exponential":
            if start_value > 0.0 and end_value > 0.0:
                ratio = end_value / start_value
                return start_value * np.power(ratio, position)
            shape = position
        else:
            raise ValueError(f"Unsupported trend model: {model}.")
        return start_value + (end_value - start_value) * shape

    @staticmethod
    def _linear_factor_path(
            dates: pd.DatetimeIndex,
            values: dict,
            anchor_factor: float,
    ) -> np.ndarray:
        """Return the baseline linear factor path."""

        return EnergyForecaster._factor_path(
            dates,
            values,
            anchor_factor,
            model="linear",)

    @staticmethod
    def _zero_share_target_path(
            dates: pd.DatetimeIndex,
            values: dict,
            anchor_value: float,
    ) -> np.ndarray:
        """Bridge a positive historical source to a zero final share."""

        if len(dates) == 0:
            return np.array([], dtype=float)
        milestone_dates = sorted(
            pd.Timestamp(date)
            for date in values)
        closure_date = (
            milestone_dates[0]
            if milestone_dates
            else dates[-1])
        start_date = dates[0]
        if closure_date <= start_date:
            return np.zeros(len(dates), dtype=float)
        x_points = np.asarray(
            [start_date.value, closure_date.value],
            dtype=float,)
        y_points = np.asarray(
            [max(float(anchor_value), 0.0), 0.0],
            dtype=float,)
        x_dates = dates.view("int64").astype(float)
        target = np.interp(
            x_dates,
            x_points,
            y_points,
            left=y_points[0],
            right=0.0,)
        target[dates >= closure_date] = 0.0
        return target

    @staticmethod
    def _match_yearly_means(
            dates: pd.DatetimeIndex,
            values: np.ndarray,
            target: np.ndarray,
    ) -> np.ndarray:
        """Scale each available calendar year to its requested mean."""

        output = np.asarray(values, dtype=float).copy()
        target_values = np.asarray(target, dtype=float)
        years = dates.year.to_numpy(dtype=int)
        for year in np.unique(years):
            mask = years == year
            current = float(np.mean(output[mask]))
            requested = float(np.mean(target_values[mask]))
            if abs(current) > 1e-12:
                output[mask] *= requested / current
        return output

    # ==== MAIN EXECUTION ====

    def run(self):
        """
        Orchestrates the entire forecasting process: projecting demand,
        projecting sources, and saving results.
        """
        # 1. Project Demand
        emit(self.config, "Projecting Demand...", "detailed")
        demand_cfg = self.config.get("demand", {})
        demand_dates = pd.to_datetime(
            self.df_data[self.config["date_column"]],
            errors="coerce")
        demand_values = pd.to_numeric(
            self.df_data["Demand"],
            errors="coerce").fillna(0.0)
        df_dem_in = pd.DataFrame(
            {"ds": demand_dates, "y": demand_values})

        df_dem_proj = self._model_projection(
            df_dem_in,
            1.0,
            "Demand",
            demand_cfg,)

        # Store demand series for use in share calculations
        self.demand_series = df_dem_proj.set_index(
            to_naive_datetime_index(df_dem_proj["ds"])
        )["y"]

        # 2. Project Sources
        results: Dict[str, pd.Series] = {}
        # For custom sources, installed capacity must not be derived from the
        # patterned production series. It must use an unpatterned capacity basis
        # so closures are visible without seasonal oscillations.
        capacity_basis_results: Dict[str, pd.Series] = {}
        sources = self.config["sources"]
        custom_cols: List[str] = []
        custom_forecaster = CustomSourceForecaster(
            self.config, self.df_data, self._maybe_apply_pattern)

        for source, props in sources.items():
            emit(self.config, f"Projecting Source: {source}", "detailed")
            share = props.get("share", 0.0)
            model = str(props.get("model", "")).lower()

            capacity_basis_res = None

            if model == "custom":
                custom_cols.append(source)
                custom_projection = custom_forecaster.project(source)
                res = custom_projection.forecast
                capacity_basis_res = custom_projection.capacity_basis
            else:
                base_y = (
                    self.df_data[source]
                    if source in self.df_data.columns
                    else pd.Series(0.0, index=self.df_data.index))
                history_values = pd.to_numeric(
                    base_y,
                    errors="coerce").fillna(0.0)
                history_data = {
                    "ds": self.df_data[self.config["date_column"]],
                    "y": history_values}
                df_hist = pd.DataFrame(history_data)

                res = self._model_projection(
                    df_hist,
                    float(share),
                    source,
                    props,)

            # --- Dynamic production limit (source-level) ---
            # Used for technologies with scheduled closures or caps.
            # The limit is applied using the real forecast dates in res["ds"],
            # not the numeric dataframe index.
            dynamic_limit = None
            lim = props.get("limit", None)
            custom_changes = props.get("custom_data", {})

            if lim is not None:
                limit_value = float(lim)
                if not np.isfinite(limit_value) or limit_value < 0.0:
                    raise ValueError(
                        f"Invalid production limit for '{source}': {lim}.")

                res["y"] = pd.to_numeric(
                    res["y"], errors="coerce"
                ).fillna(0.0)
                date_idx = to_naive_datetime_index(res["ds"])
                dynamic_limit = pd.Series(limit_value, index=date_idx)

                if custom_changes:
                    for date_text, raw_limit in sorted(
                            custom_changes.items()):
                        change_date = pd.Timestamp(date_text)
                        new_limit = float(raw_limit)
                        if not np.isfinite(new_limit) or new_limit < 0.0:
                            raise ValueError(
                                f"Invalid production limit for '{source}' "
                                f"at {date_text}: {raw_limit}.")
                        active = dynamic_limit.index >= change_date
                        dynamic_limit.loc[active] = new_limit

                res["y"] = np.minimum(
                    res["y"].to_numpy(dtype=float),
                    dynamic_limit.to_numpy(dtype=float))
                res["y"] = res["y"].clip(lower=0.0)

            idx = to_naive_datetime_index(res["ds"])
            results[source] = pd.Series(res["y"].to_numpy(), index=idx)

            # Capacity basis for custom sources:
            # - if a dynamic limit exists, it represents the available
            # production
            #   ceiling after closures and should define installed capacity;
            # - otherwise, use the unpatterned custom projection.
            # This avoids Installed_Capacity_<custom> following the seasonal
            # production pattern.
            if model == "custom" and capacity_basis_res is not None:
                if dynamic_limit is not None:
                    basis_y = dynamic_limit.reindex(idx).to_numpy(dtype=float)
                else:
                    basis_y = pd.to_numeric(
                        capacity_basis_res["y"], errors="coerce"
                    ).fillna(0.0).to_numpy(dtype=float)

                capacity_basis_results[source] = pd.Series(
                    basis_y, index=idx
                ).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(lower=0.0)

        # 3. Build and Save Final Table
        df_all = pd.DataFrame(results).sort_index()
        self._build_and_save_results(
            df_all,
            custom_cols,
            capacity_basis_results)


    # ==== INSTALLED CAPACITY CALCULATION ====

    def _infer_forecast_value_type(self) -> str:
        """
        Resolves whether the forecasted source columns represent average power
        or
        energy per period. This controls the conversion to installed capacity.

        Accepted optional values in YAML:
        - forecast_value_type: power
        - forecast_value_type: energy

        If not provided, the method uses energy_unit as a fallback:
        - MW, GW, kW -> power
        - MWh, GWh, TWh -> energy
        """
        value_type = self.config.get("forecast_value_type", None)
        if value_type is not None:
            vt = str(value_type).strip().lower()
            if vt in ("power", "mw", "average_power"):
                return "power"
            if vt in ("energy", "mwh", "production"):
                return "energy"
            raise ValueError(
                f"Unsupported forecast_value_type='{value_type}'. "
                "Use 'power' or 'energy'.")

        unit = self.config["energy_unit"]
        if is_power_unit(unit):
            return "power"
        if is_energy_unit(unit):
            return "energy"

        # Conservative default for the current LEAF-EB configuration style.
        return "power"

    def _period_hours_for_capacity(
            self,
            index: pd.Index,
            groupby: str) -> pd.Series:
        """
        Returns the number of hours represented by each row in the grouped
        forecast.
        This is only needed when forecasted values represent energy per period.
        """
        gb = str(groupby).lower()

        if gb == "hour":
            return pd.Series(1.0, index=index)
        if gb == "day":
            return pd.Series(24.0, index=index)

        if gb == "month":
            if isinstance(index, pd.MultiIndex):
                years = index.get_level_values(0).astype(int)
                months = index.get_level_values(1).astype(int)
            else:
                dt = pd.to_datetime(index)
                years = dt.year
                months = dt.month

            hours = [
                monthrange(int(year), int(month))[1] * 24.0
                for year, month in zip(years, months)]
            return pd.Series(hours, index=index)

        if gb == "year":
            years = pd.Index(index).astype(int)
            hours = [
                8784.0
                if monthrange(int(year), 2)[1] == 29
                else 8760.0
                for year in years]
            return pd.Series(hours, index=index)

        raise ValueError(
            "Installed capacity calculation currently supports "
            "groupby values Day, Month, or Year.")

    def _years_from_grouped_index(
            self,
            index: pd.Index,
            groupby: str) -> pd.Series:
        """Returns the calendar year associated with each grouped row."""
        gb = str(groupby).lower()

        if gb == "month" and isinstance(index, pd.MultiIndex):
            years = index.get_level_values(0).astype(int)
        elif gb == "year":
            years = pd.Index(index).astype(int)
        else:
            years = pd.to_datetime(index).year

        return pd.Series(years, index=index)

    def _capacity_period_timestamps(
            self, index: pd.Index, groupby: str) -> pd.DatetimeIndex:
        """Return one effective timestamp for each grouped forecast row."""
        gb = str(groupby).lower()
        if gb in {"day", "hour"}:
            return pd.DatetimeIndex(pd.to_datetime(index))
        if gb == "month":
            if isinstance(index, pd.MultiIndex):
                years = index.get_level_values(0).astype(int)
                months = index.get_level_values(1).astype(int)
            else:
                dates = pd.to_datetime(index)
                years = dates.year
                months = dates.month
            values = [
                pd.Timestamp(int(year), int(month), 1)
                + pd.offsets.MonthEnd(0)
                for year, month in zip(years, months)]
            return pd.DatetimeIndex(values)
        if gb == "year":
            values = [
                pd.Timestamp(int(year), 12, 31)
                for year in pd.Index(index)]
            return pd.DatetimeIndex(values)
        raise ValueError(
            "Explicit installed capacity supports Day, Hour, Month, "
            "or Year grouping.")

    def _explicit_installed_capacity(
            self, props: Dict, index: pd.Index, groupby: str
    ) -> Optional[pd.Series]:
        """Build installed MW directly from dated capacity additions."""
        additions = props.get("capacity_additions")
        if not isinstance(additions, dict):
            return None
        initial = float(props.get("initial_capacity", 0.0))
        timestamps = self._capacity_period_timestamps(index, groupby)
        values = np.full(len(index), initial, dtype=float)
        events = sorted(
            (pd.Timestamp(date), float(delta))
            for date, delta in additions.items())
        for event_date, delta in events:
            values[timestamps >= event_date] += delta
        if np.any(values < -1e-9):
            raise ValueError(
                "Explicit installed capacity cannot become negative.")
        return pd.Series(np.maximum(values, 0.0), index=index)

    def _add_installed_capacity_columns(
        self,
        saving: pd.DataFrame,
        groupby: str,
        capacity_basis_grouped: Optional[Dict[str, pd.Series]] = None,
    ) -> pd.DataFrame:
        """
        Adds Installed_Capacity_<source> and Installed_Capacity_Total to the
        dataframe that will be written to Forecast.xlsx.

        The calculation is intentionally placed after global balance enforcement
        so that installed-capacity columns do not enter the Total calculation.

        For production-based sources, capacity is derived from production and
        capacity_factor. Predictor does not calculate Interconnection_Limit or
        Interconnection_Capacity.
        """
        sources = self.config.get("sources", {})
        if not sources:
            return saving

        value_type = self._infer_forecast_value_type()

        emit(
            self.config,
            "Installed capacity calculation: "
            f"forecast_value_type={value_type}. Default rule: custom "
            "sources = period; other sources = annual.",
            "detailed")
        years = self._years_from_grouped_index(saving.index, groupby)
        installed_cols: List[str] = []

        if value_type == "energy":
            unit = self.config.get(
                "forecast_energy_unit", self.config.get("energy_unit", "MWh"))
            unit_factor = energy_to_mwh_factor(unit)
            hours = self._period_hours_for_capacity(saving.index, groupby)
        else:
            unit = self.config.get(
                "forecast_power_unit", self.config.get("energy_unit", "MW"))
            unit_factor = power_to_mw_factor(unit)
            hours = None

        for source, props in sources.items():
            if source not in saving.columns:
                continue

            if not isinstance(props, dict):
                continue

            out_col = f"Installed_Capacity_{source}"
            explicit = self._explicit_installed_capacity(
                props, saving.index, groupby)
            if explicit is not None:
                saving[out_col] = explicit
                installed_cols.append(out_col)
                emit(
                    self.config,
                    "Using explicit installed capacity for "
                    f"'{source}'.",
                    "detailed")
                continue

            cf = props.get("capacity_factor", None)
            if cf is None:
                emit(
                    self.config,
                    f"WARNING: Source '{source}' has no capacity_factor; "
                    f"{out_col} was not calculated.")
                continue

            cf = float(cf)
            if not np.isfinite(cf) or cf <= 0.0 or cf > 1.0:
                raise ValueError(
                    f"Invalid capacity_factor for source '{source}': {cf}. "
                    "It must be in the interval (0, 1].")

            # Internal rule to keep the YAML clean:
            # - custom sources usually represent scheduled/discrete production
            # changes, so capacity is estimated per period;
            # - all other sources use annual energy/power to avoid artificial
            # daily capacity fluctuations.
            model = str(props.get("model", "")).strip().lower()
            window = "period" if model == "custom" else "annual"

            # For custom sources, use an unpatterned capacity basis when
            # available.
            # The production column may include the seasonal pattern, but
            # installed
            # capacity should represent available capacity, not seasonal output.
            if (
                model == "custom"
                and capacity_basis_grouped is not None
                and source in capacity_basis_grouped
            ):
                prod_source = (
                    capacity_basis_grouped[source]
                    .reindex(saving.index)
                    .fillna(0.0))
                emit(
                    self.config,
                    "Using unpatterned capacity basis for "
                    f"custom source '{source}'.",
                    "detailed")
            else:
                prod_source = saving[source]

            prod = (
                pd
                .to_numeric(prod_source, errors='coerce')
                .fillna(0.0)
                .clip(lower=0.0))

            if value_type == "energy":
                production_energy = prod * unit_factor
                if window == "period":
                    saving[out_col] = production_energy / (hours * cf)
                else:
                    annual_energy = production_energy.groupby(years).sum()
                    annual_hours = hours.groupby(years).sum()
                    annual_capacity = annual_energy / (annual_hours * cf)
                    saving[out_col] = years.map(annual_capacity).to_numpy()
            else:
                production_power = prod * unit_factor
                if window == "period":
                    saving[out_col] = production_power / cf
                else:
                    annual_power = production_power.groupby(years).mean()
                    annual_capacity = annual_power / cf
                    saving[out_col] = years.map(annual_capacity).to_numpy()

            saving[out_col] = (
                saving[out_col]
                .replace([np.inf, -np.inf], np.nan)
                .fillna(0.0))
            saving[out_col] = saving[out_col].clip(lower=0.0)
            installed_cols.append(out_col)

        if installed_cols:
            saving['Installed_Capacity_Total'] = saving[installed_cols].sum(
                axis=1)

        return saving

    def _build_and_save_results(
        self,
        df_all: pd.DataFrame,
        custom_cols: List[str],
        capacity_basis_results: Optional[Dict[str, pd.Series]] = None,
    ):
        """
        Groups the final projected results by time frequency, enforces global
        balance, and saves the output to an Excel file.
        """
        groupby = self.config.get("projection_grouping", "Day")

        # Grouping and cleaning
        saving = df_all.copy()
        ts = to_naive_datetime_index(saving.index)

        if groupby.lower() == "hour":
            saving.index = ts
        elif groupby.lower() == "day":
            saving.index = ts.date
        elif groupby.lower() == 'month':
            saving = saving.groupby([ts.year, ts.month]).sum()
        elif groupby.lower() == "year": saving = saving.groupby(ts.year).sum()

        saving = (
            saving
            .apply(pd.to_numeric, errors='coerce')
            .fillna(0.0)
            .clip(lower=0.0))

        # Add demand
        dem_grp = group_series_by(self.demand_series, groupby)
        saving["Demand"] = dem_grp.reindex(saving.index).fillna(0.0).to_numpy()

        # Global Balance
        auto_metric = (
            'final_year_mean'
            if self.config.get('_objetive_granularity') == 'year'
            else 'endpoint')
        metric = (
            str(self.config.get('balance_metric', auto_metric))
            .lower()
            .strip())

        saving = enforce_global_balance(
            saving,
            balance=self.config["_demand_balance"],
            target_metric=metric,
            exclude_cols=custom_cols)

        pre_replacement = saving.copy()
        replacement = apply_custom_replacements(saving, self.config)
        saving = replacement.forecast
        replacement_diagnostics = replacement.diagnostics

        # Group custom capacity bases using the same output aggregation.
        # For the current LEAF-EB workflow these series represent energy per
        # period,
        # so grouping uses summation, consistent with source production columns.
        capacity_basis_grouped = None
        if capacity_basis_results:
            capacity_basis_grouped = {}
            for src, basis_series in capacity_basis_results.items():
                capacity_basis_grouped[src] = group_series_by(
                    basis_series,
                    groupby)

        # Installed capacity is derived after final production values are fixed.
        # This prevents capacity columns from entering Total or the balance
        # scaling.
        saving = self._add_installed_capacity_columns(
            saving,
            groupby,
            capacity_basis_grouped)
        if not replacement_diagnostics.empty:
            pre_replacement = self._add_installed_capacity_columns(
                pre_replacement,
                groupby,
                capacity_basis_grouped)

        saving = saving.reset_index()
        saving = saving.rename(columns={saving.columns[0]: "Date"})
        if not replacement_diagnostics.empty:
            pre_replacement = pre_replacement.reset_index()
            pre_replacement = pre_replacement.rename(
                columns={pre_replacement.columns[0]: "Date"})

        # Output saving
        out_dir = (
            Path(__file__).resolve().parents[2]
            / self.config["scenario_folder"]
            / self.config["scenario_subfolder"])
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / 'Forecast.xlsx'

        emit(self.config, f"Saving results to {out_file}...", "detailed")
        try:
            with pd.ExcelWriter(out_file, engine="openpyxl") as writer:
                saving.to_excel(writer, sheet_name="Forecast", index=False)
        except PermissionError as exc:
            raise PermissionError(
                f"Cannot write to '{out_file}'. Close the workbook and "
                "run LEAF again.") from exc

        # Fast internal cache for repeated simulation workers.  Excel remains
        # the user-facing file; the CSV is a disposable derived copy.
        cache_file = out_file.with_suffix(".csv")
        saving.to_csv(cache_file, index=False)
        pre_replacement_file = out_dir / "Forecast_PreReplacement.csv"
        if not replacement_diagnostics.empty:
            pre_replacement.to_csv(pre_replacement_file, index=False)
        elif pre_replacement_file.is_file():
            pre_replacement_file.unlink()

        if self.config.get("save_projection_plots", False):
            plot_frame = saving.copy()
            plot_dates = pd.to_datetime(plot_frame["Date"])
            plot_start = pd.Timestamp(self.config["start_date"])
            plot_end = pd.Timestamp(self.config["end_date"])
            plot_frame = plot_frame.loc[
                (plot_dates >= plot_start) & (plot_dates <= plot_end)]
            plot_path = (
                out_dir / "Output" / "Plots"
                / "Forecast_deterministic.png")
            save_energy_series_plot(
                plot_frame,
                plot_path,
                self.config["energy_unit"],
                "Date",)
            emit(
                self.config,
                f"Saving deterministic forecast plot to: {plot_path}",
                "detailed")

        replacement_file = out_dir / "Custom_Replacement.xlsx"
        save_replacement_diagnostics(
            replacement_diagnostics, replacement_file)
        write_resolved_config(
            self.config,
            out_dir,
            resolved={"anchors": self._resolved_anchors})

# ==== ENTRY POINT ====

def main(input_name: str):
    """Run the deterministic forecasting pipeline for one input file."""

    start_time = time.time()
    forecaster = EnergyForecaster(input_name)
    forecaster.load_data()
    forecaster.run()
    elapsed = time.time() - start_time
    emit(forecaster.config, f"Forecast completed in {elapsed:.2f} seconds")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python Predictor.py <input_name_without_extension>")
        sys.exit(1)
    main(sys.argv[1])
