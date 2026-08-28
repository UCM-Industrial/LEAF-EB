"""Projection utilities for custom sources and capacity-based replacement."""

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.forecasting.anchors import apply_anchor, get_anchor_value
from src.utilities.console import emit
from src.utilities.units import (
    convert_power, is_energy_unit, is_power_unit,
    mw_period_energy_factor)


PatternApplier = Callable[[pd.DataFrame, str], pd.DataFrame]


@dataclass
class CustomProjection:
    """Forecast and unpatterned basis for one custom source."""

    forecast: pd.DataFrame
    capacity_basis: pd.DataFrame


@dataclass
class ReplacementResult:
    """Forecast and diagnostics after custom-source replacement."""

    forecast: pd.DataFrame
    diagnostics: pd.DataFrame


def apply_custom_replacements(
        forecast: pd.DataFrame, config: Dict) -> ReplacementResult:
    """Apply every ``custom_mode: replace`` rule after balancing."""
    result = forecast.copy()
    diagnostics = pd.DataFrame(index=result.index)
    sources = config.get("sources", {})

    for custom_name, properties in sources.items():
        if not isinstance(properties, dict):
            continue
        model = str(properties.get("model", "")).strip().lower()
        mode = str(properties.get("custom_mode", "add")).strip().lower()
        if model != "custom":
            continue
        if mode not in {"add", "replace"}:
            message = (
                f"Invalid custom_mode='{mode}' for '{custom_name}'. "
                "Use 'add' or 'replace'.")
            raise ValueError(message)
        if mode == "add":
            continue

        remaining = _apply_replacement_rule(
            result, diagnostics, custom_name, properties, set(sources))
        _warn_unallocated(custom_name, remaining, config)

    production_cols = [
        name for name in sources if name in result.columns]
    if production_cols:
        result["Total"] = result[production_cols].sum(axis=1)
    return ReplacementResult(result, diagnostics)


def _apply_replacement_rule(
        forecast: pd.DataFrame, diagnostics: pd.DataFrame,
        custom_name: str, properties: Dict,
        valid_sources: set) -> pd.Series:
    """
    Reduce configured sources through annual capacity-equivalent scaling.

    The YAML remains unchanged. Replacement groups keep their configured
    priority, but allocation is solved by year instead of independently in
    every period. Each affected source receives one scale factor per year, so
    its original daily and seasonal profile is preserved.
    """
    if custom_name not in forecast.columns:
        raise ValueError(
            f"Custom source '{custom_name}' is missing from the forecast.")

    groups = properties.get("replaces")
    if not isinstance(groups, list) or not groups:
        raise ValueError(
            f"Custom source '{custom_name}' requires a replaces list.")

    requested = pd.to_numeric(
        forecast[custom_name], errors="coerce"
    ).fillna(0.0).clip(lower=0.0)

    date_values = (
        forecast["Date"] if "Date" in forecast.columns
        else pd.Index(forecast.index))
    dates = pd.DatetimeIndex(pd.to_datetime(
        date_values, errors="coerce"))

    if dates.isna().any():
        raise ValueError(
            "Custom replacement requires a Date column or a date-like "
            "forecast index.")

    years = pd.Series(
        dates.year,
        index=forecast.index,
        dtype=int,)

    replaced = pd.Series(0.0, index=forecast.index)
    unallocated = pd.Series(0.0, index=forecast.index)
    used_sources = set()

    validated_groups = []

    for group in groups:
        names = _validate_replacement_group(
            group,
            custom_name,
            forecast,
            used_sources,
            valid_sources,)
        validated_groups.append(names)

        for name in names:
            diagnostics[
                f"{custom_name}_Reduced_{name}"
            ] = 0.0

    for year in sorted(years.unique()):
        year_index = forecast.index[years == year]

        requested_year = float(
            requested.loc[year_index].sum())

        if requested_year <= 0.0:
            continue

        remaining_year = requested_year

        for names in validated_groups:
            if remaining_year <= 1e-9:
                break

            available = forecast.loc[
                year_index,
                names,
            ].apply(
                pd.to_numeric,
                errors="coerce",
            ).fillna(0.0).clip(lower=0.0)

            annual_generation = available.sum(axis=0)
            group_available = float(
                annual_generation.sum())

            if group_available <= 0.0:
                continue

            group_reduction = min(
                remaining_year,
                group_available,)

            source_weights = (
                annual_generation / group_available
            ).fillna(0.0)

            annual_reductions = (
                source_weights * group_reduction)

            for name in names:
                source_total = float(
                    annual_generation[name])

                if source_total <= 0.0:
                    continue

                reduction_fraction = min(
                    float(annual_reductions[name])
                    / source_total,
                    1.0,)

                source_values = available[name]
                source_reduction = (
                    source_values * reduction_fraction)

                forecast.loc[
                    year_index,
                    name,
                ] = (
                    source_values - source_reduction
                ).clip(lower=0.0)

                diagnostics.loc[
                    year_index,
                    f"{custom_name}_Reduced_{name}",
                ] = source_reduction

                replaced.loc[
                    year_index
                ] += source_reduction

            remaining_year -= group_reduction

        if remaining_year > 1e-9:
            requested_slice = requested.loc[
                year_index]

            requested_sum = float(
                requested_slice.sum())

            if requested_sum > 0.0:
                unallocated.loc[
                    year_index
                ] = (
                    requested_slice
                    * remaining_year
                    / requested_sum)

    diagnostics[f"{custom_name}_Requested"] = requested
    diagnostics[f"{custom_name}_Replaced"] = replaced
    diagnostics[f"{custom_name}_Unallocated"] = unallocated

    return unallocated



def _validate_replacement_group(
        group, custom_name: str, forecast: pd.DataFrame,
        used_sources: set, valid_sources: set) -> List[str]:
    """Validate one ordered replacement group."""
    if isinstance(group, list):
        names = group
    elif isinstance(group, dict):
        allocation = str(
            group.get("allocation", "proportional")
        ).strip().lower()
        if allocation != "proportional":
            raise ValueError(
                f"Unsupported allocation='{allocation}' for "
                f"'{custom_name}'.")
        names = group.get("sources")
    else:
        raise ValueError(
            f"Replacement groups for '{custom_name}' must be lists.")

    if not isinstance(names, list) or not names:
        raise ValueError(
            f"Each replacement group for '{custom_name}' needs sources.")

    for name in names:
        if name == custom_name:
            raise ValueError(
                f"Custom source '{custom_name}' cannot replace itself.")
        if name not in valid_sources:
            raise ValueError(
                f"Replacement source '{name}' is not configured.")
        if name not in forecast.columns:
            raise ValueError(
                f"Replacement source '{name}' is missing from forecast.")
        if name in used_sources:
            raise ValueError(
                f"Replacement source '{name}' appears more than once.")
        used_sources.add(name)
    return names


def _warn_unallocated(
        custom_name: str, unallocated: pd.Series, config: Dict) -> None:
    """Record replacement shortfalls and print them only when useful."""

    tolerance = 1e-9
    affected = unallocated > tolerance
    if not affected.any():
        return

    count = int(affected.sum())
    total = float(unallocated.loc[affected].sum())
    maximum = float(unallocated.loc[affected].max())
    diagnostics = config.get("_runtime_diagnostics")
    if isinstance(diagnostics, list):
        diagnostics.append({
            "Type": "custom_replacement_unallocated",
            "Source": str(custom_name),
            "Resolution": "projection",
            "Count": count,
            "Energy_Removed": total,
            "Max_Excess_Percent": np.nan,
            "Max_Date": "",
            "Maximum_Value": maximum,})

    level = "detailed" if config.get("_stochastic_run") else "standard"
    emit(
        config,
        f"WARNING: '{custom_name}' has unallocated custom generation in "
        f"{count} periods (total={total:.6f}; maximum={maximum:.6f}).",
        level)


def save_replacement_diagnostics(
        diagnostics: pd.DataFrame, output_path: Path) -> None:
    """Save replacement details separately from Forecast.xlsx."""
    if diagnostics.empty:
        return

    output = diagnostics.copy().reset_index()
    output = output.rename(columns={output.columns[0]: "Date"})
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        output.to_excel(writer, sheet_name="Replacement", index=False)


class CustomSourceForecaster:
    """Build custom step schedules outside the main Predictor class."""

    def __init__(self, config: Dict, data: pd.DataFrame,
                 pattern_applier: PatternApplier):
        """Initialize a new CustomSourceForecaster instance."""

        self.config = config
        self.data = data
        self.pattern_applier = pattern_applier

    def project(self, source_name: str) -> CustomProjection:
        """Project one custom source and preserve its capacity basis."""
        source = self.config.get("sources", {}).get(source_name, {}) or {}
        capacity = source.get("capacity_additions")
        refueling = source.get("refueling", {}) or {}
        if isinstance(capacity, dict) and bool(refueling):
            return self._project_refueling_defined_reference(
                source_name, source)
        if isinstance(capacity, dict):
            return self._project_capacity_defined_reference(
                source_name, source)

        history = self._get_history(source_name)
        capacity_basis = self._build_schedule(source_name, history)
        forecast = self.pattern_applier(
            capacity_basis.copy(), source_name)
        forecast = self._equalize_before_first_event(
            forecast, history, source_name)
        return CustomProjection(forecast, capacity_basis)

    def _project_capacity_defined_reference(
            self, source_name: str, source: Dict) -> CustomProjection:
        """Apply the historical pattern below explicit installed capacity."""

        dates = self._forecast_dates()
        installed = self._installed_capacity_schedule(source, dates)
        reference = source.get("reference_generation", {}) or {}
        capacity_factor = float(reference["capacity_factor"])
        nominal = pd.DataFrame({
            "ds": dates,
            "y": [
                self._reference_value(value, capacity_factor)
                for value in installed],})
        capacity_basis = pd.DataFrame({
            "ds": dates,
            "y": [
                self._reference_value(value, 1.0)
                for value in installed],})
        forecast = self.pattern_applier(nominal, source_name)
        forecast["y"] = np.minimum(
            pd.to_numeric(forecast["y"], errors="coerce").fillna(0.0),
            capacity_basis["y"].to_numpy(dtype=float),)
        return CustomProjection(forecast, capacity_basis)

    def _project_refueling_defined_reference(
            self, source_name: str, source: Dict) -> CustomProjection:
        """Build reference generation from explicit refuelling availability.

        For an explicit-capacity source with scheduled refuelling, LEAF does
        not assume a capacity factor. Installed capacity is converted to an
        available-capacity profile using the same deterministic refuelling
        scheduler used by the operational model. Reference generation for
        source replacement is then the energy available at full power.
        """
        if source.get("reference_generation", {}).get(
                "capacity_factor") is not None:
            raise ValueError(
                f"Source '{source_name}' defines refueling, so "
                "reference_generation.capacity_factor must be removed. "
                "Reference generation is derived from refuelling "
                "availability.")

        dates = self._forecast_dates()
        installed = self._installed_capacity_schedule(source, dates)
        temporary = pd.DataFrame({
            "Date": dates,
            f"Installed_Capacity_{source_name}": installed,})

        # Local import avoids coupling the forecasting module at import time
        # while guaranteeing identical outage timing in both stages.
        from src.technologies.nuclear.refueling import build_refueling_profiles

        operation = str(
            source.get("hourly_operation", "")
        ).strip().lower()
        power_fraction = 1.0
        if operation == "must_run":
            power_fraction = float(
                (source.get("must_run", {}) or {})["power_fraction"])

        user_input = {"sources": {source_name: source}}
        refuelled, _ = build_refueling_profiles(
            temporary,
            user_input,
            nominal_efpd=True,
            nominal_efpd_rate=power_fraction)
        available = pd.to_numeric(
            refuelled[f"Available_Capacity_{source_name}"],
            errors="coerce",
        ).fillna(0.0).to_numpy(dtype=float)

        reference_values = np.array([
            self._reference_value(value, power_fraction)
            for value in available
        ], dtype=float)
        installed_values = np.array([
            self._reference_value(value, 1.0)
            for value in installed
        ], dtype=float)

        forecast = pd.DataFrame({"ds": dates, "y": reference_values})
        capacity_basis = pd.DataFrame({
            "ds": dates, "y": installed_values})
        return CustomProjection(forecast, capacity_basis)

    @staticmethod
    def _installed_capacity_schedule(
            source: Dict, dates: pd.DatetimeIndex) -> np.ndarray:
        """Return cumulative installed MW from dated additions."""
        additions = source.get("capacity_additions", {}) or {}
        initial = float(source.get("initial_capacity", 0.0))
        values = np.full(len(dates), initial, dtype=float)
        for date, delta in sorted(
                (pd.Timestamp(date), float(delta))
                for date, delta in additions.items()):
            values[dates >= date] += delta
        if np.any(values < -1e-9):
            raise ValueError(
                "Explicit installed capacity cannot become negative.")
        return np.maximum(values, 0.0)

    def _get_history(
            self, source_name: str) -> Optional[pd.DataFrame]:
        """Return source history when the input contains the source."""
        if source_name not in self.data.columns:
            return None

        date_col = self.config["date_column"]
        values = pd.to_numeric(
            self.data[source_name], errors="coerce").fillna(0.0)
        return pd.DataFrame({"ds": self.data[date_col], "y": values})

    def _get_anchor_params(self, source_name: str) -> Tuple[str, int]:
        """Return an optional manual anchor or the automatic default."""
        source = self.config.get("sources", {}).get(source_name, {})
        if not isinstance(source, dict):
            return "auto", 1
        anchor = source.get("anchor", {})
        method = str(anchor.get("method", "auto")).lower()
        window = int(anchor.get("window", 1))
        return method, window

    def _get_events(
            self, source_name: str) -> Tuple[List[str], List[float]]:
        """Return custom event dates and values from the current schema."""
        source = self.config.get("sources", {}).get(source_name, {})
        if isinstance(source, dict):
            custom_data = source.get("custom_data")
            if isinstance(custom_data, dict):
                return list(custom_data.keys()), list(custom_data.values())
            capacity = source.get("capacity_additions")
            if isinstance(capacity, dict):
                return self._capacity_reference_events(source, capacity)

        return [], []

    def _capacity_reference_events(
            self, source: Dict, capacity: Dict
    ) -> Tuple[List[str], List[float]]:
        """Convert explicit MW additions to reference-generation events."""
        reference = source.get("reference_generation", {}) or {}
        capacity_factor = float(reference["capacity_factor"])
        values = [
            self._reference_value(float(delta), capacity_factor)
            for delta in capacity.values()]
        return list(capacity.keys()), values

    def _reference_value(
            self, power_capacity: float, capacity_factor: float) -> float:
        """Return one projection-period reference value for a MW change."""
        unit = self.config.get("energy_unit", "MWh")
        resolution = str(
            self.config.get("projection_resolution", "daily")
        ).lower()
        average_power = power_capacity * capacity_factor
        if is_power_unit(unit):
            return convert_power(average_power, "MW", unit)
        if is_energy_unit(unit):
            hours = 24.0 if resolution == "daily" else 1.0
            factor = mw_period_energy_factor(hours, unit)
            return average_power * factor
        raise ValueError(
            f"Unsupported unit '{unit}' for capacity-based custom source.")

    def _forecast_dates(self) -> pd.DatetimeIndex:
        """Build the complete forecast index for custom sources."""
        date_col = self.config["date_column"]
        historical_dates = pd.to_datetime(
            self.data[date_col], errors="coerce", utc=True)
        last_date = historical_dates.dt.tz_convert(None).max()
        frequency = self.config["projection_frequency"]
        offset = pd.tseries.frequencies.to_offset(frequency)
        end_date = self.config["_objetive_period_end"]
        return pd.date_range(last_date + offset, end_date,
                             freq=frequency)

    def _build_schedule(
            self, source_name: str,
            history: Optional[pd.DataFrame]) -> pd.DataFrame:
        """Build an unpatterned custom step schedule."""
        dates = self._forecast_dates()
        event_dates, event_values = self._get_events(source_name)
        event_dates = pd.to_datetime(event_dates).to_pydatetime().tolist()
        event_values = [float(value) for value in event_values]
        method, window = self._get_anchor_params(source_name)

        if history is not None and not history.empty:
            initial = float(get_anchor_value(
                history, method=method, window=window))
            values = self._schedule_with_history(
                dates, event_dates, event_values, initial)
        else:
            values = self._schedule_without_history(
                dates, event_dates, event_values)

        result = pd.DataFrame({"ds": dates, "y": values})
        first_year = event_dates[0].year if event_dates else None
        return apply_anchor(result, history, year=first_year,
                            method=method, window=window)

    @staticmethod
    def _apply_step_deltas(values, dates, event_dates, deltas):
        """Apply dated additive changes to a stepwise forecast array."""

        for event_date, delta in zip(event_dates, deltas):
            index = np.searchsorted(
                dates.values, np.datetime64(event_date))
            if index < len(values):
                values[index:] += float(delta)
        return values

    @staticmethod
    def _schedule_with_history(
            dates: pd.DatetimeIndex, event_dates: List,
            event_values: List[float], initial: float) -> np.ndarray:
        """Apply every configured event as a delta to the anchor."""
        event_count = len(event_dates)
        if len(event_values) == event_count:
            deltas = event_values
        elif len(event_values) == event_count - 1:
            deltas = [0.0] + event_values
        else:
            deltas = (event_values + [0.0] * event_count)[:event_count]

        values = np.full(len(dates), initial, dtype=float)
        return CustomSourceForecaster._apply_step_deltas(
            values, dates, event_dates, deltas)

    @staticmethod
    def _schedule_without_history(
            dates: pd.DatetimeIndex, event_dates: List,
            event_values: List[float]) -> np.ndarray:
        """Start at zero and apply a custom schedule without history."""
        if len(event_values) == len(event_dates):
            initial = float(event_values[0]) if event_values else 0.0
            deltas = event_values[1:]
            delta_dates = event_dates[1:]
        elif len(event_values) == len(event_dates) - 1:
            initial = 0.0
            deltas = event_values
            delta_dates = event_dates[1:]
        else:
            initial = 0.0
            deltas = []
            delta_dates = []

        values = np.zeros(len(dates), dtype=float)
        if not event_dates:
            values[:] = initial
            return values

        start = np.searchsorted(
            dates.values, np.datetime64(event_dates[0]))
        if start < len(values):
            values[start:] = initial

        return CustomSourceForecaster._apply_step_deltas(
            values, dates, delta_dates, deltas)

    def _equalize_before_first_event(
            self, forecast: pd.DataFrame,
            history: Optional[pd.DataFrame],
            source_name: str) -> pd.DataFrame:
        """Restore the historical anchor before the first custom event."""
        if history is None or history.empty or forecast.empty:
            return forecast

        method, window = self._get_anchor_params(source_name)
        anchor = get_anchor_value(
            history, method=method, window=window)
        if not np.isfinite(anchor):
            return forecast

        event_dates, _ = self._get_events(source_name)
        first_event = pd.to_datetime(
            event_dates[0]) if event_dates else None
        if first_event is None:
            mask = pd.Series(True, index=forecast.index)
        else:
            mask = forecast["ds"] < first_event

        if not mask.any():
            return forecast

        first_index = forecast.index[mask][0]
        first_value = float(forecast.loc[first_index, "y"])
        if np.isfinite(first_value) and abs(first_value) > 1e-12:
            forecast.loc[mask, "y"] *= anchor / first_value
        return forecast
