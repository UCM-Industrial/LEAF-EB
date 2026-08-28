# -*- coding: utf-8 -*-
"""Hourly expansion utilities for LEAF."""

from pathlib import Path

import numpy as np
import pandas as pd

from src.forecasting.historical_data import load_historical_dataset
from src.utilities.constants import MC_PROFILE_DATE_COLUMN
from src.utilities.console import emit
from src.utilities.name_resolution import build_name_lookup, name_key
from src.utilities.units import energy_from_mwh_factor
from src.technologies.nuclear.operation import (
    begin_efpd_day, dynamic_efpd_sources, finish_efpd_day,
    initialize_efpd_fleet_state, online_efpd_sources)
from src.technologies.nuclear.refueling import (
    apply_efpd_refueling_profiles)


HOURS_PER_DAY = 24
PATTERN_ROWS = 8760
_UNIFORM_PROFILE = np.full(
    HOURS_PER_DAY, 1.0 / HOURS_PER_DAY, dtype=np.float32)
_HISTORICAL_PROFILE_CACHE = {}
_HISTORICAL_DATASET_CACHE = {}


def enforce_daily_capacity_feasibility(
        daily_df, user_input, root_dir):
    """Apply daily capacity limits without temporal redistribution."""

    daily = _prepare_daily_data(daily_df)
    source_input = user_input.get("sources", {})
    dispatchable = _sources_by_operation(source_input, "dispatchable")
    load_following = _sources_by_operation(
        source_input, "load_following")
    pattern_cache = {}
    day_indices = _day_indices(daily["Date"])

    for name, source_data in source_input.items():
        capacity_name = f"Installed_Capacity_{name}"
        if name not in daily.columns or capacity_name not in daily.columns:
            continue

        values = pd.to_numeric(
            daily[name], errors="coerce"
        ).to_numpy(dtype=np.float64)
        operation = str(
            source_data.get("hourly_operation", "")
            if isinstance(source_data, dict) else ""
        ).strip().lower()
        available_name = f"Available_Capacity_{name}"
        refueling = (
            source_data.get("refueling", {})
            if isinstance(source_data, dict) else {}) or {}
        if (
            operation == "must_run"
            and bool(refueling)
            and available_name in daily.columns
        ):
            power_fraction = _must_run_power_fraction(name, source_data)
            unit_factor = energy_from_mwh_factor(
                user_input.get("energy_unit", "MWh"))
            values = pd.to_numeric(
                daily[available_name], errors="coerce"
            ).to_numpy(dtype=np.float64, copy=True)
            values *= HOURS_PER_DAY * unit_factor * power_fraction
            daily[name] = values
        capacities = pd.to_numeric(
            daily[capacity_name], errors="coerce"
        ).to_numpy(dtype=np.float64)

        if np.any(~np.isfinite(values)) or np.any(values < 0.0):
            raise ValueError(
                f"Daily energy for '{name}' must be finite and "
                "non-negative.")
        if np.any(~np.isfinite(capacities)) or np.any(capacities < 0.0):
            raise ValueError(
                f"Installed capacity for '{name}' must be finite and "
                "non-negative.")

        limit_factor = _capacity_limit_factor(user_input, name)
        unit_factor = energy_from_mwh_factor(
            user_input.get("energy_unit", "MWh"))
        accepted_capacities = capacities * limit_factor
        if name in dispatchable or name in load_following:
            limits = accepted_capacities * HOURS_PER_DAY * unit_factor
        else:
            profiles = _source_daily_profiles(
                daily, name, source_data, day_indices, root_dir,
                pattern_cache, user_input)
            maximum_weight = profiles.max(axis=1)
            limits = np.divide(
                accepted_capacities * unit_factor,
                maximum_weight,
                out=np.zeros_like(accepted_capacities),
                where=maximum_weight > 1e-12,)

        adjusted = np.minimum(values, limits)
        _report_capacity_clipping(
            name,
            values,
            limits,
            daily["Date"],
            "daily",
            user_input,)
        daily[name] = adjusted

    available = [
        name for name in source_input if name in daily.columns]
    if available:
        daily["Total"] = daily[available].sum(axis=1)
    return daily


def _daily_hourly_index(daily):
    """Return hourly timestamps and calendar-day indices for daily data."""

    dates = daily["Date"].to_numpy(dtype="datetime64[ns]")
    hourly_dates = np.repeat(dates, HOURS_PER_DAY)
    hourly_dates += np.tile(
        np.arange(HOURS_PER_DAY, dtype="timedelta64[h]"), len(daily))
    return hourly_dates, _day_indices(daily["Date"])


def _expand_daily_sources(
        daily, source_input, day_indices, root_dir, pattern_cache,
        dispatchable, load_following, user_input, dynamic_sources=()):
    """Expand configured daily sources into day-by-hour matrices."""

    dynamic_set = set(dynamic_sources)
    source_arrays = {}
    for name, source_data in source_input.items():
        if name not in daily.columns:
            continue
        operation = str(
            source_data.get("hourly_operation", "")
            if isinstance(source_data, dict) else ""
        ).strip().lower()
        if name in dynamic_set and operation == "must_run":
            source_arrays[name] = np.zeros(
                (len(daily), HOURS_PER_DAY), dtype=np.float32)
            continue
        limit_factor = _capacity_limit_factor(user_input, name)
        source_arrays[name] = _expand_source(
            daily, name, source_data, day_indices, root_dir,
            pattern_cache, dispatchable, load_following, limit_factor,
            user_input)
    return source_arrays


def _expand_daily_demand(
        daily, day_indices, root_dir, pattern_cache, user_input):
    """Expand daily demand to hourly values when demand is available."""

    if "Demand" not in daily.columns:
        return None
    demand_input = user_input.get("demand", {})
    pattern_file = demand_input.get("hourly_patterns")
    profiles = _profiles_for_daily_source(
        daily, "Demand", day_indices, pattern_file, root_dir,
        pattern_cache, user_input)
    daily_values = daily["Demand"].to_numpy(
        dtype=np.float32, copy=False)
    return daily_values[:, None] * profiles


def _hourly_output_data(
        hourly_dates, source_input, source_arrays, demand=None, stages=None):
    """Flatten source, demand, and dispatch arrays into output columns."""

    data = {"Date": hourly_dates}
    for name in source_input:
        values = source_arrays.get(name)
        if values is not None:
            data[name] = values.reshape(-1)
    if demand is not None:
        data["Demand"] = demand.reshape(-1)
    if stages is not None:
        for column, values in stages.items():
            data[column] = values.reshape(-1)
    return data


def expand_to_hourly(daily_df, user_input, root_dir):
    """Expand daily values to hourly values while preserving daily energy."""
    hourly_input = user_input.get("hourly_simulation", {})
    if not hourly_input.get("enabled", False):
        return daily_df.copy()

    daily = _prepare_daily_data(daily_df)
    source_input, profile_preserving, load_following, dispatchable = (
        _operation_groups(user_input))
    pattern_cache = {}

    hourly_dates, day_indices = _daily_hourly_index(daily)
    source_arrays = _expand_daily_sources(
        daily, source_input, day_indices, root_dir, pattern_cache,
        dispatchable, load_following, user_input)
    demand = _expand_daily_demand(
        daily, day_indices, root_dir, pattern_cache, user_input)

    dispatch_stages = None
    if demand is not None:
        dispatch_stages = _dispatch_arrays(
            source_arrays,
            demand,
            daily,
            profile_preserving,
            load_following,
            dispatchable,
            user_input,)

    data = _hourly_output_data(
        hourly_dates, source_input, source_arrays, demand, dispatch_stages)

    excluded = set(source_input)
    excluded.update({
        "Date", "Demand", "Total", MC_PROFILE_DATE_COLUMN
    })
    for column in daily.columns:
        if column in excluded:
            continue
        data[column] = np.repeat(
            daily[column].to_numpy(copy=False), HOURS_PER_DAY)

    hourly = pd.DataFrame(data, copy=False)
    numeric = hourly.select_dtypes(include=[np.number]).columns
    hourly[numeric] = hourly[numeric].astype(np.float32, copy=False)
    available = [name for name in source_input if name in hourly.columns]
    if available:
        hourly["Total"] = hourly[available].sum(axis=1)
    return hourly


def _source_daily_profiles(
        daily, name, source_data, day_indices, root_dir, pattern_cache,
        user_input):
    """Return configured hourly profiles for one daily source."""

    pattern_file = (
        source_data.get("hourly_patterns")
        if isinstance(source_data, dict) else None)
    if (
            isinstance(source_data, dict)
            and isinstance(source_data.get("capacity_additions"), dict)
    ):
        return _profiles_for_days(
            name, day_indices, pattern_file, root_dir, pattern_cache)
    return _profiles_for_daily_source(
        daily, name, day_indices, pattern_file, root_dir, pattern_cache,
        user_input)


def _expand_source(
        daily, name, source_data, day_indices, root_dir, pattern_cache,
        dispatchable, load_following, limit_factor, user_input):
    """Expand one source from daily energy to its hourly representation.

    Dispatchable and load-following sources are left at zero here and
    are allocated later.
    Profile-preserving sources are not dispatched against residual demand.
    Instead,
    they use the coherent historical hourly profile selected by the Monte
    Carlo bootstrap, exactly like other profile-based sources.
    """

    if name in dispatchable or name in load_following:
        return np.zeros(
            (len(daily), HOURS_PER_DAY), dtype=np.float32)

    profiles = _source_daily_profiles(
        daily, name, source_data, day_indices, root_dir, pattern_cache,
        user_input)
    daily_values = daily[name].to_numpy(
        dtype=np.float64, copy=False)
    return _expand_profiled_source(
        daily, name, daily_values, profiles, limit_factor, user_input)


def _expand_profiled_source(
        daily, name, daily_values, profiles, limit_factor, user_input):
    """Apply the hourly profile without changing its within-day shape."""

    if np.any(~np.isfinite(daily_values)):
        raise ValueError(
            f"Daily energy for '{name}' contains invalid values.")
    if np.any(daily_values < 0.0):
        raise ValueError(
            f"Daily energy for '{name}' cannot be negative.")

    expanded = daily_values[:, None] * profiles
    capacity_name = f"Installed_Capacity_{name}"

    if capacity_name not in daily.columns:
        return expanded.astype(np.float32)

    capacities = pd.to_numeric(
        daily[capacity_name], errors="coerce"
    ).to_numpy(dtype=np.float64)

    if np.any(~np.isfinite(capacities)):
        raise ValueError(
            f"Installed capacity for '{name}' contains invalid values.")
    if np.any(capacities < 0.0):
        raise ValueError(
            f"Installed capacity for '{name}' cannot be negative.")

    unit_factor = energy_from_mwh_factor(user_input.get("energy_unit", "MWh"))
    accepted_limits = capacities * limit_factor * unit_factor
    maximum = expanded.max(axis=1)
    numerical_tolerance = np.maximum(1e-6, accepted_limits * 1e-7)
    invalid = maximum > accepted_limits + numerical_tolerance

    if invalid.any():
        position = int(np.flatnonzero(invalid)[0])
        date_text = pd.Timestamp(
            daily.iloc[position]["Date"]
        ).date().isoformat()
        raise ValueError(
            f"Daily energy for '{name}' on {date_text} exceeds the "
            "profile-compatible capacity limit. Run "
            "enforce_daily_capacity_feasibility() before hourly "
            "expansion.")

    expanded = np.minimum(expanded, accepted_limits[:, None])
    return expanded.astype(np.float32)


def _dispatch_dispatchable_day(
        source_arrays, dispatchable, daily_row, demand_row, generation,
        day_position, user_input):
    """Allocate dispatchable daily energy against one day's residual need."""

    result = generation.copy()
    for name in dispatchable:
        values = source_arrays.get(name)
        if values is None:
            continue
        daily_energy = float(daily_row.get(name, 0.0))
        hourly_limit = _hourly_limit(
            daily_row, name, user_input,
            _capacity_limit_factor(user_input, name))
        deficit = np.maximum(demand_row - result, 0.0)
        dispatched = _allocate_energy_array(
            deficit, daily_energy, hourly_limit)
        values[day_position] = dispatched
        result += dispatched
    return result


def _balance_stage_arrays(
        after_profiled, after_profile_preserving, after_load_following,
        after_dispatchable):
    """Return the standard hourly balance-stage output mapping."""

    return {
        "Balance_Before_Profile_Preserving":
            after_profiled.astype(np.float32),
        "Balance_After_Profile_Preserving":
            after_profile_preserving.astype(np.float32),
        "Balance_After_Load_Following":
            after_load_following.astype(np.float32),
        "Balance_After_Dispatchable":
            after_dispatchable.astype(np.float32)}


def _dispatch_arrays(
        source_arrays, demand, daily, profile_preserving, load_following,
        dispatchable, user_input):
    """Apply profiled, load-following, and dispatchable operating stages.

    Profile-preserving sources retain their sampled hourly profiles.
    Load-following
    sources are then dispatched against residual demand according to their
    configured energy policy. Dispatchable sources are allocated last against
    any remaining hourly need.
    """

    source_order = [name for name in source_arrays]

    base_generation = np.zeros_like(demand, dtype=np.float64)
    excluded = set(profile_preserving) | set(load_following) | set(dispatchable)
    for name in source_order:
        if name in excluded:
            continue
        base_generation += source_arrays[name]

    after_profiled = base_generation - demand
    generation = base_generation.copy()

    for name in profile_preserving:
        values = source_arrays.get(name)
        if values is not None:
            generation += values

    after_profile_preserving = generation - demand

    for name in load_following:
        values = source_arrays.get(name)
        if values is None:
            continue
        dispatched = _dispatch_load_following_source(
            name,
            demand,
            generation,
            daily,
            user_input,)
        values[:] = dispatched
        generation += dispatched

    after_load_following = generation - demand
    after_dispatchable = np.zeros_like(demand, dtype=np.float64)

    for day_pos in range(len(daily)):
        daily_row = daily.iloc[day_pos]
        day_generation = _dispatch_dispatchable_day(
            source_arrays, dispatchable, daily_row, demand[day_pos],
            generation[day_pos], day_pos, user_input)
        after_dispatchable[day_pos] = day_generation - demand[day_pos]

    return _balance_stage_arrays(
        after_profiled, after_profile_preserving, after_load_following,
        after_dispatchable)


def _prepare_daily_data(daily_df):
    """Validate and normalize a daily forecast before hourly expansion."""

    if "Date" not in daily_df.columns:
        raise ValueError("Forecast must contain a 'Date' column.")
    prepared = daily_df.copy()
    prepared["Date"] = pd.to_datetime(prepared["Date"], errors="coerce")
    if prepared["Date"].isna().any():
        raise ValueError("Forecast contains invalid dates.")
    return prepared.sort_values("Date").reset_index(drop=True)


def _day_indices(dates):
    """Map dates to zero-based positions on a common non-leap year."""

    months = dates.dt.month.to_numpy(dtype=int)
    days = dates.dt.day.to_numpy(dtype=int)
    days = np.where((months == 2) & (days == 29), 28, days)
    references = pd.to_datetime({
        "year": np.full(len(dates), 2021, dtype=int),
        "month": months,
        "day": days,
    })
    return references.dt.dayofyear.to_numpy(dtype=int) - 1


def _profiles_for_daily_source(
        daily, column, day_indices, pattern_file, root_dir, pattern_cache,
        user_input):
    """Return sampled historical profiles when available for one MC run."""

    fallback = _profiles_for_days(
        column, day_indices, pattern_file, root_dir, pattern_cache)
    if user_input is None:
        return fallback
    if MC_PROFILE_DATE_COLUMN not in daily.columns:
        return fallback
    if user_input.get("external_hourly_profile_file"):
        return fallback

    monte_carlo = user_input.get("monte_carlo", {}) or {}
    requested = monte_carlo.get("sources", [])
    requested_keys = {name_key(value) for value in requested}
    if name_key(column) not in requested_keys:
        return fallback

    sampled_dates = pd.to_datetime(
        daily[MC_PROFILE_DATE_COLUMN], errors="coerce")
    if sampled_dates.isna().any():
        raise ValueError(
            "Monte Carlo profile dates contain invalid timestamps.")

    return _sampled_historical_profiles(
        column, sampled_dates, fallback, user_input, root_dir)


def historical_daily_profile_table(
        frame, date_column, column):
    """Return normalized 24-hour profiles indexed by historical day."""

    lookup = build_name_lookup(
        frame.columns, "historical hourly Monte Carlo profiles")
    actual_column = lookup.get(name_key(column))
    if actual_column is None:
        raise ValueError(
            f"Historical hourly profile source '{column}' was not found.")

    dates = pd.DatetimeIndex(frame[date_column])
    values = pd.to_numeric(frame[actual_column], errors="coerce")
    table = pd.DataFrame({
        "day": dates.normalize(),
        "hour": dates.hour,
        "value": values.to_numpy(dtype=float),
    })
    matrix = table.pivot_table(
        index="day", columns="hour", values="value", aggfunc="mean")
    matrix = matrix.reindex(columns=range(HOURS_PER_DAY))
    totals = matrix.sum(axis=1, min_count=HOURS_PER_DAY)
    valid = (
        matrix.notna().all(axis=1)
        & np.isfinite(totals)
        & (totals > 1e-12))
    matrix = matrix.loc[valid].div(totals.loc[valid], axis=0)
    return matrix.astype(np.float32)


def _select_historical_profiles(matrix, sampled_dates, fallback):
    """Select historical daily shapes, preserving fallback where invalid."""

    result = np.asarray(fallback, dtype=np.float32).copy()
    if not isinstance(matrix, pd.DataFrame) or matrix.empty:
        return result

    selected = pd.DatetimeIndex(sampled_dates).normalize()
    sampled = matrix.reindex(selected)
    valid_rows = sampled.notna().all(axis=1).to_numpy()
    if valid_rows.any():
        sampled_values = sampled.to_numpy(dtype=np.float32, copy=False)
        result[valid_rows] = sampled_values[valid_rows]
    return result


def sample_historical_daily_profiles(
        frame, date_column, column, sampled_dates, fallback):
    """Use selected historical days, retaining fallback for invalid days."""

    matrix = historical_daily_profile_table(
        frame, date_column, column)
    return _select_historical_profiles(matrix, sampled_dates, fallback)


def _sampled_historical_profiles(
        column, sampled_dates, fallback, user_input, root_dir):
    """Return normalized hourly shapes for bootstrap-selected history days.

    The Monte Carlo residual and the hourly profile are taken from the same
    historical day. If the configured historical data are not hourly, the
    deterministic calendar profile is retained. Data-loading errors are not
    hidden because silently reverting to the fallback would change the
    stochastic method without informing the caller.
    """

    data_name = str(user_input.get("historical_data_file", ""))
    data_path = _resolve_path(root_dir, data_name).resolve()
    dataset_key = str(data_path)
    profile_key = dataset_key, name_key(column)

    if profile_key not in _HISTORICAL_PROFILE_CACHE:
        dataset = _HISTORICAL_DATASET_CACHE.get(dataset_key)
        if dataset is None:
            try:
                dataset = load_historical_dataset(
                    user_input, context="hourly Monte Carlo profiles")
            except (FileNotFoundError, TypeError, ValueError) as exc:
                raise RuntimeError(
                    "Cannot load historical data for Monte Carlo hourly "
                    f"profiles: {data_path}") from exc
            _HISTORICAL_DATASET_CACHE[dataset_key] = dataset

        if dataset.input_resolution != "hourly":
            return fallback
        frame = dataset.raw
        matrix = historical_daily_profile_table(
            frame, dataset.date_column, column)
        _HISTORICAL_PROFILE_CACHE[profile_key] = matrix

    matrix = _HISTORICAL_PROFILE_CACHE.get(profile_key)
    return _select_historical_profiles(matrix, sampled_dates, fallback)


def _profiles_for_days(column, day_indices, pattern_file, root_dir,
                       pattern_cache):
    """Return deterministic hourly profiles for the requested calendar days."""

    if not pattern_file:
        return np.broadcast_to(
            _UNIFORM_PROFILE, (len(day_indices), HOURS_PER_DAY))

    pattern_path = _resolve_path(root_dir, pattern_file)
    if pattern_path not in pattern_cache:
        pattern_cache[pattern_path] = _load_pattern_file(pattern_path)

    pattern_data = pattern_cache[pattern_path]
    lookup = build_name_lookup(
        pattern_data.columns, f"hourly pattern '{pattern_path}'")
    actual_column = lookup.get(name_key(column))
    if actual_column is None:
        raise ValueError(
            f"Hourly pattern column '{column}' was not found in "
            f"{pattern_path}")

    values = pd.to_numeric(
        pattern_data[actual_column], errors="coerce"
    ).to_numpy(dtype=np.float32)
    matrix = values.reshape(365, HOURS_PER_DAY)
    profiles = matrix[day_indices].copy()
    if not np.isfinite(profiles).all():
        raise ValueError(
            f"Hourly pattern for '{column}' contains invalid values.")
    if (profiles < 0).any():
        raise ValueError(
            f"Hourly pattern for '{column}' contains negative values.")

    totals = profiles.sum(axis=1)
    if (totals <= 0).any():
        raise ValueError(
            f"Hourly pattern for '{column}' has a zero daily sum.")
    return profiles / totals[:, None]


def _sources_by_operation(source_input, operation):
    """Return sources in one hourly operating mode, sorted by priority."""

    selected = []
    for name, source_data in source_input.items():
        if not isinstance(source_data, dict):
            continue
        if source_data.get("hourly_operation") != operation:
            continue
        selected.append((source_data.get("dispatch_priority", 999), name))
    selected.sort()
    return [name for _, name in selected]


def _operation_groups(user_input):
    """Return source configuration and ordered hourly operation groups."""

    source_input = user_input.get("sources", {})
    return (
        source_input,
        _sources_by_operation(source_input, "profile_preserving"),
        _sources_by_operation(source_input, "load_following"),
        _sources_by_operation(source_input, "dispatchable"),)


def _validate_causal_load_following(user_input, load_following):
    """Require residual-following control for causal EFPD operation."""

    for name in load_following:
        policy = _load_following_settings(
            user_input, name)["energy_policy"]
        if policy != "follow_residual":
            raise ValueError(
                "EFPD-based causal refuelling requires load-following "
                "energy_policy: follow_residual for every load-following "
                f"source; '{name}' uses {policy!r}.")


def _hourly_limit(
        daily_row, source, user_input, limit_factor=1.0):
    """Return the accepted hourly production limit in the energy unit."""

    capacity_column = f"Installed_Capacity_{source}"
    hourly_power_limit = float(daily_row.get(capacity_column, np.inf))
    if hourly_power_limit < 0:
        raise ValueError(
            f"Installed capacity for '{source}' cannot be negative.")
    unit_factor = energy_from_mwh_factor(user_input.get("energy_unit", "MWh"))
    return hourly_power_limit * float(limit_factor) * unit_factor


def _must_run_power_fraction(source, source_data):
    """Return available-power fraction for refuelling-aware must-run."""

    settings = source_data.get("must_run", {}) or {}
    if "power_fraction" not in settings:
        raise ValueError(
            f"Must-run source '{source}' requires an explicit "
            "must_run.power_fraction.")
    fraction = float(settings["power_fraction"])
    if not np.isfinite(fraction) or fraction <= 0.0 or fraction > 1.0:
        raise ValueError(
            f"Must-run power fraction for '{source}' must be in (0, 1].")
    return fraction


def _capacity_limit_factor(user_input, source):
    """Return a validated capacity-tolerance multiplier for one source."""

    simulation = user_input.get("simulation", {}) or {}
    sources = user_input.get("sources", {}) or {}
    source_input = sources.get(source, {}) or {}
    raw_percent = source_input.get(
        "capacity_tolerance",
        simulation.get("capacity_tolerance", 0.0),)
    try:
        percent = float(raw_percent)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            f"Invalid capacity tolerance for '{source}': "
            f"{raw_percent}.") from exc
    if not np.isfinite(percent) or not 0.0 <= percent <= 100.0:
        raise ValueError(
            f"Capacity tolerance for '{source}' must be between "
            "0 and 100 percent.")
    return 1.0 + percent / 100.0


def _report_capacity_clipping(
        source, values, limits, dates, resolution, user_input):
    """Record local capacity clipping without flooding the console."""

    values = np.asarray(values, dtype=float)
    limits = np.asarray(limits, dtype=float)
    numerical = np.maximum(1e-6, np.abs(limits) * 1e-7)
    excess = np.maximum(values - limits, 0.0)
    clipped = excess > numerical
    if not clipped.any():
        return

    positions = np.flatnonzero(clipped)
    ratios = np.full(values.shape, np.inf, dtype=float)
    positive = limits > numerical
    ratios[positive] = excess[positive] / limits[positive]
    maximum_position = int(positions[np.argmax(ratios[positions])])
    maximum_percent = 100.0 * ratios[maximum_position]
    maximum_date = pd.Timestamp(dates.iloc[maximum_position])
    total = float(excess[clipped].sum())
    count = int(clipped.sum())
    date_text = maximum_date.isoformat()
    energy_unit = str(user_input.get("energy_unit", "energy units"))

    diagnostics = user_input.get("_runtime_diagnostics")
    if isinstance(diagnostics, list):
        diagnostics.append({
            "Type": "capacity_clipping",
            "Source": str(source),
            "Resolution": str(resolution),
            "Count": count,
            "Energy_Removed": total,
            "Energy_Unit": energy_unit,
            "Max_Excess_Percent": float(maximum_percent),
            "Max_Date": date_text,})

    emit(
        user_input,
        f"WARNING: Capacity clipping for '{source}': {count} "
        f"{resolution} steps, {total:.6f} {energy_unit} removed; maximum "
        f"excess={maximum_percent:.3f}% at {date_text}. Energy was not "
        "transferred to another time step.",
        "detailed",
        allow_worker=True)


def _allocate_energy_array(deficit, available_energy, hourly_limit):
    """Allocate daily energy preferentially to hourly deficits."""

    values = np.asarray(deficit, dtype=float).copy()
    dispatch = np.zeros(HOURS_PER_DAY, dtype=float)
    remaining = min(float(available_energy), float(values.sum()))
    if available_energy < 0:
        raise ValueError("Daily available energy cannot be negative.")
    if remaining <= 0 or hourly_limit <= 0:
        return dispatch

    open_hours = values > 1e-9
    while remaining > 1e-9 and open_hours.any():
        weights = values[open_hours]
        weights = weights / weights.sum()
        allocation = remaining * weights
        room = hourly_limit - dispatch[open_hours]
        allocation = np.minimum(allocation, room)
        allocation = np.minimum(allocation, values[open_hours])
        used = float(allocation.sum())
        if used <= 1e-9:
            break
        dispatch[open_hours] += allocation
        values[open_hours] -= allocation
        remaining -= used
        open_hours = (
            (values > 1e-9)
            & (dispatch < hourly_limit - 1e-9))
    return dispatch




def _load_following_settings(user_input, source):
    """Return normalized operating parameters for one load-following source."""

    source_data = user_input.get("sources", {}).get(source, {}) or {}
    settings = source_data.get("load_following", {}) or {}
    required = (
        "energy_policy", "control_mode", "minimum_power_fraction",
        "maximum_power_fraction")
    missing = [name for name in required if name not in settings]
    if missing:
        raise ValueError(
            f"Load-following source '{source}' is missing explicit settings: "
            + ", ".join(missing) + ".")

    control_mode = str(settings["control_mode"]).strip().lower()
    output = {
        "energy_policy": str(settings["energy_policy"]).strip().lower(),
        "control_mode": control_mode,
        "minimum_power_fraction": float(
            settings["minimum_power_fraction"]),
        "maximum_power_fraction": float(
            settings["maximum_power_fraction"])}
    if control_mode == "constrained_hourly":
        constrained = (
            "ramp_up_rate", "ramp_down_rate",
            "deep_reduction_threshold_fraction",
            "deep_reduction_request_duration", "max_deep_reduction_cycles")
        missing = [name for name in constrained if name not in settings]
        if missing:
            raise ValueError(
                f"Load-following source '{source}' is missing constrained "
                "settings: " + ", ".join(missing) + ".")
        output.update({
            "ramp_up_rate": float(settings["ramp_up_rate"]),
            "ramp_down_rate": float(settings["ramp_down_rate"]),
            "deep_reduction_threshold_fraction": float(
                settings["deep_reduction_threshold_fraction"]),
            "deep_reduction_request_duration": int(
                settings["deep_reduction_request_duration"]),
            "max_deep_reduction_cycles": int(
                settings["max_deep_reduction_cycles"])})
    return output


def _dispatch_load_following_source(
        source, demand, generation, daily, user_input):
    """Dispatch one source against residual demand within power bounds."""

    settings = _load_following_settings(user_input, source)
    capacity_column = _load_following_capacity_column(daily, source)
    power_capacities = pd.to_numeric(
        daily[capacity_column], errors="coerce"
    ).to_numpy(dtype=np.float64)
    if np.any(~np.isfinite(power_capacities)) or np.any(power_capacities < 0.0):
        raise ValueError(
            f"Load-following capacity for '{source}' is invalid.")

    unit_factor = energy_from_mwh_factor(user_input.get("energy_unit", "MWh"))
    capacities = power_capacities * unit_factor
    minimum_fraction = settings["minimum_power_fraction"]
    maximum_fraction = settings["maximum_power_fraction"]
    lower = capacities[:, None] * minimum_fraction
    upper = capacities[:, None] * maximum_fraction
    residual = demand.astype(np.float64) - generation.astype(np.float64)

    if settings["energy_policy"] == "follow_residual":
        if settings["control_mode"] == "direct":
            output = np.clip(residual, lower, upper)
            return output.astype(np.float32)
        if settings["control_mode"] == "constrained_hourly":
            output = _dispatch_constrained_load_following(
                residual, capacities, lower, upper, daily, settings)
            return output.astype(np.float32)
        raise ValueError(
            f"Unsupported load-following control mode for '{source}': "
            f"{settings['control_mode']}.")
    if settings["energy_policy"] != "preserve_annual":
        raise ValueError(
            f"Unsupported load-following energy policy for '{source}': "
            f"{settings['energy_policy']}.")

    dates = pd.to_datetime(daily["Date"])
    years = dates.dt.year.to_numpy(dtype=int)
    output = np.zeros_like(demand, dtype=np.float64)
    for year in np.unique(years):
        day_mask = years == year
        target = pd.to_numeric(
            daily.loc[day_mask, source], errors="coerce"
        ).fillna(0.0).sum()
        dispatched = _allocate_preserved_annual_energy(
            residual[day_mask].reshape(-1),
            float(target),
            lower[day_mask].repeat(HOURS_PER_DAY, axis=1).reshape(-1),
            upper[day_mask].repeat(HOURS_PER_DAY, axis=1).reshape(-1),
            source,
            int(year),)
        output[day_mask] = dispatched.reshape(-1, HOURS_PER_DAY)
    return output.astype(np.float32)


def _dispatch_constrained_load_following(
        residual, capacities, lower, upper, daily, settings):
    """Apply causal hourly load following with simple operating limits."""

    shape = residual.shape
    lower_hourly = np.broadcast_to(lower, shape).reshape(-1)
    upper_hourly = np.broadcast_to(upper, shape).reshape(-1)
    capacity_hourly = np.repeat(capacities, HOURS_PER_DAY)
    raw_target = np.clip(
        residual, lower, upper).reshape(-1).astype(np.float64)
    dates = _hourly_dates_from_daily(daily)

    ramp_up = (
        capacity_hourly * settings["ramp_up_rate"])
    ramp_down = (
        capacity_hourly * settings["ramp_down_rate"])
    deep_threshold = (
        capacity_hourly
        * settings["deep_reduction_threshold_fraction"])
    persistence_required = settings[
        "deep_reduction_request_duration"]
    max_deep_cycles = settings["max_deep_reduction_cycles"]

    output = np.zeros_like(raw_target)
    previous = float(upper_hourly[0]) if len(output) else 0.0
    previous_capacity = (
        float(capacity_hourly[0]) if len(output) else 0.0)
    deep_active = False
    persistence = 0
    cycles_today = 0
    active_day = None

    for position, target in enumerate(raw_target):
        day = dates[position].date()
        if day != active_day:
            active_day = day
            cycles_today = 0

        threshold = deep_threshold[position]
        deep_request = target < threshold - 1e-9
        if deep_request:
            persistence += 1
        else:
            persistence = 0

        permitted_target = float(target)
        if deep_request and not deep_active:
            persistent = persistence >= persistence_required
            cycle_available = cycles_today < max_deep_cycles
            if not persistent or not cycle_available:
                permitted_target = max(permitted_target, threshold)

        current_capacity = float(capacity_hourly[position])
        added_capacity = max(current_capacity - previous_capacity, 0.0)
        ramp_reference = min(
            previous + added_capacity,
            float(upper_hourly[position]),)
        ramp_low = ramp_reference - ramp_down[position]
        ramp_high = ramp_reference + ramp_up[position]
        value = float(np.clip(permitted_target, ramp_low, ramp_high))
        value = float(np.clip(
            value, lower_hourly[position], upper_hourly[position]))

        now_deep = value < threshold - 1e-9
        if now_deep and not deep_active:
            cycles_today += 1
        deep_active = now_deep
        output[position] = value
        previous = value
        previous_capacity = current_capacity

    return output.reshape(shape)


def _hourly_dates_from_daily(daily):
    """Expand daily forecast dates into one timestamp per modeled hour."""

    dates = pd.to_datetime(daily["Date"]).to_numpy(dtype="datetime64[ns]")
    hourly = np.repeat(dates, HOURS_PER_DAY)
    hourly += np.tile(
        np.arange(HOURS_PER_DAY, dtype="timedelta64[h]"), len(dates))
    return pd.to_datetime(hourly)


def _load_following_capacity_column(daily, source):
    """Prefer refuelling-adjusted capacity when a profile is available."""

    available = f"Available_Capacity_{source}"
    installed = f"Installed_Capacity_{source}"
    if available in daily.columns:
        return available
    if installed in daily.columns:
        return installed
    raise ValueError(
        f"Load-following source '{source}' requires {installed}.")


def _allocate_preserved_annual_energy(
        residual, target_energy, lower, upper, source, year):
    """Project residual demand onto power bounds at fixed annual energy.

    The solution has the water-filling form clip(residual + offset, lower,
    upper). The offset is obtained by bisection so that the hourly values sum
    to the prescribed annual energy.
    """

    residual = np.asarray(residual, dtype=np.float64)
    lower = np.asarray(lower, dtype=np.float64)
    upper = np.asarray(upper, dtype=np.float64)

    if residual.shape != lower.shape or lower.shape != upper.shape:
        raise ValueError(
            "Load-following residual and power-bound arrays must match.")
    if np.any(~np.isfinite(residual)):
        raise ValueError(
            f"Residual demand for '{source}' in {year} is not finite.")
    if np.any(~np.isfinite(lower)) or np.any(~np.isfinite(upper)):
        raise ValueError(
            f"Load-following bounds for '{source}' in {year} are invalid.")
    if np.any(lower < 0.0) or np.any(upper < lower):
        raise ValueError(
            f"Load-following bounds for '{source}' in {year} are invalid.")

    minimum_energy = float(lower.sum())
    maximum_energy = float(upper.sum())
    tolerance = max(1e-6, abs(target_energy) * 1e-10)
    if target_energy < minimum_energy - tolerance:
        raise ValueError(
            f"Annual target for '{source}' in {year} is below the minimum "
            "load-following energy permitted by minimum_power_fraction.")
    if target_energy > maximum_energy + tolerance:
        raise ValueError(
            f"Annual target for '{source}' in {year} exceeds the maximum "
            "load-following energy permitted by installed capacity.")

    if maximum_energy <= tolerance:
        return np.zeros_like(residual)

    low = float(np.min(lower - residual)) - 1.0
    high = float(np.max(upper - residual)) + 1.0
    for _ in range(80):
        offset = 0.5 * (low + high)
        trial = np.clip(residual + offset, lower, upper)
        if float(trial.sum()) < target_energy:
            low = offset
        else:
            high = offset

    values = np.clip(
        residual + 0.5 * (low + high), lower, upper)
    difference = float(target_energy - values.sum())
    if abs(difference) > tolerance:
        if difference > 0.0:
            room = upper - values
        else:
            room = values - lower
        available = float(room.sum())
        if available > 0.0:
            values += np.sign(difference) * room * (
                abs(difference) / available)
            values = np.clip(values, lower, upper)

    final_difference = float(target_energy - values.sum())
    if abs(final_difference) > max(tolerance, 1e-4):
        raise RuntimeError(
            f"Could not preserve annual energy for '{source}' in {year}. "
            f"Residual error={final_difference:.6f} energy units.")
    return values


def _load_pattern_file(pattern_path):
    """Load and validate an 8760-row deterministic hourly pattern.

    Internally generated Excel profiles may have a CSV sidecar.  The CSV is
    used only when it is at least as recent as the Excel source, so manually
    edited workbooks cannot be shadowed by a stale cache.
    """

    if not pattern_path.is_file():
        raise FileNotFoundError(
            f"Hourly pattern file not found: {pattern_path}")
    cache_path = pattern_path.with_suffix(".csv")
    use_cache = cache_path.is_file() and (
        cache_path.stat().st_mtime_ns
        >= pattern_path.stat().st_mtime_ns)
    if use_cache:
        pattern_data = pd.read_csv(cache_path)
    else:
        pattern_data = pd.read_excel(pattern_path, engine="openpyxl")
    if len(pattern_data) != PATTERN_ROWS:
        raise ValueError(
            "Hourly pattern file must contain exactly 8760 rows: "
            f"{pattern_path}")
    return pattern_data


def _resolve_path(root_dir, file_name):
    """Resolve a configured file path relative to the project root."""

    path = Path(file_name)
    return path if path.is_absolute() else Path(root_dir) / path


def prepare_operational_data(forecast_df, user_input, root_dir):
    """Prepare a daily or hourly forecast for the operational model."""

    projection = user_input["projection_resolution"]
    simulation = user_input["simulation_resolution"]

    if projection == "daily":
        daily = enforce_daily_capacity_feasibility(
            forecast_df,
            user_input,
            root_dir,)
        if simulation == "daily":
            if dynamic_efpd_sources(user_input):
                raise ValueError(
                    "EFPD-based offline refuelling requires "
                    "simulation_resolution: hourly.")
            return _attach_online_efpd_diagnostics(daily, user_input)
        if dynamic_efpd_sources(user_input):
            hourly = _expand_with_dynamic_efpd(
                daily, user_input, root_dir)
        else:
            hourly = expand_to_hourly(daily, user_input, root_dir)
        return _attach_online_efpd_diagnostics(hourly, user_input)

    return prepare_hourly_forecast(forecast_df, user_input)


def prepare_hourly_forecast(forecast_df, user_input):
    """Validate, constrain and dispatch an hourly forecast directly."""

    if dynamic_efpd_sources(user_input):
        hourly = _prepare_hourly_with_dynamic_efpd(
            forecast_df, user_input)
    else:
        hourly = _prepare_hourly_forecast_once(forecast_df, user_input)
    return _attach_online_efpd_diagnostics(hourly, user_input)


def _prepare_hourly_forecast_once(forecast_df, user_input):
    """Prepare one hourly pass using the currently supplied capacities."""

    hourly = _prepare_hourly_data(forecast_df)
    hourly = _enforce_hourly_capacity_feasibility(
        hourly,
        user_input,)
    hourly = _dispatch_hourly_sources(hourly, user_input)
    available = [
        name for name in user_input.get("sources", {})
        if name in hourly.columns]
    if available:
        hourly["Total"] = hourly[available].sum(axis=1)
    return hourly


def _attach_online_efpd_diagnostics(frame, user_input):
    """Add online-refuelling EFPD diagnostics to operational output."""

    sources = online_efpd_sources(user_input)
    if not sources:
        return frame
    output, _ = apply_efpd_refueling_profiles(
        frame, user_input, sources)
    return output


def _initialize_dynamic_efpd_columns(frame, user_input):
    """Start dynamic EFPD iteration with all installed units available."""

    output = frame.copy()
    for source in dynamic_efpd_sources(user_input):
        installed = f"Installed_Capacity_{source}"
        if installed not in output.columns:
            raise ValueError(
                f"EFPD refuelling for '{source}' requires {installed}.")
        output[f"Refueling_Units_{source}"] = 0
        output[f"Refueling_Capacity_{source}"] = 0.0
        output[f"Available_Capacity_{source}"] = output[installed]
        output[f"Mean_Accumulated_EFPD_{source}"] = 0.0
    return output


def _dynamic_profile_columns(user_input):
    """Return operational columns updated by EFPD iteration."""

    columns = []
    for source in dynamic_efpd_sources(user_input):
        columns.extend([
            f"Refueling_Units_{source}",
            f"Refueling_Capacity_{source}",
            f"Available_Capacity_{source}",
            f"Mean_Accumulated_EFPD_{source}",])
    return columns

def _causal_operation_groups(user_input):
    """Return operation groups and validate causal load following."""

    groups = _operation_groups(user_input)
    _validate_causal_load_following(user_input, groups[2])
    return *groups, dynamic_efpd_sources(user_input)


def _expand_with_dynamic_efpd(daily, user_input, root_dir):
    """Expand and operate EFPD-refuelled fleets causally.

    EFPD outages depend on generation already produced. A fixed-point
    iteration can oscillate because moving an outage changes generation and
    therefore moves the next outage again. The correct model is causal: the
    fleet operates with today's available capacity, today's generation adds
    EFPD, and a newly triggered outage starts on the next modeled day.
    """

    # ``prepare_operational_data`` has already applied daily capacity
    # feasibility before entering this causal expansion.  Repeating it here
    # duplicated the same work for every EFPD simulation.
    daily = _initialize_dynamic_efpd_columns(
        _prepare_daily_data(daily), user_input)
    source_input, _, load_following, dispatchable, dynamic_sources = (
        _causal_operation_groups(user_input))

    pattern_cache = {}
    hourly_dates, day_indices = _daily_hourly_index(daily)
    source_arrays = _expand_daily_sources(
        daily, source_input, day_indices, root_dir, pattern_cache,
        dispatchable, load_following, user_input, dynamic_sources)
    demand = _expand_daily_demand(
        daily, day_indices, root_dir, pattern_cache, user_input)
    if demand is None:
        raise ValueError(
            "EFPD-based operational refuelling requires Demand.")

    stages, profile_state = _run_causal_efpd_dispatch(
        source_arrays, demand, daily, user_input)

    data = _hourly_output_data(
        hourly_dates, source_input, source_arrays, demand, stages)

    excluded_columns = set(source_input)
    excluded_columns.update({
        "Date", "Demand", "Total", MC_PROFILE_DATE_COLUMN})
    dynamic_columns = set(_dynamic_profile_columns(user_input))
    excluded_columns.update(dynamic_columns)
    for column in daily.columns:
        if column in excluded_columns:
            continue
        data[column] = np.repeat(
            daily[column].to_numpy(copy=False), HOURS_PER_DAY)

    for source, state in profile_state.items():
        data[f"Refueling_Units_{source}"] = np.repeat(
            state["refueling_units"], HOURS_PER_DAY)
        data[f"Refueling_Capacity_{source}"] = np.repeat(
            state["refueling_power_capacity"], HOURS_PER_DAY)
        data[f"Available_Capacity_{source}"] = np.repeat(
            state["available_power_capacity"], HOURS_PER_DAY)
        data[f"Mean_Accumulated_EFPD_{source}"] = np.repeat(
            state["mean_lifetime_efpd"], HOURS_PER_DAY)

    hourly = pd.DataFrame(data, copy=False)
    numeric = hourly.select_dtypes(include=[np.number]).columns
    hourly[numeric] = hourly[numeric].astype(np.float32, copy=False)
    available = [
        name for name in source_input if name in hourly.columns]
    if available:
        hourly["Total"] = hourly[available].sum(axis=1)
    event_rows = [
        row for state in profile_state.values()
        for row in state.get("event_rows", [])]
    hourly.attrs["operational_refueling_schedule"] = event_rows
    return hourly


def _run_causal_efpd_dispatch(
        source_arrays, demand, daily, user_input):
    """Operate prepared daily source arrays with causal EFPD state."""

    source_input, profile_preserving, load_following, dispatchable = (
        _operation_groups(user_input))
    dynamic_sources = dynamic_efpd_sources(user_input)
    dynamic_set = set(dynamic_sources)

    profile_state = {
        source: initialize_efpd_fleet_state(source, daily, user_input)
        for source in dynamic_sources}
    load_states = {
        source: _new_load_following_state() for source in load_following}
    load_settings = {
        source: _load_following_settings(user_input, source)
        for source in load_following}
    load_unit_factor = energy_from_mwh_factor(
        user_input.get("energy_unit", "MWh"))

    after_profiled = np.zeros_like(demand, dtype=np.float64)
    after_profile_preserving = np.zeros_like(demand, dtype=np.float64)
    after_load_following = np.zeros_like(demand, dtype=np.float64)
    after_dispatchable = np.zeros_like(demand, dtype=np.float64)

    excluded = set(profile_preserving) | set(load_following) | set(dispatchable)
    base_names = [
        name for name in source_arrays
        if name not in excluded and name not in dynamic_set]

    for day_pos in range(len(daily)):
        daily_row = daily.iloc[day_pos]
        generation = np.zeros(HOURS_PER_DAY, dtype=np.float64)
        for name in base_names:
            generation += source_arrays[name][day_pos]

        for name in dynamic_sources:
            source_data = source_input.get(name, {}) or {}
            operation = str(
                source_data.get("hourly_operation", "")
            ).strip().lower()
            availability = begin_efpd_day(
                profile_state[name], daily_row, name)
            if operation == "must_run":
                fraction = _must_run_power_fraction(name, source_data)
                unit_factor = energy_from_mwh_factor(
                    user_input.get("energy_unit", "MWh"))
                values = np.full(
                    HOURS_PER_DAY,
                    availability["available_capacity"] * unit_factor * fraction,
                    dtype=np.float32)
                source_arrays[name][day_pos] = values
                generation += values

        after_profiled[day_pos] = generation - demand[day_pos]

        for name in profile_preserving:
            values = source_arrays.get(name)
            if values is not None:
                generation += values[day_pos]
        after_profile_preserving[day_pos] = generation - demand[day_pos]

        for name in load_following:
            values = source_arrays.get(name)
            if values is None:
                continue
            if name in dynamic_set:
                power_capacity = profile_state[name]["today_available_capacity"]
            else:
                capacity_column = _load_following_capacity_column(
                    daily, name)
                power_capacity = float(daily_row[capacity_column])
            dispatched = _dispatch_load_following_day(
                name, demand[day_pos], generation, power_capacity, user_input,
                load_states[name], settings=load_settings[name],
                unit_factor=load_unit_factor)
            values[day_pos] = dispatched
            generation += dispatched
        after_load_following[day_pos] = generation - demand[day_pos]

        day_generation = _dispatch_dispatchable_day(
            source_arrays, dispatchable, daily_row, demand[day_pos],
            generation, day_pos, user_input)
        after_dispatchable[day_pos] = day_generation - demand[day_pos]

        for name in dynamic_sources:
            values = source_arrays.get(name)
            energy = (
                float(values[day_pos].sum())
                if values is not None else 0.0)
            finish_efpd_day(
                profile_state[name], energy, user_input)

    stages = _balance_stage_arrays(
        after_profiled, after_profile_preserving, after_load_following,
        after_dispatchable)
    return stages, profile_state


def _new_load_following_state():
    """Return persistent state for causal constrained load following."""

    return {
        "previous": None,
        "previous_capacity": None,
        "deep_active": False,
        "persistence": 0,}


def _dispatch_load_following_day(
        source, demand, generation, power_capacity, user_input, state,
        settings=None, unit_factor=None):
    """Dispatch one load-following source for one chronological day."""

    if settings is None:
        settings = _load_following_settings(user_input, source)
    if settings["energy_policy"] != "follow_residual":
        raise ValueError(
            f"Causal load following for '{source}' requires "
            "energy_policy: follow_residual.")
    if unit_factor is None:
        unit_factor = energy_from_mwh_factor(
            user_input.get("energy_unit", "MWh"))
    capacity = float(power_capacity) * unit_factor
    lower = capacity * settings["minimum_power_fraction"]
    upper = capacity * settings["maximum_power_fraction"]
    residual = np.asarray(demand, dtype=float) - np.asarray(
        generation, dtype=float)
    raw_target = np.clip(residual, lower, upper)

    if settings["control_mode"] == "direct":
        if len(raw_target):
            state["previous"] = float(raw_target[-1])
            state["previous_capacity"] = capacity
        state["deep_active"] = False
        state["persistence"] = 0
        return raw_target.astype(np.float32)
    if settings["control_mode"] != "constrained_hourly":
        raise ValueError(
            f"Unsupported load-following control mode for '{source}': "
            f"{settings['control_mode']}.")

    ramp_up = capacity * settings["ramp_up_rate"]
    ramp_down = capacity * settings["ramp_down_rate"]
    threshold = (
        capacity * settings["deep_reduction_threshold_fraction"])
    persistence_required = settings[
        "deep_reduction_request_duration"]
    max_deep_cycles = settings["max_deep_reduction_cycles"]
    output = np.zeros(HOURS_PER_DAY, dtype=np.float64)
    if state["previous"] is None:
        state["previous"] = float(upper)
        state["previous_capacity"] = capacity
    cycles_today = 0

    for position, target in enumerate(raw_target):
        deep_request = target < threshold - 1e-9
        if deep_request:
            state["persistence"] += 1
        else:
            state["persistence"] = 0

        permitted_target = float(target)
        if deep_request and not state["deep_active"]:
            persistent = (
                state["persistence"] >= persistence_required)
            cycle_available = cycles_today < max_deep_cycles
            if not persistent or not cycle_available:
                permitted_target = max(permitted_target, threshold)

        previous_capacity = float(state["previous_capacity"] or 0.0)
        added_capacity = max(capacity - previous_capacity, 0.0)
        ramp_reference = min(
            float(state["previous"]) + added_capacity, float(upper))
        ramp_low = ramp_reference - ramp_down
        ramp_high = ramp_reference + ramp_up
        value = min(max(permitted_target, ramp_low), ramp_high)
        value = min(max(value, lower), upper)

        now_deep = value < threshold - 1e-9
        if now_deep and not state["deep_active"]:
            cycles_today += 1
        state["deep_active"] = now_deep
        state["previous"] = value
        state["previous_capacity"] = capacity
        output[position] = value
    return output.astype(np.float32)


def _build_hourly_dispatch_inputs(
        hourly, source_input, profile_preserving, load_following, dispatchable,
        dynamic_sources=()):
    """Build daily dispatch matrices from a complete hourly forecast."""

    day_keys = hourly["Date"].dt.normalize()
    day_index = hourly.groupby(day_keys).size().index
    day_count = len(day_index)
    demand = hourly["Demand"].to_numpy(dtype=np.float32)
    demand = demand.reshape(day_count, HOURS_PER_DAY)
    source_arrays = {}
    daily_data = {"Date": day_index}
    dynamic_set = set(dynamic_sources)

    for name, source_data in source_input.items():
        if name not in hourly.columns:
            continue
        values = hourly[name].to_numpy(dtype=np.float32)
        matrix = values.reshape(day_count, HOURS_PER_DAY)
        operation = str(
            source_data.get("hourly_operation", "")
            if isinstance(source_data, dict) else ""
        ).strip().lower()

        if name in dynamic_set and operation == "must_run":
            source_arrays[name] = np.zeros_like(matrix)
        elif name in dispatchable or name in load_following:
            daily_data[name] = matrix.sum(axis=1)
            source_arrays[name] = np.zeros_like(matrix)
        else:
            source_arrays[name] = matrix.copy()
            if name in profile_preserving:
                daily_data[name] = matrix.sum(axis=1)

        installed = f"Installed_Capacity_{name}"
        if installed in hourly.columns:
            capacity = hourly[installed].to_numpy(dtype=float)
            capacity = capacity.reshape(day_count, HOURS_PER_DAY)
            daily_data[installed] = capacity.max(axis=1)

        available = f"Available_Capacity_{name}"
        if available in hourly.columns and name not in dynamic_set:
            capacity = hourly[available].to_numpy(dtype=float)
            capacity = capacity.reshape(day_count, HOURS_PER_DAY)
            daily_data[available] = capacity.min(axis=1)

    return demand, source_arrays, pd.DataFrame(daily_data)


def _prepare_hourly_with_dynamic_efpd(forecast_df, user_input):
    """Operate direct-hourly EFPD fleets causally, one day at a time."""

    hourly = _prepare_hourly_data(forecast_df)
    hourly = _enforce_hourly_capacity_feasibility(hourly, user_input)
    day_keys = hourly["Date"].dt.normalize()
    day_counts = hourly.groupby(day_keys).size()
    if not day_counts.eq(HOURS_PER_DAY).all():
        raise ValueError(
            "Direct-hourly EFPD operation requires 24 rows per day.")
    if "Demand" not in hourly.columns:
        raise ValueError(
            "EFPD-based operational refuelling requires Demand.")

    (source_input, profile_preserving, load_following, dispatchable,
     dynamic_sources) = _causal_operation_groups(user_input)

    demand, source_arrays, daily = _build_hourly_dispatch_inputs(
        hourly, source_input, profile_preserving, load_following, dispatchable,
        dynamic_sources)
    stages, profile_state = _run_causal_efpd_dispatch(
        source_arrays, demand, daily, user_input)

    output = hourly.copy()
    for name, values in source_arrays.items():
        output[name] = values.reshape(-1)
    for column, values in stages.items():
        output[column] = values.reshape(-1)
    for source, state in profile_state.items():
        output[f"Refueling_Units_{source}"] = np.repeat(
            state["refueling_units"], HOURS_PER_DAY)
        output[f"Refueling_Capacity_{source}"] = np.repeat(
            state["refueling_power_capacity"], HOURS_PER_DAY)
        output[f"Available_Capacity_{source}"] = np.repeat(
            state["available_power_capacity"], HOURS_PER_DAY)
        output[f"Mean_Accumulated_EFPD_{source}"] = np.repeat(
            state["mean_lifetime_efpd"], HOURS_PER_DAY)

    available_sources = [
        name for name in source_input if name in output.columns]
    if available_sources:
        output["Total"] = output[available_sources].sum(axis=1)
    event_rows = [
        row for state in profile_state.values()
        for row in state.get("event_rows", [])]
    output.attrs["operational_refueling_schedule"] = event_rows
    return output


def _prepare_hourly_data(hourly_df):
    """Validate a forecast containing one row per hour."""

    if "Date" not in hourly_df.columns:
        raise ValueError("Forecast must contain a 'Date' column.")
    prepared = hourly_df.copy()
    prepared["Date"] = pd.to_datetime(
        prepared["Date"],
        errors="coerce",)
    if prepared["Date"].isna().any():
        raise ValueError("Forecast contains invalid timestamps.")
    prepared = prepared.sort_values("Date").reset_index(drop=True)
    if prepared["Date"].duplicated().any():
        raise ValueError("Hourly Forecast contains duplicate timestamps.")
    differences = prepared["Date"].diff().dropna()
    if not differences.eq(pd.Timedelta(hours=1)).all():
        raise ValueError(
            "Hourly Forecast must contain a continuous one-hour time axis.")
    return prepared


def _enforce_hourly_capacity_feasibility(hourly, user_input):
    """Apply hourly capacity limits without temporal redistribution."""

    output = hourly.copy()
    for name in user_input.get("sources", {}):
        capacity_name = f"Installed_Capacity_{name}"
        if name not in output.columns or capacity_name not in output.columns:
            continue
        values = pd.to_numeric(
            output[name],
            errors="coerce",
        ).to_numpy(dtype=float)
        source_data = user_input.get("sources", {}).get(name, {}) or {}
        operation = str(
            source_data.get("hourly_operation", "")
        ).strip().lower()
        available_name = f"Available_Capacity_{name}"
        refueling = source_data.get("refueling", {}) or {}
        if (
            operation == "must_run"
            and bool(refueling)
            and available_name in output.columns
        ):
            power_fraction = _must_run_power_fraction(name, source_data)
            unit_factor = energy_from_mwh_factor(
                user_input.get("energy_unit", "MWh"))
            values = pd.to_numeric(
                output[available_name], errors="coerce"
            ).to_numpy(dtype=float)
            values *= unit_factor * power_fraction
            output[name] = values
        capacities = pd.to_numeric(
            output[capacity_name],
            errors="coerce",
        ).to_numpy(dtype=float)
        if np.any(~np.isfinite(values)):
            raise ValueError(
                f"Hourly values for '{name}' must be finite.")
        if np.any(~np.isfinite(capacities)):
            raise ValueError(
                f"Hourly capacities for '{name}' must be finite.")
        limit_factor = _capacity_limit_factor(user_input, name)
        unit_factor = energy_from_mwh_factor(
            user_input.get("energy_unit", "MWh"))
        limits = capacities * limit_factor * unit_factor
        if np.any(values < 0.0) or np.any(limits < 0.0):
            raise ValueError(
                f"Hourly values and capacities for '{name}' "
                "must be non-negative.")
        _report_capacity_clipping(
            name,
            values,
            limits,
            output["Date"],
            "hourly",
            user_input,)
        output[name] = np.minimum(values, limits)
    return output


def _dispatch_hourly_sources(hourly, user_input):
    """Dispatch profile_preserving and dispatchable sources within each day."""

    source_input, profile_preserving, load_following, dispatchable = (
        _operation_groups(user_input))
    if not profile_preserving and not load_following and not dispatchable:
        return hourly
    if "Demand" not in hourly.columns:
        raise ValueError(
            "Hourly dispatch requires a Demand column.")

    day_counts = hourly.groupby(
        hourly["Date"].dt.normalize(),
    ).size()
    if not day_counts.eq(HOURS_PER_DAY).all():
        raise ValueError(
            "Hourly operational dispatch requires 24 rows per day.")

    demand, source_arrays, daily = _build_hourly_dispatch_inputs(
        hourly, source_input, profile_preserving, load_following, dispatchable)
    dispatch_stages = _dispatch_arrays(
        source_arrays,
        demand,
        daily,
        profile_preserving,
        load_following,
        dispatchable,
        user_input,)
    output = hourly.copy()
    for name, values in source_arrays.items():
        output[name] = values.reshape(-1)
    for column, values in dispatch_stages.items():
        output[column] = values.reshape(-1)
    return output
