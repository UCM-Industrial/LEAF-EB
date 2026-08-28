# -*- coding: utf-8 -*-
"""Build refuelling availability profiles for explicit-capacity fleets."""

from heapq import heappop, heappush
from pathlib import Path

import numpy as np
import pandas as pd

from src.utilities.units import energy_to_mwh_factor
from src.technologies.nuclear.config import (
    is_dynamic_efpd_refueling, is_online_efpd_refueling,
    refueling_settings)


_DAYS_PER_YEAR = 365.2425


def build_refueling_profiles(
        forecast_df, user_input, nominal_efpd=False,
        nominal_efpd_rate=1.0):
    """Add deterministic refuelling availability columns.

    Calendar-based offline refuelling is deterministic and is therefore
    built before Monte Carlo sampling. ``schedule: auto`` keeps each unit's
    own clock and allows simultaneous outages. ``schedule: staggered``
    enforces a non-overlapping fleet schedule when that constraint is used.

    EFPD-based offline refuelling is operational because load following
    changes the EFPD clock. It is omitted here unless ``nominal_efpd`` is
    requested, in which case a full-power nominal profile is generated for
    scenario construction. Online refuelling never removes capacity.
    """

    output = forecast_df.copy()
    schedules = []
    for source, source_input in user_input.get("sources", {}).items():
        settings = refueling_settings(source_input)
        if not settings["enabled"]:
            continue
        if settings["mode"] == "online":
            profile = _online_profile(output, source)
            schedule = _empty_schedule()
        elif settings["basis"] == "burnup" and not nominal_efpd:
            continue
        else:
            profile, schedule = _build_source_profile(
                output, source, settings, nominal_efpd=nominal_efpd,
                nominal_efpd_rate=nominal_efpd_rate)
        for column in profile.columns:
            if column != "Date":
                output[column] = profile[column].to_numpy(copy=False)
        schedules.append(schedule)

    schedule = _combine_schedules(schedules)
    return output, schedule


def save_refueling_schedule(
        schedule, output_dir, start_date=None, end_date=None):
    """Save deterministic outage events overlapping the scenario period."""

    if schedule.empty:
        return None
    selected = schedule.copy()
    starts = pd.to_datetime(selected["Outage_Start"])
    ends = pd.to_datetime(selected["Outage_End"])
    mask = pd.Series(True, index=selected.index)
    if start_date:
        mask &= ends >= pd.Timestamp(start_date)
    if end_date:
        mask &= starts <= pd.Timestamp(end_date)
    selected = selected.loc[mask].reset_index(drop=True)
    output_path = Path(output_dir)
    path = output_path / "Refueling_Schedule.csv"
    selected.to_csv(path, index=False)
    return path


def save_refueling_profile(forecast_df, output_dir):
    """Save deterministic daily refuelling and available-capacity data."""

    prefixes = (
        "Refueling_Units_", "Refueling_Capacity_", "Available_Capacity_")
    columns = [
        column for column in forecast_df.columns
        if str(column).startswith(prefixes)]
    if not columns or "Date" not in forecast_df.columns:
        return None
    output = forecast_df[["Date", *columns]].copy()
    output["Date"] = pd.to_datetime(output["Date"])
    output_path = Path(output_dir)
    path = output_path / "Refueling_Profile.csv"
    output.to_csv(path, index=False)
    nuclear_columns = [
        column for column in columns
        if str(column).endswith("_Nuclear")]
    if nuclear_columns and len(nuclear_columns) == len(columns):
        output.to_csv(
            output_path / "Nuclear_Refueling_Profile.csv", index=False)
    return path


def apply_efpd_refueling_profiles(
        hourly_df, user_input, source_names=None):
    """Derive EFPD-driven outage profiles from operational generation.

    The generated profile is causal at daily resolution: EFPD accumulated
    during one modeled day can trigger an outage starting the next day. Units
    commissioned on the same date may therefore refuel simultaneously.
    """

    output = hourly_df.copy()
    sources = user_input.get("sources", {})
    selected = source_names or list(sources)
    schedules = []
    for source in selected:
        source_input = sources.get(source, {}) or {}
        settings = refueling_settings(source_input)
        if is_dynamic_efpd_refueling(source_input):
            profile, schedule = _efpd_profile_from_generation(
                output, source, settings, user_input)
        elif is_online_efpd_refueling(source_input):
            profile = _online_efpd_profile_from_generation(
                output, source, settings, user_input)
            schedule = _empty_schedule()
        else:
            continue
        output = _merge_profile_to_resolution(output, profile)
        schedules.append(schedule)
    return output, _combine_schedules(schedules)


def _online_profile(forecast, source):
    """Return a zero-outage profile for an online-refuelled fleet."""

    capacity_column = f"Installed_Capacity_{source}"
    if capacity_column not in forecast.columns:
        raise ValueError(
            f"Refuelling source '{source}' requires {capacity_column}.")
    dates = pd.to_datetime(forecast["Date"], errors="coerce")
    capacity = pd.to_numeric(
        forecast[capacity_column], errors="coerce").to_numpy(dtype=float)
    zeros = np.zeros(len(forecast), dtype=int)
    return pd.DataFrame({
        "Date": dates,
        f"Refueling_Units_{source}": zeros,
        f"Refueling_Capacity_{source}": zeros.astype(float),
        f"Available_Capacity_{source}": capacity,})


def _build_source_profile(
        forecast, source, settings, nominal_efpd=False,
        nominal_efpd_rate=1.0):
    """Build one deterministic source availability profile."""

    capacity_column = f"Installed_Capacity_{source}"
    if capacity_column not in forecast.columns:
        raise ValueError(
            f"Refuelling source '{source}' requires {capacity_column}.")
    if "Date" not in forecast.columns:
        raise ValueError("Refuelling profile requires a Date column.")

    dates = pd.to_datetime(forecast["Date"], errors="coerce")
    if dates.isna().any():
        raise ValueError("Refuelling profile contains invalid dates.")
    if not dates.is_monotonic_increasing:
        raise ValueError("Refuelling profile dates must be increasing.")

    capacities = pd.to_numeric(
        forecast[capacity_column], errors="coerce"
    ).to_numpy(dtype=float)
    if np.any(~np.isfinite(capacities)) or np.any(capacities < 0.0):
        raise ValueError(
            f"Installed capacity for '{source}' is invalid.")

    unit_capacity = settings["unit_capacity"]
    units = _capacity_to_units(capacities, unit_capacity, source)
    if settings["schedule"] == "staggered":
        _validate_staggered_feasibility(units, settings, source)
    unit_windows = _unit_operating_windows(dates, units, source)
    event_rows = _schedule_unit_outages(
        dates, units, unit_windows, settings, source,
        nominal_efpd=nominal_efpd,
        nominal_efpd_rate=nominal_efpd_rate)
    refueling_units = _profile_outages(dates, event_rows)

    available = capacities - refueling_units * unit_capacity
    if np.any(available < -1e-6):
        raise ValueError(
            f"Refuelling schedule for '{source}' removes more capacity "
            "than installed.")
    available = np.maximum(available, 0.0)

    profile = pd.DataFrame({
        "Date": dates,
        f"Refueling_Units_{source}": refueling_units,
        f"Refueling_Capacity_{source}":
            refueling_units.astype(float) * unit_capacity,
        f"Available_Capacity_{source}": available,})
    return profile, pd.DataFrame(event_rows)


def _capacity_to_units(capacities, unit_capacity, source):
    """Convert installed MW into an exact integer unit count."""

    units = np.rint(capacities / unit_capacity).astype(int)
    reconstructed = units.astype(float) * unit_capacity
    tolerance = np.maximum(1e-6, unit_capacity * 1e-6)
    if np.any(np.abs(reconstructed - capacities) > tolerance):
        raise ValueError(
            f"Installed capacity for '{source}' must be an integer multiple "
            "of unit_capacity.")
    return units


def _validate_staggered_feasibility(units, settings, source):
    """Reject non-overlap schedules that cannot fit within a cycle."""

    max_units = int(np.max(units)) if len(units) else 0
    if max_units <= 0:
        return
    if settings["basis"] == "calendar":
        cycle_days = (
            _DAYS_PER_YEAR * float(settings["operating_cycle"]) / 12.0)
    else:
        cycle_days = float(_cycle_efpd(settings))
    required_days = max_units * settings["outage_duration"]
    if required_days > cycle_days + 1e-9:
        spacing = cycle_days / max_units
        raise ValueError(
            f"Refuelling settings for '{source}' require overlapping "
            "outages: outage_duration={settings['outage_duration']}, "
            f"cycle/unit spacing={spacing:.2f} days.")


def _unit_operating_windows(dates, units, source):
    """Infer commissioning and retirement dates from installed capacity."""

    windows = []
    active = []
    next_id = 1
    previous_count = 0

    for position, current_value in enumerate(units):
        current_count = int(current_value)
        date = pd.Timestamp(dates.iloc[position]).normalize()
        if current_count > previous_count:
            for _ in range(current_count - previous_count):
                windows.append({
                    "Unit": f"{source}_{next_id:02d}",
                    "Commissioning_Date": date,
                    "Retirement_Date": None,})
                active.append(len(windows) - 1)
                next_id += 1
        elif current_count < previous_count:
            retire_count = previous_count - current_count
            for _ in range(retire_count):
                if not active:
                    raise ValueError(
                        f"Installed unit history for '{source}' is invalid.")
                window_index = active.pop()
                windows[window_index]["Retirement_Date"] = date
        previous_count = current_count

    return windows


def _schedule_unit_outages(
        dates, units, unit_windows, settings, source,
        nominal_efpd=False, nominal_efpd_rate=1.0):
    """Schedule recurring deterministic outages for each unit."""

    if not unit_windows or settings["mode"] == "online":
        return []

    horizon_end = pd.Timestamp(dates.iloc[-1]).normalize()
    outage_duration = settings["outage_duration"]
    schedule_mode = settings["schedule"]
    queue = []

    for unit_index, window in enumerate(unit_windows):
        nominal = _first_nominal_outage(
            window["Commissioning_Date"], settings, nominal_efpd,
            nominal_efpd_rate)
        heappush(queue, (nominal, unit_index, 1))

    occupied = []
    rows = []
    while queue:
        nominal, unit_index, refueling_number = heappop(queue)
        window = unit_windows[unit_index]
        retirement = window["Retirement_Date"]
        if retirement is not None and nominal >= retirement:
            continue
        if nominal > horizon_end:
            continue

        actual = pd.Timestamp(nominal).normalize()
        if schedule_mode == "staggered":
            actual = _first_free_start(actual, outage_duration, occupied)
        outage_end = actual + pd.Timedelta(days=outage_duration - 1)
        if retirement is not None and outage_end >= retirement:
            continue
        if actual > horizon_end:
            continue

        if schedule_mode == "staggered":
            occupied.append((actual, outage_end))
            occupied.sort(key=lambda item: item[0])
        installed_units = _installed_units_on_date(
            dates, units, actual)
        row = {
            "Source": source,
            "Unit": window["Unit"],
            "Commissioning_Date":
                window["Commissioning_Date"].date().isoformat(),
            "Refueling_Number": refueling_number,
            "Nominal_Outage_Start": nominal.date().isoformat(),
            "Outage_Start": actual.date().isoformat(),
            "Outage_End": outage_end.date().isoformat(),
            "Schedule_Shift": int((actual - nominal).days),
            "Outage_Duration": outage_duration,
            "Operating_Cycle": settings.get("operating_cycle"),
            "Fuel_Batches": settings.get("fuel_batches"),
            "Cycle_EFPD": _cycle_efpd(settings),
            "Installed_Units": installed_units,}
        rows.append(row)

        if (
                settings["basis"] == "burnup"
                and nominal_efpd
                and schedule_mode == "auto"):
            next_nominal = _nominal_efpd_outage(
                window["Commissioning_Date"],
                settings,
                refueling_number + 1,
                nominal_efpd_rate)
        else:
            next_nominal = _next_nominal_outage(
                actual, settings, nominal_efpd_rate)
        if retirement is None or next_nominal < retirement:
            heappush(
                queue,
                (next_nominal, unit_index, refueling_number + 1))

    return rows


def _first_nominal_outage(
        date, settings, nominal_efpd, nominal_efpd_rate=1.0):
    """Return the first nominal outage date for deterministic scheduling."""

    if settings["basis"] == "calendar":
        return _add_cycle(date, settings["operating_cycle"])
    if not nominal_efpd:
        raise ValueError(
            "EFPD refuelling requires operational scheduling.")
    return _nominal_efpd_outage(
        date, settings, 1, nominal_efpd_rate)


def _nominal_efpd_outage(
        commissioning_date, settings, refueling_number,
        nominal_efpd_rate=1.0):
    """Return the exact full-power date of an EFPD refuelling event.

    Fractional EFPD is carried from one cycle to the next. This mirrors the
    causal must-run clock, instead of rounding every individual cycle up to
    a whole day and accumulating a calendar drift.
    """

    number = int(refueling_number)
    if number <= 0:
        raise ValueError("Refuelling number must be positive.")
    rate = float(nominal_efpd_rate)
    if not np.isfinite(rate) or rate <= 0.0 or rate > 1.0:
        raise ValueError("nominal_efpd_rate must be in (0, 1].")
    operating_days = int(np.ceil(
        number * _cycle_efpd(settings) / rate))
    outage_duration = int(settings.get("outage_duration", 0) or 0)
    elapsed_days = operating_days + (number - 1) * outage_duration
    return pd.Timestamp(commissioning_date) + pd.Timedelta(days=elapsed_days)


def _next_nominal_outage(
        date, settings, nominal_efpd_rate=1.0):
    """Advance a deterministic outage by one configured cycle."""

    if settings["basis"] == "calendar":
        resumed = pd.Timestamp(date) + pd.Timedelta(
            days=int(settings.get("outage_duration", 0) or 0))
        return _add_cycle(resumed, settings["operating_cycle"])
    rate = float(nominal_efpd_rate)
    if not np.isfinite(rate) or rate <= 0.0 or rate > 1.0:
        raise ValueError("nominal_efpd_rate must be in (0, 1].")
    operating_days = int(np.ceil(_cycle_efpd(settings) / rate))
    outage_duration = int(settings.get("outage_duration", 0) or 0)
    return pd.Timestamp(date) + pd.Timedelta(
        days=operating_days + outage_duration)


def _cycle_efpd(settings):
    """Return EFPD between batch refuelling events."""

    if settings.get("basis") != "burnup":
        return None
    return (
        float(settings["residence_efpd"])
        / float(settings["fuel_batches"]))


def _add_cycle(date, operating_cycle):
    """Advance a date by a calendar refuelling cycle."""

    rounded = int(round(operating_cycle))
    if abs(operating_cycle - rounded) < 1e-9:
        return pd.Timestamp(date) + pd.DateOffset(months=rounded)
    cycle_days = int(round(_DAYS_PER_YEAR * operating_cycle / 12.0))
    return pd.Timestamp(date) + pd.Timedelta(days=cycle_days)


def _first_free_start(nominal, outage_duration, occupied):
    """Return the first feasible non-overlapping outage block."""

    start = pd.Timestamp(nominal).normalize()
    duration = pd.Timedelta(days=outage_duration - 1)
    while True:
        end = start + duration
        conflict = None
        for occupied_start, occupied_end in occupied:
            if start <= occupied_end and end >= occupied_start:
                conflict = (occupied_start, occupied_end)
                break
        if conflict is None:
            return start
        start = conflict[1] + pd.Timedelta(days=1)


def _installed_units_on_date(dates, units, date):
    """Return installed unit count on a calendar date."""

    values = dates.to_numpy(dtype="datetime64[ns]")
    position = int(np.searchsorted(values, np.datetime64(date), side="right"))
    position = min(max(position - 1, 0), len(units) - 1)
    return int(units[position])


def _profile_outages(dates, event_rows):
    """Convert outage events into a count of unavailable units."""

    result = np.zeros(len(dates), dtype=int)
    values = dates.to_numpy(dtype="datetime64[ns]")
    for row in event_rows:
        start = np.datetime64(pd.Timestamp(row["Outage_Start"]))
        stop = np.datetime64(
            pd.Timestamp(row["Outage_End"]) + pd.Timedelta(days=1))
        start_index = int(np.searchsorted(values, start, side="left"))
        stop_index = int(np.searchsorted(values, stop, side="left"))
        if stop_index > start_index:
            result[start_index:stop_index] += 1
    return result


def _daily_generation_capacity(
        hourly, source, capacity_column, user_input):
    """Aggregate hourly generation and installed capacity by modeled day."""

    frame = hourly[["Date", source, capacity_column]].copy()
    frame["Date"] = pd.to_datetime(frame["Date"])
    frame["Day"] = frame["Date"].dt.normalize()
    factor = energy_to_mwh_factor(user_input.get("energy_unit", "MWh"))
    grouped = frame.groupby("Day")
    generation = grouped[source].sum() * factor
    capacity = grouped[capacity_column].max().reindex(generation.index)
    return generation, capacity.to_numpy(dtype=float)


def _efpd_profile_from_generation(
        hourly, source, settings, user_input):
    """Build a causal daily EFPD outage profile from actual generation."""

    if "Date" not in hourly.columns or source not in hourly.columns:
        raise ValueError(
            f"EFPD refuelling for '{source}' requires Date and generation.")
    capacity_column = f"Installed_Capacity_{source}"
    if capacity_column not in hourly.columns:
        raise ValueError(
            f"EFPD refuelling for '{source}' requires {capacity_column}.")

    daily_generation, capacities = _daily_generation_capacity(
        hourly, source, capacity_column, user_input)
    dates = daily_generation.index
    unit_capacity = settings["unit_capacity"]
    installed_counts = _capacity_to_units(
        capacities, unit_capacity, source)

    threshold = _cycle_efpd(settings)
    outage_duration = int(settings["outage_duration"])
    units = []
    next_id = 1
    previous_count = 0
    event_rows = []
    refueling_counts = np.zeros(len(dates), dtype=int)
    fleet_efpd = np.zeros(len(dates), dtype=float)

    for position, day in enumerate(dates):
        current_count = int(installed_counts[position])
        if current_count > previous_count:
            for _ in range(current_count - previous_count):
                units.append({
                    "Unit": f"{source}_{next_id:02d}",
                    "Commissioning_Date": pd.Timestamp(day),
                    "cycle_efpd": 0.0,
                    "lifetime_efpd": 0.0,
                    "outage_remaining": 0,
                    "refueling_number": 0,
                    "retired": False,})
                next_id += 1
        elif current_count < previous_count:
            retire_count = previous_count - current_count
            candidates = [unit for unit in units if not unit["retired"]]
            for unit in reversed(candidates[-retire_count:]):
                unit["retired"] = True
                unit["outage_remaining"] = 0
        previous_count = current_count

        active = [unit for unit in units if not unit["retired"]]
        offline = [
            unit for unit in active if unit["outage_remaining"] > 0]
        available = [
            unit for unit in active if unit["outage_remaining"] <= 0]
        refueling_counts[position] = len(offline)

        available_capacity = len(available) * unit_capacity
        maximum_energy = available_capacity * 24.0
        actual_energy = max(float(daily_generation.iloc[position]), 0.0)
        usable_energy = min(actual_energy, maximum_energy)
        if available_capacity > 0.0:
            increment = usable_energy / (available_capacity * 24.0)
        else:
            increment = 0.0

        for unit in available:
            unit["cycle_efpd"] += increment
            unit["lifetime_efpd"] += increment

        active_lifetime = [
            unit["lifetime_efpd"] for unit in active]
        fleet_efpd[position] = (
            float(np.mean(active_lifetime)) if active_lifetime else 0.0)

        for unit in offline:
            unit["outage_remaining"] -= 1

        next_day = pd.Timestamp(day) + pd.Timedelta(days=1)
        for unit in available:
            if unit["cycle_efpd"] + 1e-12 < threshold:
                continue
            unit["cycle_efpd"] -= threshold
            unit["refueling_number"] += 1
            unit["outage_remaining"] = outage_duration
            outage_end = next_day + pd.Timedelta(days=outage_duration - 1)
            event_rows.append({
                "Source": source,
                "Unit": unit["Unit"],
                "Commissioning_Date":
                    unit["Commissioning_Date"].date().isoformat(),
                "Refueling_Number": unit["refueling_number"],
                "Nominal_Outage_Start": next_day.date().isoformat(),
                "Outage_Start": next_day.date().isoformat(),
                "Outage_End": outage_end.date().isoformat(),
                "Schedule_Shift": 0,
                "Outage_Duration": outage_duration,
                "Operating_Cycle": None,
                "Fuel_Batches": settings.get("fuel_batches"),
                "Cycle_EFPD": threshold,
                "Installed_Units": current_count,
                "EFPD_At_Trigger": threshold,})

    available_capacity = (
        capacities - refueling_counts.astype(float) * unit_capacity)
    available_capacity = np.maximum(available_capacity, 0.0)
    profile = pd.DataFrame({
        "Date": dates,
        f"Refueling_Units_{source}": refueling_counts,
        f"Refueling_Capacity_{source}":
            refueling_counts.astype(float) * unit_capacity,
        f"Available_Capacity_{source}": available_capacity,
        f"Mean_Accumulated_EFPD_{source}": fleet_efpd,})
    return profile, pd.DataFrame(event_rows)


def _online_efpd_profile_from_generation(
        hourly, source, settings, user_input):
    """Track per-unit lifetime EFPD for an online-refuelled fleet."""

    capacity_column = f"Installed_Capacity_{source}"
    if capacity_column not in hourly.columns or source not in hourly.columns:
        raise ValueError(
            f"Online EFPD tracking for '{source}' requires generation and "
            f"{capacity_column}.")

    daily_generation, capacities = _daily_generation_capacity(
        hourly, source, capacity_column, user_input)
    dates = daily_generation.index
    unit_capacity = float(settings["unit_capacity"])
    installed_counts = _capacity_to_units(
        capacities, unit_capacity, source)

    units = []
    previous_count = 0
    lifetime_mean = np.zeros(len(dates), dtype=float)
    for position, current_value in enumerate(installed_counts):
        current_count = int(current_value)
        if current_count > previous_count:
            units.extend([0.0] * (current_count - previous_count))
        elif current_count < previous_count:
            retire_count = previous_count - current_count
            if retire_count > len(units):
                raise ValueError(
                    f"Installed unit history for '{source}' is invalid.")
            del units[-retire_count:]
        previous_count = current_count

        maximum_energy = current_count * unit_capacity * 24.0
        actual_energy = max(float(daily_generation.iloc[position]), 0.0)
        usable_energy = min(actual_energy, maximum_energy)
        increment = (
            usable_energy / maximum_energy if maximum_energy > 0.0
            else 0.0)
        if units:
            units = [value + increment for value in units]
            lifetime_mean[position] = float(np.mean(units))

    zeros = np.zeros(len(dates), dtype=int)
    return pd.DataFrame({
        "Date": dates,
        f"Refueling_Units_{source}": zeros,
        f"Refueling_Capacity_{source}": zeros.astype(float),
        f"Available_Capacity_{source}": capacities,
        f"Mean_Accumulated_EFPD_{source}": lifetime_mean,})


def _merge_profile_to_resolution(frame, profile):
    """Merge a daily refuelling profile onto daily or hourly data."""

    output = frame.copy()
    dates = pd.to_datetime(output["Date"])
    keys = dates.dt.normalize()
    lookup = profile.copy()
    lookup["Date"] = pd.to_datetime(lookup["Date"]).dt.normalize()
    lookup = lookup.set_index("Date")
    for column in lookup.columns:
        output[column] = keys.map(lookup[column])
    return output


def _empty_schedule():
    """Return an empty schedule with the stable output schema."""

    return pd.DataFrame(columns=[
        "Source", "Unit", "Commissioning_Date", "Refueling_Number",
        "Nominal_Outage_Start", "Outage_Start", "Outage_End",
        "Schedule_Shift", "Outage_Duration", "Operating_Cycle",
        "Fuel_Batches", "Cycle_EFPD", "Installed_Units",
        "EFPD_At_Trigger",])


def _combine_schedules(schedules):
    """Combine non-empty event tables into one stable output table."""

    available = [table for table in schedules if not table.empty]
    if not available:
        return _empty_schedule()
    result = pd.concat(available, ignore_index=True, sort=False)
    return result.sort_values(
        ["Outage_Start", "Source", "Unit"]
    ).reset_index(drop=True)
