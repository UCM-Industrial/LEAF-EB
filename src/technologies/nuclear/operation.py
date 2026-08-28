"""Maintain causal EFPD fleet state during system operation."""

import numpy as np
import pandas as pd

from src.technologies.nuclear.config import (
    is_dynamic_efpd_refueling, is_online_efpd_refueling,
    refueling_settings)
from src.utilities.units import energy_from_mwh_factor


HOURS_PER_DAY = 24


def dynamic_efpd_sources(user_input):
    """Return sources whose outage dates depend on generated EFPD."""

    return [
        name for name, source in user_input.get("sources", {}).items()
        if is_dynamic_efpd_refueling(source)]


def online_efpd_sources(user_input):
    """Return sources that track EFPD without scheduled outages."""

    return [
        name for name, source in user_input.get("sources", {}).items()
        if is_online_efpd_refueling(source)]

def initialize_efpd_fleet_state(source, daily, user_input):
    """Create causal unit and output state for one EFPD fleet."""

    source_input = user_input.get("sources", {}).get(source, {}) or {}
    settings = refueling_settings(source_input)
    unit_capacity = float(settings["unit_capacity"])
    installed_column = f"Installed_Capacity_{source}"
    installed = pd.to_numeric(
        daily[installed_column], errors="coerce").to_numpy(dtype=float)
    units = np.rint(installed / unit_capacity).astype(int)
    tolerance = max(1e-6, unit_capacity * 1e-6)
    if np.any(np.abs(units * unit_capacity - installed) > tolerance):
        raise ValueError(
            f"Installed capacity for '{source}' must be an integer "
            "multiple of unit_capacity.")
    threshold = (
        float(settings["residence_efpd"])
        / float(settings["fuel_batches"]))
    return {
        "source": source,
        "settings": settings,
        "threshold": threshold,
        "installed_counts": units,
        "units": [],
        "next_id": 1,
        "previous_count": 0,
        "today_available": [],
        "today_offline": [],
        "today_available_capacity": 0.0,
        "refueling_units": np.zeros(len(daily), dtype=int),
        "refueling_power_capacity": np.zeros(len(daily), dtype=float),
        "available_power_capacity": np.zeros(len(daily), dtype=float),
        "mean_lifetime_efpd": np.zeros(len(daily), dtype=float),
        "event_rows": [],
        "current_date": None,
        "day_position": -1,}


def begin_efpd_day(state, daily_row, source):
    """Apply commissioning/retirement and expose today's availability."""

    state["day_position"] += 1
    position = state["day_position"]
    current_count = int(state["installed_counts"][position])
    previous_count = int(state["previous_count"])
    date = pd.Timestamp(daily_row["Date"]).normalize()
    state["current_date"] = date
    if current_count > previous_count:
        for _ in range(current_count - previous_count):
            state["units"].append({
                "Unit": f"{source}_{state['next_id']:02d}",
                "Commissioning_Date": date,
                "cycle_efpd": 0.0,
                "lifetime_efpd": 0.0,
                "refueling_number": 0,
                "outage_remaining": 0,
                "retired": False,})
            state["next_id"] += 1
    elif current_count < previous_count:
        retire_count = previous_count - current_count
        active_candidates = [
            unit for unit in state["units"] if not unit["retired"]]
        for unit in reversed(active_candidates[-retire_count:]):
            unit["retired"] = True
            unit["outage_remaining"] = 0
    state["previous_count"] = current_count

    active = [
        unit for unit in state["units"] if not unit["retired"]]
    offline = [
        unit for unit in active if unit["outage_remaining"] > 0]
    available = [
        unit for unit in active if unit["outage_remaining"] <= 0]
    unit_capacity = float(state["settings"]["unit_capacity"])
    available_capacity = len(available) * unit_capacity
    state["today_available"] = available
    state["today_offline"] = offline
    state["today_available_capacity"] = available_capacity
    state["refueling_units"][position] = len(offline)
    state["refueling_power_capacity"][position] = (
        len(offline) * unit_capacity)
    state["available_power_capacity"][position] = available_capacity
    return {"available_capacity": available_capacity}


def finish_efpd_day(state, generated_energy, user_input):
    """Accumulate EFPD and trigger tomorrow's outages causally."""

    position = state["day_position"]
    available = state["today_available"]
    offline = state["today_offline"]
    available_capacity = float(state["today_available_capacity"])
    unit_factor = energy_from_mwh_factor(
        user_input.get("energy_unit", "MWh"))
    maximum_energy = available_capacity * HOURS_PER_DAY * unit_factor
    usable_energy = min(max(float(generated_energy), 0.0), maximum_energy)
    increment = (
        usable_energy / maximum_energy if maximum_energy > 0.0 else 0.0)
    for unit in available:
        unit["cycle_efpd"] += increment
        unit["lifetime_efpd"] += increment

    active_lifetime = [
        unit["lifetime_efpd"] for unit in state["units"]
        if not unit["retired"]]
    state["mean_lifetime_efpd"][position] = (
        float(np.mean(active_lifetime)) if active_lifetime else 0.0)

    for unit in offline:
        unit["outage_remaining"] -= 1
    outage_duration = int(state["settings"]["outage_duration"])
    threshold = float(state["threshold"])
    for unit in available:
        if unit["cycle_efpd"] + 1e-12 < threshold:
            continue
        unit["cycle_efpd"] -= threshold
        unit["refueling_number"] += 1
        unit["outage_remaining"] = outage_duration
        start = pd.Timestamp(state["current_date"]) + pd.Timedelta(days=1)
        end = start + pd.Timedelta(days=outage_duration - 1)
        state["event_rows"].append({
            "Source": state["source"],
            "Unit": unit["Unit"],
            "Commissioning_Date": unit["Commissioning_Date"],
            "Refueling_Number": unit["refueling_number"],
            "Outage_Start": start,
            "Outage_End": end,
            "Outage_Duration": outage_duration,
            "Cycle_EFPD": threshold,})
