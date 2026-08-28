"""Track nuclear fuel batches and discharge burnup."""

import numpy as np
import pandas as pd

from src.technologies.nuclear.config import refueling_settings
from src.utilities.units import energy_to_mwh_factor


def attach_calendar_fuel_diagnostics(
        operational_df, user_input, schedule):
    """Attach calendar-refuelling fuel discharge diagnostics.

    Calendar dates determine the outages. Actual fleet generation determines
    the EFPD accumulated by each available equivalent unit. Fuel batches are
    internal bookkeeping only; the public input defines their count.
    """

    output = operational_df.copy()
    output.attrs.update(getattr(operational_df, "attrs", {}))
    if schedule is None:
        schedule = pd.DataFrame()
    schedule = pd.DataFrame(schedule).copy()
    if not schedule.empty:
        output.attrs["operational_refueling_schedule"] = (
            schedule.to_dict("records"))

    discharge_tables = []
    annual_tables = []
    for source, source_input in user_input.get("sources", {}).items():
        settings = refueling_settings(source_input)
        if not settings.get("enabled"):
            continue
        if settings.get("mode") != "offline":
            continue
        if settings.get("basis") != "calendar":
            continue
        if source not in output.columns:
            continue
        source_schedule = schedule.loc[
            schedule.get("Source", pd.Series(dtype=object)).eq(source)
        ].copy() if not schedule.empty else pd.DataFrame()
        discharge, annual = _calendar_fuel_discharge_for_source(
            output, source, source_input, settings, source_schedule,
            user_input)
        if not discharge.empty:
            discharge_tables.append(discharge)
        if not annual.empty:
            annual_tables.append(annual)

    discharges = (
        pd.concat(discharge_tables, ignore_index=True)
        if discharge_tables else pd.DataFrame())
    annual = (
        pd.concat(annual_tables, ignore_index=True)
        if annual_tables else pd.DataFrame())
    output.attrs["fuel_discharge_events"] = discharges.to_dict("records")
    output.attrs["annual_fuel_summary"] = annual.to_dict("records")
    return output


def _calendar_fuel_discharge_for_source(
        frame, source, source_input, settings, schedule, user_input):
    """Calculate discharge burnup for one calendar-driven fleet."""

    fuel_cycle = source_input.get("fuel_cycle", {}) or {}
    thermal_power = float(fuel_cycle.get("thermal_power", 0.0))
    core_mass = float(fuel_cycle.get("core_fuel_mass", 0.0))
    fuel_batches = int(settings.get("fuel_batches", 0) or 0)
    unit_capacity = float(settings.get("unit_capacity", 0.0))
    if (
            thermal_power <= 0.0 or core_mass <= 0.0
            or fuel_batches <= 0 or unit_capacity <= 0.0):
        return pd.DataFrame(), pd.DataFrame()

    dates = pd.to_datetime(frame["Date"], errors="raise")
    work = pd.DataFrame({
        "Day": dates.dt.normalize(),
        "Generation": pd.to_numeric(
            frame[source], errors="coerce").fillna(0.0),})
    installed_column = f"Installed_Capacity_{source}"
    if installed_column not in frame.columns:
        raise ValueError(
            f"Fuel diagnostics for '{source}' require "
            f"{installed_column}.")
    work["Installed"] = pd.to_numeric(
        frame[installed_column], errors="coerce").fillna(0.0)
    available_column = f"Available_Capacity_{source}"
    if available_column in frame.columns:
        work["Available"] = pd.to_numeric(
            frame[available_column], errors="coerce").fillna(0.0)
    else:
        work["Available"] = work["Installed"]

    factor = energy_to_mwh_factor(user_input.get("energy_unit", "MWh"))
    daily = work.groupby("Day", sort=True).agg({
        "Generation": "sum",
        "Installed": "last",
        "Available": "last",})
    daily["Generation_MWh"] = daily["Generation"] * factor
    denominator = daily["Available"] * 24.0
    daily["Fleet_EFPD"] = np.divide(
        daily["Generation_MWh"].to_numpy(dtype=float),
        denominator.to_numpy(dtype=float),
        out=np.zeros(len(daily), dtype=float),
        where=denominator.to_numpy(dtype=float) > 0.0)
    if np.any(daily["Fleet_EFPD"].to_numpy(dtype=float) < -1e-9):
        raise ValueError(
            f"Fuel diagnostics for '{source}' found negative generation.")

    if schedule.empty:
        return pd.DataFrame(), _annual_fuel_summary(
            daily, source, pd.DataFrame(), unit_capacity)

    schedule = schedule.copy()
    for column in ("Commissioning_Date", "Outage_Start", "Outage_End"):
        schedule[column] = pd.to_datetime(
            schedule[column], errors="raise").dt.normalize()
    schedule = schedule.sort_values(
        ["Outage_Start", "Unit", "Refueling_Number"]).reset_index(drop=True)

    batch_mass = core_mass / fuel_batches
    burnup_per_efpd = thermal_power / core_mass / 1000.0
    unit_states = {}
    for unit, group in schedule.groupby("Unit", sort=False):
        commissioning = pd.Timestamp(
            group["Commissioning_Date"].min()).normalize()
        unit_states[unit] = {
            "commissioning": commissioning,
            "batches": [
                {"burnup": 0.0, "loaded": commissioning}
                for _ in range(fuel_batches)],
            "cycle_efpd": 0.0,
            "discharge_number": 0,}

    starts = {}
    outages = {}
    for row in schedule.to_dict("records"):
        start = pd.Timestamp(row["Outage_Start"]).normalize()
        end = pd.Timestamp(row["Outage_End"]).normalize()
        starts.setdefault(start, []).append(row)
        outages.setdefault(row["Unit"], []).append((start, end))

    discharge_rows = []
    for day, values in daily.iterrows():
        day = pd.Timestamp(day).normalize()
        for event in starts.get(day, []):
            unit = event["Unit"]
            state = unit_states[unit]
            batch = state["batches"].pop(0)
            state["discharge_number"] += 1
            discharge_rows.append({
                "Source": source,
                "Unit": unit,
                "Refueling_Number": int(event["Refueling_Number"]),
                "Discharge_Date": day,
                "Discharged_Fuel": batch_mass,
                "Fuel_Mass_Unit": "tHM",
                "Discharge_Burnup": float(batch["burnup"]),
                "Burnup_Unit": "GWd/tHM",
                "Cycle_EFPD": float(state["cycle_efpd"]),
                "Fuel_Batches": fuel_batches,})
            state["batches"].append({
                "burnup": 0.0,
                "loaded": day,})
            state["cycle_efpd"] = 0.0

        increment = max(float(values["Fleet_EFPD"]), 0.0)
        if increment <= 0.0:
            continue
        for unit, state in unit_states.items():
            if day < state["commissioning"]:
                continue
            offline = any(
                start <= day <= end
                for start, end in outages.get(unit, []))
            if offline:
                continue
            state["cycle_efpd"] += increment
            delta_burnup = burnup_per_efpd * increment
            for batch in state["batches"]:
                batch["burnup"] += delta_burnup

    discharge = pd.DataFrame(discharge_rows)
    annual = _annual_fuel_summary(
        daily, source, discharge, unit_capacity)
    return discharge, annual


def _annual_fuel_summary(daily, source, discharge, unit_capacity):
    """Return one annual physical summary for a nuclear fleet."""

    work = daily.reset_index().copy()
    work["Year"] = pd.to_datetime(work["Day"]).dt.year
    work["Potential_MWh"] = work["Installed"] * 24.0
    annual = work.groupby("Year", as_index=False).agg({
        "Generation_MWh": "sum",
        "Potential_MWh": "sum",
        "Installed": "last",})
    annual = annual.loc[
        (annual["Installed"] > 0.0)
        | (annual["Generation_MWh"] > 0.0)].copy()
    if annual.empty:
        return annual
    generation = annual["Generation_MWh"].to_numpy(dtype=float)
    potential = annual["Potential_MWh"].to_numpy(dtype=float)
    annual["Load_Factor"] = np.divide(
        generation, potential,
        out=np.zeros_like(generation), where=potential > 0.0)
    annual["Source"] = source
    annual["Installed_Power"] = annual.pop("Installed")
    annual["Installed_Units"] = np.rint(
        annual["Installed_Power"] / unit_capacity).astype(int)
    annual["Refueling_Events"] = 0
    annual["Discharged_Fuel"] = 0.0
    annual["Fuel_Mass_Unit"] = "tHM"
    annual["Mean_Discharge_Burnup"] = np.nan
    annual["Burnup_Unit"] = "GWd/tHM"

    if not discharge.empty:
        events = discharge.copy()
        events["Year"] = pd.to_datetime(events["Discharge_Date"]).dt.year
        grouped = events.groupby("Year")
        counts = grouped.size().to_dict()
        masses = grouped["Discharged_Fuel"].sum().to_dict()
        weighted = {}
        for year, group in grouped:
            mass = group["Discharged_Fuel"].to_numpy(dtype=float)
            burnup = group["Discharge_Burnup"].to_numpy(
                dtype=float)
            weighted[int(year)] = (
                float(np.average(burnup, weights=mass))
                if mass.sum() > 0.0 else np.nan)
        for index, year in annual["Year"].items():
            year = int(year)
            annual.at[index, "Refueling_Events"] = int(
                counts.get(year, 0))
            annual.at[index, "Discharged_Fuel"] = float(
                masses.get(year, 0.0))
            annual.at[index, "Mean_Discharge_Burnup"] = (
                weighted.get(year, np.nan))

    columns = [
        "Year", "Source", "Installed_Power", "Installed_Units",
        "Load_Factor", "Refueling_Events", "Discharged_Fuel",
        "Fuel_Mass_Unit", "Mean_Discharge_Burnup", "Burnup_Unit"]
    return annual[columns].reset_index(drop=True)
