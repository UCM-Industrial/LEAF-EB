"""Export annual nuclear fleet data to the ANICCA interface."""

from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis.scenario import get_output_level


def save_anicca_input(hourly_df, simulation_id, output_dir, user):
    """Write the six-column annual LEAF-to-ANICCA interface."""

    if simulation_id != 0:
        return None
    if get_output_level(user) == "comparison":
        return None
    sources = user.get("sources", {}) or {}
    fleets = []
    for source, config in sources.items():
        if not isinstance(config, dict):
            continue
        refueling = config.get("refueling", {}) or {}
        fuel_cycle = config.get("fuel_cycle", {}) or {}
        if not bool(refueling) and not fuel_cycle:
            continue
        if source in hourly_df.columns:
            fleets.append(source)
    if not fleets:
        return None

    path = Path(output_dir) / "ANICCA_Input.xlsx"
    path.parent.mkdir(parents=True, exist_ok=True)
    used_names = set()
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for source in fleets:
            config = sources[source]
            table = _anicca_fleet_table(
                hourly_df, source, config, user)
            sheet_name = _safe_sheet_name(source, used_names)
            table.to_excel(writer, sheet_name=sheet_name, index=False)
    return path

def _anicca_fleet_table(hourly_df, source, config, user):
    """Return one annual six-column fleet table."""

    installed_column = f"Installed_Capacity_{source}"
    if installed_column not in hourly_df.columns:
        raise ValueError(
            f"Nuclear fleet output for '{source}' requires "
            f"{installed_column}.")
    refueling = config.get("refueling", {}) or {}
    unit_capacity = float(config.get("unit_capacity", 0.0))
    if unit_capacity <= 0.0:
        raise ValueError(
            f"Nuclear fleet output for '{source}' requires positive "
            "unit_capacity.")

    frame = hourly_df[["Date", source, installed_column]].copy()
    frame["Date"] = pd.to_datetime(frame["Date"])
    frame["Year"] = frame["Date"].dt.year
    energy_factor = _energy_to_mwh(user.get("energy_unit", "MWh"))
    step_hours = _modeled_step_hours(frame["Date"])
    frame["Generation_Energy"] = (
        pd.to_numeric(frame[source], errors="coerce").fillna(0.0)
        * energy_factor)
    frame["Potential_Energy"] = (
        pd.to_numeric(frame[installed_column], errors="coerce").fillna(0.0)
        * step_hours)

    annual = frame.groupby("Year", as_index=False).agg({
        "Generation_Energy": "sum",
        "Potential_Energy": "sum",
        installed_column: "last",})
    generation = annual["Generation_Energy"].to_numpy(dtype=float)
    potential = annual["Potential_Energy"].to_numpy(dtype=float)
    annual["LoadFactor"] = np.divide(
        generation, potential,
        out=np.zeros_like(generation), where=potential > 0.0)

    unit_counts = np.rint(
        pd.to_numeric(frame[installed_column], errors="coerce").fillna(0.0)
        / unit_capacity).astype(int)
    initial_capacity = float(
        config.get("initial_capacity", 0.0) or 0.0)
    previous = int(round(initial_capacity / unit_capacity))
    daily_units = pd.DataFrame({
        "Day": frame["Date"].dt.normalize().to_numpy(),
        "Units": unit_counts.to_numpy()}).groupby(
            "Day", sort=True)["Units"].last()
    changes = daily_units.diff()
    if not changes.empty:
        changes.iloc[0] = daily_units.iloc[0] - previous
    changes = changes.loc[changes.ne(0)].astype(int)
    change_years = changes.index.year
    positive = changes.loc[changes.gt(0)]
    negative = changes.loc[changes.lt(0)]
    in_by_year = positive.groupby(
        change_years[changes.gt(0)]).sum().astype(int).to_dict()
    out_by_year = (-negative).groupby(
        change_years[changes.lt(0)]).sum().astype(int).to_dict()

    burnup = _fleet_burnup(config)
    result = pd.DataFrame({
        "Year": annual["Year"].astype(int),
        "Installed_Power": annual[installed_column].astype(float),
        "Burnup": burnup,
        "Load_Factor": annual["LoadFactor"].astype(float),
        "Reactors_In": [
            int(in_by_year.get(int(year), 0)) for year in annual["Year"]],
        "Reactors_Out": [
            int(out_by_year.get(int(year), 0)) for year in annual["Year"]],})
    return result


def _fleet_burnup(config):
    """Return the ANICCA burnup value when its meaning is unambiguous."""

    refueling = config.get("refueling", {}) or {}
    if refueling.get("operating_cycle") is not None:
        return float("nan")
    fuel_cycle = config.get("fuel_cycle", {}) or {}
    target = fuel_cycle.get("target_burnup")
    if target is None:
        return float("nan")
    return float(target)


def _safe_sheet_name(name, used_names):
    """Return a unique Excel-compatible sheet name."""

    invalid = set('[]:*?/\\\\')
    cleaned = "".join("_" if char in invalid else char for char in str(name))
    cleaned = cleaned[:31] or "Fleet"
    candidate = cleaned
    index = 2
    while candidate in used_names:
        suffix = f"_{index}"
        candidate = f"{cleaned[:31-len(suffix)]}{suffix}"
        index += 1
    used_names.add(candidate)
    return candidate


def _modeled_step_hours(dates):
    """Infer duration represented by one operational row."""

    values = pd.to_datetime(dates).sort_values()
    differences = values.diff().dropna()
    if differences.empty:
        return 1.0
    hours = differences.dt.total_seconds() / 3600.0
    median = float(hours.median())
    return median if median > 0.0 else 1.0


def _energy_to_mwh(unit):
    """Return active energy-unit to MWh conversion without circular imports."""

    from src.utilities.units import energy_to_mwh_factor
    return energy_to_mwh_factor(unit)
