"""Calculate annual, monthly, event, duration, storage, and ramp metrics."""

from pathlib import Path
import shutil
import json
import hashlib

import numpy as np
import pandas as pd

from src.utilities.units import (
    canonical_energy_unit, energy_conversion_factor,
    energy_from_mwh_factor, normalize_energy_unit,
    normalize_quantity_unit)


ANALYSIS_FOLDER = "Analysis"
TEMP_FOLDER = "Analysis_temp"
OUTPUT_LEVELS = {"comparison", "analysis", "detailed"}
SUMMARY_FOLDER = "Summary"
PROFILES_FOLDER = "Profiles"
FULL_FOLDER = "Detailed"

COMPARISON_SYSTEM_METRICS = [
    "Demand", "Generation", "Initial_Surplus_Electricity",
    "Initial_Residual_Supply_Requirement", "BESS_Charge",
    "BESS_Discharge_to_System", "Supply_From_Fuel_Reconversion",
    "Residual_Supply_Requirement",
    "Peak_Initial_Residual_Supply_Requirement",
    "Peak_Residual_Supply_Requirement"]
COMPARISON_BESS_METRICS = [
    "BESS_Power_Capacity", "BESS_Energy_Capacity", "BESS_Charge",
    "BESS_Discharge_to_System", "Equivalent_Full_Cycles"]
PUBLIC_METRIC_NAMES = {
    "Initial_Surplus_Electricity": "Initial_Surplus_Energy",
    "Initial_Residual_Supply_Requirement":
        "Initial_Positive_Residual_Load_Energy",
    "Residual_Supply_Requirement":
        "Remaining_Positive_Residual_Load_Energy",
    "Peak_Initial_Residual_Supply_Requirement":
        "Peak_Initial_Residual_Load",
    "Peak_Residual_Supply_Requirement":
        "Peak_Remaining_Residual_Load",
    "BESS_Discharge_to_System": "Battery_Discharge",
    "Supply_From_Fuel_Reconversion": "Reconversion_Generation",
    "Equivalent_Full_Cycles": "Full_Equivalent_Cycles",
    "P95_Ramp_Magnitude":
        "P95_Residual_Load_Ramp_Magnitude",
    "P99_Ramp_Magnitude":
        "P99_Residual_Load_Ramp_Magnitude",
    "Maximum_Ramp_Magnitude":
        "Maximum_Residual_Load_Ramp_Magnitude"}
RAMP_PUBLIC_METRIC_NAMES = {
    "P95_Ramp_Magnitude":
        "P95_Residual_Load_Ramp_Magnitude",
    "P99_Ramp_Magnitude":
        "P99_Residual_Load_Ramp_Magnitude",
    "Maximum_Ramp_Magnitude":
        "Maximum_Residual_Load_Ramp_Magnitude"}


def get_output_config(user):
    """Return the normalized three-level output configuration."""

    output = user.get("output")
    if not isinstance(output, dict):
        output = {}

    level = str(output.get("level", "analysis")).strip().lower()
    if level not in OUTPUT_LEVELS:
        allowed = ", ".join(sorted(OUTPUT_LEVELS))
        raise ValueError(
            "output.level must be one of: " + allowed + ".")

    threshold = output.get("event_threshold", 1e-9)
    try:
        threshold = float(threshold)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "output.event_threshold must be numeric.") from exc
    if threshold < 0.0:
        raise ValueError(
            "output.event_threshold cannot be negative.")

    return {
        "level": level,
        "event_threshold": threshold,}



def get_output_level(user):
    """Return comparison, analysis, or detailed."""

    return get_output_config(user)["level"]


def should_save_detailed_output(user, simulation_id=None):
    """Return whether detailed time histories should be retained."""

    del simulation_id
    return get_output_level(user) == "detailed"




def get_analysis_paths(output_dir, create=False):
    """Return the root and temporary directories for scenario analysis."""

    analysis_dir = Path(output_dir) / ANALYSIS_FOLDER
    temp_dir = analysis_dir / TEMP_FOLDER

    if create:
        temp_dir.mkdir(parents=True, exist_ok=True)

    return analysis_dir, temp_dir


def get_simulation_analysis_dir(output_dir, simulation_id, create=False):
    """Return the temporary analysis directory for one simulation."""

    _, temp_dir = get_analysis_paths(output_dir, create=create)
    simulation_dir = temp_dir / f"{simulation_id:06d}"

    if create:
        simulation_dir.mkdir(parents=True, exist_ok=True)

    return simulation_dir


def prepare_analysis_batch(output_dir, simulation_ids, resume=True):
    """Initialize analysis folders and remove stale state when required."""

    _, temp_dir = get_analysis_paths(output_dir, create=not resume)

    if resume:
        return

    for simulation_id in simulation_ids:
        simulation_dir = temp_dir / f"{simulation_id:06d}"
        if simulation_dir.is_dir():
            shutil.rmtree(simulation_dir)


def _analysis_signature(user):
    """Build a stable signature for settings that affect analysis outputs."""

    config = get_output_config(user)
    raw = json.dumps(config, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _success_path(output_dir, simulation_id):
    """Return the completion-marker path for one simulation."""

    simulation_dir = get_simulation_analysis_dir(
        output_dir, simulation_id, create=False)
    return simulation_dir / "_SUCCESS.json"


def clear_simulation_analysis(output_dir, simulation_id):
    """Remove temporary analysis data for one simulation."""

    simulation_dir = get_simulation_analysis_dir(
        output_dir, simulation_id, create=False)

    if simulation_dir.is_dir():
        shutil.rmtree(simulation_dir)


def mark_simulation_complete(
        output_dir, simulation_id, user, run_id=""):
    """Write the completion marker for one analyzed simulation."""

    simulation_dir = get_simulation_analysis_dir(
        output_dir, simulation_id, create=True)
    files = sorted(
        path.name for path in simulation_dir.iterdir()
        if path.suffix in {".parquet", ".pkl", ".csv"})
    payload = {
        "simulation": int(simulation_id),
        "run_id": str(run_id),
        "analysis_signature": _analysis_signature(user),
        "files": files,
        "completed_at": pd.Timestamp.now().isoformat()}
    success_path = simulation_dir / "_SUCCESS.json"
    temp_path = simulation_dir / "_SUCCESS.json.tmp"
    temp_path.write_text(
        json.dumps(payload, sort_keys=True), encoding="utf-8")
    temp_path.replace(success_path)


def simulation_is_complete(
        output_dir, simulation_id, user, run_id=""):
    """Return whether one simulation has complete compatible analysis."""

    success_path = _success_path(output_dir, simulation_id)

    if not success_path.is_file():
        return False

    try:
        payload = json.loads(success_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False

    if int(payload.get("simulation", -1)) != int(simulation_id):
        return False

    if run_id and str(payload.get("run_id", "")) != str(run_id):
        return False

    signature = str(payload.get("analysis_signature", ""))
    if signature != _analysis_signature(user):
        return False

    simulation_dir = success_path.parent
    files = payload.get("files", [])

    if not isinstance(files, list):
        return False

    return all((simulation_dir / name).is_file() for name in files)


def get_completed_simulations(
        output_dir, simulation_ids, user, run_id=""):
    """Return simulations with explicit compatible completion markers."""

    return {
        simulation_id for simulation_id in simulation_ids
        if simulation_is_complete(
            output_dir, simulation_id, user, run_id)}


def clear_analysis_temp(output_dir):
    """Remove all temporary per-simulation analysis tables."""

    analysis_dir = Path(output_dir) / ANALYSIS_FOLDER
    temp_dir = analysis_dir / TEMP_FOLDER

    if temp_dir.is_dir():
        shutil.rmtree(temp_dir)

    if analysis_dir.is_dir() and not any(analysis_dir.iterdir()):
        analysis_dir.rmdir()


def analyze_simulation(output, generation, simulation_id,
                       output_dir, user):
    """Calculate the results required by the selected output level."""

    clear_simulation_analysis(output_dir, simulation_id)
    analysis_dir = get_simulation_analysis_dir(
        output_dir, simulation_id, create=True)
    level = get_output_level(user)

    output = _prepare_frame(output)
    generation = _prepare_frame(generation)
    frame = _combine_aligned_frames(output, generation)

    annual_system = _system_summary(frame, simulation_id, "year")
    _append_output(
        analysis_dir, "Annual_System_Summary.csv", annual_system)
    annual_sources = _source_summary(
        generation, simulation_id, user, "year")
    _append_output(
        analysis_dir, "Annual_Source_Summary.csv", annual_sources)

    annual_nuclear = _nuclear_generation_summary(
        generation, simulation_id, user, "year")
    _append_output(
        analysis_dir, "Annual_Nuclear_Generation.csv", annual_nuclear)

    energy_flows = _energy_flow_summary(frame, simulation_id)
    _append_output(
        analysis_dir, "Energy_Flow_Summary.csv", energy_flows)

    storage = _storage_summary(frame, simulation_id)
    _append_output(analysis_dir, "Storage_Summary.csv", storage)
    bess_lifetime = _bess_lifetime_summary(
        frame, simulation_id, user)
    _append_output(
        analysis_dir, "BESS_Lifetime_Summary.csv", bess_lifetime)

    commodities = _commodity_summary(frame, simulation_id, user)
    _append_output(
        analysis_dir, "Annual_Commodity_Summary.csv", commodities)

    annual_emissions = _emissions_summary(frame, simulation_id, "year")
    _append_output(
        analysis_dir, "Annual_Emissions_Summary.csv", annual_emissions)

    ramps = _ramp_summary(frame, generation, simulation_id, user)
    _append_output(analysis_dir, "Ramp_Summary.csv", ramps)

    if level in {"analysis", "detailed"}:
        daily_publication = _daily_publication_profile(
            frame, simulation_id, user)
        _append_output(
            analysis_dir, "Daily_Publication_Profile.csv",
            daily_publication)

        monthly_system = _system_summary(frame, simulation_id, "month")
        _append_output(
            analysis_dir, "Monthly_System_Summary.csv", monthly_system)
        monthly_sources = _source_summary(
            generation, simulation_id, user, "month")
        _append_output(
            analysis_dir, "Monthly_Source_Summary.csv", monthly_sources)

        temporal = _temporal_profiles(
            frame, generation, simulation_id, user)
        _append_output(
            analysis_dir, "Temporal_Profile_Summary.csv", temporal)

        bess_operation = _bess_operation_summary(
            frame, simulation_id, user)
        _append_output(
            analysis_dir, "BESS_Operation.csv", bess_operation)
        bess_soc = _bess_soc_distribution(
            frame, simulation_id, user)
        _append_output(
            analysis_dir, "BESS_SOC_Distribution.csv", bess_soc)
        bess_hourly, bess_monthly = _bess_temporal_profiles(
            frame, simulation_id, user)
        _append_output(
            analysis_dir, "BESS_Hourly_Profile.csv", bess_hourly)
        _append_output(
            analysis_dir, "BESS_Monthly_Profile.csv", bess_monthly)

        capacity = _commodity_capacity_summary(
            frame, simulation_id, user)
        _append_output(
            analysis_dir, "Commodity_Operation.csv", capacity)
        hourly_profile = _hourly_commodity_profile(
            frame, simulation_id, user)
        _append_output(
            analysis_dir, "Commodity_Hourly_Profile.csv", hourly_profile)
        monthly_profile = _monthly_commodity_profile(
            frame, simulation_id, user)
        _append_output(
            analysis_dir, "Commodity_Monthly_Profile.csv", monthly_profile)

        monthly_emissions = _emissions_summary(
            frame, simulation_id, "month")
        _append_output(
            analysis_dir, "Monthly_Emissions_Summary.csv",
            monthly_emissions)

        annual_use = _annual_technology_use(
            frame, generation, simulation_id, user)
        _append_output(
            analysis_dir,
            "Annual_Technology_Use_By_Simulation.csv",
            annual_use)
        event_summary = _technology_event_summary(
            frame, generation, simulation_id, user)
        _append_output(
            analysis_dir,
            "Technology_Event_Summary.csv",
            event_summary)

        initial_deficit_summary = _event_statistics(
            frame, simulation_id, "initial_deficit", user)
        _append_output(
            analysis_dir,
            "Annual_Initial_Positive_Residual_Load_Episode_Summary.csv",
            initial_deficit_summary)
        deficit_summary = _event_statistics(
            frame, simulation_id, "deficit", user)
        _append_output(
            analysis_dir,
            "Annual_Remaining_Positive_Residual_Load_Episode_Summary.csv",
            deficit_summary)
        surplus_summary = _event_statistics(
            frame, simulation_id, "surplus", user)
        _append_output(
            analysis_dir, "Annual_Surplus_Episode_Summary.csv",
            surplus_summary)

        duration = _duration_curve_summary(frame, simulation_id)
        _append_output(
            analysis_dir, "Residual_Load_Duration_Curve_Summary.csv", duration)

        daily_nuclear = _nuclear_daily_generation_summary(
            generation, simulation_id, user)
        _append_output(
            analysis_dir, "Daily_Nuclear_Generation.csv", daily_nuclear)
        monthly_nuclear = _nuclear_generation_summary(
            generation, simulation_id, user, "month")
        _append_output(
            analysis_dir, "Monthly_Nuclear_Generation.csv", monthly_nuclear)
        refueling_schedule = _operational_refueling_schedule(
            generation, simulation_id)
        _append_output(
            analysis_dir,
            "Nuclear_Refueling_Schedule_By_Simulation.csv",
            refueling_schedule)
        fuel_discharge = _fuel_discharge_events(
            generation, simulation_id)
        _append_output(
            analysis_dir,
            "Nuclear_Fuel_Discharge_By_Simulation.csv",
            fuel_discharge)
        annual_fuel = _annual_fuel_summary(
            generation, simulation_id)
        _append_output(
            analysis_dir,
            "Annual_Nuclear_Fuel_Summary_By_Simulation.csv",
            annual_fuel)
        refueling_impact = _refueling_outage_impact_by_simulation(
            frame, simulation_id, user)
        _append_output(
            analysis_dir,
            "Nuclear_Refueling_Impact_By_Simulation.csv",
            refueling_impact)

    if level == "detailed":
        hourly_use = _hourly_technology_use(
            frame, generation, simulation_id, user)
        _append_output(
            analysis_dir, "Hourly_Technology_Profile.csv", hourly_use)
        technology_events = _technology_use_events(
            frame, generation, simulation_id, user)
        _append_output(
            analysis_dir, "Technology_Use_Events.csv",
            technology_events)
        deficits = _event_summary(frame, simulation_id, "deficit", user)
        _append_output(
            analysis_dir, "Remaining_Positive_Residual_Load_Episodes.csv",
            deficits)
        surplus = _event_summary(frame, simulation_id, "surplus", user)
        _append_output(
            analysis_dir, "Surplus_Episodes.csv", surplus)

    del frame

def _prepare_frame(frame):
    """Normalize dates and ordering before metric calculations."""

    result = frame.copy(deep=False)
    if not pd.api.types.is_datetime64_any_dtype(result["Date"]):
        result = result.copy()
        result["Date"] = pd.to_datetime(result["Date"])
    if not result["Date"].is_monotonic_increasing:
        result = result.sort_values("Date").reset_index(drop=True)
    return result


def _time_step_hours(frame):
    """Return the uniform duration represented by one row."""

    dates = pd.to_datetime(frame["Date"], errors="coerce")
    if dates.isna().any():
        raise ValueError("Analysis data contain invalid timestamps.")
    if len(dates) < 2:
        return 1.0
    differences = dates.diff().dropna().dt.total_seconds() / 3600.0
    if (differences <= 0.0).any():
        raise ValueError(
            "Analysis timestamps must be strictly increasing.")
    step = float(differences.median())
    tolerance = max(1e-9, abs(step) * 1e-9)
    if not np.allclose(differences, step, atol=tolerance, rtol=0.0):
        raise ValueError(
            "Analysis timestamps must use a uniform time step.")
    return step


def _final_balance_series(frame):
    """Return the most advanced available system-balance stage."""

    prefixes = [
        "Final_Electricity_Balance",
        "Electricity_Balance_After_Commodity_Production",
        "Electricity_Balance_After_Storage_and_Interconnections",
        "Electricity_Balance_After_BESS",
        "Electricity_Balance_After_Interconnections",
        "Raw_Electricity_Balance",]
    for prefix in prefixes:
        series = _series_by_prefix(frame, prefix)
        if series is not None:
            return series
    return None


def _combine_aligned_frames(output, generation):
    """Align operational and generation frames on their common time axis."""

    if len(output) != len(generation):
        message = ("Output and generation have different row counts: "
                   f"{len(output)} != {len(generation)}.")
        raise ValueError(message)

    output_dates = output["Date"].to_numpy(copy=False)
    generation_dates = generation["Date"].to_numpy(copy=False)

    if not np.array_equal(output_dates, generation_dates):
        message = "Output and generation dates are not aligned."
        raise ValueError(message)

    missing_columns = [column for column in generation.columns
                       if column != "Date"
                       and column not in output.columns]

    if not missing_columns:
        return output

    frame = output.copy(deep=False)

    for column in missing_columns:
        frame[column] = generation[column].to_numpy(copy=False)

    return frame


def _add_calendar_groups(work, dates, frequency):
    """Add year/month grouping columns and return their ordered names."""

    work["Year"] = dates.dt.year.to_numpy()
    if frequency != "month":
        return ["Year"]
    work["Month"] = dates.dt.month.to_numpy()
    return ["Year", "Month"]


def _system_summary(frame, simulation_id, frequency):
    """Aggregate system-level balance indicators by calendar period."""

    dates = frame["Date"]
    metric_frame = pd.DataFrame(_system_metric_series(frame))
    group_cols = _add_calendar_groups(metric_frame, dates, frequency)

    grouped = metric_frame.groupby(group_cols, as_index=False)
    summary = grouped.sum()
    step_hours = _time_step_hours(frame)
    peaks = grouped[[
        "Initial_Surplus_Electricity",
        "Initial_Residual_Supply_Requirement",
        "Remaining_Surplus_Electricity",
        "Residual_Supply_Requirement",
    ]].max().rename(columns={
        "Initial_Surplus_Electricity":
            "Peak_Initial_Surplus_Electricity",
        "Initial_Residual_Supply_Requirement":
            "Peak_Initial_Residual_Supply_Requirement",
        "Remaining_Surplus_Electricity":
            "Peak_Remaining_Surplus_Electricity",
        "Residual_Supply_Requirement":
            "Peak_Residual_Supply_Requirement",})
    peak_columns = [
        "Peak_Initial_Surplus_Electricity",
        "Peak_Initial_Residual_Supply_Requirement",
        "Peak_Remaining_Surplus_Electricity",
        "Peak_Residual_Supply_Requirement"]
    peaks[peak_columns] = peaks[peak_columns] / step_hours
    summary = summary.merge(peaks, on=group_cols, how="left")
    summary.insert(0, "Simulation", simulation_id)
    summary["Hours"] = grouped.size()["size"].to_numpy() * step_hours
    return summary


def _system_metric_series(frame):
    """Return the system series used by balance summary calculations."""

    raw = _series_by_prefix(frame, "Raw_Electricity_Balance")
    final = _series_by_prefix(frame, "Final_Electricity_Balance")
    covered = _series_by_prefix(frame, "Supply_From_Fuel_Reconversion")
    uncovered = _series_by_prefix(frame, "Residual_Supply_Requirement")

    if final is None:
        final = _final_balance_series(frame)

    before_profile_preserving = _series_by_prefix(
        frame, "Balance_Before_Profile_Preserving")
    after_profile_preserving = _series_by_prefix(
        frame, "Balance_After_Profile_Preserving")
    after_dispatchable = _series_by_prefix(
        frame, "Balance_After_Dispatchable")

    return {
        "Demand": _series_or_zero(frame, "Demand"),
        "Generation": _series_or_zero(frame, "Total"),
        "Before_Profile_Preserving_Surplus": _positive(
            before_profile_preserving, len(frame)),
        "Before_Profile_Preserving_Residual_Supply_Requirement": _negative(
            before_profile_preserving, len(frame)),
        "After_Profile_Preserving_Surplus": _positive(
            after_profile_preserving, len(frame)),
        "After_Profile_Preserving_Residual_Supply_Requirement": _negative(
            after_profile_preserving, len(frame)),
        "After_Dispatchable_Surplus": _positive(
            after_dispatchable, len(frame)),
        "After_Dispatchable_Residual_Supply_Requirement": _negative(
            after_dispatchable, len(frame)),
        "Initial_Surplus_Electricity": _positive(raw, len(frame)),
        "Initial_Residual_Supply_Requirement": _negative(raw, len(frame)),
        "Imports": _series_by_prefix_or_zero(frame, "Imports"),
        "Exports": _series_by_prefix_or_zero(frame, "Exports"),
        "BESS_Charge": _series_by_prefix_or_zero(frame, "BESS_Charge"),
        "BESS_Discharge_to_System": _series_by_prefix_or_zero(
            frame, "BESS_Discharge_to_System"),
        "Supply_From_Fuel_Reconversion": _zero_if_none(covered, len(frame)),
        "Residual_Supply_Requirement": _zero_if_none(uncovered, len(frame)),
        "Remaining_Surplus_Electricity": _positive(final, len(frame))}


def _daily_publication_profile(frame, simulation_id, user):
    """Build compact daily series needed for publication figures.

    Energy-flow columns are summed over each day. State variables such as
    BESS state of charge and commodity inventories retain the final hourly
    value of the day. One daily table per realization permits exact ensemble
    quantiles without retaining every complete hourly history.
    """

    dates = pd.to_datetime(frame["Date"], errors="raise")
    work = pd.DataFrame(_system_metric_series(frame))
    work.insert(0, "Date", dates.dt.normalize().to_numpy())

    flow_prefixes = {
        "BESS_Energy_Withdrawn": "BESS_Energy_Withdrawn",
        "Fuel_Reconversion_Generation": "Fuel_Reconversion_Generation",
    }
    for name, prefix in flow_prefixes.items():
        work[name] = _series_by_prefix_or_zero(frame, prefix).to_numpy()

    commodities = user.get("Commodities_Production", {}) or {}
    for commodity in commodities:
        production = _column_by_prefix(frame, commodity)
        if production is not None:
            work[f"{commodity}_Production"] = pd.to_numeric(
                frame[production], errors="coerce").fillna(0.0).to_numpy()

    summed = work.groupby("Date", sort=True, as_index=False).sum()

    state_columns = {}
    for column in frame.columns:
        name = str(column)
        if name.startswith("BESS_SOC") or name.startswith("Inventory_"):
            state_columns[name] = pd.to_numeric(
                frame[column], errors="coerce").fillna(0.0).to_numpy()

    if state_columns:
        states = pd.DataFrame(state_columns)
        states.insert(0, "Date", dates.dt.normalize().to_numpy())
        states = states.groupby("Date", sort=True, as_index=False).last()
        summed = summed.merge(
            states, on="Date", how="left", validate="one_to_one")

    summed.insert(0, "Simulation", int(simulation_id))
    return summed


def _source_summary(generation, simulation_id, user, frequency):
    """Aggregate generation by source and calendar period."""

    dates = generation["Date"]
    sources = [
        source for source in user.get("sources", {})
        if source in generation.columns]

    if not sources:
        return pd.DataFrame()

    columns = {
        source: pd.to_numeric(
            generation[source], errors="coerce").fillna(0.0)
        for source in sources}
    work = pd.DataFrame(columns)
    group_cols = _add_calendar_groups(work, dates, frequency)

    grouped = work.groupby(group_cols, as_index=False)[sources].sum()
    result = grouped.melt(
        id_vars=group_cols, var_name="Source", value_name="Generation")
    result.insert(0, "Simulation", simulation_id)
    return result


def _nuclear_source_names(user):
    """Return configured sources represented as nuclear fleets."""

    sources = user.get("sources", {}) or {}
    names = []
    for source, config in sources.items():
        if not isinstance(config, dict):
            continue
        refueling = config.get("refueling", {}) or {}
        fuel_cycle = config.get("fuel_cycle", {}) or {}
        if bool(refueling) or bool(fuel_cycle):
            names.append(source)
    return names


def _nuclear_daily_generation_summary(
        generation, simulation_id, user):
    """Aggregate final operational nuclear generation by calendar day."""

    nuclear_sources = set(_nuclear_source_names(user))
    sources = [
        source for source in user.get("sources", {})
        if source in nuclear_sources and source in generation.columns]
    if not sources:
        return pd.DataFrame()

    dates = pd.to_datetime(generation["Date"], errors="raise").dt.normalize()
    work = pd.DataFrame({"Date": dates})
    for source in sources:
        work[source] = pd.to_numeric(
            generation[source], errors="coerce").fillna(0.0)

    grouped = work.groupby("Date", as_index=False, sort=True)[sources].sum()
    result = grouped.melt(
        id_vars=["Date"], var_name="Source", value_name="Generation")
    result.insert(0, "Simulation", int(simulation_id))
    return result


def _nuclear_generation_summary(
        generation, simulation_id, user, frequency):
    """Aggregate final operational nuclear generation by calendar period."""

    nuclear_sources = set(_nuclear_source_names(user))
    if not nuclear_sources:
        return pd.DataFrame()
    summary = _source_summary(
        generation, simulation_id, user, frequency)
    if summary.empty:
        return summary
    summary = summary[summary["Source"].isin(nuclear_sources)].copy()
    if summary.empty:
        return summary
    energy_unit = str(user.get("energy_unit", "MWh")).strip()
    summary["Energy_Unit"] = energy_unit
    if frequency == "year":
        step_hours = _time_step_hours(generation)
        mwh_to_active = energy_from_mwh_factor(energy_unit)
        potential_rows = []
        dates = pd.to_datetime(generation["Date"], errors="raise")
        for source in sorted(nuclear_sources):
            capacity_column = f"Installed_Capacity_{source}"
            if capacity_column not in generation.columns:
                continue
            installed = pd.to_numeric(
                generation[capacity_column], errors="coerce").fillna(0.0)
            potential = installed * step_hours * mwh_to_active
            annual = pd.DataFrame({
                "Year": dates.dt.year.to_numpy(),
                "Potential_Generation": potential.to_numpy()
            }).groupby("Year", as_index=False)[
                "Potential_Generation"].sum()
            annual["Source"] = source
            potential_rows.append(annual)
        if potential_rows:
            potential = pd.concat(potential_rows, ignore_index=True)
            summary = summary.merge(
                potential, on=["Year", "Source"], how="left")
            summary["Load_Factor"] = np.divide(
                summary["Generation"], summary["Potential_Generation"],
                out=np.zeros(len(summary), dtype=float),
                where=summary["Potential_Generation"].to_numpy() > 0.0)
    return summary


def _operational_refueling_schedule(generation, simulation_id):
    """Return unit-level operational refuelling events for one simulation."""

    schedule = generation.attrs.get("operational_refueling_schedule")
    if schedule is None or len(schedule) == 0:
        return pd.DataFrame()
    result = pd.DataFrame(schedule).copy()
    result.insert(0, "Simulation", int(simulation_id))
    for column in (
            "Commissioning_Date", "Outage_Start", "Outage_End"):
        if column in result.columns:
            result[column] = pd.to_datetime(
                result[column], errors="coerce").dt.normalize()
    if "Date" in generation.columns and "Outage_Start" in result.columns:
        horizon_end = pd.to_datetime(
            generation["Date"], errors="coerce").max()
        if pd.notna(horizon_end):
            result = result.loc[
                result["Outage_Start"] <= horizon_end.normalize()].copy()
    return result


def _fuel_discharge_events(generation, simulation_id):
    """Return physical fuel discharges calculated by LEAF."""

    rows = generation.attrs.get("fuel_discharge_events")
    if not rows:
        return pd.DataFrame()
    result = pd.DataFrame(rows).copy()
    result.insert(0, "Simulation", int(simulation_id))
    if "Discharge_Date" in result.columns:
        result["Discharge_Date"] = pd.to_datetime(
            result["Discharge_Date"], errors="coerce").dt.normalize()
    return result


def _annual_fuel_summary(generation, simulation_id):
    """Return annual fleet fuel results without ANICCA assumptions."""

    rows = generation.attrs.get("annual_fuel_summary")
    if not rows:
        return pd.DataFrame()
    result = pd.DataFrame(rows).copy()
    result.insert(0, "Simulation", int(simulation_id))
    return result


def _temporal_profiles(frame, generation, simulation_id, user):
    """Build compact MC profiles and detailed inspection profiles."""

    dates = frame["Date"]
    metrics = _system_metric_series(frame)
    profile = pd.DataFrame(metrics)
    profile["Year"] = dates.dt.year.to_numpy()
    profile["Month"] = dates.dt.month.to_numpy()
    profile["Hour"] = dates.dt.hour.to_numpy()

    sources = [
        source for source in user.get("sources", {})
        if source in generation.columns]

    for source in sources:
        values = pd.to_numeric(
            generation[source], errors="coerce")
        profile[source] = values.fillna(0.0)

    group_cols = ["Year", "Month", "Hour"]
    grouped = profile.groupby(group_cols, sort=True)
    means = grouped.mean().reset_index()
    value_cols = [
        column for column in means.columns
        if column not in group_cols]
    mean_names = {
        column: f"{column}_Mean" for column in value_cols}
    means = means.rename(columns=mean_names)

    if not should_save_detailed_output(user, simulation_id):
        means.insert(0, "Simulation", simulation_id)
        return means

    quantiles = _profile_quantiles(user)
    lower_probability = float(quantiles[0])
    upper_probability = float(quantiles[-1])
    lower = grouped.quantile(lower_probability).reset_index()
    upper = grouped.quantile(upper_probability).reset_index()
    lower_label = _percentile_label(lower_probability)
    upper_label = _percentile_label(upper_probability)
    lower_names = {
        column: f"{column}_{lower_label}" for column in value_cols}
    upper_names = {
        column: f"{column}_{upper_label}" for column in value_cols}
    lower = lower.rename(columns=lower_names)
    upper = upper.rename(columns=upper_names)

    result = means.merge(lower, on=group_cols)
    result = result.merge(upper, on=group_cols)
    result.insert(0, "Simulation", simulation_id)
    return result


def _energy_flow_summary(frame, simulation_id):
    """Summarize cumulative electricity flows between balancing stages."""

    years = frame["Date"].dt.year.to_numpy()

    stages = {
        "Before_Profile_Preserving": "Balance_Before_Profile_Preserving",
        "After_Profile_Preserving": "Balance_After_Profile_Preserving",
        "After_Dispatchable": "Balance_After_Dispatchable",
        "Initial": "Raw_Electricity_Balance",
        "After_Interconnections":
            "Electricity_Balance_After_Interconnections",
        "After_BESS": "Electricity_Balance_After_BESS",
        "After_Storage_and_Interconnections":
            "Electricity_Balance_After_Storage_and_Interconnections",
        "After_Commodity_Production":
            "Electricity_Balance_After_Commodity_Production",
        "Final": "Final_Electricity_Balance"}

    columns = {}
    for stage, prefix in stages.items():
        balance = _series_by_prefix(frame, prefix)
        columns[f"{stage}_Surplus_Electricity"] = _positive(
            balance, len(frame))
        columns[f"{stage}_Residual_Supply_Requirement"] = _negative(
            balance, len(frame))

    flow_prefixes = {
        "Imports": "Imports",
        "Exports": "Exports",
        "BESS_Charge": "BESS_Charge",
        "BESS_Energy_Withdrawn": "BESS_Energy_Withdrawn",
        "BESS_Discharge_to_System": "BESS_Discharge_to_System",
        "Fuel_Reconversion_Generation": "Fuel_Reconversion_Generation",
        "Supply_From_Fuel_Reconversion": "Supply_From_Fuel_Reconversion",
        "Residual_Supply_Requirement": "Residual_Supply_Requirement"}

    for name, prefix in flow_prefixes.items():
        columns[name] = _series_by_prefix_or_zero(
            frame, prefix)

    result = pd.DataFrame(columns)
    result["Year"] = years
    result = result.groupby("Year", as_index=False).sum()
    result.insert(0, "Simulation", simulation_id)

    return result


def _numeric_series_by_year(values, years):
    """Group one numeric series by calendar year."""

    data = pd.DataFrame({
        "Year": years,
        "Value": pd.to_numeric(values, errors="coerce").to_numpy()})
    return data.groupby("Year")["Value"]


def _simulation_rows(rows, simulation_id):
    """Concatenate summary rows and attach their simulation identifier."""

    if not rows:
        return pd.DataFrame()
    result = pd.concat(rows, ignore_index=True)
    result.insert(0, "Simulation", simulation_id)
    return result


def _storage_summary(frame, simulation_id):
    """Summarize BESS charging, delivery, losses, and state of charge."""

    years = frame["Date"].dt.year.to_numpy()
    prefixes = ["BESS_SOC", "BESS_Energy_Capacity", "Inventory_"]
    rows = []

    for column in frame.columns:
        if not any(column.startswith(prefix) for prefix in prefixes):
            continue

        grouped = _numeric_series_by_year(frame[column], years)
        summary = pd.DataFrame({
            "Mean": grouped.mean(), "Maximum": grouped.max(),
            "Final": grouped.last()}).reset_index()
        summary["Variable"] = column
        rows.append(summary)

    return _simulation_rows(rows, simulation_id)



def _bess_lifetime_config(user):
    """Return common optional long-term BESS lifetime assumptions."""

    bess = user.get("BESS")
    if not isinstance(bess, dict) or not bess:
        return None

    single_keys = {
        "model", "value", "values", "duration", "efficiency",
        "lifetime"}
    if single_keys.intersection(bess):
        configs = [bess]
    else:
        configs = [
            value for value in bess.values()
            if isinstance(value, dict)]

    lifetime_blocks = [
        cfg.get("lifetime") for cfg in configs
        if isinstance(cfg.get("lifetime"), dict)]
    if not lifetime_blocks:
        return None

    first = lifetime_blocks[0]
    calendar = first.get("calendar_years")
    cycle = first.get("cycle_life_efc")
    return {
        "calendar_years": (
            float(calendar) if calendar is not None else np.inf),
        "cycle_life_efc": (
            float(cycle) if cycle is not None else np.inf),}


def _trim_bess_cohorts(cohorts, excess_power):
    """Remove prescribed capacity reductions from the oldest cohorts."""

    remaining = max(float(excess_power), 0.0)
    cohorts.sort(key=lambda item: item["commission_year"])
    for cohort in cohorts:
        if remaining <= 1e-9:
            break
        removed = min(cohort["power_capacity"], remaining)
        cohort["power_capacity"] -= removed
        remaining -= removed
    cohorts[:] = [
        cohort for cohort in cohorts
        if cohort["power_capacity"] > 1e-9]


def _bess_replacement_schedule(annual, lifetime):
    """Estimate annual BESS renewals with a simple cohort model."""

    rows = []
    cohorts = []
    previous_capacity = 0.0
    calendar_life = lifetime["calendar_years"]
    cycle_life = lifetime["cycle_life_efc"]
    cumulative_replacement = 0.0

    for row in annual.itertuples(index=False):
        year = int(row.Year)
        capacity = max(float(row.BESS_Power_Capacity), 0.0)
        delta = capacity - previous_capacity
        if delta > 1e-9:
            cohorts.append({
                "power_capacity": delta,
                "commission_year": year,
                "efc": 0.0,})
        elif delta < -1e-9:
            _trim_bess_cohorts(cohorts, -delta)

        annual_efc = max(float(row.Equivalent_Full_Cycles), 0.0)
        calendar_replacement = 0.0
        cycle_replacement = 0.0
        both_limits_replacement = 0.0

        for cohort in cohorts:
            cohort["efc"] += annual_efc
            calendar_due = (
                np.isfinite(calendar_life)
                and year - cohort["commission_year"] >= calendar_life)
            cycle_due = (
                np.isfinite(cycle_life)
                and cohort["efc"] >= cycle_life)
            if not calendar_due and not cycle_due:
                continue

            capacity_due = cohort["power_capacity"]
            if calendar_due and cycle_due:
                both_limits_replacement += capacity_due
            elif calendar_due:
                calendar_replacement += capacity_due
            else:
                cycle_replacement += capacity_due

            cohort["commission_year"] = year
            cohort["efc"] = 0.0

        total = (
            calendar_replacement + cycle_replacement
            + both_limits_replacement)
        cumulative_replacement += total
        rows.append({
            "Year": year,
            "Calendar_Replacement": calendar_replacement,
            "Cycle_Replacement": cycle_replacement,
            "Both_Limits_Replacement": both_limits_replacement,
            "Total_Replacement": total,
            "Cumulative_Replacement": cumulative_replacement,})
        previous_capacity = capacity

    return pd.DataFrame(rows)


def _bess_lifetime_summary(frame, simulation_id, user):
    """Summarize BESS cycling and optional long-term replacement needs."""

    discharge = _series_by_prefix(frame, "BESS_Energy_Withdrawn")
    delivered = _series_by_prefix(frame, "BESS_Discharge_to_System")
    charge = _series_by_prefix(frame, "BESS_Charge")
    energy_capacity = _series_by_prefix(
        frame, "BESS_Energy_Capacity")
    soc = _series_by_prefix(frame, "BESS_SOC")
    power_capacity = _series_by_prefix(frame, "BESS_Power_Capacity")

    if discharge is None or energy_capacity is None:
        return pd.DataFrame()

    dates = pd.to_datetime(frame["Date"])
    capacity_values = pd.to_numeric(
        energy_capacity, errors="coerce").fillna(0.0)
    discharge_values = pd.to_numeric(
        discharge, errors="coerce").fillna(0.0)
    valid_capacity = capacity_values > 1e-12
    efc_increment = pd.Series(0.0, index=frame.index)
    efc_increment.loc[valid_capacity] = (
        discharge_values.loc[valid_capacity]
        / capacity_values.loc[valid_capacity])

    soc_fraction = pd.Series(np.nan, index=frame.index, dtype=float)
    if soc is not None:
        soc_values = pd.to_numeric(
            soc, errors="coerce").fillna(0.0)
        soc_fraction.loc[valid_capacity] = (
            soc_values.loc[valid_capacity]
            / capacity_values.loc[valid_capacity])

    data = pd.DataFrame({
        "Year": dates.dt.year.to_numpy(),
        "BESS_Power_Capacity": (
            pd.to_numeric(power_capacity, errors="coerce").fillna(0.0)
            if power_capacity is not None else 0.0),
        "BESS_Energy_Capacity": capacity_values,
        "BESS_Charge": (
            pd.to_numeric(charge, errors="coerce").fillna(0.0)
            if charge is not None else 0.0),
        "BESS_Energy_Withdrawn": discharge_values,
        "BESS_Discharge_to_System": (
            pd.to_numeric(delivered, errors="coerce").fillna(0.0)
            if delivered is not None else 0.0),
        "Equivalent_Full_Cycles": efc_increment,
        "SOC_Fraction": soc_fraction,})

    grouped = data.groupby("Year", sort=True)
    annual = pd.DataFrame({
        "BESS_Power_Capacity": grouped["BESS_Power_Capacity"].last(),
        "BESS_Energy_Capacity": grouped[
            "BESS_Energy_Capacity"].last(),
        "BESS_Charge": grouped["BESS_Charge"].sum(),
        "BESS_Energy_Withdrawn": grouped[
            "BESS_Energy_Withdrawn"].sum(),
        "BESS_Discharge_to_System": grouped[
            "BESS_Discharge_to_System"].sum(),
        "Equivalent_Full_Cycles": grouped[
            "Equivalent_Full_Cycles"].sum(),
        "Mean_SOC_Fraction": grouped["SOC_Fraction"].mean(),
    }).reset_index()
    annual["Cumulative_EFC"] = annual[
        "Equivalent_Full_Cycles"].cumsum()

    lifetime = _bess_lifetime_config(user)
    if lifetime is None:
        annual["Calendar_Life_Years"] = np.nan
        annual["Cycle_Life_EFC"] = np.nan
        annual["Cycle_Limited_Life_Years"] = np.nan
        annual["Estimated_Effective_Life_Years"] = np.nan
        annual["Calendar_Replacement"] = np.nan
        annual["Cycle_Replacement"] = np.nan
        annual["Both_Limits_Replacement"] = np.nan
        annual["Total_Replacement"] = np.nan
        annual["Cumulative_Replacement"] = np.nan
    else:
        calendar_life = lifetime["calendar_years"]
        cycle_life = lifetime["cycle_life_efc"]
        annual["Calendar_Life_Years"] = (
            calendar_life if np.isfinite(calendar_life) else np.nan)
        annual["Cycle_Life_EFC"] = (
            cycle_life if np.isfinite(cycle_life) else np.nan)
        annual_efc = annual["Equivalent_Full_Cycles"].to_numpy(float)
        cycle_years = np.divide(
            cycle_life,
            annual_efc,
            out=np.full(len(annual), np.inf),
            where=annual_efc > 1e-12,)
        annual["Cycle_Limited_Life_Years"] = np.where(
            np.isfinite(cycle_years), cycle_years, np.nan)
        effective = np.minimum(calendar_life, cycle_years)
        annual["Estimated_Effective_Life_Years"] = np.where(
            np.isfinite(effective), effective, np.nan)
        replacements = _bess_replacement_schedule(annual, lifetime)
        annual = annual.merge(replacements, on="Year", how="left")

    annual.insert(0, "Simulation", simulation_id)
    annual["Energy_Unit"] = canonical_energy_unit(
        user.get("energy_unit", "MWh"))
    annual["Power_Unit"] = "MW"
    return annual



def _active_period_statistics(active, step_hours):
    """Return count and duration statistics for contiguous active periods."""

    values = np.asarray(active, dtype=bool)
    edges = np.diff(values.astype(np.int8), prepend=0, append=0)
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)
    if starts.size == 0:
        return {
            "Count": 0,
            "Mean_Duration_Hours": 0.0,
            "P95_Duration_Hours": 0.0,
            "Maximum_Duration_Hours": 0.0,}
    durations = (ends - starts).astype(float) * float(step_hours)
    return {
        "Count": int(starts.size),
        "Mean_Duration_Hours": float(durations.mean()),
        "P95_Duration_Hours": float(np.quantile(durations, 0.95)),
        "Maximum_Duration_Hours": float(durations.max()),}


def _bess_operation_summary(frame, simulation_id, user):
    """Summarize annual BESS energy, power, state, and operating periods."""

    charge = _series_by_prefix(frame, "BESS_Charge")
    withdrawn = _series_by_prefix(frame, "BESS_Energy_Withdrawn")
    discharge = _series_by_prefix(frame, "BESS_Discharge_to_System")
    soc = _series_by_prefix(frame, "BESS_SOC")
    energy_capacity = _series_by_prefix(frame, "BESS_Energy_Capacity")
    power_capacity = _series_by_prefix(frame, "BESS_Power_Capacity")
    if charge is None and discharge is None and soc is None:
        return pd.DataFrame()

    dates = pd.to_datetime(frame["Date"], errors="raise")
    step_hours = _time_step_hours(frame)
    threshold = get_output_config(user)["event_threshold"]
    unit = canonical_energy_unit(user.get("energy_unit", "MWh"))
    to_mwh = energy_conversion_factor(unit, "MWh")

    def numeric(series):
        if series is None:
            return np.zeros(len(frame), dtype=float)
        return pd.to_numeric(
            series, errors="coerce").fillna(0.0).to_numpy(dtype=float)

    charge_values = numeric(charge)
    withdrawn_values = numeric(withdrawn)
    discharge_values = numeric(discharge)
    soc_values = numeric(soc)
    capacity_values = numeric(energy_capacity)
    power_values = numeric(power_capacity)
    valid_capacity = capacity_values > threshold
    soc_fraction = np.full(len(frame), np.nan, dtype=float)
    soc_fraction[valid_capacity] = (
        soc_values[valid_capacity] / capacity_values[valid_capacity])
    years = dates.dt.year.to_numpy(dtype=int)
    rows = []

    for year in np.unique(years):
        mask = years == year
        c = charge_values[mask]
        w = withdrawn_values[mask]
        d = discharge_values[mask]
        sf = soc_fraction[mask]
        pc = power_values[mask]
        active_charge = c > threshold
        active_discharge = d > threshold
        charge_events = _active_period_statistics(active_charge, step_hours)
        discharge_events = _active_period_statistics(
            active_discharge, step_hours)
        near_empty = np.isfinite(sf) & (sf <= 0.10)
        near_full = np.isfinite(sf) & (sf >= 0.90)
        empty_events = _active_period_statistics(near_empty, step_hours)
        full_events = _active_period_statistics(near_full, step_hours)
        hours = float(mask.sum()) * step_hours
        peak_charge = float(c.max(initial=0.0) * to_mwh / step_hours)
        peak_discharge = float(d.max(initial=0.0) * to_mwh / step_hours)
        power_limit_energy = pc * step_hours / to_mwh
        charge_limited = (
            active_charge & (pc > threshold)
            & (c >= 0.999 * power_limit_energy))
        withdrawn_year = w if w.size else np.zeros_like(c)
        discharge_limited = (
            withdrawn_year > threshold) & (pc > threshold)
        discharge_limited &= (
            withdrawn_year >= 0.999 * power_limit_energy)
        efc = np.divide(
            w, capacity_values[mask],
            out=np.zeros_like(w),
            where=capacity_values[mask] > threshold).sum()
        finite_sf = sf[np.isfinite(sf)]
        rows.append({
            "Simulation": int(simulation_id),
            "Year": int(year),
            "Charging_Energy": float(c.sum()),
            "Discharged_Energy": float(d.sum()),
            "Battery_Energy_Withdrawn": float(w.sum()),
            "Energy_Throughput": float(c.sum() + w.sum()),
            "Equivalent_Full_Cycles": float(efc),
            "Peak_Charging_Power": peak_charge,
            "Peak_Discharge_Power": peak_discharge,
            "Charging_Hours": float(active_charge.sum()) * step_hours,
            "Discharging_Hours": float(active_discharge.sum()) * step_hours,
            "Idle_Hours": float(
                (~(active_charge | active_discharge)).sum()) * step_hours,
            "Charging_Events": charge_events["Count"],
            "Mean_Charging_Event_Hours": charge_events[
                "Mean_Duration_Hours"],
            "Maximum_Charging_Event_Hours": charge_events[
                "Maximum_Duration_Hours"],
            "Discharging_Events": discharge_events["Count"],
            "Mean_Discharging_Event_Hours": discharge_events[
                "Mean_Duration_Hours"],
            "Maximum_Discharging_Event_Hours": discharge_events[
                "Maximum_Duration_Hours"],
            "Mean_SOC_Fraction": (
                float(finite_sf.mean()) if finite_sf.size else np.nan),
            "Minimum_SOC_Fraction": (
                float(finite_sf.min()) if finite_sf.size else np.nan),
            "Maximum_SOC_Fraction": (
                float(finite_sf.max()) if finite_sf.size else np.nan),
            "Hours_SOC_At_or_Below_10pct": float(
                near_empty.sum()) * step_hours,
            "Hours_SOC_At_or_Above_90pct": float(
                near_full.sum()) * step_hours,
            "Longest_SOC_At_or_Below_10pct_Hours": empty_events[
                "Maximum_Duration_Hours"],
            "Longest_SOC_At_or_Above_90pct_Hours": full_events[
                "Maximum_Duration_Hours"],
            "Hours_At_Charge_Power_Limit": float(
                charge_limited.sum()) * step_hours,
            "Hours_At_Discharge_Power_Limit": float(
                discharge_limited.sum()) * step_hours,
            "Hours": hours,
            "Energy_Unit": unit,
            "Power_Unit": "MW",})
    return pd.DataFrame(rows)


def _bess_soc_distribution(frame, simulation_id, user):
    """Return annual BESS state-of-charge occupancy in ten percent bins."""

    soc = _series_by_prefix(frame, "BESS_SOC")
    capacity = _series_by_prefix(frame, "BESS_Energy_Capacity")
    if soc is None or capacity is None:
        return pd.DataFrame()
    dates = pd.to_datetime(frame["Date"], errors="raise")
    step_hours = _time_step_hours(frame)
    threshold = get_output_config(user)["event_threshold"]
    soc_values = pd.to_numeric(
        soc, errors="coerce").fillna(0.0).to_numpy(dtype=float)
    capacity_values = pd.to_numeric(
        capacity, errors="coerce").fillna(0.0).to_numpy(dtype=float)
    valid = capacity_values > threshold
    fraction = np.full(len(frame), np.nan, dtype=float)
    fraction[valid] = np.clip(
        soc_values[valid] / capacity_values[valid], 0.0, 1.0)
    years = dates.dt.year.to_numpy(dtype=int)
    edges = np.linspace(0.0, 1.0, 11)
    rows = []
    for year in np.unique(years):
        values = fraction[(years == year) & np.isfinite(fraction)]
        counts, _ = np.histogram(values, bins=edges)
        total = counts.sum()
        for index, count in enumerate(counts):
            rows.append({
                "Simulation": int(simulation_id),
                "Year": int(year),
                "SOC_Bin_Lower": float(edges[index]),
                "SOC_Bin_Upper": float(edges[index + 1]),
                "Hours": float(count) * step_hours,
                "Fraction_of_Available_Hours": (
                    float(count / total) if total else 0.0),})
    return pd.DataFrame(rows)


def _bess_temporal_profiles(frame, simulation_id, user):
    """Return BESS hour-of-day and month profiles for compact analysis."""

    charge = _series_by_prefix(frame, "BESS_Charge")
    discharge = _series_by_prefix(frame, "BESS_Discharge_to_System")
    soc = _series_by_prefix(frame, "BESS_SOC")
    capacity = _series_by_prefix(frame, "BESS_Energy_Capacity")
    if charge is None and discharge is None and soc is None:
        return pd.DataFrame(), pd.DataFrame()
    dates = pd.to_datetime(frame["Date"], errors="raise")
    threshold = get_output_config(user)["event_threshold"]

    def series_or_zero(value):
        if value is None:
            return pd.Series(0.0, index=frame.index)
        return pd.to_numeric(value, errors="coerce").fillna(0.0)

    charge_values = series_or_zero(charge)
    discharge_values = series_or_zero(discharge)
    soc_values = series_or_zero(soc)
    capacity_values = series_or_zero(capacity)
    soc_fraction = pd.Series(np.nan, index=frame.index, dtype=float)
    valid = capacity_values > threshold
    soc_fraction.loc[valid] = (
        soc_values.loc[valid] / capacity_values.loc[valid])
    data = pd.DataFrame({
        "Year": dates.dt.year.to_numpy(),
        "Month": dates.dt.month.to_numpy(),
        "Hour": dates.dt.hour.to_numpy(),
        "Charge": charge_values.to_numpy(),
        "Discharge": discharge_values.to_numpy(),
        "SOC_Fraction": soc_fraction.to_numpy(),})
    data["Charging"] = data["Charge"] > threshold
    data["Discharging"] = data["Discharge"] > threshold

    hourly = data.groupby(["Year", "Hour"], as_index=False).agg(
        Mean_Charge=("Charge", "mean"),
        Mean_Discharge=("Discharge", "mean"),
        Mean_SOC_Fraction=("SOC_Fraction", "mean"),
        Charging_Hours_Fraction=("Charging", "mean"),
        Discharging_Hours_Fraction=("Discharging", "mean"),)
    hourly.insert(0, "Simulation", int(simulation_id))

    monthly = data.groupby(["Year", "Month"], as_index=False).agg(
        Charging_Energy=("Charge", "sum"),
        Discharged_Energy=("Discharge", "sum"),
        Mean_SOC_Fraction=("SOC_Fraction", "mean"),
        Charging_Hours_Fraction=("Charging", "mean"),
        Discharging_Hours_Fraction=("Discharging", "mean"),)
    monthly.insert(0, "Simulation", int(simulation_id))
    return hourly, monthly

def _commodity_quantity_unit(user, commodity):
    """Return the configured physical quantity unit for a commodity."""

    units = user.get("_commodity_quantity_units", {}) or {}
    return normalize_quantity_unit(units.get(commodity, "kg"))


def _commodity_summary(frame, simulation_id, user):
    """Aggregate commodity production, use, sales, and inventories."""

    years = frame["Date"].dt.year.to_numpy()
    commodities = list(user.get("Commodities_Production", {}).keys())
    rows = []

    for commodity in commodities:
        variables = {
            "Produced": commodity,
            "Stored": f"Stored_{commodity}",
            "Sold": f"Sold_{commodity}",
            "Expired": f"Expired_{commodity}",
            "Burned": f"Burned_{commodity}",
            "Inventory": f"Inventory_{commodity}"}

        for metric, prefix in variables.items():
            column = _column_by_prefix(frame, prefix)
            if column is None:
                continue

            grouped = _numeric_series_by_year(frame[column], years)
            values = grouped.last() if metric == "Inventory" else grouped.sum()
            result = values.reset_index(name="Value")
            result["Commodity"] = commodity
            result["Quantity_Unit"] = _commodity_quantity_unit(
                user, commodity)
            result["Metric"] = metric
            rows.append(result)

    return _simulation_rows(rows, simulation_id)


def _commodity_capacity_summary(frame, simulation_id, user):
    """Summarize annual commodity production peaks and operating periods."""

    dates = pd.to_datetime(frame["Date"], errors="raise")
    years = dates.dt.year.to_numpy(dtype=int)
    step_hours = _time_step_hours(frame)
    threshold = get_output_config(user)["event_threshold"]
    rows = []

    for commodity in user.get("Commodities_Production", {}):
        column = _column_by_prefix(frame, commodity)
        if column is None:
            continue
        values = pd.to_numeric(
            frame[column], errors="coerce").fillna(0.0).to_numpy(float)
        for year in np.unique(years):
            mask = years == year
            amount = values[mask]
            rate = amount / step_hours
            active = amount > threshold
            events = _active_period_statistics(active, step_hours)
            active_rate = rate[active]
            year_dates = dates[mask]
            peak_index = int(np.argmax(rate)) if rate.size else 0
            peak_time = year_dates.iloc[peak_index] if rate.size else pd.NaT
            rows.append({
                "Simulation": int(simulation_id),
                "Year": int(year),
                "Commodity": commodity,
                "Quantity_Unit": _commodity_quantity_unit(user, commodity),
                "Annual_Production": float(amount.sum()),
                "Mean_Production_Rate": float(rate.mean()),
                "Mean_Rate_When_Active": (
                    float(active_rate.mean()) if active_rate.size else 0.0),
                "Maximum_Production_Rate": (
                    float(rate.max()) if rate.size else 0.0),
                "P95_Production_Rate": (
                    float(np.quantile(rate, 0.95)) if rate.size else 0.0),
                "P99_Production_Rate": (
                    float(np.quantile(rate, 0.99)) if rate.size else 0.0),
                "Peak_Production_Time": peak_time,
                "Peak_Production_Month": (
                    int(peak_time.month) if pd.notna(peak_time) else np.nan),
                "Peak_Production_Hour": (
                    int(peak_time.hour) if pd.notna(peak_time) else np.nan),
                "Active_Hours": float(active.sum()) * step_hours,
                "Active_Hours_Fraction": (
                    float(active.mean()) if active.size else 0.0),
                "Operating_Periods": events["Count"],
                "Mean_Operating_Period_Hours": events[
                    "Mean_Duration_Hours"],
                "P95_Operating_Period_Hours": events[
                    "P95_Duration_Hours"],
                "Maximum_Operating_Period_Hours": events[
                    "Maximum_Duration_Hours"],})
    return pd.DataFrame(rows)


def _monthly_commodity_profile(frame, simulation_id, user):
    """Aggregate commodity production by calendar month."""

    dates = pd.to_datetime(frame["Date"], errors="raise")
    threshold = get_output_config(user)["event_threshold"]
    rows = []
    for commodity in user.get("Commodities_Production", {}):
        column = _column_by_prefix(frame, commodity)
        if column is None:
            continue
        values = pd.to_numeric(
            frame[column], errors="coerce").fillna(0.0)
        data = pd.DataFrame({
            "Year": dates.dt.year.to_numpy(),
            "Month": dates.dt.month.to_numpy(),
            "Production": values.to_numpy(),})
        data["Active"] = data["Production"] > threshold
        summary = data.groupby(
            ["Year", "Month"], as_index=False).agg(
                Production=("Production", "sum"),
                Mean_Production=("Production", "mean"),
                Maximum_Production=("Production", "max"),
                Active_Hours_Fraction=("Active", "mean"),)
        summary.insert(
            0, "Quantity_Unit", _commodity_quantity_unit(user, commodity))
        summary.insert(0, "Commodity", commodity)
        summary.insert(0, "Simulation", int(simulation_id))
        rows.append(summary)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)

def _hourly_commodity_profile(frame, simulation_id, user):
    """Aggregate commodity activity by hour of day."""

    dates = pd.to_datetime(frame["Date"])
    threshold = get_output_config(user)["event_threshold"]
    rows = []

    for commodity in user.get("Commodities_Production", {}):
        column = _column_by_prefix(frame, commodity)

        if column is None:
            continue

        production = pd.to_numeric(
            frame[column], errors="coerce").fillna(0.0)
        data = pd.DataFrame({
            "Year": dates.dt.year,
            "Hour": dates.dt.hour,
            "Production": production})

        grouped = data.groupby(["Year", "Hour"])["Production"]
        summary = grouped.agg(
            Mean_Hourly_Production="mean",
            Total_Production="sum",
            Active_Hours_Fraction=lambda values: (
                values.gt(threshold).mean())).reset_index()

        annual_total = summary.groupby(
            "Year")["Total_Production"].transform("sum")
        summary["Share_of_Annual_Production"] = np.where(
            annual_total > 0.0,
            summary["Total_Production"] / annual_total,
            0.0)
        summary = summary.drop(columns="Total_Production")
        summary.insert(0, "Quantity_Unit",
                       _commodity_quantity_unit(user, commodity))
        summary.insert(0, "Commodity", commodity)
        summary.insert(0, "Simulation", simulation_id)
        rows.append(summary)

    if not rows:
        return pd.DataFrame()

    return pd.concat(rows, ignore_index=True)


def _emissions_summary(frame, simulation_id, frequency):
    """Aggregate modeled emissions and carbon flows by calendar period."""

    dates = frame["Date"]
    sum_columns = [
        column for column in frame.columns
        if str(column).startswith("Emissions_")
        or column in {
            "Electricity_Emissions (kgCO2e)",
            "Atmospheric_CO2_Gross_Capture (kg)",
            "CO2_Released_from_CH4_Synthesis (kg)",
            "Atmospheric_CO2_Captured (kg)",
            "CO2_Released_from_CH4 (kg)",
            "CO2_in_Sold_CH4 (kg)",
            "CO2_in_Expired_CH4 (kg)",
            "Retained_Atmospheric_CO2 (kg)",
            "Gross_Emissions (kgCO2e)",
            "Net_Emissions (kgCO2e)"}]
    inventory_column = "CO2_in_CH4_Inventory (kg)"

    if not sum_columns and inventory_column not in frame.columns:
        return pd.DataFrame()

    work = pd.DataFrame({
        column: pd.to_numeric(
            frame[column], errors="coerce").fillna(0.0)
        for column in sum_columns})
    group_cols = _add_calendar_groups(work, dates, frequency)

    if sum_columns:
        result = work.groupby(
            group_cols, as_index=False)[sum_columns].sum()
    else:
        result = work[group_cols].drop_duplicates().reset_index(drop=True)

    if inventory_column in frame.columns:
        inventory_data = pd.DataFrame({
            **{column: work[column] for column in group_cols},
            inventory_column: pd.to_numeric(
                frame[inventory_column], errors="coerce").fillna(0.0)})
        inventory = inventory_data.groupby(
            group_cols, as_index=False)[inventory_column].last()
        result = result.merge(inventory, on=group_cols, how="left")

    result.insert(0, "Simulation", simulation_id)
    return result

def _technology_series(frame, generation, user):
    """Collect technology-use time series available in model outputs."""

    technologies = {}

    for source, source_input in user.get("sources", {}).items():
        if source not in generation.columns:
            continue

        operation = source_input.get("hourly_operation")
        category = "Generation_Source"

        if operation in {
            "profile_preserving", "load_following", "dispatchable"
        }:
            category = "Balancing_Source"

        values = pd.to_numeric(
            generation[source], errors="coerce").fillna(0.0)
        technologies[(category, source)] = values

    commodities = user.get("Commodities_Production", {})

    for commodity in commodities:
        column = _column_by_prefix(
            frame, f"Burned_{commodity}")

        if column is None:
            continue

        values = pd.to_numeric(
            frame[column], errors="coerce").fillna(0.0)
        technologies[
            ("Fuel_Reconversion", commodity)
        ] = values

    generic_variables = {
        ("Storage", "BESS_Charge"): "BESS_Charge",
        ("Storage", "BESS_Discharge_to_System"): "BESS_Discharge_to_System",
        ("Interconnection", "Imports"): "Imports",
        ("Interconnection", "Exports"): "Exports",
        ("System_Condition", "Residual_Supply_Requirement"):
            "Residual_Supply_Requirement"}

    for key, prefix in generic_variables.items():
        values = _series_by_prefix(frame, prefix)

        if values is not None:
            technologies[key] = values

    final = _series_by_prefix(
        frame, "Final_Electricity_Balance")

    if final is not None:
        technologies[
            ("System_Condition", "Remaining_Surplus_Electricity")
        ] = _positive(final, len(frame))
        technologies[
            ("System_Condition", "Residual_Supply_Requirement")
        ] = _negative(final, len(frame))

    return technologies


def _technology_group_metrics(group, active_group):
    """Return shared activity and output metrics for one technology group."""

    return {
        "Observations": len(group),
        "Active_Observations": int(group["Active"].sum()),
        "Total_Output": group["Value"].sum(),
        "Mean_Output": group["Value"].mean(),
        "Mean_When_Active": _active_mean(active_group),
        "Peak_Output": group["Value"].max()}


def _annual_technology_use(
        frame, generation, simulation_id, user):
    """Summarize annual technology use, activity, and peak requirements."""

    threshold = get_output_config(user)["event_threshold"]
    dates = pd.to_datetime(frame["Date"])
    years = dates.dt.year.to_numpy()
    days = dates.dt.normalize().to_numpy()
    technologies = _technology_series(frame, generation, user)
    rows = []

    for (category, technology), series in technologies.items():
        values = pd.to_numeric(
            series, errors="coerce").fillna(0.0).to_numpy()
        active = values > threshold
        data = pd.DataFrame({
            "Year": years, "Day": days, "Value": values,
            "Active": active})
        grouped = data.groupby("Year", sort=True)
        summary = grouped.agg(
            Observations=("Value", "size"),
            Active_Observations=("Active", "sum"),
            Total_Output=("Value", "sum"),
            Mean_Output=("Value", "mean"),
            Peak_Output=("Value", "max")).reset_index()
        active_mean = data["Value"].where(data["Active"]).groupby(
            data["Year"]).mean().fillna(0.0)
        active_days = data.loc[data["Active"]].groupby(
            "Year")["Day"].nunique()
        summary["Mean_When_Active"] = summary["Year"].map(
            active_mean).fillna(0.0)
        summary["Active_Days"] = summary["Year"].map(
            active_days).fillna(0).astype(int)
        summary.insert(0, "Technology", technology)
        summary.insert(0, "Category", category)
        summary.insert(0, "Simulation", simulation_id)
        rows.append(summary)

    if not rows:
        return pd.DataFrame()
    columns = [
        "Simulation", "Year", "Category", "Technology",
        "Active_Days", "Observations", "Active_Observations",
        "Total_Output", "Mean_Output", "Mean_When_Active",
        "Peak_Output"]
    return pd.concat(rows, ignore_index=True)[columns]


def _hourly_technology_use(
        frame, generation, simulation_id, user):
    """Summarize technology use by hour of day."""

    threshold = get_output_config(user)["event_threshold"]
    dates = pd.to_datetime(frame["Date"])
    group_data = pd.DataFrame({
        "Year": dates.dt.year.to_numpy(),
        "Month": dates.dt.month.to_numpy(),
        "Hour": dates.dt.hour.to_numpy()})
    group_cols = ["Year", "Month", "Hour"]
    technologies = _technology_series(frame, generation, user)
    rows = []

    for (category, technology), series in technologies.items():
        data = group_data.copy()
        data["Value"] = pd.to_numeric(
            series, errors="coerce").fillna(0.0).to_numpy()
        data["Active"] = data["Value"] > threshold
        grouped = data.groupby(group_cols, sort=True)
        summary = grouped.agg(
            Observations=("Value", "size"),
            Active_Observations=("Active", "sum"),
            Total_Output=("Value", "sum"),
            Mean_Output=("Value", "mean"),
            Peak_Output=("Value", "max")).reset_index()
        active_mean = data["Value"].where(data["Active"]).groupby(
            [data[column] for column in group_cols]).mean()
        active_mean = active_mean.rename("Mean_When_Active").reset_index()
        summary = summary.merge(active_mean, on=group_cols, how="left")
        summary["Mean_When_Active"] = summary[
            "Mean_When_Active"].fillna(0.0)
        summary.insert(0, "Technology", technology)
        summary.insert(0, "Category", category)
        summary.insert(0, "Simulation", simulation_id)
        rows.append(summary)

    if not rows:
        return pd.DataFrame()
    columns = [
        "Simulation", "Year", "Month", "Hour", "Category",
        "Technology", "Observations", "Active_Observations",
        "Total_Output", "Mean_Output", "Mean_When_Active",
        "Peak_Output"]
    return pd.concat(rows, ignore_index=True)[columns]


def _technology_active_inputs(frame, generation, user):
    """Yield normalized non-generation technology-use series."""

    threshold = get_output_config(user)["event_threshold"]
    for (category, technology), series in _technology_series(
            frame, generation, user).items():
        if category == "Generation_Source":
            continue
        values = pd.to_numeric(
            series, errors="coerce").fillna(0.0).reset_index(drop=True)
        active = values > threshold
        yield category, technology, values, active


def _technology_event_inputs(frame, generation, user):
    """Yield technology-use series with groups for detailed event tables."""

    for category, technology, values, active in _technology_active_inputs(
            frame, generation, user):
        groups = active.ne(active.shift(fill_value=False)).cumsum()
        yield category, technology, values, active, groups


def _technology_use_events(
        frame, generation, simulation_id, user):
    """Identify contiguous periods of positive technology use."""

    dates = pd.to_datetime(frame["Date"]).reset_index(drop=True)
    step_hours = _time_step_hours(frame)
    rows = []

    for category, technology, values, active, groups in (
            _technology_event_inputs(frame, generation, user)):
        if not active.any():
            continue
        data = pd.DataFrame({
            "Date": dates.loc[active].to_numpy(),
            "Value": values.loc[active].to_numpy(),
            "Event": groups.loc[active].to_numpy()})
        events = data.groupby("Event", sort=True).agg(
            Start=("Date", "first"), End=("Date", "last"),
            Duration_Hours=("Value", "size"),
            Total_Output=("Value", "sum"),
            Peak_Output=("Value", "max"),
            Mean_Output=("Value", "mean")).reset_index(drop=True)
        events.insert(0, "Event", np.arange(1, len(events) + 1))
        events.insert(0, "Technology", technology)
        events.insert(0, "Category", category)
        events.insert(0, "Simulation", simulation_id)
        events["Duration_Hours"] = (
            events["Duration_Hours"].astype(float) * step_hours)
        events["Start_Year"] = events["Start"].dt.year
        events["Start_Month"] = events["Start"].dt.month
        events["Start_Hour"] = events["Start"].dt.hour
        rows.append(events)

    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def _technology_event_summary(
        frame, generation, simulation_id, user):
    """Summarize technology-use events with bounded NumPy operations."""

    dates = pd.to_datetime(frame["Date"], errors="coerce")
    year_values = dates.dt.year.to_numpy(dtype=np.int32, copy=False)
    years = np.unique(year_values)
    year_edges = np.flatnonzero(
        np.r_[True, year_values[1:] != year_values[:-1], True])
    step_hours = _time_step_hours(frame)
    rows = []

    for category, technology, values, active in _technology_active_inputs(
            frame, generation, user):
        numeric = values.to_numpy(dtype=float, copy=False)
        active_values = active.to_numpy(dtype=bool, copy=False)

        for left, right in zip(year_edges[:-1], year_edges[1:]):
            year = int(year_values[left])
            subset_active = active_values[left:right]
            subset_values = numeric[left:right]
            edges = np.diff(
                subset_active.astype(np.int8), prepend=0, append=0)
            starts = np.flatnonzero(edges == 1)
            ends = np.flatnonzero(edges == -1)

            if starts.size == 0:
                rows.append({
                    "Simulation": simulation_id,
                    "Year": year,
                    "Category": category,
                    "Technology": technology,
                    "Event_Count": 0,
                    "Active_Hours": 0.0,
                    "Total_Output": 0.0,
                    "Mean_Duration_Hours": 0.0,
                    "Maximum_Duration_Hours": 0.0,
                    "P95_Duration_Hours": 0.0,
                    "Mean_Event_Output": 0.0,
                    "Maximum_Event_Output": 0.0,
                    "Maximum_Peak_Output": 0.0,})
                continue

            lengths = ends - starts
            durations = lengths.astype(float) * step_hours
            cumulative = np.concatenate((
                [0.0], np.cumsum(subset_values, dtype=float)))
            event_output = cumulative[ends] - cumulative[starts]
            peak_output = np.maximum.reduceat(subset_values, starts)
            rows.append({
                "Simulation": simulation_id,
                "Year": year,
                "Category": category,
                "Technology": technology,
                "Event_Count": int(starts.size),
                "Active_Hours": float(durations.sum()),
                "Total_Output": float(event_output.sum()),
                "Mean_Duration_Hours": float(durations.mean()),
                "Maximum_Duration_Hours": float(durations.max()),
                "P95_Duration_Hours": float(np.quantile(durations, 0.95)),
                "Mean_Event_Output": float(event_output.mean()),
                "Maximum_Event_Output": float(event_output.max()),
                "Maximum_Peak_Output": float(peak_output.max()),})

    columns = [
        "Simulation", "Year", "Category", "Technology",
        "Event_Count", "Active_Hours", "Total_Output",
        "Mean_Duration_Hours", "Maximum_Duration_Hours",
        "P95_Duration_Hours", "Mean_Event_Output",
        "Maximum_Event_Output", "Maximum_Peak_Output"]
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows, columns=columns)


def _active_mean(active_group):
    """Return the mean of positive finite values in one group."""

    if active_group.empty:
        return 0.0

    return active_group["Value"].mean()


def _event_statistics(frame, simulation_id, event_type, user=None):
    """Return compact annual statistics for deficit or surplus events."""

    dates = pd.to_datetime(frame["Date"], errors="coerce")
    step_hours = _time_step_hours(frame)
    threshold = get_output_config(user or {})["event_threshold"]
    if event_type == "initial_deficit":
        balance = _series_by_prefix(frame, "Raw_Electricity_Balance")
        series = _negative(balance, len(frame))
    elif event_type == "deficit":
        series = _series_by_prefix(frame, "Residual_Supply_Requirement")
        if series is None:
            balance = _final_balance_series(frame)
            series = _negative(balance, len(frame))
    else:
        balance = _final_balance_series(frame)
        series = _positive(balance, len(frame))
    if series is None:
        return pd.DataFrame()

    values = pd.to_numeric(
        series, errors="coerce").fillna(0.0).to_numpy(dtype=float)
    years = dates.dt.year.to_numpy(dtype=np.int32, copy=False)
    year_edges = np.flatnonzero(
        np.r_[True, years[1:] != years[:-1], True])
    rows = []

    for left, right in zip(year_edges[:-1], year_edges[1:]):
        year = int(years[left])
        year_values = values[left:right]
        active = year_values > threshold
        edges = np.diff(active.astype(np.int8), prepend=0, append=0)
        starts = np.flatnonzero(edges == 1)
        ends = np.flatnonzero(edges == -1)

        if starts.size == 0:
            rows.append({
                "Simulation": simulation_id, "Year": year,
                "Event_Count": 0, "Active_Hours": 0.0,
                "Total_Energy": 0.0, "Mean_Duration_Hours": 0.0,
                "Maximum_Duration_Hours": 0.0,
                "P95_Duration_Hours": 0.0,
                "Mean_Event_Energy": 0.0,
                "Maximum_Event_Energy": 0.0,
                "Maximum_Peak_Power": 0.0})
            continue

        lengths = ends - starts
        durations = lengths.astype(float) * step_hours
        cumulative = np.concatenate(
            ([0.0], np.cumsum(year_values, dtype=float)))
        energy = cumulative[ends] - cumulative[starts]
        peak = np.maximum.reduceat(year_values, starts)
        rows.append({
            "Simulation": simulation_id, "Year": year,
            "Event_Count": int(starts.size),
            "Active_Hours": float(durations.sum()),
            "Total_Energy": float(energy.sum()),
            "Mean_Duration_Hours": float(durations.mean()),
            "Maximum_Duration_Hours": float(durations.max()),
            "P95_Duration_Hours": float(np.quantile(durations, 0.95)),
            "Mean_Event_Energy": float(energy.mean()),
            "Maximum_Event_Energy": float(energy.max()),
            "Maximum_Peak_Power": float(peak.max() / step_hours)})

    return pd.DataFrame(rows)


def _event_summary(frame, simulation_id, event_type, user=None):
    """Return events split by calendar year and measured in real time."""

    dates = pd.to_datetime(frame["Date"], errors="coerce")
    step_hours = _time_step_hours(frame)
    threshold = get_output_config(user or {})["event_threshold"]
    if event_type == "deficit":
        series = _series_by_prefix(frame, "Residual_Supply_Requirement")
        if series is None:
            balance = _final_balance_series(frame)
            series = _negative(balance, len(frame))
    else:
        balance = _final_balance_series(frame)
        series = _positive(balance, len(frame))
    if series is None:
        return pd.DataFrame()

    values = pd.to_numeric(
        series, errors="coerce").fillna(0.0).to_numpy(dtype=float)
    date_values = dates.to_numpy(copy=False)
    years = dates.dt.year.to_numpy(dtype=np.int32, copy=False)
    year_edges = np.flatnonzero(
        np.r_[True, years[1:] != years[:-1], True])
    frames = []
    event_offset = 0

    for left, right in zip(year_edges[:-1], year_edges[1:]):
        year = int(years[left])
        year_values = values[left:right]
        active = year_values > threshold
        edges = np.diff(active.astype(np.int8), prepend=0, append=0)
        starts = np.flatnonzero(edges == 1)
        ends = np.flatnonzero(edges == -1)

        if starts.size == 0:
            frames.append(pd.DataFrame({
                "Simulation": [simulation_id],
                "Event": [0],
                "Is_Event": [False],
                "Start": [pd.NaT],
                "End": [pd.NaT],
                "Duration_Hours": [0.0],
                "Energy": [0.0],
                "Peak_Power": [0.0],
                "Mean_Power": [0.0],
                "Start_Year": [year]}))
            continue

        lengths = ends - starts
        cumulative = np.concatenate(
            ([0.0], np.cumsum(year_values, dtype=float)))
        energy = cumulative[ends] - cumulative[starts]
        peak = np.maximum.reduceat(year_values, starts)
        count = starts.size
        event_ids = np.arange(
            event_offset + 1, event_offset + count + 1, dtype=np.int64)
        event_offset += count
        global_starts = left + starts
        global_ends = left + ends - 1

        frames.append(pd.DataFrame({
            "Simulation": np.full(count, simulation_id, dtype=np.int64),
            "Event": event_ids,
            "Is_Event": np.ones(count, dtype=bool),
            "Start": date_values[global_starts],
            "End": date_values[global_ends],
            "Duration_Hours": lengths.astype(float) * step_hours,
            "Energy": energy,
            "Peak_Power": peak / step_hours,
            "Mean_Power": energy / (lengths.astype(float) * step_hours),
            "Start_Year": np.full(count, year, dtype=np.int32)}))

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)

def _residual_load_energy_series(frame, stage):
    """Return signed residual-load energy for one balance stage.

    Positive values indicate demand exceeding represented supply; negative
    values indicate surplus. The series is derived from the electricity
    balance already produced by the simulation and does not alter dispatch.
    """

    if stage == "Initial":
        balance = _series_by_prefix(frame, "Raw_Electricity_Balance")
    elif stage == "Remaining":
        balance = _final_balance_series(frame)
    else:
        raise ValueError(f"Unknown residual-load stage: {stage}")
    if balance is None:
        return None
    values = pd.to_numeric(balance, errors="coerce")
    return -values


def _average_power(series, step_hours):
    """Convert interval energy to average power over the represented step."""

    if series is None:
        return None
    values = pd.to_numeric(series, errors="coerce")
    return values / float(step_hours)


def _duration_curve_summary(frame, simulation_id):
    """Build initial and remaining residual-load duration-curve points."""

    step_hours = _time_step_hours(frame)
    probabilities = np.array(
        [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99])
    rows = []
    for stage in ("Initial", "Remaining"):
        residual_energy = _residual_load_energy_series(frame, stage)
        residual_power = _average_power(residual_energy, step_hours)
        if residual_power is None:
            continue
        values = pd.to_numeric(
            residual_power, errors="coerce").dropna().to_numpy(
                dtype=float, copy=False)
        if values.size == 0:
            continue
        quantiles = np.quantile(values, 1.0 - probabilities)
        rows.extend({
            "Simulation": simulation_id,
            "Stage": stage,
            "Exceedance": float(probability),
            "Residual_Load": float(value),
            "Power_Unit": "MW"}
            for probability, value in zip(probabilities, quantiles))
    return pd.DataFrame(rows)


def _ramp_summary(frame, generation, simulation_id, user):
    """Return consecutive-step ramp statistics in average-power units.

    Ramps are calculated from average power over each represented interval.
    For hourly simulations this is the conventional one-hour change in MW/h.
    P95, P99, and the maximum of the absolute ramp magnitude characterize
    progressively more extreme short-term changes without changing the
    underlying simulation.
    """

    step_hours = _time_step_hours(frame)
    variables = {
        "Demand": _series_or_none(frame, "Demand"),
        "Generation": _series_or_none(frame, "Total"),
        "Initial_Residual_Load": _residual_load_energy_series(
            frame, "Initial"),
        "Remaining_Residual_Load": _residual_load_energy_series(
            frame, "Remaining")}
    for source in user.get("sources", {}):
        if source in generation.columns:
            variables[source] = generation[source]

    rows = []
    for name, series in variables.items():
        average_power = _average_power(series, step_hours)
        if average_power is None:
            continue
        values = pd.to_numeric(
            average_power, errors="coerce").to_numpy(
                dtype=float, copy=False)
        if values.size < 2:
            continue
        valid = np.isfinite(values[:-1]) & np.isfinite(values[1:])
        if not valid.any():
            continue
        ramps = (values[1:] - values[:-1])[valid] / step_hours
        magnitude = np.abs(ramps)
        rows.append({
            "Simulation": simulation_id,
            "Variable": name,
            "Time_Step_Hours": step_hours,
            "Maximum_Up_Ramp_Rate": float(ramps.max()),
            "Maximum_Down_Ramp_Rate": float(ramps.min()),
            "P95_Up_Ramp_Rate": float(np.quantile(ramps, 0.95)),
            "P05_Down_Ramp_Rate": float(np.quantile(ramps, 0.05)),
            "P95_Ramp_Magnitude": float(np.quantile(magnitude, 0.95)),
            "P99_Ramp_Magnitude": float(np.quantile(magnitude, 0.99)),
            "Maximum_Ramp_Magnitude": float(magnitude.max()),
            "Mean_Ramp_Magnitude": float(magnitude.mean()),
            "Power_Unit": "MW",
            "Rate_Time_Unit": "h"})
    return pd.DataFrame(rows)


def _parquet_engine_available():
    """Return True when the optional parquet engine is installed."""
    try:
        import pyarrow  # noqa: F401
    except ImportError:
        return False
    return True


def _temporary_suffix():
    """Use parquet when available and fast pickle otherwise."""
    if _parquet_engine_available():
        return ".parquet"
    return ".pkl"


def _temporary_name(file_name):
    """Return the temporary table name for the active backend."""
    return f"{Path(file_name).stem}{_temporary_suffix()}"


def _read_temporary_table(path, columns=None):
    """Read a temporary analysis table using its file suffix."""
    path = Path(path)
    if path.suffix == ".parquet":
        return pd.read_parquet(path, columns=columns)
    if path.suffix == ".pkl":
        frame = pd.read_pickle(path)
        return frame if columns is None else frame.loc[:, columns]
    return pd.read_csv(path, usecols=columns)


def _table_folder(file_name):
    """Route each analysis table to its output-detail folder."""

    profile_tables = {
        "Daily_Publication_Profile.csv",
        "Temporal_Profile_Summary.csv",
        "Commodity_Hourly_Profile.csv",
        "Commodity_Monthly_Profile.csv",
        "BESS_SOC_Distribution.csv",
        "BESS_Hourly_Profile.csv",
        "BESS_Monthly_Profile.csv"}
    full_tables = {
        "Hourly_Technology_Profile.csv",
        "Technology_Use_Events.csv", "Remaining_Positive_Residual_Load_Episodes.csv",
        "Surplus_Episodes.csv"}
    if file_name in profile_tables:
        return PROFILES_FOLDER
    if file_name in full_tables:
        return FULL_FOLDER
    return SUMMARY_FOLDER


def _read_final_output(base_path):
    """Read the existing final analysis table in any supported format."""

    base_path = Path(base_path)
    candidates = [
        base_path.with_suffix(".xlsx"),
        base_path.with_suffix(".csv"),
        base_path.with_suffix(".parquet")]

    for path in candidates:
        if not path.is_file():
            continue
        if path.suffix == ".xlsx":
            return pd.read_excel(path)
        if path.suffix == ".csv":
            return pd.read_csv(path)
        return pd.read_parquet(path)

    return None


def _remove_alternative_outputs(base_path, keep_suffix):
    """Remove obsolete copies of a table stored in other formats."""

    base_path = Path(base_path)

    for suffix in (".xlsx", ".csv", ".parquet"):
        path = base_path.with_suffix(suffix)
        if suffix != keep_suffix and path.is_file():
            path.unlink()


def _write_final_output(base_path, frame):
    """Write one final table without routing large cell grids to Excel."""

    if frame is None or frame.empty:
        return

    base_path = Path(base_path)
    rows = len(frame)
    columns = len(frame.columns)
    cells = rows * columns
    excel_ok = rows <= 100_000 and cells <= 750_000

    if excel_ok:
        path = base_path.with_suffix(".xlsx")
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_excel(path, index=False, engine="openpyxl")
        _remove_alternative_outputs(base_path, ".xlsx")
        return

    if _parquet_engine_available():
        path = base_path.with_suffix(".parquet")
        _write_parquet_atomic(path, frame)
        _remove_alternative_outputs(base_path, ".parquet")
        return

    path = base_path.with_suffix(".csv")
    _write_csv_atomic(path, frame)
    _remove_alternative_outputs(base_path, ".csv")


def _keep_analysis_temp(user):
    """Return whether temporary analysis tables should be retained."""

    monte_carlo = user.get("monte_carlo", {}) or {}
    return bool(monte_carlo.get("keep_analysis_temp", False))


def _convergence_checkpoints(simulation_count):
    """Return compact Monte Carlo checkpoints up to the available count."""

    count = int(simulation_count)
    if count <= 0:
        return []
    preferred = [50, 100, 200, 300, 500, 750, 1000]
    checkpoints = [value for value in preferred if value <= count]
    if not checkpoints or checkpoints[-1] != count:
        checkpoints.append(count)
    return sorted(set(checkpoints))


def _refresh_monte_carlo_outputs(analysis_dir, user):
    """Refresh Monte Carlo convergence summaries."""

    summary_dir = analysis_dir.parent / SUMMARY_FOLDER
    annual_base = summary_dir / "Annual_System_Summary"
    annual = _read_final_output(annual_base)

    if annual is None or annual.empty:
        return

    annual = annual[annual["Simulation"] > 0]
    if annual.empty:
        return

    metrics = [
        metric for metric in COMPARISON_SYSTEM_METRICS
        if metric in annual.columns]
    convergence_rows = []
    quantiles = _profile_quantiles(user)
    lower_probability = float(quantiles[0])
    upper_probability = float(quantiles[-1])
    lower_label = _percentile_label(lower_probability)
    upper_label = _percentile_label(upper_probability)
    confidence = 1.0 - 2.0 * lower_probability

    for year, year_data in annual.groupby("Year"):
        year_data = year_data.sort_values("Simulation")
        for metric in metrics:
            metric_data = year_data[["Simulation", metric]].copy()
            metric_data[metric] = pd.to_numeric(
                metric_data[metric], errors="coerce")
            metric_data = metric_data.dropna(subset=[metric])
            if metric_data.empty:
                continue
            values = metric_data[metric]
            for checkpoint in _convergence_checkpoints(len(values)):
                subset = metric_data.iloc[:checkpoint]
                sample = subset[metric]
                std = sample.std(ddof=1)
                convergence_rows.append({
                    "Year": year,
                    "Metric": metric,
                    "Simulations": int(checkpoint),
                    "Last_Simulation": int(
                        subset["Simulation"].iloc[-1]),
                    "Confidence_Level": confidence,
                    "Cumulative_Mean": sample.mean(),
                    "Cumulative_Std": std,
                    "Standard_Error": (
                        std / np.sqrt(checkpoint)
                        if checkpoint > 1 else np.nan),
                    lower_label: sample.quantile(lower_probability),
                    "P50": sample.quantile(0.50),
                    upper_label: sample.quantile(upper_probability)})

    convergence = pd.DataFrame(convergence_rows)
    convergence_base = summary_dir / "Monte_Carlo_Convergence"
    _write_final_output(convergence_base, convergence)


def _write_parquet_atomic(path, frame):
    """Write a parquet table atomically through a temporary file."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(
        temp_path, index=False, compression="zstd")
    temp_path.replace(path)


def _write_csv_atomic(path, frame):
    """Write a CSV table atomically through a temporary file."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temp_path, index=False)
    temp_path.replace(path)


def _write_pickle_atomic(path, frame):
    """Write a pickle table atomically for fast temporary storage."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    frame.to_pickle(temp_path)
    temp_path.replace(path)


def _append_output(analysis_dir, file_name, frame):
    """Append one simulation table to its temporary analysis file."""

    if frame is None or frame.empty:
        return

    path = Path(analysis_dir) / _temporary_name(file_name)
    if path.suffix == ".parquet":
        _write_parquet_atomic(path, frame)
    elif path.suffix == ".pkl":
        _write_pickle_atomic(path, frame)
    else:
        _write_csv_atomic(path, frame)


def _profile_group_columns(file_name):
    """Return identifier columns used to align one profile table."""

    groups = {
        "Daily_Publication_Profile": ["Date"],
        "Temporal_Profile_Summary": [
            "Year", "Month", "Hour"],
        "Commodity_Hourly_Profile": [
            "Commodity", "Quantity_Unit", "Year", "Hour"],
        "Commodity_Monthly_Profile": [
            "Commodity", "Quantity_Unit", "Year", "Month"],
        "BESS_SOC_Distribution": [
            "Year", "SOC_Bin_Lower", "SOC_Bin_Upper"],
        "BESS_Hourly_Profile": ["Year", "Hour"],
        "BESS_Monthly_Profile": ["Year", "Month"],
        "Hourly_Technology_Profile": [
            "Year", "Month", "Hour", "Category", "Technology"],
        "Monthly_System_Summary": ["Year", "Month"],
        "Monthly_Source_Summary": ["Year", "Month", "Source"],
        "Monthly_Emissions_Summary": ["Year", "Month"]}
    return groups[Path(file_name).stem]


def _profile_quantiles(user):
    """Return sorted ensemble quantiles requested by the user."""

    monte_carlo = user.get("monte_carlo", {}) or {}
    confidence = monte_carlo.get("confidence_level", 0.95)

    try:
        confidence = float(confidence)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Monte Carlo confidence level must be numeric.") from exc

    if not 0.0 < confidence < 1.0:
        raise ValueError(
            "Monte Carlo confidence level must be between 0 and 1.")

    lower = (1.0 - confidence) / 2.0
    upper = 1.0 - lower
    return np.array([lower, 0.5, upper], dtype=float)


def _percentile_label(probability):
    """Format one probability as a stable percentile column label."""

    percentage = 100.0 * float(probability)

    if np.isclose(percentage, round(percentage), atol=1e-10):
        return f"P{int(round(percentage)):02d}"

    tenths = 10.0 * percentage
    if np.isclose(tenths, round(tenths), atol=1e-10):
        return f"P{int(round(tenths)):03d}"

    text = f"{percentage:.3f}".rstrip("0").rstrip(".")
    return f"P{text.replace('.', '')}"


def _profile_value_columns(frame, group_cols):
    """Return numeric profile columns eligible for ensemble statistics."""

    excluded = {"Simulation", *group_cols}
    value_cols = [
        column for column in frame.columns
        if column not in excluded]

    for column in value_cols:
        converted = pd.to_numeric(frame[column], errors="coerce")
        invalid = frame[column].notna() & converted.isna()

        if invalid.any():
            raise ValueError(
                f"Profile column '{column}' must be numeric.")

    return value_cols


def _profile_simulation_id(path):
    """Extract the simulation identifier from a temporary profile path."""

    try:
        return int(Path(path).parent.name)
    except ValueError as exc:
        raise ValueError(
            f"Invalid profile simulation folder: {path}") from exc


def _monte_carlo_profile_paths(paths):
    """Return profile paths belonging to Monte Carlo simulations only."""

    by_simulation = {}

    for path in paths:
        simulation_id = _profile_simulation_id(path)

        if simulation_id <= 0:
            continue
        if simulation_id in by_simulation:
            raise ValueError(
                "Duplicate profile data for simulation "
                f"{simulation_id}.")

        by_simulation[simulation_id] = Path(path)

    return [
        by_simulation[simulation_id]
        for simulation_id in sorted(by_simulation)]


def _sorted_profile_frame(path, group_cols, value_cols):
    """Read one profile table in its stored group order."""

    columns = [*group_cols, *value_cols]
    frame = _read_temporary_table(path, columns=columns)
    return frame.reset_index(drop=True)

def _profile_keys_match(reference, current, group_cols):
    """Check that two profile tables describe the same ordered keys."""

    if len(reference) != len(current):
        return False

    for column in group_cols:
        left = reference[column].to_numpy(copy=False)
        right = current[column].to_numpy(copy=False)

        if not np.array_equal(left, right):
            return False

    return True


def _profile_statistics_exact(
        paths, group_cols, value_cols, reference, work_path, user):
    """Calculate exact profile statistics in memory-bounded batches."""

    simulation_count = len(paths)
    group_count = len(reference)
    metric_count = len(value_cols)
    quantiles = _profile_quantiles(user)
    quantile_labels = [
        _percentile_label(value) for value in quantiles]
    result = {
        column: reference[column].to_numpy(copy=True)
        for column in group_cols}
    result["Simulations"] = np.full(
        group_count, simulation_count, dtype=np.int32)
    totals = {}

    target_matrix_bytes = 128 * 1024 ** 2
    bytes_per_metric = max(
        simulation_count * group_count * 8, 1)
    batch_size = max(
        1, min(metric_count, target_matrix_bytes // bytes_per_metric))

    for batch_start in range(0, metric_count, batch_size):
        batch_end = min(batch_start + batch_size, metric_count)
        batch_cols = value_cols[batch_start:batch_end]
        matrix = np.empty(
            (simulation_count, group_count, len(batch_cols)),
            dtype=np.float64)

        for row, path in enumerate(paths):
            frame = _sorted_profile_frame(
                path, group_cols, batch_cols)
            if not _profile_keys_match(reference, frame, group_cols):
                frame = frame.sort_values(group_cols).reset_index(drop=True)
            if not _profile_keys_match(reference, frame, group_cols):
                simulation_id = _profile_simulation_id(path)
                raise ValueError(
                    "Profile groups are inconsistent for simulation "
                    f"{simulation_id} in {path.name}.")
            matrix[row, :, :] = frame[batch_cols].to_numpy(
                dtype=float, copy=False)
            del frame

        count = np.zeros(
            (group_count, len(batch_cols)), dtype=np.int32)
        mean = np.full((group_count, len(batch_cols)), np.nan)
        std = np.full((group_count, len(batch_cols)), np.nan)
        minimum = np.full((group_count, len(batch_cols)), np.nan)
        maximum = np.full((group_count, len(batch_cols)), np.nan)
        percentile = np.full(
            (len(quantiles), group_count, len(batch_cols)), np.nan)
        target_chunk_bytes = 32 * 1024 ** 2
        bytes_per_group = max(
            simulation_count * len(batch_cols) * 8, 1)
        group_chunk = max(
            64, min(
                group_count, target_chunk_bytes // bytes_per_group))

        for group_start in range(0, group_count, group_chunk):
            group_end = min(
                group_start + group_chunk, group_count)
            values = matrix[:, group_start:group_end, :]
            valid = np.isfinite(values)

            if valid.all():
                count[group_start:group_end, :] = simulation_count
                mean[group_start:group_end, :] = values.mean(axis=0)
                if simulation_count > 1:
                    std[group_start:group_end, :] = values.std(
                        axis=0, ddof=1)
                minimum[group_start:group_end, :] = values.min(axis=0)
                maximum[group_start:group_end, :] = values.max(axis=0)
                percentile[:, group_start:group_end, :] = np.quantile(
                    values, quantiles, axis=0)
                continue

            local_count = valid.sum(axis=0)
            count[group_start:group_end, :] = local_count
            safe = np.where(valid, values, np.nan)
            with np.errstate(invalid="ignore", divide="ignore"):
                mean[group_start:group_end, :] = np.nanmean(
                    safe, axis=0)
                std[group_start:group_end, :] = np.nanstd(
                    safe, axis=0, ddof=1)
                minimum[group_start:group_end, :] = np.nanmin(
                    safe, axis=0)
                maximum[group_start:group_end, :] = np.nanmax(
                    safe, axis=0)
                percentile[:, group_start:group_end, :] = np.nanquantile(
                    safe, quantiles, axis=0)

        for position, column in enumerate(batch_cols):
            result[f"{column}_Count"] = count[:, position]
            result[f"{column}_Mean"] = mean[:, position]
            result[f"{column}_Std"] = std[:, position]
            for q_position, label in enumerate(quantile_labels):
                result[f"{column}_{label}"] = percentile[
                    q_position, :, position]
            result[f"{column}_Minimum"] = minimum[:, position]
            result[f"{column}_Maximum"] = maximum[:, position]
            if column in {"Observations", "Active_Observations"}:
                totals[column] = np.nansum(
                    matrix[:, :, position], axis=0)

        del matrix
        del count, mean, std, minimum, maximum, percentile

    observations = totals.get("Observations")
    active = totals.get("Active_Observations")
    if observations is not None and active is not None:
        result["Probability_Active"] = np.divide(
            active, observations,
            out=np.zeros(group_count, dtype=float),
            where=observations > 0.0)

    return pd.DataFrame(result)

def _temporary_row_count(path):
    """Return the row count of one temporary analysis table."""

    path = Path(path)
    if path.suffix == ".parquet":
        import pyarrow.parquet as parquet
        return parquet.ParquetFile(path).metadata.num_rows
    if path.suffix == ".pkl":
        return len(pd.read_pickle(path))

    with path.open("rb") as stream:
        return max(sum(1 for _ in stream) - 1, 0)


def _simulation_table_paths(temp_dir, complete_ids, file_name):
    """Return existing temporary table paths for selected simulations."""

    paths = []

    for simulation_id in complete_ids:
        path = temp_dir / f"{simulation_id:06d}" / file_name
        if path.is_file():
            paths.append(path)

    return paths


def _clear_final_output(base_path):
    """Remove existing final copies of one analysis table."""

    base_path = Path(base_path)

    for suffix in (".xlsx", ".csv", ".parquet"):
        path = base_path.with_suffix(suffix)
        if path.is_file():
            path.unlink()


def _stream_regular_output(paths, final_base):
    """Write a large regular table incrementally with bounded memory."""

    final_base = Path(final_base)
    final_base.parent.mkdir(parents=True, exist_ok=True)

    if _parquet_engine_available():
        import pyarrow as pa
        import pyarrow.parquet as parquet

        path = final_base.with_suffix(".parquet")
        temp_path = path.with_suffix(".parquet.tmp")
        temp_path.unlink(missing_ok=True)
        writer = None
        try:
            for source_path in paths:
                frame = _read_temporary_table(source_path)
                table = pa.Table.from_pandas(
                    frame, preserve_index=False)
                if writer is None:
                    writer = parquet.ParquetWriter(
                        temp_path, table.schema, compression="zstd")
                writer.write_table(table)
                del table
                del frame
            if writer is not None:
                writer.close()
                writer = None
            temp_path.replace(path)
            _remove_alternative_outputs(final_base, ".parquet")
            return
        finally:
            if writer is not None:
                writer.close()
            if temp_path.is_file():
                temp_path.unlink()

    path = final_base.with_suffix(".csv")
    temp_path = path.with_suffix(".csv.tmp")
    temp_path.unlink(missing_ok=True)
    first = True

    with temp_path.open(
            "w", encoding="utf-8", newline="") as stream:
        for source_path in paths:
            frame = _read_temporary_table(source_path)
            frame.to_csv(stream, header=first, index=False)
            first = False
            del frame

    temp_path.replace(path)
    _remove_alternative_outputs(final_base, ".csv")


def _consolidate_regular_table(paths, final_base):
    """Consolidate regular tables without accumulating large frames."""

    if not paths:
        return

    total_rows = sum(_temporary_row_count(path) for path in paths)
    first = _read_temporary_table(paths[0])
    columns = len(first.columns)
    del first
    cells = total_rows * columns
    excel_ok = total_rows <= 100_000 and cells <= 750_000
    _clear_final_output(final_base)

    if excel_ok:
        frames = [_read_temporary_table(path) for path in paths]
        combined = pd.concat(frames, ignore_index=True)
        _write_final_output(final_base, combined)
        del combined
        del frames
        return

    _stream_regular_output(paths, final_base)

def _consolidate_profile_table(
        paths, file_name, final_base, work_path, user):
    """Aggregate aligned profile tables into ensemble statistics."""

    monte_carlo_paths = _monte_carlo_profile_paths(paths)

    if not monte_carlo_paths:
        return

    group_cols = _profile_group_columns(file_name)
    first = _read_temporary_table(monte_carlo_paths[0])
    first = first.sort_values(group_cols).reset_index(drop=True)

    if first.duplicated(group_cols).any():
        simulation_id = _profile_simulation_id(
            monte_carlo_paths[0])
        raise ValueError(
            "Duplicated profile groups were found for simulation "
            f"{simulation_id} in {monte_carlo_paths[0].name}.")

    value_cols = _profile_value_columns(first, group_cols)
    if Path(file_name).stem == "Temporal_Profile_Summary":
        value_cols = [
            column for column in value_cols
            if str(column).endswith("_Mean")]
    reference = first[group_cols].copy()
    del first

    if not value_cols:
        return

    statistics = _profile_statistics_exact(
        monte_carlo_paths,
        group_cols,
        value_cols,
        reference,
        work_path,
        user)

    deterministic_paths = [
        path for path in paths
        if _profile_simulation_id(path) == 0]
    if deterministic_paths:
        deterministic = _sorted_profile_frame(
            deterministic_paths[0], group_cols, value_cols)
        if not _profile_keys_match(
                reference, deterministic, group_cols):
            deterministic = deterministic.sort_values(
                group_cols).reset_index(drop=True)
        if _profile_keys_match(reference, deterministic, group_cols):
            for column in value_cols:
                statistics[f"Deterministic_{column}"] = deterministic[
                    column].to_numpy(dtype=float, copy=False)
        del deterministic

    _clear_final_output(final_base)
    _write_final_output(final_base, statistics)
    del statistics
    del reference



def _nuclear_refueling_config(user):
    """Return one calendar-based refuelling fleet when unambiguous."""

    sources = user.get("sources", {})
    if not isinstance(sources, dict):
        return None, None

    candidates = []
    for source_name, source_cfg in sources.items():
        if not isinstance(source_cfg, dict):
            continue
        refueling = source_cfg.get("refueling", {})
        if not isinstance(refueling, dict) or not refueling:
            continue
        mode = str(refueling.get("mode", "offline")).strip().lower()
        if mode != "offline" or refueling.get("operating_cycle") is None:
            continue
        candidates.append((source_name, source_cfg))

    if len(candidates) != 1:
        return None, None
    return candidates[0]


def _first_capacity_addition_year(source_cfg):
    """Return the first year with explicit installed-capacity addition."""

    additions = source_cfg.get("capacity_additions", {})
    if not isinstance(additions, dict) or not additions:
        return None

    dates = pd.to_datetime(list(additions), errors="coerce")
    dates = dates[~pd.isna(dates)]
    if len(dates) == 0:
        return None
    return int(dates.min().year)


def _refueling_month_sequence(commissioning_month, operating_cycle):
    """Return recurring refueling months implied by one commissioning month."""

    month = int(commissioning_month)
    cycle = int(round(float(operating_cycle)))
    if month < 1 or month > 12 or cycle <= 0:
        return []

    sequence = []
    current = ((month - 1 + cycle) % 12) + 1

    while current not in sequence:
        sequence.append(current)
        current = ((current - 1 + cycle) % 12) + 1

    return sequence


def _percentile_triplet(values):
    """Return P2.5, P50, and P97.5 for one numeric series."""

    series = pd.Series(values, dtype=float).dropna()
    if series.empty:
        return np.nan, np.nan, np.nan
    return tuple(
        float(series.quantile(value))
        for value in (0.025, 0.5, 0.975))


_REFUELING_IMPACT_MEANS = {
    "Outage_Energy": ("Outage_Energy", "mean"),
    "Initial_Residual_Supply_Requirement_Increase": (
        "Initial_Residual_Supply_Requirement_Increase", "mean"),
    "Initial_Surplus_Electricity_Reduction": (
        "Initial_Surplus_Electricity_Reduction", "mean"),
    "Outage_Energy_Share_Creating_Residual_Supply": (
        "Outage_Energy_Share_Creating_Residual_Supply", "mean"),}


def _refueling_outage_impact_by_simulation(frame, simulation_id, user):
    """Estimate the monthly impact of one standard refuelling outage.

    The diagnostic removes one configured unit from the first day of each
    eligible month while all other modelled generation is unchanged. Energy
    values remain in the scenario energy unit; the unit is reported in a
    separate column rather than encoded in metric names.
    """

    source_name, source_cfg = _nuclear_refueling_config(user)
    if source_name is None:
        return None

    refueling = source_cfg.get("refueling", {}) or {}
    try:
        unit_capacity = float(source_cfg.get("unit_capacity", 0.0))
        outage_duration = int(refueling.get("outage_duration", 0))
    except (TypeError, ValueError):
        return None
    if unit_capacity <= 0.0 or outage_duration <= 0:
        return None

    raw = _series_by_prefix(frame, "Raw_Electricity_Balance")
    if raw is None:
        return None

    installed_column = f"Installed_Capacity_{source_name}"
    if installed_column not in frame.columns:
        return None

    dates = pd.to_datetime(frame["Date"], errors="coerce")
    if dates.isna().any() or frame.empty:
        return None

    step_hours = _time_step_hours(frame)
    energy_unit = normalize_energy_unit(user.get("energy_unit", "MWh"))
    mwh_to_active = energy_from_mwh_factor(energy_unit)
    loss_per_step = unit_capacity * step_hours * mwh_to_active
    expected_steps = int(round(outage_duration * 24.0 / step_hours))
    if expected_steps <= 0:
        return None

    installed = pd.to_numeric(
        frame[installed_column], errors="coerce").fillna(0.0).to_numpy()
    raw_values = pd.to_numeric(raw, errors="coerce").fillna(0.0).to_numpy()
    date_values = dates.to_numpy(dtype="datetime64[ns]")
    first_date = pd.Timestamp(dates.iloc[0])
    last_date = pd.Timestamp(dates.iloc[-1])
    step_delta = pd.Timedelta(hours=step_hours)
    first_year = _first_capacity_addition_year(source_cfg)
    if first_year is None:
        first_year = int(first_date.year)

    event_rows = []
    for year in range(first_year, int(last_date.year) + 1):
        for month in range(1, 13):
            start_date = pd.Timestamp(year=year, month=month, day=1)
            end_date = start_date + pd.Timedelta(days=outage_duration)
            if start_date < first_date or end_date > last_date + step_delta:
                continue

            start64 = start_date.to_datetime64()
            start_pos = int(np.searchsorted(date_values, start64, side="left"))
            end_pos = start_pos + expected_steps
            if end_pos > len(date_values) or start_pos >= len(date_values):
                continue
            if date_values[start_pos] != start64:
                continue

            expected_end = end_date.to_datetime64()
            if end_pos < len(date_values):
                if date_values[end_pos] != expected_end:
                    continue
            elif expected_end != last_date.to_datetime64() + np.timedelta64(
                    int(round(step_hours * 3600.0)), "s"):
                continue

            installed_window = installed[start_pos:end_pos]
            if np.any(installed_window < unit_capacity - 1e-9):
                continue

            balance = raw_values[start_pos:end_pos]
            original_requirement = np.maximum(-balance, 0.0)
            outage_balance = balance - loss_per_step
            outage_requirement = np.maximum(-outage_balance, 0.0)
            original_surplus = np.maximum(balance, 0.0)
            outage_surplus = np.maximum(outage_balance, 0.0)

            requirement_increase = float(
                np.sum(outage_requirement - original_requirement))
            surplus_reduction = float(
                np.sum(original_surplus - outage_surplus))
            outage_energy = loss_per_step * expected_steps
            impact_share = (
                100.0 * requirement_increase / outage_energy
                if outage_energy > 0.0 else np.nan)

            event_rows.append({
                "Simulation": int(simulation_id),
                "Year": int(year),
                "Month": int(month),
                "Outage_Energy": outage_energy,
                "Initial_Residual_Supply_Requirement_Increase":
                    requirement_increase,
                "Initial_Surplus_Electricity_Reduction":
                    surplus_reduction,
                "Outage_Energy_Share_Creating_Residual_Supply":
                    impact_share,})

    if not event_rows:
        return None

    events = pd.DataFrame(event_rows)
    result = events.groupby(
        ["Simulation", "Month"], as_index=False).agg(
            **_REFUELING_IMPACT_MEANS,
            Years_Evaluated=("Year", "count"))
    result["Energy_Unit"] = canonical_energy_unit(energy_unit)
    result["Share_Unit"] = "%"
    return result


def _build_refueling_month_guidance(impact_by_simulation, user):
    """Build simple percentile guidance from the outage stress test."""

    source_name, source_cfg = _nuclear_refueling_config(user)
    if source_name is None or impact_by_simulation is None:
        return None, None

    impact = impact_by_simulation.copy()
    if impact.empty or "Simulation" not in impact:
        return None, None

    mc_ids = sorted(
        value for value in impact["Simulation"].unique() if value > 0)
    if mc_ids:
        impact = impact[impact["Simulation"].isin(mc_ids)]
    else:
        impact = impact[impact["Simulation"] == 0]
    if impact.empty:
        return None, None

    refueling = source_cfg.get("refueling", {}) or {}
    try:
        operating_cycle = float(refueling.get("operating_cycle", 0.0))
    except (TypeError, ValueError):
        return None, None
    if operating_cycle <= 0.0:
        return None, None

    month_names = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun",
        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    metrics = [
        "Initial_Residual_Supply_Requirement_Increase",
        "Initial_Surplus_Electricity_Reduction",
        "Outage_Energy_Share_Creating_Residual_Supply"]

    month_rows = []
    for month in range(1, 13):
        subset = impact[impact["Month"] == month]
        row = {
            "Month": month,
            "Month_Name": month_names[month - 1],
            "Outage_Energy": float(
                subset["Outage_Energy"].median())
                if not subset.empty else np.nan,
            "Simulation_Count": int(subset["Simulation"].nunique()),
            "Energy_Unit": str(user.get("energy_unit", "MWh")),
            "Share_Unit": "%",
            "Mean_Years_Evaluated": float(
                subset["Years_Evaluated"].mean())
                if not subset.empty else np.nan}
        for metric in metrics:
            p025, p50, p975 = _percentile_triplet(subset[metric])
            row[f"{metric}_P2.5"] = p025
            row[f"{metric}_P50"] = p50
            row[f"{metric}_P97.5"] = p975
        month_rows.append(row)

    monthly_guidance = pd.DataFrame(month_rows)
    monthly_guidance["Indicative_Rank"] = (
        monthly_guidance[
            "Outage_Energy_Share_Creating_Residual_Supply_P50"]
        .rank(method="dense", ascending=True)
        .astype("Int64"))

    commissioning_rows = []
    for commissioning_month in range(1, 13):
        refueling_months = _refueling_month_sequence(
            commissioning_month, operating_cycle)
        if not refueling_months:
            continue

        per_simulation = impact[
            impact["Month"].isin(refueling_months)]
        per_simulation = per_simulation.groupby(
            "Simulation", as_index=False).agg(**_REFUELING_IMPACT_MEANS)

        row = {
            "Commissioning_Month": commissioning_month,
            "Commissioning_Month_Name": month_names[
                commissioning_month - 1],
            "Operating_Cycle": operating_cycle,
            "Operating_Cycle_Unit": "month",
            "First_Refueling_Month": refueling_months[0],
            "First_Refueling_Month_Name": month_names[
                refueling_months[0] - 1],
            "Recurring_Refueling_Months": " / ".join(
                month_names[value - 1] for value in refueling_months),
            "Outage_Energy": float(
                per_simulation["Outage_Energy"].median())
                if not per_simulation.empty else np.nan,
            "Simulation_Count": int(
                per_simulation["Simulation"].nunique()),
            "Energy_Unit": str(user.get("energy_unit", "MWh")),
            "Share_Unit": "%"}
        for metric in metrics:
            p025, p50, p975 = _percentile_triplet(
                per_simulation[metric])
            row[f"{metric}_P2.5"] = p025
            row[f"{metric}_P50"] = p50
            row[f"{metric}_P97.5"] = p975
        commissioning_rows.append(row)

    commissioning_guidance = pd.DataFrame(commissioning_rows)
    if not commissioning_guidance.empty:
        commissioning_guidance["Indicative_Rank"] = (
            commissioning_guidance[
                "Outage_Energy_Share_Creating_Residual_Supply_P50"]
            .rank(method="dense", ascending=True)
            .astype("Int64"))

    return monthly_guidance, commissioning_guidance


def _build_refueling_month_diagnostics(
        monthly_system, monthly_sources, user):
    """Build the detailed monthly pattern table used for interpretation."""

    source_name, source_cfg = _nuclear_refueling_config(user)
    if source_name is None:
        return None

    system = monthly_system.copy()
    sources = monthly_sources.copy()
    if "Simulation" not in system or "Month" not in system:
        return None
    if "Simulation" not in sources or "Month" not in sources:
        return None

    mc_ids = sorted(
        value for value in system["Simulation"].unique() if value > 0)
    if mc_ids:
        system = system[system["Simulation"].isin(mc_ids)]
        sources = sources[sources["Simulation"].isin(mc_ids)]
    else:
        system = system[system["Simulation"] == 0]
        sources = sources[sources["Simulation"] == 0]

    first_year = _first_capacity_addition_year(source_cfg)
    if first_year is not None and "Year" in system and "Year" in sources:
        system = system[system["Year"] >= first_year]
        sources = sources[sources["Year"] >= first_year]
    if system.empty or sources.empty:
        return None

    system_agg = system.groupby(
        ["Simulation", "Month"], as_index=False).agg(
            Demand=("Demand", "sum"),
            Initial_Surplus_Electricity=(
                "Initial_Surplus_Electricity", "sum"),
            Initial_Residual_Supply_Requirement=(
                "Initial_Residual_Supply_Requirement", "sum"))
    non_nuclear = sources[sources["Source"] != source_name]
    non_nuclear = non_nuclear.groupby(
        ["Simulation", "Month"], as_index=False).agg(
            Non_Nuclear_Generation=("Generation", "sum"))
    pattern = system_agg.merge(
        non_nuclear, on=["Simulation", "Month"], how="left")
    pattern["Non_Nuclear_Generation"] = pattern[
        "Non_Nuclear_Generation"].fillna(0.0)

    demand = pattern["Demand"].replace(0.0, np.nan)
    pattern["Initial_Surplus_Share_of_Demand"] = (
        100.0 * pattern["Initial_Surplus_Electricity"] / demand)
    pattern["Initial_Residual_Supply_Share_of_Demand"] = (
        100.0 * pattern["Initial_Residual_Supply_Requirement"] / demand)
    pattern["Residual_Before_Nuclear_Share_of_Demand"] = (
        100.0
        * (pattern["Demand"] - pattern["Non_Nuclear_Generation"])
        / demand)

    month_names = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun",
        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    metrics = [
        "Initial_Surplus_Share_of_Demand",
        "Initial_Residual_Supply_Share_of_Demand",
        "Residual_Before_Nuclear_Share_of_Demand"]
    rows = []
    for month in range(1, 13):
        subset = pattern[pattern["Month"] == month]
        row = {
            "Month": month,
            "Month_Name": month_names[month - 1],
            "Simulation_Count": int(subset["Simulation"].nunique()),
            "Share_Unit": "%"}
        for metric in metrics:
            p025, p50, p975 = _percentile_triplet(subset[metric])
            row[f"{metric}_P2.5"] = p025
            row[f"{metric}_P50"] = p50
            row[f"{metric}_P97.5"] = p975
        rows.append(row)
    return pd.DataFrame(rows)


def _write_nuclear_month_guidance(output_root, user):
    """Write simple outage guidance plus an optional diagnostic table."""

    source_name, _ = _nuclear_refueling_config(user)
    if source_name is None:
        return

    summary_dir = Path(output_root) / SUMMARY_FOLDER
    impact = _read_final_output(
        summary_dir / "Nuclear_Refueling_Impact_By_Simulation")
    if impact is not None and not impact.empty:
        monthly, commissioning = _build_refueling_month_guidance(
            impact, user)
        if monthly is not None and not monthly.empty:
            _write_final_output(
                summary_dir / "Nuclear_Refueling_Month_Guidance",
                monthly)
        if commissioning is not None and not commissioning.empty:
            _write_final_output(
                summary_dir / "Nuclear_Commissioning_Month_Guidance",
                commissioning)

    monthly_system = _read_final_output(
        summary_dir / "Monthly_System_Summary")
    monthly_sources = _read_final_output(
        summary_dir / "Monthly_Source_Summary")
    if monthly_system is None or monthly_sources is None:
        return
    diagnostics = _build_refueling_month_diagnostics(
        monthly_system, monthly_sources, user)
    if diagnostics is not None and not diagnostics.empty:
        _write_final_output(
            summary_dir / "Nuclear_Refueling_Month_Diagnostics",
            diagnostics)


def save_simulation_diagnostics(output_dir, simulation_id, diagnostics):
    """Persist compact runtime diagnostics for one simulation."""

    simulation_dir = get_simulation_analysis_dir(
        output_dir, simulation_id, create=True)
    path = simulation_dir / "Runtime_Diagnostics.json"
    rows = diagnostics if isinstance(diagnostics, list) else []
    payload = {
        "simulation": int(simulation_id),
        "diagnostics": rows,}
    with path.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, default=str)


def _runtime_diagnostics_table(temp_dir, simulation_ids):
    """Read structured runtime diagnostics from completed simulations."""

    rows = []
    for simulation_id in simulation_ids:
        path = (
            Path(temp_dir) / f"{int(simulation_id):06d}"
            / "Runtime_Diagnostics.json")
        if not path.is_file():
            continue
        with path.open(encoding="utf-8") as stream:
            payload = json.load(stream)
        for item in payload.get("diagnostics", []):
            if not isinstance(item, dict):
                continue
            row = {"Simulation": int(simulation_id)}
            row.update(item)
            rows.append(row)
    return pd.DataFrame(rows)


def _capacity_clipping_summary(raw, simulation_ids):
    """Aggregate capacity clipping across deterministic and MC cases."""

    if raw.empty or "Type" not in raw.columns:
        return pd.DataFrame()
    data = raw.loc[raw["Type"] == "capacity_clipping"].copy()
    if data.empty:
        return pd.DataFrame()

    mc_requested = sum(int(value) > 0 for value in simulation_ids)
    summaries = []
    group_cols = ["Source", "Resolution", "Energy_Unit"]
    for keys, subset in data.groupby(group_cols, sort=True, dropna=False):
        source, resolution, energy_unit = keys
        simulations = pd.to_numeric(
            subset["Simulation"], errors="coerce").fillna(-1).astype(int)
        mc_subset = subset.loc[simulations > 0]
        energy = pd.to_numeric(
            subset["Energy_Removed"], errors="coerce").fillna(0.0)
        mc_energy = pd.to_numeric(
            mc_subset.get("Energy_Removed", pd.Series(dtype=float)),
            errors="coerce").fillna(0.0)
        excess = pd.to_numeric(
            subset["Max_Excess_Percent"], errors="coerce")
        worst_index = excess.idxmax() if excess.notna().any() else None
        worst_date = ""
        if worst_index is not None and "Max_Date" in subset.columns:
            worst_date = str(subset.loc[worst_index, "Max_Date"])
        affected_mc = int((simulations > 0).sum())
        deterministic_energy = energy.loc[simulations == 0]
        summaries.append({
            "Source": source,
            "Resolution": resolution,
            "Energy_Unit": energy_unit,
            "Deterministic_Affected": bool((simulations == 0).any()),
            "Deterministic_Energy_Removed": (
                float(deterministic_energy.sum())
                if not deterministic_energy.empty else 0.0),
            "MC_Affected": affected_mc,
            "MC_Requested": int(mc_requested),
            "MC_Affected_Percent": (
                100.0 * affected_mc / mc_requested
                if mc_requested else 0.0),
            "Median_MC_Energy_Removed": (
                float(mc_energy.median()) if not mc_energy.empty else 0.0),
            "P95_MC_Energy_Removed": (
                float(mc_energy.quantile(0.95))
                if not mc_energy.empty else 0.0),
            "Max_MC_Energy_Removed": (
                float(mc_energy.max()) if not mc_energy.empty else 0.0),
            "Maximum_Excess_Percent": (
                float(excess.max()) if excess.notna().any() else np.nan),
            "Worst_Date": worst_date,})
    return pd.DataFrame(summaries)


def _nuclear_generation_statistics(frame, group_cols, user):
    """Return deterministic and Monte Carlo nuclear annual statistics."""

    if frame.empty:
        return pd.DataFrame()
    work = frame.copy()
    work["Simulation"] = pd.to_numeric(
        work["Simulation"], errors="raise").astype(int)
    work["Generation"] = pd.to_numeric(
        work["Generation"], errors="raise")
    if "Load_Factor" in work.columns:
        work["Load_Factor"] = pd.to_numeric(
            work["Load_Factor"], errors="coerce")
    deterministic = work[work["Simulation"] == 0].copy()
    monte_carlo = work[work["Simulation"] > 0].copy()

    keys = list(group_cols)
    base = deterministic[keys + ["Generation"]].rename(
        columns={"Generation": "Deterministic_Generation"})
    if base.empty and not monte_carlo.empty:
        base = monte_carlo[keys].drop_duplicates().copy()
        base["Deterministic_Generation"] = np.nan

    if monte_carlo.empty:
        for column in (
                "MC_Mean", "MC_Std", "MC_P025", "MC_P50",
                "MC_P975", "MC_Minimum", "MC_Maximum"):
            base[column] = np.nan
        base["MC_Simulations"] = 0
    else:
        grouped = monte_carlo.groupby(keys, sort=True)["Generation"]
        moments = grouped.agg(
            MC_Mean="mean", MC_Std="std", MC_Minimum="min",
            MC_Maximum="max", MC_Simulations="count").reset_index()
        probabilities = _profile_quantiles(user)
        quantiles = grouped.quantile(probabilities).unstack()
        quantiles.columns = [
            f"MC_{_percentile_label(value)}"
            for value in probabilities]
        quantiles = quantiles.reset_index()
        base = base.merge(moments, on=keys, how="outer")
        base = base.merge(quantiles, on=keys, how="outer")

    if "Load_Factor" in work.columns:
        det_lf = deterministic[keys + ["Load_Factor"]].rename(
            columns={"Load_Factor": "Deterministic_Load_Factor"})
        if det_lf.empty and not monte_carlo.empty:
            det_lf = monte_carlo[keys].drop_duplicates().copy()
            det_lf["Deterministic_Load_Factor"] = np.nan
        base = base.merge(det_lf, on=keys, how="outer")

        if monte_carlo.empty:
            for column in (
                    "MC_Load_Factor_Mean", "MC_Load_Factor_Std",
                    "MC_Load_Factor_P025", "MC_Load_Factor_P50",
                    "MC_Load_Factor_P975", "MC_Load_Factor_Minimum",
                    "MC_Load_Factor_Maximum"):
                base[column] = np.nan
            base["MC_Load_Factor_Simulations"] = 0
        else:
            grouped_lf = monte_carlo.groupby(
                keys, sort=True)["Load_Factor"]
            lf_moments = grouped_lf.agg(
                MC_Load_Factor_Mean="mean",
                MC_Load_Factor_Std="std",
                MC_Load_Factor_Minimum="min",
                MC_Load_Factor_Maximum="max",
                MC_Load_Factor_Simulations="count").reset_index()
            probabilities = _profile_quantiles(user)
            lf_quantiles = grouped_lf.quantile(
                probabilities).unstack()
            lf_quantiles.columns = [
                f"MC_Load_Factor_{_percentile_label(value)}"
                for value in probabilities]
            lf_quantiles = lf_quantiles.reset_index()
            base = base.merge(lf_moments, on=keys, how="outer")
            base = base.merge(lf_quantiles, on=keys, how="outer")

    ordered = [
        *keys, "Deterministic_Generation", "MC_Mean", "MC_Std",
        "MC_P025", "MC_P50", "MC_P975", "MC_Minimum",
        "MC_Maximum", "MC_Simulations"]
    load_factor_columns = [
        "Deterministic_Load_Factor", "MC_Load_Factor_Mean",
        "MC_Load_Factor_Std", "MC_Load_Factor_P025",
        "MC_Load_Factor_P50", "MC_Load_Factor_P975",
        "MC_Load_Factor_Minimum", "MC_Load_Factor_Maximum",
        "MC_Load_Factor_Simulations"]
    if "Load_Factor" in work.columns:
        ordered.extend(load_factor_columns)
    for column in ordered:
        if column not in base.columns:
            base[column] = np.nan
    base = base[ordered].sort_values(keys).reset_index(drop=True)
    base["Energy_Unit"] = str(
        user.get("energy_unit", "MWh")).strip()
    return base


def _nuclear_convergence_statistics(annual_raw, user):
    """Return compact convergence for nuclear generation and load factor."""

    if annual_raw.empty:
        return pd.DataFrame()
    work = annual_raw.copy()
    work["Simulation"] = pd.to_numeric(
        work["Simulation"], errors="raise").astype(int)
    work = work[work["Simulation"] > 0].copy()
    if work.empty:
        return pd.DataFrame()

    probabilities = _profile_quantiles(user)
    lower_probability = float(probabilities[0])
    upper_probability = float(probabilities[-1])
    lower_label = _percentile_label(lower_probability)
    upper_label = _percentile_label(upper_probability)
    confidence = 1.0 - 2.0 * lower_probability
    energy_unit = str(user.get("energy_unit", "MWh")).strip()
    metrics = [("Generation", energy_unit)]
    if "Load_Factor" in work.columns:
        metrics.append(("Load_Factor", "fraction"))

    rows = []
    for (year, source), group in work.groupby(
            ["Year", "Source"], sort=True):
        group = group.sort_values("Simulation")
        for metric, unit in metrics:
            metric_data = group[["Simulation", metric]].copy()
            metric_data[metric] = pd.to_numeric(
                metric_data[metric], errors="coerce")
            metric_data = metric_data.dropna(subset=[metric])
            if metric_data.empty:
                continue
            for checkpoint in _convergence_checkpoints(
                    len(metric_data)):
                subset = metric_data.iloc[:checkpoint]
                sample = subset[metric]
                std = sample.std(ddof=1)
                rows.append({
                    "Year": year,
                    "Source": source,
                    "Metric": metric,
                    "Unit": unit,
                    "Simulations": int(checkpoint),
                    "Last_Simulation": int(
                        subset["Simulation"].iloc[-1]),
                    "Confidence_Level": confidence,
                    "Cumulative_Mean": sample.mean(),
                    "Cumulative_Std": std,
                    "Standard_Error": (
                        std / np.sqrt(checkpoint)
                        if checkpoint > 1 else np.nan),
                    lower_label: sample.quantile(lower_probability),
                    "P50": sample.quantile(0.50),
                    upper_label: sample.quantile(upper_probability)})
    return pd.DataFrame(rows)

def _nearest_date_quantile(values, probability):
    """Return a date quantile selected from observed daily event dates."""

    dates = pd.to_datetime(values, errors="coerce").dropna().sort_values()
    if dates.empty:
        return pd.NaT
    position = int(np.floor(
        float(probability) * (len(dates) - 1) + 0.5))
    position = min(max(position, 0), len(dates) - 1)
    return pd.Timestamp(dates.iloc[position]).normalize()


def _refueling_uncertainty_statistics(frame, user):
    """Summarize Monte Carlo uncertainty in unit-level outage dates."""

    if frame.empty:
        return pd.DataFrame()
    work = frame.copy()
    work["Simulation"] = pd.to_numeric(
        work["Simulation"], errors="raise").astype(int)
    for column in ("Commissioning_Date", "Outage_Start", "Outage_End"):
        if column in work.columns:
            work[column] = pd.to_datetime(
                work[column], errors="coerce").dt.normalize()

    keys = ["Source", "Unit", "Refueling_Number"]
    probabilities = _profile_quantiles(user)
    labels = [_percentile_label(value) for value in probabilities]
    rows = []
    for values, subset in work.groupby(keys, sort=True, dropna=False):
        deterministic = subset.loc[subset["Simulation"] == 0]
        monte_carlo = subset.loc[subset["Simulation"] > 0]
        commissioning = pd.to_datetime(
            subset.get("Commissioning_Date"), errors="coerce").dropna()
        row = {
            "Source": values[0],
            "Unit": values[1],
            "Refueling_Number": values[2],
            "Commissioning_Date": (
                commissioning.iloc[0] if not commissioning.empty else pd.NaT),
            "Deterministic_Start": (
                deterministic["Outage_Start"].iloc[0]
                if not deterministic.empty else pd.NaT),
            "Deterministic_End": (
                deterministic["Outage_End"].iloc[0]
                if not deterministic.empty else pd.NaT),
            "MC_Simulations": int(monte_carlo["Simulation"].nunique()),}
        for probability, label in zip(probabilities, labels):
            row[f"MC_{label}_Start"] = _nearest_date_quantile(
                monte_carlo["Outage_Start"], probability)
            row[f"MC_{label}_End"] = _nearest_date_quantile(
                monte_carlo["Outage_End"], probability)
        rows.append(row)

    result = pd.DataFrame(rows)
    ordered = [
        "Source", "Unit", "Refueling_Number", "Commissioning_Date",
        "Deterministic_Start", "MC_P025_Start", "MC_P50_Start",
        "MC_P975_Start", "Deterministic_End", "MC_P025_End",
        "MC_P50_End", "MC_P975_End", "MC_Simulations"]
    for column in ordered:
        if column not in result.columns:
            is_date = "Start" in column or "End" in column
            result[column] = pd.NaT if is_date else np.nan
    return result[ordered].sort_values(keys).reset_index(drop=True)


def _nuclear_daily_statistics(paths, user):
    """Return exact per-day nuclear generation statistics across MC runs."""

    if not paths:
        return pd.DataFrame()
    group_cols = ["Date", "Source"]
    value_cols = ["Generation"]
    monte_carlo_paths = _monte_carlo_profile_paths(paths)
    deterministic_paths = [
        path for path in paths
        if _profile_simulation_id(path) == 0]

    if monte_carlo_paths:
        first = _read_temporary_table(
            monte_carlo_paths[0], columns=[*group_cols, *value_cols])
        first = first.sort_values(group_cols).reset_index(drop=True)
        if first.duplicated(group_cols).any():
            raise ValueError(
                "Duplicated daily nuclear groups were found in "
                f"{monte_carlo_paths[0].name}.")
        reference = first[group_cols].copy()
        del first
        statistics = _profile_statistics_exact(
            monte_carlo_paths, group_cols, value_cols, reference,
            Path(paths[0]).parent / "Daily_Nuclear_Statistics.tmp", user)
    elif deterministic_paths:
        deterministic = _read_temporary_table(
            deterministic_paths[0], columns=[*group_cols, *value_cols])
        deterministic = deterministic.sort_values(
            group_cols).reset_index(drop=True)
        statistics = deterministic[group_cols].copy()
        statistics["Generation_Count"] = 0
        for suffix in (
                "Mean", "Std", "P025", "P50", "P975",
                "Minimum", "Maximum"):
            statistics[f"Generation_{suffix}"] = np.nan
    else:
        return pd.DataFrame()

    if deterministic_paths:
        deterministic = _read_temporary_table(
            deterministic_paths[0], columns=[*group_cols, *value_cols])
        deterministic = deterministic.sort_values(
            group_cols).reset_index(drop=True)
        deterministic = deterministic.rename(
            columns={"Generation": "Deterministic_Generation"})
        statistics = statistics.merge(
            deterministic, on=group_cols, how="left")
    else:
        statistics["Deterministic_Generation"] = np.nan

    rename = {
        "Generation_Count": "MC_Simulations",
        "Generation_Mean": "MC_Mean",
        "Generation_Std": "MC_Std",
        "Generation_P025": "MC_P025",
        "Generation_P50": "MC_P50",
        "Generation_P975": "MC_P975",
        "Generation_Minimum": "MC_Minimum",
        "Generation_Maximum": "MC_Maximum"}
    statistics = statistics.rename(columns=rename)
    statistics = statistics.drop(
        columns=["Simulations"], errors="ignore")
    ordered = [
        "Date", "Source", "Deterministic_Generation", "MC_Mean",
        "MC_Std", "MC_P025", "MC_P50", "MC_P975", "MC_Minimum",
        "MC_Maximum", "MC_Simulations"]
    for column in ordered:
        if column not in statistics.columns:
            statistics[column] = np.nan
    statistics = statistics[ordered].sort_values(
        group_cols).reset_index(drop=True)
    statistics["Date"] = pd.to_datetime(
        statistics["Date"], errors="raise").dt.normalize()
    statistics["Energy_Unit"] = str(
        user.get("energy_unit", "MWh")).strip()
    return statistics


def _nuclear_generation_workbook(
        temp_dir, complete_ids, output_dir, user):
    """Write final operational nuclear generation and MC uncertainty."""

    daily_name = _temporary_name("Daily_Nuclear_Generation.csv")
    annual_name = _temporary_name("Annual_Nuclear_Generation.csv")
    monthly_name = _temporary_name("Monthly_Nuclear_Generation.csv")
    refueling_name = _temporary_name(
        "Nuclear_Refueling_Schedule_By_Simulation.csv")
    daily_paths = _simulation_table_paths(
        temp_dir, complete_ids, daily_name)
    annual_paths = _simulation_table_paths(
        temp_dir, complete_ids, annual_name)
    monthly_paths = _simulation_table_paths(
        temp_dir, complete_ids, monthly_name)
    refueling_paths = _simulation_table_paths(
        temp_dir, complete_ids, refueling_name)
    if not annual_paths:
        return None

    daily = _nuclear_daily_statistics(daily_paths, user)

    annual_frames = [
        _read_temporary_table(path) for path in annual_paths]
    annual_raw = pd.concat(annual_frames, ignore_index=True)
    del annual_frames
    annual = _nuclear_generation_statistics(
        annual_raw, ["Year", "Source"], user)
    annual_convergence = _nuclear_convergence_statistics(
        annual_raw, user)

    monthly = pd.DataFrame()
    if monthly_paths:
        monthly_frames = [
            _read_temporary_table(path) for path in monthly_paths]
        monthly_raw = pd.concat(monthly_frames, ignore_index=True)
        del monthly_frames
        monthly = _nuclear_generation_statistics(
            monthly_raw, ["Year", "Month", "Source"], user)
        del monthly_raw

    refueling_raw = pd.DataFrame()
    refueling_statistics = pd.DataFrame()
    if refueling_paths:
        refueling_frames = [
            _read_temporary_table(path) for path in refueling_paths]
        refueling_raw = pd.concat(refueling_frames, ignore_index=True)
        del refueling_frames
        refueling_statistics = _refueling_uncertainty_statistics(
            refueling_raw, user)

    mc_count = int(annual_raw.loc[
        annual_raw["Simulation"] > 0, "Simulation"].nunique())
    sources = ", ".join(sorted(annual_raw["Source"].astype(str).unique()))
    metadata = pd.DataFrame({
        "Field": [
            "Definition", "Energy_Unit", "Confidence_Level",
            "Monte_Carlo_Simulations", "Nuclear_Sources",
            "Daily_Statistics", "Refueling_Date_Quantiles"],
        "Value": [
            "Final operational generation after load following and "
            "refuelling",
            str(user.get("energy_unit", "MWh")).strip(),
            float((user.get("monte_carlo", {}) or {}).get(
                "confidence_level", 0.95)),
            mc_count, sources,
            "Per-date statistics across Monte Carlo simulations",
            "Nearest observed Monte Carlo event date"]})

    path = Path(output_dir) / "Nuclear_Generation.xlsx"
    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        annual.to_excel(writer, sheet_name="Annual", index=False)
        if not monthly.empty:
            monthly.to_excel(writer, sheet_name="Monthly", index=False)
        if not daily.empty:
            daily.to_excel(
                writer, sheet_name="Daily_Statistics", index=False)
        annual_raw = annual_raw.sort_values(
            ["Simulation", "Year", "Source"])
        annual_raw.to_excel(
            writer, sheet_name="Annual_By_Simulation", index=False)
        if not annual_convergence.empty:
            annual_convergence.to_excel(
                writer, sheet_name="Monte_Carlo_Convergence", index=False)
        if not refueling_statistics.empty:
            refueling_statistics.to_excel(
                writer, sheet_name="Refueling_Uncertainty", index=False)
        if not refueling_raw.empty:
            refueling_raw = refueling_raw.sort_values(
                ["Simulation", "Source", "Unit", "Refueling_Number"])
            refueling_raw.to_excel(
                writer, sheet_name="Refueling_By_Simulation", index=False)
        metadata.to_excel(writer, sheet_name="Metadata", index=False)
        for worksheet in writer.book.worksheets:
            worksheet.freeze_panes = "A2"
            if worksheet.max_row > 1 and worksheet.max_column > 1:
                worksheet.auto_filter.ref = worksheet.dimensions
            for column in worksheet.columns:
                width = max(
                    len(str(cell.value)) if cell.value is not None else 0
                    for cell in column)
                worksheet.column_dimensions[column[0].column_letter].width = (
                    min(max(width + 2, 10), 28))
    return path



def _final_table(output_root, folder, stem):
    """Read one consolidated internal result table when it exists."""

    return _read_final_output(Path(output_root) / folder / stem)


def _statistics_table(frame, group_cols, user, value_cols=None):
    """Return deterministic and Monte Carlo statistics in long form."""

    if frame is None or frame.empty or "Simulation" not in frame.columns:
        return pd.DataFrame()
    work = frame.copy()
    group_cols = [column for column in group_cols if column in work.columns]
    if value_cols is None:
        excluded = {"Simulation", *group_cols}
        value_cols = [
            column for column in work.columns
            if column not in excluded
            and pd.api.types.is_numeric_dtype(work[column])]
    quantiles = _profile_quantiles(user)
    lower = float(quantiles[0])
    upper = float(quantiles[-1])
    rows = []

    if group_cols:
        groups = work.groupby(group_cols, dropna=False, sort=True)
    else:
        groups = [((), work)]
    for key, group in groups:
        if not isinstance(key, tuple):
            key = (key,)
        identifiers = dict(zip(group_cols, key))
        deterministic = group.loc[group["Simulation"] == 0]
        monte_carlo = group.loc[group["Simulation"] > 0]
        for metric in value_cols:
            det_values = pd.to_numeric(
                deterministic[metric], errors="coerce").dropna()
            mc_values = pd.to_numeric(
                monte_carlo[metric], errors="coerce").dropna()
            row = {
                **identifiers,
                "Metric": metric,
                "Deterministic": (
                    float(det_values.iloc[0])
                    if not det_values.empty else np.nan),
                "MC_Mean": (
                    float(mc_values.mean())
                    if not mc_values.empty else np.nan),
                "MC_Std": (
                    float(mc_values.std(ddof=1))
                    if len(mc_values) > 1 else np.nan),
                "P025": (
                    float(mc_values.quantile(lower))
                    if not mc_values.empty else np.nan),
                "P50": (
                    float(mc_values.quantile(0.5))
                    if not mc_values.empty else np.nan),
                "P975": (
                    float(mc_values.quantile(upper))
                    if not mc_values.empty else np.nan),
                "MC_Simulations": int(len(mc_values)),}
            rows.append(row)
    return pd.DataFrame(rows)



def _public_metric_names(frame):
    """Apply canonical names to user-facing metric columns."""

    if frame is None or frame.empty or "Metric" not in frame.columns:
        return frame
    result = frame.copy()
    result["Metric"] = result["Metric"].replace(PUBLIC_METRIC_NAMES)
    return result


def _select_metric_rows(frame, metrics):
    """Return only selected metric rows from a long statistics table."""

    if frame is None or frame.empty or "Metric" not in frame.columns:
        return pd.DataFrame()
    return frame.loc[frame["Metric"].isin(metrics)].copy()


def _select_emission_rows(frame):
    """Return net emissions only for the comparison workbook."""

    if frame is None or frame.empty or "Metric" not in frame.columns:
        return pd.DataFrame()
    mask = frame["Metric"].astype(str).str.startswith("Net_Emissions")
    return frame.loc[mask].copy()


def _select_ramp_rows(frame):
    """Return canonical remaining-residual-load ramp metrics."""

    if frame is None or frame.empty:
        return pd.DataFrame()
    required = {"Variable", "Metric"}
    if not required.issubset(frame.columns):
        return pd.DataFrame()
    metrics = set(RAMP_PUBLIC_METRIC_NAMES)
    mask = (
        frame["Variable"].isin(
            ["Initial_Residual_Load", "Remaining_Residual_Load"])
        & frame["Metric"].isin(metrics))
    result = frame.loc[mask].copy()
    result["Metric"] = result["Metric"].replace(RAMP_PUBLIC_METRIC_NAMES)
    result["Unit"] = "MW/h"
    return result

def _commodity_statistics(frame, user):
    """Return annual statistics for each commodity flow metric."""

    if frame is None or frame.empty:
        return pd.DataFrame()
    quantiles = _profile_quantiles(user)
    lower = float(quantiles[0])
    upper = float(quantiles[-1])
    groups = ["Year", "Commodity", "Quantity_Unit", "Metric"]
    rows = []
    for key, group in frame.groupby(groups, dropna=False, sort=True):
        identifiers = dict(zip(groups, key))
        deterministic = pd.to_numeric(
            group.loc[group["Simulation"] == 0, "Value"],
            errors="coerce").dropna()
        mc = pd.to_numeric(
            group.loc[group["Simulation"] > 0, "Value"],
            errors="coerce").dropna()
        rows.append({
            **identifiers,
            "Deterministic": (
                float(deterministic.iloc[0])
                if not deterministic.empty else np.nan),
            "MC_Mean": float(mc.mean()) if not mc.empty else np.nan,
            "MC_Std": (
                float(mc.std(ddof=1)) if len(mc) > 1 else np.nan),
            "P025": (
                float(mc.quantile(lower)) if not mc.empty else np.nan),
            "P50": (
                float(mc.quantile(0.5)) if not mc.empty else np.nan),
            "P975": (
                float(mc.quantile(upper)) if not mc.empty else np.nan),
            "MC_Simulations": int(len(mc)),})
    return pd.DataFrame(rows)

def _samples_from_table(frame, family, item_cols=None):
    """Convert one simulation table to the compact long sample schema."""

    if frame is None or frame.empty or "Simulation" not in frame.columns:
        return pd.DataFrame()
    work = frame.copy()
    item_cols = [
        column for column in (item_cols or []) if column in work.columns]
    if item_cols:
        work["_Item"] = work[item_cols].astype(str).agg(" | ".join, axis=1)
    else:
        work["_Item"] = "System"

    identifier_cols = {"Simulation", "Year", "_Item", *item_cols}
    value_cols = [
        column for column in work.columns
        if column not in identifier_cols
        and pd.api.types.is_numeric_dtype(work[column])]
    if not value_cols:
        return pd.DataFrame()
    id_vars = ["Simulation"]
    if "Year" in work.columns:
        id_vars.append("Year")
    id_vars.append("_Item")
    result = work[id_vars + value_cols].melt(
        id_vars=id_vars, var_name="Metric", value_name="Sample_Value")
    result = result.rename(columns={"Sample_Value": "Value"})
    result = result.rename(columns={"_Item": "Item"})
    result.insert(1, "Family", family)
    if "Year" not in result.columns:
        result["Year"] = pd.Series(
            pd.NA, index=result.index, dtype="Int64")
    columns = [
        "Simulation", "Year", "Family", "Item", "Metric", "Value"]
    return result[columns]

def _write_samples(output_root, frames):
    """Write annual per-simulation samples in one compact machine table."""

    valid = [frame for frame in frames if frame is not None and not frame.empty]
    if not valid:
        return None
    samples = pd.concat(valid, ignore_index=True)
    samples["Simulation"] = pd.to_numeric(
        samples["Simulation"], errors="raise").astype(int)
    base = Path(output_root) / "Samples"
    if _parquet_engine_available():
        path = base.with_suffix(".parquet")
        samples.to_parquet(path, index=False, compression="zstd")
    else:
        path = base.with_suffix(".csv.gz")
        samples.to_csv(path, index=False, compression="gzip")
    return path


def _write_workbook(path, sheets):
    """Write non-empty result sheets with consistent basic formatting."""

    valid = {
        name[:31]: frame for name, frame in sheets.items()
        if frame is not None and not frame.empty}
    if not valid:
        return None
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for name, frame in valid.items():
            frame.to_excel(writer, sheet_name=name, index=False)
        for worksheet in writer.book.worksheets:
            worksheet.freeze_panes = "A2"
            if worksheet.max_row > 1 and worksheet.max_column > 1:
                worksheet.auto_filter.ref = worksheet.dimensions
            for column in worksheet.columns:
                width = max(
                    len(str(cell.value)) if cell.value is not None else 0
                    for cell in column)
                letter = column[0].column_letter
                worksheet.column_dimensions[letter].width = min(
                    max(width + 2, 10), 28)
    return path


def _commodity_peak_timing(frame):
    """Count the month/hour combinations containing annual commodity peaks."""

    if frame is None or frame.empty:
        return pd.DataFrame()
    required = {
        "Simulation", "Year", "Commodity",
        "Peak_Production_Month", "Peak_Production_Hour"}
    if not required.issubset(frame.columns):
        return pd.DataFrame()
    mc = frame.loc[frame["Simulation"] > 0].copy()
    if mc.empty:
        return pd.DataFrame()
    groups = [
        "Year", "Commodity", "Peak_Production_Month",
        "Peak_Production_Hour"]
    counts = mc.groupby(groups, dropna=False).size().reset_index(name="Count")
    totals = counts.groupby(
        ["Year", "Commodity"])["Count"].transform("sum")
    counts["Fraction"] = np.divide(
        counts["Count"], totals,
        out=np.zeros(len(counts), dtype=float), where=totals > 0)
    return counts


def _public_output_metadata(user, complete_ids):
    """Return compact metadata describing one consolidated output set."""

    mc_ids = [value for value in complete_ids if int(value) > 0]
    monte_carlo = user.get("monte_carlo", {}) or {}
    return pd.DataFrame({
        "Field": [
            "Output_Level", "Energy_Unit", "Power_Unit", "Ramp_Unit",
            "Monte_Carlo_Simulations", "Seed", "Confidence_Level",
            "Start_Date", "End_Date"],
        "Value": [
            get_output_level(user),
            canonical_energy_unit(user.get("energy_unit", "MWh")),
            "MW", "MW/h", len(mc_ids),
            monte_carlo.get("seed", 12345),
            monte_carlo.get("confidence_level", 0.95),
            user.get("start_date"), user.get("end_date")]})


def _assign_public_units(frame, unit_map, default=None):
    """Attach explicit units to a public long-form metric table."""

    if frame is None or frame.empty or "Metric" not in frame.columns:
        return frame
    result = frame.copy()
    result["Unit"] = result["Metric"].map(unit_map)
    if default is not None:
        result["Unit"] = result["Unit"].fillna(default)
    return result


def _build_public_outputs(output_root, complete_ids, user):
    """Collapse internal tables into comparison, analysis, or detailed outputs."""

    root = Path(output_root)
    level = get_output_level(user)
    annual_system = _final_table(
        root, SUMMARY_FOLDER, "Annual_System_Summary")
    annual_sources = _final_table(
        root, SUMMARY_FOLDER, "Annual_Source_Summary")
    energy_flows = _final_table(
        root, SUMMARY_FOLDER, "Energy_Flow_Summary")
    bess = _final_table(root, SUMMARY_FOLDER, "BESS_Lifetime_Summary")
    commodities = _final_table(
        root, SUMMARY_FOLDER, "Annual_Commodity_Summary")
    emissions = _final_table(
        root, SUMMARY_FOLDER, "Annual_Emissions_Summary")
    ramps = _final_table(root, SUMMARY_FOLDER, "Ramp_Summary")
    convergence = _final_table(
        root, SUMMARY_FOLDER, "Monte_Carlo_Convergence")
    annual_fuel = _final_table(
        root, SUMMARY_FOLDER,
        "Annual_Nuclear_Fuel_Summary_By_Simulation")
    fuel_discharge = _final_table(
        root, SUMMARY_FOLDER,
        "Nuclear_Fuel_Discharge_By_Simulation")

    nuclear_stats = pd.DataFrame()
    nuclear_samples = pd.DataFrame()
    nuclear_book = root / "Nuclear_Generation.xlsx"
    if nuclear_book.is_file():
        nuclear_stats = pd.read_excel(
            nuclear_book, sheet_name="Annual", engine="openpyxl")
        nuclear_samples = pd.read_excel(
            nuclear_book, sheet_name="Annual_By_Simulation",
            engine="openpyxl")

    energy_unit = canonical_energy_unit(user.get("energy_unit", "MWh"))
    system_results = _public_metric_names(_statistics_table(
        annual_system, ["Year"], user,
        value_cols=COMPARISON_SYSTEM_METRICS))
    system_results = _assign_public_units(system_results, {
        "Demand": energy_unit,
        "Generation": energy_unit,
        "Initial_Surplus_Energy": energy_unit,
        "Initial_Positive_Residual_Load_Energy": energy_unit,
        "BESS_Charge": energy_unit,
        "Battery_Discharge": energy_unit,
        "Reconversion_Generation": energy_unit,
        "Remaining_Positive_Residual_Load_Energy": energy_unit,
        "Peak_Initial_Residual_Load": "MW",
        "Peak_Remaining_Residual_Load": "MW"})
    bess_results = _public_metric_names(_statistics_table(
        bess, ["Year"], user, value_cols=COMPARISON_BESS_METRICS))
    bess_results = _assign_public_units(bess_results, {
        "BESS_Power_Capacity": "MW",
        "BESS_Energy_Capacity": energy_unit,
        "BESS_Charge": energy_unit,
        "Battery_Discharge": energy_unit,
        "Full_Equivalent_Cycles": "cycles/year"})
    emission_results = _select_emission_rows(
        _statistics_table(emissions, ["Year"], user))
    ramp_results = _select_ramp_rows(
        _statistics_table(ramps, ["Variable"], user))
    nuclear_fuel_results = _statistics_table(
        annual_fuel,
        ["Year", "Source", "Fuel_Mass_Unit", "Burnup_Unit"],
        user)
    results = {
        "System": system_results,
        "Generation": _statistics_table(
            annual_sources, ["Year", "Source"], user,
            value_cols=["Generation"] if annual_sources is not None else None),
        "BESS": bess_results,
        "Commodities": _commodity_statistics(commodities, user),
        "Emissions": emission_results,
        "Ramps": ramp_results,
        "Nuclear": nuclear_stats,
        "Nuclear_Fuel": nuclear_fuel_results,
        "Convergence": _public_metric_names(convergence),
        "Metadata": _public_output_metadata(user, complete_ids),}
    _write_workbook(root / "Results.xlsx", results)

    system_samples = annual_system[[
        column for column in [
            "Simulation", "Year", *COMPARISON_SYSTEM_METRICS]
        if annual_system is not None and column in annual_system.columns
    ]].copy() if annual_system is not None else pd.DataFrame()
    bess_samples = bess[[
        column for column in [
            "Simulation", "Year", *COMPARISON_BESS_METRICS]
        if bess is not None and column in bess.columns
    ]].copy() if bess is not None else pd.DataFrame()
    if emissions is not None and not emissions.empty:
        emission_columns = [
            column for column in emissions.columns
            if column in {"Simulation", "Year"}
            or str(column).startswith("Net_Emissions")]
        emission_samples = emissions[emission_columns].copy()
    else:
        emission_samples = pd.DataFrame()
    if ramps is not None and not ramps.empty and "Variable" in ramps.columns:
        ramp_samples = ramps.loc[
            ramps["Variable"].isin(
                ["Initial_Residual_Load", "Remaining_Residual_Load"]),
            [column for column in [
                "Simulation", "Variable", "P95_Ramp_Magnitude",
                "P99_Ramp_Magnitude", "Maximum_Ramp_Magnitude"]
             if column in ramps.columns]].copy()
    else:
        ramp_samples = pd.DataFrame()

    sample_frames = [
        _public_metric_names(_samples_from_table(system_samples, "System")),
        _samples_from_table(annual_sources, "Generation", ["Source"]),
        _public_metric_names(_samples_from_table(bess_samples, "BESS")),
        _samples_from_table(
            commodities, "Commodity", ["Commodity", "Metric"]),
        _samples_from_table(emission_samples, "Emissions"),
        _public_metric_names(
            _samples_from_table(ramp_samples, "Ramps", ["Variable"])),
        _samples_from_table(nuclear_samples, "Nuclear", ["Source"]),
        _samples_from_table(
            annual_fuel, "Nuclear_Fuel_Annual", ["Source"]),
        _samples_from_table(
            fuel_discharge, "Nuclear_Fuel_Discharge",
            ["Source", "Unit", "Refueling_Number", "Discharge_Date"]),]

    if level in {"analysis", "detailed"}:
        initial_deficit = _final_table(
            root, SUMMARY_FOLDER,
            "Annual_Initial_Positive_Residual_Load_Episode_Summary")
        deficit = _final_table(
            root, SUMMARY_FOLDER,
            "Annual_Remaining_Positive_Residual_Load_Episode_Summary")
        surplus = _final_table(
            root, SUMMARY_FOLDER, "Annual_Surplus_Episode_Summary")
        bess_operation = _final_table(
            root, SUMMARY_FOLDER, "BESS_Operation")
        commodity_operation = _final_table(
            root, SUMMARY_FOLDER, "Commodity_Operation")
        technology_use = _final_table(
            root, SUMMARY_FOLDER,
            "Annual_Technology_Use_By_Simulation")
        sample_frames.extend([
            _public_metric_names(
                _samples_from_table(energy_flows, "Energy_Flow")),
            _samples_from_table(emissions, "Emissions_All"),
            _samples_from_table(ramps, "Ramps_All", ["Variable"]),
            _samples_from_table(
                initial_deficit, "Initial_Positive_Residual_Load_Episodes"),
            _samples_from_table(
                deficit, "Remaining_Positive_Residual_Load_Episodes"),
            _samples_from_table(surplus, "Surplus_Episodes"),
            _samples_from_table(bess_operation, "BESS_Operation"),
            _samples_from_table(
                commodity_operation, "Commodity_Operation", ["Commodity"]),
            _samples_from_table(
                technology_use, "Technology_Use",
                ["Category", "Technology"]),])

        tallies = {
            "Daily_Publication": _final_table(
                root, PROFILES_FOLDER, "Daily_Publication_Statistics"),
            "Energy_Flows": _public_metric_names(_statistics_table(
                energy_flows, ["Year"], user)),
            "BESS_Lifetime": _public_metric_names(_statistics_table(
                bess, ["Year"], user)),
            "Emissions": _statistics_table(emissions, ["Year"], user),
            "Ramps": _statistics_table(ramps, ["Variable"], user),
            "Monthly_System": _final_table(
                root, SUMMARY_FOLDER, "Monthly_System_Statistics"),
            "Monthly_Generation": _final_table(
                root, SUMMARY_FOLDER, "Monthly_Source_Statistics"),
            "Initial_Positive_Residual_Load_Episodes": _statistics_table(
                initial_deficit, ["Year"], user),
            "Remaining_Positive_Residual_Load_Episodes": _statistics_table(
                deficit, ["Year"], user),
            "Surplus_Episodes": _statistics_table(
                surplus, ["Year"], user),
            "BESS_Operation": _statistics_table(
                bess_operation, ["Year"], user),
            "BESS_SOC": _final_table(
                root, PROFILES_FOLDER, "BESS_SOC_Distribution"),
            "BESS_Hourly": _final_table(
                root, PROFILES_FOLDER, "BESS_Hourly_Statistics"),
            "BESS_Monthly": _final_table(
                root, PROFILES_FOLDER, "BESS_Monthly_Statistics"),
            "Commodity_Operation": _statistics_table(
                commodity_operation,
                ["Year", "Commodity", "Quantity_Unit"], user),
            "Commodity_Peak_Timing": _commodity_peak_timing(
                commodity_operation),
            "Commodity_Hourly": _final_table(
                root, PROFILES_FOLDER, "Commodity_Hourly_Statistics"),
            "Commodity_Monthly": _final_table(
                root, PROFILES_FOLDER, "Commodity_Monthly_Statistics"),
            "System_Profile": _final_table(
                root, PROFILES_FOLDER, "Temporal_Profile_Statistics"),
            "Technology_Events": _statistics_table(
                _final_table(
                    root, SUMMARY_FOLDER, "Technology_Event_Summary"),
                ["Year", "Category", "Technology"], user),
            "Residual_Load_Duration_Curve": _statistics_table(
                _final_table(
                    root, SUMMARY_FOLDER,
                    "Residual_Load_Duration_Curve_Summary"),
                ["Stage", "Exceedance"], user,
                value_cols=["Residual_Load"]),
            "Nuclear_Fuel_Annual": nuclear_fuel_results,
            "Nuclear_Fuel_Events": _statistics_table(
                fuel_discharge,
                ["Source", "Unit", "Refueling_Number", "Discharge_Date",
                 "Fuel_Mass_Unit", "Burnup_Unit"],
                user,
                value_cols=["Discharged_Fuel", "Discharge_Burnup",
                            "Cycle_EFPD"]),}

        if nuclear_book.is_file():
            workbook = pd.ExcelFile(nuclear_book, engine="openpyxl")
            available = set(workbook.sheet_names)
            if "Daily_Statistics" in available:
                tallies["Nuclear_Daily"] = pd.read_excel(
                    workbook, sheet_name="Daily_Statistics")
            if "Refueling_Uncertainty" in available:
                tallies["Refueling"] = pd.read_excel(
                    workbook, sheet_name="Refueling_Uncertainty")
            if "Refueling_By_Simulation" in available:
                tallies["Refueling_Samples"] = pd.read_excel(
                    workbook, sheet_name="Refueling_By_Simulation")
            if "Monte_Carlo_Convergence" in available:
                tallies["Nuclear_Convergence"] = pd.read_excel(
                    workbook, sheet_name="Monte_Carlo_Convergence")
            workbook.close()
        refueling_impact = _final_table(
            root, SUMMARY_FOLDER,
            "Nuclear_Refueling_Impact_By_Simulation")
        if refueling_impact is not None and not refueling_impact.empty:
            tallies["Refueling_Impact"] = _statistics_table(
                refueling_impact, ["Month"], user)
        _write_workbook(root / "Tallies.xlsx", tallies)

    _write_samples(root, sample_frames)
    nuclear_book.unlink(missing_ok=True)
    for folder in (SUMMARY_FOLDER, PROFILES_FOLDER):
        path = root / folder
        if path.is_dir():
            shutil.rmtree(path)
    if level != "detailed":
        detailed = root / FULL_FOLDER
        if detailed.is_dir():
            shutil.rmtree(detailed)


def finalize_runtime_diagnostics(output_dir, temp_dir, simulation_ids):
    """Write the always-present runtime diagnostics workbook."""

    raw = _runtime_diagnostics_table(temp_dir, simulation_ids)
    summary = _capacity_clipping_summary(raw, simulation_ids)
    path = Path(output_dir) / "Diagnostics.xlsx"
    path.parent.mkdir(parents=True, exist_ok=True)
    overview = pd.DataFrame({
        "Field": ["Completed_Simulations", "Runtime_Diagnostic_Rows"],
        "Value": [len(simulation_ids), len(raw)]})
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        overview.to_excel(writer, sheet_name="Overview", index=False)
        if not summary.empty:
            summary.to_excel(
                writer, sheet_name="Capacity_Clipping", index=False)
        if not raw.empty:
            raw.to_excel(writer, sheet_name="Raw_Diagnostics", index=False)
    return summary


def finalize_analysis_batch(output_dir, simulation_ids, user):
    """Consolidate completed simulation analyses and clean temporary data."""

    analysis_dir, temp_dir = get_analysis_paths(
        output_dir, create=False)
    output_root = Path(output_dir)
    complete_ids = sorted(
        get_completed_simulations(
            output_dir, simulation_ids, user))
    missing_ids = sorted(set(simulation_ids) - set(complete_ids))

    if missing_ids:
        preview = ", ".join(str(value) for value in missing_ids[:20])
        if len(missing_ids) > 20:
            preview += ", ..."
        message = (
            "Analysis consolidation stopped. Missing or incomplete "
            f"simulations: {preview}")
        raise RuntimeError(message)

    file_names = set()

    for simulation_id in complete_ids:
        simulation_dir = temp_dir / f"{simulation_id:06d}"
        if simulation_dir.is_dir():
            file_names.update(
                path.name for path in simulation_dir.iterdir()
                if path.suffix in {".parquet", ".pkl", ".csv"})

    accumulator_dir = temp_dir / "_accumulators"

    def consolidation_priority(file_name):
        """Run memory-intensive profile aggregation before table joins."""

        stem = Path(file_name).stem
        if stem == "Temporal_Profile_Summary":
            return 0, stem
        original_name = f"{stem}.csv"
        if _table_folder(original_name) == PROFILES_FOLDER:
            return 1, stem
        return 2, stem

    _nuclear_generation_workbook(
        temp_dir, complete_ids, output_root, user)
    nuclear_stems = {
        "Daily_Nuclear_Generation", "Annual_Nuclear_Generation",
        "Monthly_Nuclear_Generation",
        "Nuclear_Refueling_Schedule_By_Simulation"}
    ordered_files = [
        file_name for file_name in sorted(
            file_names, key=consolidation_priority)
        if Path(file_name).stem not in nuclear_stems]
    compact_monthly = {
        "Monthly_System_Summary.csv",
        "Monthly_Source_Summary.csv",
        "Monthly_Emissions_Summary.csv"}
    for file_name in ordered_files:
        stem = Path(file_name).stem
        original_name = f"{stem}.csv"
        folder = _table_folder(original_name)
        paths = _simulation_table_paths(
            temp_dir, complete_ids, file_name)
        use_statistics = (
            folder == PROFILES_FOLDER
            or original_name in compact_monthly)

        if use_statistics:
            if stem.endswith("_Summary"):
                statistics_name = stem.replace(
                    "_Summary", "_Statistics")
            else:
                statistics_name = stem.replace(
                    "Profile", "Statistics")
            final_base = output_root / folder / statistics_name
            accumulator_path = accumulator_dir / file_name
            _consolidate_profile_table(
                paths, file_name, final_base,
                accumulator_path, user)

            if original_name in compact_monthly:
                deterministic_paths = [
                    path for path in paths
                    if _profile_simulation_id(path) == 0]
                if deterministic_paths:
                    deterministic_base = output_root / folder / stem
                    _consolidate_regular_table(
                        deterministic_paths, deterministic_base)
            continue

        final_base = output_root / folder / stem
        _consolidate_regular_table(paths, final_base)

    refresh_monte_carlo_outputs(analysis_dir, user)
    diagnostics_summary = finalize_runtime_diagnostics(
        output_dir, temp_dir, complete_ids)
    _build_public_outputs(output_root, complete_ids, user)

    if not _keep_analysis_temp(user):
        clear_analysis_temp(output_dir)

    for folder in (SUMMARY_FOLDER, PROFILES_FOLDER, FULL_FOLDER):
        folder_path = output_root / folder
        if folder_path.is_dir() and not any(folder_path.iterdir()):
            folder_path.rmdir()
    return diagnostics_summary

def refresh_monte_carlo_outputs(analysis_dir, user):
    """Refresh Monte Carlo convergence summaries."""

    _refresh_monte_carlo_outputs(Path(analysis_dir), user)


def _column_by_prefix(frame, prefix):
    """Return an exact metric column, allowing only a unit suffix."""

    matches = []
    for column in frame.columns:
        text = str(column)
        if text == prefix or text.startswith(f"{prefix} ("):
            matches.append(column)
    if len(matches) > 1:
        names = ", ".join(str(column) for column in matches)
        raise ValueError(
            f"Ambiguous metric columns for '{prefix}': {names}.")
    return matches[0] if matches else None


def _series_by_prefix(frame, prefix):
    """Return a numeric series matched by an exact metric prefix."""

    column = _column_by_prefix(frame, prefix)

    if column is None:
        return None

    return pd.to_numeric(frame[column], errors="coerce").fillna(0.0)


def _series_by_prefix_or_zero(frame, prefix):
    """Return a matched metric series or zeros when it is absent."""

    series = _series_by_prefix(frame, prefix)

    return _zero_if_none(series, len(frame))


def _series_or_none(frame, column):
    """Return one numeric column or None when the column is absent."""

    if column not in frame.columns:
        return None

    return pd.to_numeric(frame[column], errors="coerce").fillna(0.0)


def _series_or_zero(frame, column):
    """Return one numeric column or zeros when the column is absent."""

    series = _series_or_none(frame, column)

    return _zero_if_none(series, len(frame))


def _zero_if_none(series, length):
    """Replace a missing series with a zero-valued series."""

    if series is None:
        return pd.Series(np.zeros(length), dtype=float)

    return pd.Series(series).reset_index(drop=True)


def _positive(series, length):
    """Return the positive part of a numeric series."""

    values = _zero_if_none(series, length)

    return values.clip(lower=0.0)


def _negative(series, length):
    """Return the magnitude of the negative part of a numeric series."""

    values = _zero_if_none(series, length)

    return -values.clip(upper=0.0)
