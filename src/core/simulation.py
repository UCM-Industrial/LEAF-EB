"""Coordinate LEAF-EB Monte Carlo and electricity-balance simulations.

This module loads the deterministic forecast, prepares stochastic inputs,
launches isolated worker processes, and delegates hourly balancing, storage,
commodity conversion, emissions, and result analysis to focused modules.
"""

from multiprocessing import cpu_count
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import traceback

import numpy as np
import pandas as pd
import yaml

from src.analysis import scenario
from src.analysis.emissions import calculate_emissions
from src.core import output as output_io
from src.core.hourly import prepare_operational_data
from src.core.monte_carlo import (
    load_cov_matrix as load_monthly_cov_matrix,
    perturb_df_energies,
    residual_params_dict)
from src.interfaces.anicca import save_anicca_input
from src.technologies.flexibility import (
    build_storage_config, configure_flexibility, surplus)
from src.technologies.nuclear.fuel import (
    attach_calendar_fuel_diagnostics)
from src.technologies.nuclear.refueling import build_refueling_profiles
from src.forecasting.custom_sources import apply_custom_replacements
from src.utilities.bootstrap_tools import (
    aggregate_stationary_block_lengths,
    annual_variability_scales)
from src.utilities.configuration import (
    load_config_file, write_resolved_config)
from src.utilities.console import emit
from src.utilities.memory import (
    memory_safe_worker_count, release_unused_memory)
from src.utilities.name_resolution import (
    build_name_lookup,
    name_key,
    normalize_frame_columns,
    normalize_input_frame,
    normalize_source_axis,
    normalize_source_values)
from src.utilities.units import (
    commodity_quantity_unit,
    kwh_per_quantity_to_energy_factor,
    normalize_energy_unit)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
User = {}
Database = {}
BALANCE_TOLERANCE = 1e-9
RUN_ID = ""
REFUELING_SCHEDULE = pd.DataFrame()


def _stochastic_run_enabled() -> bool:
    """Return whether the current worker is a stochastic realization."""

    return bool(User.get("_stochastic_run", False))


def _technology_uncertainty_enabled() -> bool:
    """Return whether technology-parameter uncertainty is enabled.

    Temporal Monte Carlo sampling and technology uncertainty are separate
    choices. Database distributions are sampled only when this explicit
    option is true. Deterministic runs use fixed values or triangular modes.
    """

    monte_carlo = User.get("monte_carlo", {}) or {}
    return (
        _stochastic_run_enabled()
        and bool(monte_carlo.get("technology_uncertainty", False))
    )


def _database_value(values, sample_key=None):
    """Resolve one Database.yaml scalar or referenced distribution.

    ``[value]`` is fixed and ``[min, mode, max]`` is triangular. Named
    stochastic parameters are sampled once per Monte Carlo simulation and
    then reused consistently.
    """

    if not isinstance(values, (list, tuple)) or not values:
        raise ValueError(
            "Database technology values must be non-empty lists or tuples.")
    if len(values) not in {1, 3}:
        raise ValueError(
            "Database values must contain one fixed value or three "
            "triangular parameters [min, mode, max].")

    numeric = [float(value) for value in values]
    if not all(np.isfinite(value) for value in numeric):
        raise ValueError("Database technology values must be finite.")

    if len(numeric) == 1:
        return numeric[0]

    low = numeric[0]
    high = numeric[-1]
    if low > high:
        raise ValueError("Database distribution bounds must be ordered.")

    mode = numeric[1]
    if not low <= mode <= high:
        raise ValueError(
            "Triangular Database values must satisfy min <= mode <= max.")
    if not _technology_uncertainty_enabled():
        return mode
    sampler = lambda: random.triangular(low, high, mode)

    if sample_key is None:
        return sampler()
    samples = User.setdefault("_technology_samples", {})
    if sample_key not in samples:
        samples[sample_key] = sampler()
    return float(samples[sample_key])


def load_user_config_from_path(cfg_path: str):
    """Load the current-schema configuration from one YAML path."""

    return load_config_file(cfg_path)


def load_database():
    """Load the configured commodity technology database."""

    data_path = Path(User.get("database_path", "data/Database.yaml"))
    if not data_path.is_absolute():
        data_path = PROJECT_ROOT / data_path
    data_path = data_path.resolve()
    with data_path.open("r", encoding="utf-8") as stream:
        emit(User, f"Data file loaded: {data_path}", "detailed")
        return yaml.safe_load(stream)

def load_forecast():
    """Load and normalize the deterministic scenario forecast.

    ``Forecast.xlsx`` remains the user-facing scenario output.  When the
    matching ``Forecast.csv`` cache is present and at least as recent,
    workers use it to avoid repeated Excel parsing during Monte Carlo runs.
    """

    module_dir = Path(__file__).resolve().parents[2]
    forecast_path = module_dir / User["scenario_folder"]
    forecast_path /= User["scenario_subfolder"]
    forecast_path /= "Forecast.xlsx"
    if not forecast_path.is_file():
        raise FileNotFoundError(f"No Forecast file found at {forecast_path}")

    cache_path = forecast_path.with_suffix(".csv")
    use_cache = cache_path.is_file() and (
        cache_path.stat().st_mtime_ns
        >= forecast_path.stat().st_mtime_ns)
    if use_cache:
        forecast = pd.read_csv(cache_path)
        context = "Forecast.csv"
    else:
        forecast = pd.read_excel(forecast_path, engine="openpyxl")
        context = "Forecast.xlsx"
    return normalize_input_frame(forecast, User, context)




def _has_custom_replacement():
    """Return whether the scenario contains a custom replacement source."""

    for source in User.get("sources", {}).values():
        if not isinstance(source, dict):
            continue
        model = str(source.get("model", "")).strip().lower()
        mode = str(source.get("custom_mode", "add")).strip().lower()
        if model == "custom" and mode == "replace":
            return True
    return False


def load_pre_replacement_forecast(reference_df):
    """Load the unreduced forecast used for paired Monte Carlo matching."""

    module_dir = Path(__file__).resolve().parents[2]
    path = module_dir / User["scenario_folder"]
    path /= User["scenario_subfolder"]
    path /= "Forecast_PreReplacement.csv"
    if not path.is_file():
        raise FileNotFoundError(
            "Custom replacement Monte Carlo requires "
            f"'{path.name}'. Rerun Runner.py so Predictor regenerates the "
            "scenario forecast.")
    forecast = pd.read_csv(path)
    forecast = normalize_input_frame(
        forecast, User, "Forecast_PreReplacement.csv")
    reference_dates = pd.to_datetime(reference_df["Date"])
    dates = pd.to_datetime(forecast["Date"])
    start = reference_dates.min()
    end = reference_dates.max()
    selected = forecast.loc[(dates >= start) & (dates <= end)].copy()
    if len(selected) != len(reference_df):
        raise ValueError(
            "Forecast_PreReplacement.csv does not match the selected "
            "scenario period. Rerun Runner.py to regenerate the forecast.")
    return selected.reset_index(drop=True)


def get_commodities_production_config(required=True):
    """Return the configured commodity-production mapping."""
    if "Commodities_Production" in User:
        return User["Commodities_Production"] or {}

    if required:
        raise ValueError("Missing 'Commodities_Production' in the input YAML.")

    return {}


def _technology_consumption(commodity, technology, conversion):
    """Return one technology's electricity use within its commodity scope."""

    electricity = Database.get("Electricity_Consumption", {}) or {}
    commodity_data = electricity.get(commodity)
    if not isinstance(commodity_data, dict):
        raise ValueError(
            f"Commodity '{commodity}' has no Electricity_Consumption "
            "definition in Database.yaml.")
    if technology not in commodity_data:
        raise ValueError(
            f"Technology '{technology}' is not defined for commodity "
            f"'{commodity}' in Database.yaml.")
    sample_key = f"Electricity_Consumption.{commodity}.{technology}"
    return _database_value(
        commodity_data[technology], sample_key) * conversion


def technology_values():
    """Return pathway electricity use in database input order."""

    energy_unit = User.get("energy_unit", "MWh")
    conversion = kwh_per_quantity_to_energy_factor(energy_unit)
    single_values = []
    dependent_values = []
    production_inputs = Database.get("Production_Inputs", {}) or {}

    for commodity, config in get_commodities_production_config().items():
        direct = _technology_consumption(
            commodity, config["technology"], conversion)
        inputs = config.get("inputs", {}) or {}
        requirements = production_inputs.get(commodity)

        if inputs and not isinstance(requirements, dict):
            raise ValueError(
                f"Commodity '{commodity}' configures input technologies but "
                "has no Production_Inputs definition in Database.yaml.")
        if not inputs:
            single_values.append(direct)
            continue

        expected = list(requirements)
        missing = [name for name in expected if name not in inputs]
        extra = [name for name in inputs if name not in requirements]
        if missing or extra:
            details = []
            if missing:
                details.append("missing: " + ", ".join(missing))
            if extra:
                details.append("unexpected: " + ", ".join(extra))
            raise ValueError(
                f"Production-input technologies for '{commodity}' do not "
                "match Database.yaml (" + "; ".join(details) + ").")

        consumptions = [direct]
        for input_name in expected:
            consumptions.append(_technology_consumption(
                input_name, inputs[input_name], conversion))
        dependent_values.append(consumptions)

    return single_values, dependent_values

def production_share():
    """Return commodity production shares."""

    single_perc = []
    dependent_perc = []

    for config in get_commodities_production_config().values():
        share = float(config["share"])

        if config.get("inputs"):
            dependent_perc.append(share)
        else:
            single_perc.append(share)

    total = sum(single_perc) + sum(dependent_perc)

    if abs(total - 1.0) > 1e-9:
        raise ValueError(
            "Commodities_Production shares must sum to 1.")

    return single_perc, dependent_perc

def final_production(perturbed_df_energies_data):
    """Convert surplus electricity into commodity production."""

    energy_unit = normalize_energy_unit(User.get("energy_unit", "MWh"))
    input_balance_col = (
        "Electricity_Balance_After_Storage_and_Interconnections "
        f"({energy_unit})")
    production_col = (
        f"Commodity_Production_Electricity ({energy_unit})")
    output_balance_col = (
        "Electricity_Balance_After_Commodity_Production "
        f"({energy_unit})")

    sur = surplus(perturbed_df_energies_data)
    config = get_commodities_production_config(required=False)

    # Commodity production is optional. When no pathway is configured,
    # preserve the post-storage/interconnection balance unchanged. This
    # allows basic electricity-balance and BESS-only public examples.
    if not config:
        input_balance = pd.to_numeric(
            sur[input_balance_col], errors="coerce").fillna(0.0)
        sur[production_col] = 0.0
        sur[output_balance_col] = input_balance.mask(
            input_balance.abs() <= BALANCE_TOLERANCE, 0.0)
        return sur

    single_cons, dependent_cons = technology_values()
    single_shares, dependent_shares = production_share()

    single_names = [name for name, values in config.items()
        if not values.get("inputs")]

    dependent_names = [name for name, values in config.items()
        if values.get("inputs")]

    dependent_total_cons = []

    for name, consumptions in zip(dependent_names, dependent_cons):
        production_inputs = Database["Production_Inputs"][name]
        direct_consumption = float(consumptions[0])
        auxiliary_cons = consumptions[1:]
        input_requirements = list(production_inputs.items())

        if len(auxiliary_cons) != len(input_requirements):
            message = (
                f"Invalid production-input definition for commodity "
                f"'{name}'.")
            raise ValueError(message)

        total_consumption = direct_consumption

        for (input_name, input_requirement), aux_cons in zip(
                input_requirements, auxiliary_cons):
            sample_key = f"Production_Inputs.{name}.{input_name}"
            input_factor = _database_value(
                input_requirement, sample_key)
            total_consumption += input_factor * aux_cons

        dependent_total_cons.append(total_consumption)

    commodity_names = single_names + dependent_names
    shares = [*single_shares, *dependent_shares]
    consumptions = [*single_cons, *dependent_total_cons]

    for name in commodity_names:
        sur[name] = 0.0

    input_balance = pd.to_numeric(
        sur[input_balance_col], errors="coerce").fillna(0.0)
    surplus_energy = input_balance.clip(lower=0.0)
    allocated_energy = pd.Series(0.0, index=sur.index)
    for name, share, consumption in zip(
            commodity_names, shares, consumptions):
        if consumption <= 0.0:
            continue
        pathway_energy = surplus_energy * share
        sur[name] = pathway_energy / consumption
        allocated_energy += pathway_energy

    allocated_energy = np.minimum(allocated_energy, surplus_energy)
    remaining_surplus = surplus_energy - allocated_energy
    remaining_surplus = remaining_surplus.clip(lower=0.0)
    remaining_surplus = remaining_surplus.mask(
        remaining_surplus <= BALANCE_TOLERANCE, 0.0)

    output_balance = input_balance.copy()
    positive_mask = input_balance > BALANCE_TOLERANCE
    sur[production_col] = allocated_energy
    output_balance.loc[positive_mask] = remaining_surplus.loc[positive_mask]
    output_balance = output_balance.mask(
        output_balance.abs() <= BALANCE_TOLERANCE, 0.0)
    sur[output_balance_col] = output_balance

    return sur

# -----------------------------
# Simple FIFO storage
# -----------------------------


def _fuel_to_electricity_config():
    """Resolve fuel shares and reconversion technologies.

    The public schema is ``commodity: {share, technology}``. Both fields are
    explicit so reconversion never depends on Database.yaml mapping order.
    """

    raw = User.get("Fuel_to_Electricity", {}) or {}
    if not isinstance(raw, dict):
        raise ValueError("Fuel_to_Electricity must be a mapping.")
    resolved = {}
    database = Database.get("Commodities_to_Electricity", {}) or {}
    for commodity, setting in raw.items():
        technologies = database.get(commodity, {}) or {}
        if not isinstance(setting, dict):
            raise ValueError(
                f"Fuel_to_Electricity.{commodity} must define share and "
                "technology explicitly.")
        share = float(setting.get("share", 0.0))
        technology = str(setting.get("technology", "")).strip()
        if not technology:
            raise ValueError(
                f"Fuel_to_Electricity.{commodity}.technology is required.")
        if share < 0.0:
            raise ValueError(
                f"Fuel_to_Electricity.{commodity}.share cannot be negative.")
        if technology not in technologies:
            raise ValueError(
                f"Fuel-to-electricity technology '{technology}' for "
                f"'{commodity}' was not found in Database.yaml.")
        resolved[commodity] = {
            "share": share, "technology": technology}
    return resolved


def get_conversion_factors():
    """Return configured commodity-to-electricity factors.

    Database energy densities are stored in kWh per physical quantity.
    Technology distributions are sampled only when explicit technology
    uncertainty is enabled; deterministic runs use their documented central
    values.
    """

    energy_unit = User.get("energy_unit", "MWh")
    conv = kwh_per_quantity_to_energy_factor(energy_unit)
    result_dict = {}
    database = Database.get("Commodities_to_Electricity", {}) or {}

    for commodity, setting in _fuel_to_electricity_config().items():
        technology = setting["technology"]
        values = database[commodity][technology]
        if not isinstance(values, (list, tuple)) or len(values) != 2:
            raise ValueError(
                f"Invalid reconversion definition for {commodity}/{technology}.")
        prefix = f"Commodities_to_Electricity.{commodity}.{technology}"
        efficiency = _database_value(
            values[0], f"{prefix}.efficiency")
        energy_density = _database_value(
            values[1], f"{prefix}.energy_density") * conv
        result_dict[commodity] = efficiency * energy_density

    return result_dict


def _zero_commodity_arrays(names, size):
    """Return one zeroed float32 output array per commodity."""

    return {name: np.zeros(size, dtype=np.float32) for name in names}


def to_electricity_with_storage(perturbed_df_energies_data):
    """Produce, store and burn commodities using FIFO storage."""

    energy_unit = normalize_energy_unit(User.get("energy_unit", "MWh"))
    balance_col = (
        "Electricity_Balance_After_Commodity_Production "
        f"({energy_unit})")

    df = final_production(perturbed_df_energies_data)
    conv = get_conversion_factors()
    fuel_to_electricity = _fuel_to_electricity_config()

    production_cfg = get_commodities_production_config(
        required=False)
    commodity_cols = [
        name for name in production_cfg if name in df.columns]

    supply_col = f"Supply_From_Fuel_Reconversion ({energy_unit})"
    residual_col = f"Residual_Supply_Requirement ({energy_unit})"
    final_col = f"Final_Electricity_Balance ({energy_unit})"

    if not commodity_cols:
        df[f"Fuel_Reconversion_Generation ({energy_unit})"] = 0.0
        df[supply_col] = 0.0
        df[final_col] = df[balance_col]
        df[residual_col] = -df[final_col].clip(upper=0.0)
        return df

    storages = build_storage_config(commodity_cols, User)
    row_count = len(df)
    dates = pd.to_datetime(df["Date"]).to_numpy()
    simulation_cfg = User.get("simulation", {})
    initial_cfg = {}
    if isinstance(simulation_cfg, dict):
        initial_cfg = simulation_cfg.get("initial_conditions", {}) or {}
    inventory_cfg = {}
    if isinstance(initial_cfg, dict):
        configured_inventory = initial_cfg.get(
            "commodity_inventory", {}) or {}
        if isinstance(configured_inventory, dict):
            inventory_cfg.update(configured_inventory)
    if row_count and isinstance(inventory_cfg, dict):
        for name, quantity in inventory_cfg.items():
            if name in storages and float(quantity) > 0.0:
                storages[name].add(dates[0], float(quantity))
    balances = df[balance_col].to_numpy(dtype=np.float32, copy=False)

    produced = {
        name: df[name].to_numpy(dtype=np.float32, copy=False)
        for name in commodity_cols}
    stored = _zero_commodity_arrays(commodity_cols, row_count)
    sold = _zero_commodity_arrays(commodity_cols, row_count)
    expired = _zero_commodity_arrays(commodity_cols, row_count)
    burned = _zero_commodity_arrays(commodity_cols, row_count)
    potential = _zero_commodity_arrays(commodity_cols, row_count)
    inventory = _zero_commodity_arrays(commodity_cols, row_count)

    supply_from_reconversion = np.zeros(row_count, dtype=np.float32)
    reconversion_generation = np.zeros(row_count, dtype=np.float32)

    burn_list = [
        (name, float(fuel_to_electricity[name]["share"]))
        for name in commodity_cols
        if (
            name in fuel_to_electricity
            and float(fuel_to_electricity[name]["share"]) > 0.0
            and name in conv
        )]
    total_share = sum(share for _, share in burn_list)

    if total_share > 0:
        burn_list = [
            (name, share / total_share)
            for name, share in burn_list]

    for pos in range(row_count):
        date = dates[pos]

        for name in commodity_cols:
            expired[name][pos] = storages[name].expire(date)
            quantity = float(produced[name][pos])
            stored_mass, sold_mass = storages[name].add(
                date, quantity)
            stored[name][pos] = stored_mass
            sold[name][pos] = sold_mass

        balance = float(balances[pos])
        if balance < -BALANCE_TOLERANCE:
            deficit = -balance
            remaining = deficit

            if total_share > 0:
                for pass_id in (1, 2):
                    for name, share in burn_list:
                        if remaining <= BALANCE_TOLERANCE:
                            break

                        need_energy = remaining
                        if pass_id == 1:
                            need_energy *= share

                        factor = float(conv.get(name, 0.0))
                        if factor <= 0:
                            continue

                        withdrawn = storages[name].withdraw(
                            need_energy / factor)
                        if withdrawn <= 0:
                            continue

                        energy = withdrawn * factor
                        burned[name][pos] += withdrawn
                        potential[name][pos] += energy
                        supply_from_reconversion[pos] += energy
                        remaining = max(remaining - energy, 0.0)

            supply_from_reconversion[pos] = min(
                supply_from_reconversion[pos], deficit)

        for name in commodity_cols:
            inventory[name][pos] = storages[name].total_quantity
            reconversion_generation[pos] += potential[name][pos]

    for name in commodity_cols:
        quantity_unit = commodity_quantity_unit(Database, name)
        df[f"Stored_{name} ({quantity_unit})"] = stored[name]
        df[f"Sold_{name} ({quantity_unit})"] = sold[name]
        df[f"Expired_{name} ({quantity_unit})"] = expired[name]
        df[f"Burned_{name} ({quantity_unit})"] = burned[name]
        df[f"Reconversion_Generation_{name} ({energy_unit})"] = (
            potential[name])
        df[f"Inventory_{name} ({quantity_unit})"] = inventory[name]

    df[f"Fuel_Reconversion_Generation ({energy_unit})"] = (
        reconversion_generation)
    df[supply_col] = supply_from_reconversion

    negative_mask = balances < -BALANCE_TOLERANCE
    final_balance = balances.copy()
    final_balance[negative_mask] += (
        supply_from_reconversion[negative_mask])
    final_balance[np.abs(final_balance) <= BALANCE_TOLERANCE] = 0.0

    df[final_col] = final_balance
    residual_requirement = np.maximum(-final_balance, 0.0)
    residual_requirement[
        residual_requirement <= BALANCE_TOLERANCE] = 0.0
    df[residual_col] = residual_requirement

    return df

def process_output(perturbed_df_energies_data, simulation_id):
    """Apply storage, emissions, analysis, and output persistence."""

    stage_start = time.perf_counter()
    emit(User, f"Simulation {simulation_id}: storage calculation started.",
        "detailed", allow_worker=True)
    output = to_electricity_with_storage(perturbed_df_energies_data)
    storage_time = time.perf_counter() - stage_start
    emit(User, f"Simulation {simulation_id}: storage calculation finished.",
        "detailed", allow_worker=True)
    release_unused_memory()

    stage_start = time.perf_counter()
    output = calculate_emissions(
        output, perturbed_df_energies_data, User, Database)
    emissions_time = time.perf_counter() - stage_start
    release_unused_memory()

    output_dir, temp_dir, _ = output_io.get_output_paths(
        __file__, User)

    stage_start = time.perf_counter()
    emit(User, f"Simulation {simulation_id}: analysis started.",
        "detailed", allow_worker=True)
    scenario.analyze_simulation(
        output, perturbed_df_energies_data,
        simulation_id, output_dir, User)
    analysis_time = time.perf_counter() - stage_start
    emit(User, f"Simulation {simulation_id}: analysis finished.",
        "detailed", allow_worker=True)
    release_unused_memory()

    stage_start = time.perf_counter()
    if scenario.should_save_detailed_output(User, simulation_id):
        emit(User, f"Simulation {simulation_id}: detailed output started.",
            "detailed", allow_worker=True)
    output_io.save_system_operation_output(
        output, simulation_id, output_dir, User)
    output_time = time.perf_counter() - stage_start
    if scenario.should_save_detailed_output(User, simulation_id):
        emit(User, f"Simulation {simulation_id}: detailed output finished.",
            "detailed", allow_worker=True)

    del output
    release_unused_memory()
    return {
        "storage": storage_time, "emissions": emissions_time,
        "analysis": analysis_time, "output": output_time}


def _apply_deterministic_stochastic_ceilings(perturbed, deterministic):
    """Cap max-anchored stochastic sources at their nominal projection."""

    output = perturbed.copy()
    for source, source_input in User.get("sources", {}).items():
        if not isinstance(source_input, dict):
            continue
        anchor = source_input.get("anchor", {}) or {}
        method = str(anchor.get("method", "auto")).strip().lower()
        explicit_capacity = isinstance(
            source_input.get("capacity_additions"), dict)
        if method != "max" and not explicit_capacity:
            continue
        if source not in output.columns or source not in deterministic.columns:
            continue
        output[source] = np.minimum(
            pd.to_numeric(output[source], errors="coerce").fillna(0.0),
            pd.to_numeric(deterministic[source], errors="coerce").fillna(0.0),)
    return output


def run_simulation(
        simulation_id, df_energies, residual_params, cov_matrix_df,
        is_monte_carlo=True):
    """Run one deterministic or Monte Carlo electricity-balance case."""

    global RUN_ID
    total_start = time.perf_counter()
    User["_active_simulation_id"] = int(simulation_id)
    User["_runtime_diagnostics"] = []
    output_dir, _, forecast_dir = output_io.get_output_paths(
        __file__, User)
    monte_carlo = User.get("monte_carlo", {}) or {}
    seed = int(monte_carlo.get("seed", 12345))
    np.random.seed(seed + simulation_id)
    random.seed(seed + simulation_id + 1000)
    daily_df = df_energies.copy()
    original_stochastic_run = User.get("_stochastic_run", False)
    original_technology_samples = User.get("_technology_samples")
    stochastic_run = (
        is_monte_carlo and bool(monte_carlo.get("enabled", False)))
    User["_stochastic_run"] = stochastic_run
    User["_technology_samples"] = {}
    use_stochastic_forecast = stochastic_run

    try:
        stage_start = time.perf_counter()
        if use_stochastic_forecast and residual_params is not None:
            emit(User, f"Simulation {simulation_id}", "detailed",
                allow_worker=True)
            if _has_custom_replacement():
                daily_df = load_pre_replacement_forecast(df_energies)
            daily_df = perturb_df_energies(
                forecast_df=daily_df, residual_params=residual_params,
                confidence_level=monte_carlo.get("confidence_level"),
                cov_matrix=cov_matrix_df,
                preserve_annual_targets=monte_carlo.get(
                    "preserve_annual_targets", False))
            if _has_custom_replacement():
                daily_df = apply_custom_replacements(
                    daily_df, User).forecast
            daily_df = _apply_deterministic_stochastic_ceilings(
                daily_df, df_energies)
            source_names = list(User.get("sources", {}))
            available = [
                name for name in source_names if name in daily_df.columns]
            if not available:
                raise ValueError(
                    "No source columns found after Monte Carlo.")
            daily_df[available] = daily_df[available].clip(lower=0.0)
            daily_df["Total"] = daily_df[available].sum(axis=1)
        perturbation_time = time.perf_counter() - stage_start
        del residual_params, cov_matrix_df
        release_unused_memory()

        root_dir = Path(__file__).resolve().parents[2]
        if use_stochastic_forecast:
            output_io.save_perturbed_forecast(
                daily_df, simulation_id, forecast_dir, User)

        stage_start = time.perf_counter()
        hourly_df = prepare_operational_data(
            daily_df,
            User,
            root_dir,)
        hourly_df = attach_calendar_fuel_diagnostics(
            hourly_df, User, REFUELING_SCHEDULE)
        hourly_time = time.perf_counter() - stage_start
        forecast_rows = len(daily_df)
        del daily_df
        release_unused_memory()

        emit(
            User,
            f"Simulation {simulation_id}: {forecast_rows} forecast rows, "
            f"{len(hourly_df)} operational rows.",
            "detailed", allow_worker=True)
        stage_start = time.perf_counter()
        output_io.save_hourly_generation_output(
            hourly_df, simulation_id, output_dir, User)
        save_anicca_input(
            hourly_df, simulation_id, output_dir, User)
        pre_output_time = time.perf_counter() - stage_start
        stage_times = process_output(hourly_df, simulation_id)
        total_time = time.perf_counter() - total_start
        scenario.save_simulation_diagnostics(
            output_dir, simulation_id,
            User.get("_runtime_diagnostics", []))
        scenario.mark_simulation_complete(
            output_dir, simulation_id, User, RUN_ID)
        timing_message = (
            f"Simulation {simulation_id} timings: "
            f"perturbation={perturbation_time:.2f}s, "
            f"hourly={hourly_time:.2f}s, "
            f"pre_output={pre_output_time:.2f}s, "
            f"storage={stage_times.get('storage', 0.0):.2f}s, "
            f"emissions={stage_times.get('emissions', 0.0):.2f}s, "
            f"analysis={stage_times['analysis']:.2f}s, "
            f"output={stage_times['output']:.2f}s, "
            f"total={total_time:.2f}s.")
        timing_level = "detailed"
        emit(User, timing_message, timing_level, allow_worker=True)
        del hourly_df
        release_unused_memory()
        return simulation_id
    except Exception:
        scenario.clear_simulation_analysis(
            output_dir, simulation_id)
        traceback.print_exc()
        raise
    finally:
        User["_stochastic_run"] = original_stochastic_run
        if original_technology_samples is None:
            User.pop("_technology_samples", None)
        else:
            User["_technology_samples"] = original_technology_samples


def _selected_bootstrap_length(
        metadata: pd.DataFrame, history_columns: list[str]
) -> float | None:
    """Aggregate block lengths for the active stochastic sources."""

    if metadata.empty or not history_columns:
        return None

    selected = metadata.loc[
        metadata["column"].isin(history_columns)
    ].copy()
    estimates = pd.to_numeric(
        selected.get("stationary_block_length"),
        errors="coerce",
    ).dropna()

    if estimates.empty:
        estimates = pd.to_numeric(
            metadata.get("common_block_length"),
            errors="coerce",
        ).dropna()

    if estimates.empty:
        return None

    return aggregate_stationary_block_lengths(
        estimates.to_numpy(dtype=float))


def _normalize_source_dictionary(
        values: dict, source_names: list[str], context: str) -> dict:
    """Normalize source keys in one parameter dictionary."""

    lookup = build_name_lookup(
        [*source_names, "Demand"], f"source names for {context}")
    normalized = {}
    for raw_name, item in values.items():
        canonical = lookup.get(name_key(raw_name), raw_name)
        if canonical in normalized:
            raise ValueError(
                f"Duplicate source '{canonical}' after case normalization "
                f"in {context}.")
        normalized[canonical] = item
    return normalized


def _normalize_residual_source_names(
        residuals: object, source_names: list[str]) -> object:
    """Normalize source keys in residual-parameter mappings."""

    if not isinstance(residuals, dict):
        return residuals

    has_periods = any(
        isinstance(key, int) or key in {"global", "slow"}
        for key in residuals)
    if not has_periods:
        return _normalize_source_dictionary(
            residuals, source_names, "residual parameters")

    normalized = {}
    for key, values in residuals.items():
        if key == "bootstrap" or not isinstance(values, dict):
            normalized[key] = values
            continue
        normalized[key] = _normalize_source_dictionary(
            values, source_names, f"residual parameters '{key}'")
    return normalized


def _normalize_covariance_source_names(
        covariance: object, source_names: list[str]) -> object:
    """Normalize source labels in covariance matrices."""

    if isinstance(covariance, dict):
        normalized = {}
        for key, matrix in covariance.items():
            normalized[key] = normalize_source_axis(
                matrix, source_names, f"covariance '{key}'")
        return normalized
    return normalize_source_axis(
        covariance, source_names, "covariance matrix")


def _prepare_monte_carlo_inputs(forecast, num_monte_carlo):
    """Load and filter stochastic inputs for one simulation batch."""

    module_dir = Path(__file__).resolve().parents[2]
    scenario_dir = module_dir / User["scenario_folder"]
    scenario_dir /= User["scenario_subfolder"]
    monte_carlo = User.get("monte_carlo", {}) or {}
    stochastic_enabled = bool(monte_carlo.get("enabled", False))

    if num_monte_carlo <= 0 or not stochastic_enabled:
        return None, None

    source_names = list(User.get("sources", {}))
    residual_path = scenario_dir / "Residual_distribution.xlsx"
    residuals = residual_params_dict(residual_path)
    residuals = _normalize_residual_source_names(
        residuals, source_names)
    cov_path = scenario_dir / "Cov_matrix.xlsx"
    covariance_data = load_monthly_cov_matrix(cov_path)
    covariance_data = _normalize_covariance_source_names(
        covariance_data, source_names)

    if isinstance(residuals, dict):
        period_keys = [
            key for key in residuals if isinstance(key, int)]

        if period_keys:
            residual_sources = residuals[period_keys[0]]
        elif "global" in residuals:
            residual_sources = residuals["global"]
        else:
            residual_sources = residuals
    else:
        residual_sources = residuals

    requested = monte_carlo.get(
        "sources", list(User.get("sources", {})))
    technologies = [
        source for source in requested
        if source in forecast.columns and source in residual_sources]

    if isinstance(residuals, dict) and (
            any(isinstance(key, int) for key in residuals)
            or "global" in residuals or "slow" in residuals
    ):
        filtered_residuals = {}

        for key, values in residuals.items():
            if not isinstance(values, dict):
                continue

            filtered_residuals[key] = {
                source: values[source]
                for source in technologies if source in values}

        residuals = filtered_residuals
    else:
        residuals = {
            source: residuals[source]
            for source in technologies if source in residuals}

    if isinstance(covariance_data, dict):
        covariance = {}

        for key, matrix in covariance_data.items():
            available = [
                source for source in technologies
                if source in matrix.index and source in matrix.columns]
            covariance[key] = matrix.loc[available, available]
    else:
        covariance = covariance_data.loc[technologies, technologies]

    history_path = scenario_dir / "Residual_history.xlsx"

    if history_path.is_file() and isinstance(residuals, dict):
        with pd.ExcelFile(
                history_path, engine="openpyxl") as workbook:
            local_relative = "LocalRelative" in workbook.sheet_names
        history_sheet = (
            "LocalRelative" if local_relative else "FastRelative")
        history = pd.read_excel(
            history_path, sheet_name=history_sheet, engine="openpyxl")
        history = normalize_frame_columns(
            history, ["Date", *source_names, "Demand"],
            f"Residual_history.xlsx {history_sheet}")
        metadata = pd.read_excel(
            history_path, sheet_name="Bootstrap", engine="openpyxl")
        if "column" in metadata.columns:
            metadata["column"] = normalize_source_values(
                metadata["column"], source_names,
                "Residual_history.xlsx Bootstrap")
        history_columns = [
            source for source in technologies if source in history.columns]

        if history_columns and not metadata.empty:
            date_column = "Date"

            if date_column not in history.columns:
                date_column = str(history.columns[0])

            mean_length = _selected_bootstrap_length(
                metadata, history_columns)

            if mean_length is not None:
                history_dates = pd.to_datetime(
                    history[date_column]).to_numpy()
                history_values = history[history_columns].to_numpy(
                    dtype=float)
                annual_scales = annual_variability_scales(
                    history_values, history_dates)
                residuals["bootstrap"] = {
                    "history": history[[date_column, *history_columns]],
                    "date_column": date_column,
                    "mean_block_length": mean_length,
                    "relative_basis": (
                        "local" if local_relative else "global"),
                    "annual_variability_scales": annual_scales,}

    return residuals, covariance


def _load_worker_forecast():
    """Load deterministic refuelling profiles before scenario filtering."""

    global REFUELING_SCHEDULE
    forecast = load_forecast()
    forecast, REFUELING_SCHEDULE = build_refueling_profiles(
        forecast, User)
    date_column = "Date" if "Date" in forecast.columns else "ds"
    dates = pd.to_datetime(forecast[date_column])
    mask = pd.Series(True, index=forecast.index)
    start_date = User.get("start_date")
    end_date = User.get("end_date")
    if start_date:
        mask &= dates >= pd.to_datetime(start_date)
    if end_date:
        mask &= dates <= pd.to_datetime(end_date)
    return forecast.loc[mask].copy()


def run_external_worker_batch(
        config_path, simulation_ids, is_monte_carlo, run_id):
    """Initialize one worker once and run several simulations in sequence."""

    global df_energies, RUN_ID
    LEAFSimulator(config_path)
    RUN_ID = str(run_id)
    df_energies = _load_worker_forecast()
    residuals, covariance = _prepare_monte_carlo_inputs(
        df_energies, 1 if is_monte_carlo else 0)
    results = []
    for simulation_id in simulation_ids:
        results.append(run_simulation(
            int(simulation_id),
            df_energies,
            residuals,
            covariance,
            is_monte_carlo))
    return results


def run_external_worker(config_path, simulation_id,
                        is_monte_carlo, run_id):
    """Initialize and execute one isolated simulation worker."""

    return run_external_worker_batch(
        config_path, [simulation_id], is_monte_carlo, run_id)[0]


def _worker_command(config_path, simulation_ids,
                    is_monte_carlo, run_id):
    """Build the subprocess command for one persistent worker batch."""

    if isinstance(simulation_ids, int):
        simulation_ids = [simulation_ids]
    id_text = ",".join(str(value) for value in simulation_ids)
    command = [
        sys.executable,
        "-u",
        "-m",
        "src.core.worker",
        "--config",
        str(config_path),
        "--simulation-ids",
        id_text,
        "--run-id",
        str(run_id)]
    if is_monte_carlo:
        command.append("--monte-carlo")
    return command


def _worker_environment(process_count):
    """Return an environment that avoids BLAS oversubscription."""

    if int(process_count) <= 1:
        return None
    environment = os.environ.copy()
    for variable in (
        "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
    ):
        environment[variable] = "1"
    return environment


def _worker_subprocess_kwargs(process_count):
    """Return stable subprocess settings for isolated LEAF workers."""

    kwargs = {"cwd": str(PROJECT_ROOT)}
    environment = _worker_environment(process_count)
    if environment is not None:
        kwargs["env"] = environment
    return kwargs


def _run_external_workers(
        config_path, tasks, run_id, process_count,
        progress_output_dir=None):
    """Launch persistent worker subprocesses with bounded parallelism."""

    tasks = list(tasks)
    if not tasks:
        return
    modes = {bool(is_monte_carlo) for _, is_monte_carlo in tasks}
    if len(modes) != 1:
        raise ValueError(
            "Worker batches cannot mix deterministic and Monte Carlo runs.")
    is_monte_carlo = modes.pop()
    worker_count = max(1, min(int(process_count), len(tasks)))
    batches = [[] for _ in range(worker_count)]
    for position, (simulation_id, _) in enumerate(tasks):
        batches[position % worker_count].append(simulation_id)

    worker_kwargs = _worker_subprocess_kwargs(worker_count)
    active = []
    for simulation_ids in batches:
        command = _worker_command(
            config_path, simulation_ids, is_monte_carlo, run_id)
        process = subprocess.Popen(command, **worker_kwargs)
        active.append((simulation_ids, process))

    failed = []
    progress_ids = [int(simulation_id) for simulation_id, _ in tasks]
    progress_total = len(progress_ids)
    next_progress = 10
    while active:
        remaining = []
        for simulation_ids, process in active:
            return_code = process.poll()
            if return_code is None:
                remaining.append((simulation_ids, process))
            elif return_code != 0:
                failed.append((simulation_ids, return_code))
        active = remaining
        if progress_output_dir is not None and progress_total:
            completed_count = 0
            for simulation_id in progress_ids:
                success = scenario.get_simulation_analysis_dir(
                    progress_output_dir, simulation_id,
                    create=False) / "_SUCCESS.json"
                completed_count += int(success.is_file())
            percent = int(100 * completed_count / progress_total)
            if percent >= next_progress:
                emit(
                    User,
                    f"Monte Carlo progress: {completed_count}/"
                    f"{progress_total} completed ({percent}%).")
                while next_progress <= percent:
                    next_progress += 10
        if active:
            time.sleep(0.1)

    if failed:
        details = "; ".join(
            f"{','.join(map(str, simulation_ids))}:{return_code}"
            for simulation_ids, return_code in failed)
        raise RuntimeError(
            f"Commodity worker batches failed: {details}")


def _print_diagnostics_summary(summary, output_dir):
    """Print one compact warning summary after ensemble consolidation."""

    if summary is None or summary.empty:
        return
    emit(
        User,
        "Warnings summary: capacity clipping occurred; detailed values are "
        "saved in Output/Summary/Simulation_Diagnostics.xlsx.")
    for _, row in summary.iterrows():
        source = row.get("Source", "Unknown")
        deterministic = bool(row.get("Deterministic_Affected", False))
        affected = int(row.get("MC_Affected", 0))
        requested = int(row.get("MC_Requested", 0))
        maximum = float(row.get("Maximum_Excess_Percent", 0.0))
        det_text = "yes" if deterministic else "no"
        emit(
            User,
            f"  {source}: deterministic={det_text}; MC={affected}/"
            f"{requested}; maximum excess={maximum:.2f}%.")


def simulation(num_monte_carlo):
    """Coordinate deterministic and Monte Carlo worker processes."""

    global df_energies, User, Database, RUN_ID

    monte_carlo = User.get("monte_carlo", {}) or {}
    stochastic = bool(monte_carlo.get("enabled", False))

    if num_monte_carlo > 0 and not stochastic:
        emit(
            User,
            "WARNING: Monte Carlo simulations were requested, but "
            "stochastic simulation is disabled; running only simulation 0.")
        num_monte_carlo = 0

    monte_carlo_ids = list(range(1, num_monte_carlo + 1))
    simulation_ids = [0] + monte_carlo_ids
    output_dir, temp_dir, _ = output_io.get_output_paths(
        __file__, User)
    forecast_path = Path(output_dir).parent / "Forecast.xlsx"
    RUN_ID = output_io.build_run_id(User, forecast_path)
    emit(User, f"Run identifier: {RUN_ID}", "detailed")
    database_path = Path(User.get("database_path", "data/Database.yaml"))
    if not database_path.is_absolute():
        database_path = PROJECT_ROOT / database_path
    write_resolved_config(
        User,
        Path(output_dir).parent,
        run_id=RUN_ID,
        database_path=database_path.resolve())
    config_path = User["_config_path"]

    with output_io.ScenarioRunLock(output_dir):
        resume = bool(monte_carlo.get("resume", True))
        scenario.prepare_analysis_batch(
            output_dir, simulation_ids, resume=resume)
        completed = scenario.get_completed_simulations(
            output_dir, simulation_ids, User, RUN_ID)

        if 0 not in completed or not resume:
            _run_external_workers(
                config_path,
                [(0, False)],
                RUN_ID,
                1,)

        pending_ids = [
            value for value in monte_carlo_ids
            if value not in completed or not resume]
        if pending_ids:
            requested = int(monte_carlo.get("workers", cpu_count()))
            process_count, memory_info = memory_safe_worker_count(
                requested, len(pending_ids), cpu_count())
            message = f"Running {len(pending_ids)} pending Monte Carlo "
            message += f"simulations with {process_count} processes."
            emit(User, message)
            available_gib = memory_info.get("available_gib")
            memory_limit = memory_info.get("memory_limit")
            base_limit = min(
                max(1, requested), len(pending_ids), cpu_count())
            if (
                available_gib is not None
                and memory_limit is not None
                and process_count < base_limit
            ):
                emit(
                    User,
                    "Memory guard reduced parallel workers: "
                    f"{base_limit} -> {process_count} "
                    f"(available RAM {available_gib:.1f} GiB).")
            tasks = [
                (simulation_id, True)
                for simulation_id in pending_ids]
            _run_external_workers(
                config_path,
                tasks,
                RUN_ID,
                process_count,
                progress_output_dir=output_dir,)
        elif num_monte_carlo > 0:
            emit(
                User,
                "All requested Monte Carlo simulations are complete.")

        diagnostics_summary = scenario.finalize_analysis_batch(
            output_dir, simulation_ids, User)
        _print_diagnostics_summary(diagnostics_summary, output_dir)

class LEAFSimulator:
    """Load one scenario and execute its LEAF-EB scenario workflow."""

    def __init__(self, cfg_path: str):
        """Load normalized scenario and scenario settings."""

        global User, Database

        self.cfg_path = str(cfg_path)
        self.cfg_path_resolved = str(Path(cfg_path).resolve())

        full_cfg = load_user_config_from_path(self.cfg_path)
        if isinstance(full_cfg, dict):
            commodities_cfg = full_cfg.get("commodities_input", {})
        else:
            commodities_cfg = {}

        User = dict(full_cfg) if isinstance(full_cfg, dict) else {}
        User["_config_path"] = self.cfg_path_resolved
        if isinstance(commodities_cfg, dict):
            User.update(commodities_cfg)

        configure_flexibility(User)
        module_dir = Path(__file__).resolve().parents[2]

        Database = load_database()
        configured_units = Database.get("Commodity_Units", {}) or {}
        User["_commodity_quantity_units"] = {
            name: commodity_quantity_unit(Database, name)
            for name in configured_units}

        scenario_folder = full_cfg.get("scenario_folder")
        scenario_folder = scenario_folder or User.get("scenario_folder")
        scenario_subfolder = full_cfg.get("scenario_subfolder")
        scenario_subfolder = scenario_subfolder or User.get(
            "scenario_subfolder")

        self.scenario_dir = module_dir / scenario_folder
        self.scenario_dir /= scenario_subfolder
        self.scenario_dir = self.scenario_dir.resolve()

    def run(self):
        """Load the forecast and run the configured simulations."""

        global df_energies

        df_energies = load_forecast()

        date_column = "Date" if "Date" in df_energies.columns else "ds"
        dates = pd.to_datetime(df_energies[date_column])

        start_date = User.get("start_date")
        end_date = User.get("end_date")
        mask = pd.Series(True, index=df_energies.index)

        if start_date:
            start_date = pd.to_datetime(start_date)
            mask &= dates >= start_date
        if end_date:
            end_date = pd.to_datetime(end_date)
            mask &= dates <= end_date
        df_energies = df_energies.loc[mask].copy()

        if df_energies.empty:
            raise ValueError(
                "No Forecast rows remain after applying scenario dates.")


        first_date = df_energies[date_column].min()
        last_date = df_energies[date_column].max()

        emit(
            User,
            f"Simulation period: {first_date} to {last_date}, "
            f"{len(df_energies)} forecast rows.")


        monte_carlo = User.get("monte_carlo", {}) or {}
        num_monte_carlo = int(monte_carlo.get("simulations", 0))
        simulation(num_monte_carlo)
