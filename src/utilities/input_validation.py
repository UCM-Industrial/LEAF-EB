"""Validate the current LEAF-EB input schema and temporal choices."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from src.forecasting.historical_data import load_historical_dataset
from src.utilities.name_resolution import build_name_lookup


_ALLOWED_HOURLY_OPERATIONS = {
    "must_run",
    "dispatchable",
    "profile_preserving",
    "load_following",}

_ALLOWED_SOURCE_FIELDS = {
    "share",
    "capacity_factor",
    "capacity_tolerance",
    "model",
    "values",
    "anchor",
    "limit",
    "emission_factor_co2",
    "hourly_operation",
    "dispatch_priority",
    "must_run",
    "load_following",
    "unit_capacity",
    "refueling",
    "fuel_cycle",
    "initial_capacity",
    "capacity_additions",
    "reference_generation",
    "custom_mode",
    "custom_data",
    "replaces",
    "technology_template",
    "hourly_patterns",}

_ALLOWED_COMMODITIES_FIELDS = {
    "run_commodities",
    "database_path",
    "Fuel_to_Electricity",
    "Commodities_Production",
    "Commodity_Storage",
    "BESS",
    "Dispatch_Order",
    "Interconnections",}

_ALLOWED_BESS_FIELDS = {
    "model",
    "value",
    "values",
    "duration",
    "efficiency",
    "lifetime",}

_ALLOWED_BESS_LIFETIME_FIELDS = {
    "calendar_years",
    "cycle_life_efc",}


def validate_input(user_input: dict, root_dir: Path) -> None:
    """Validate one normalized current-schema input."""

    root_dir = Path(root_dir).resolve()
    _validate_scenario(user_input)
    _validate_initial_conditions(user_input)
    _validate_capacity_tolerance(user_input)
    dataset = load_historical_dataset(user_input)
    _validate_temporal_configuration(
        user_input,
        dataset.input_resolution,
        root_dir,)
    _validate_required_columns(user_input, dataset.raw.columns)
    _validate_historical_values(user_input, dataset.raw)
    _validate_demand(user_input)
    _validate_sources(user_input)
    _validate_emission_configuration(user_input)
    _validate_monte_carlo(user_input)
    _validate_commodities_input(user_input, root_dir)
    _validate_output(user_input.get("output"))


def _validate_scenario(user_input: dict) -> None:
    """Validate scenario dates and output identifiers."""

    start = pd.Timestamp(user_input["start_date"])
    end = pd.Timestamp(user_input["end_date"])
    if end < start:
        raise ValueError(
            "scenario.end_date must not precede scenario.start_date.")

    for key in ("scenario_folder", "scenario_subfolder"):
        if not str(user_input.get(key, "")).strip():
            raise ValueError(f"{key} cannot be empty.")



def _validate_initial_conditions(user_input: dict) -> None:
    """Validate optional storage states for short operational cases."""

    simulation = user_input.get("simulation", {})
    conditions = simulation.get("initial_conditions", {})
    if conditions is None:
        return
    if not isinstance(conditions, dict):
        raise ValueError(
            "simulation.initial_conditions must be a mapping.")
    fraction = _as_float(
        conditions.get("bess_state_of_charge", 0.0),
        "simulation.initial_conditions.bess_state_of_charge",)
    if fraction < 0.0 or fraction > 1.0:
        raise ValueError(
            "bess_state_of_charge must be between zero and one.")
    for inventory_key in ("commodity_inventory",):
        inventories = conditions.get(inventory_key, {}) or {}
        if not isinstance(inventories, dict):
            raise ValueError(
                f"{inventory_key} must be a mapping.")
        for name, value in inventories.items():
            quantity = _as_float(
                value,
                f"{inventory_key}.{name}",)
            if quantity < 0.0:
                raise ValueError(
                    f"{inventory_key}.{name} cannot be negative.")


def _validate_capacity_tolerance(user_input: dict) -> None:
    """Validate global and source-specific capacity tolerances."""

    simulation = user_input.get("simulation", {}) or {}
    global_value = simulation.get("capacity_tolerance", 0.0)
    _validate_tolerance_value(
        global_value,
        "simulation.capacity_tolerance",)

    for source_name, source_input in user_input.get("sources", {}).items():
        if not isinstance(source_input, dict):
            continue
        if "capacity_tolerance" not in source_input:
            continue
        _validate_tolerance_value(
            source_input["capacity_tolerance"],
            f"sources.{source_name}.capacity_tolerance",)


def _validate_tolerance_value(value: object, label: str) -> None:
    """Validate one tolerance expressed as percentage points."""

    percent = _as_float(value, label)
    if percent < 0.0 or percent > 100.0:
        raise ValueError(
            f"{label} must be between 0 and 100 percent.")


def _validate_temporal_configuration(
        user_input: dict, input_resolution: str, root_dir: Path
) -> None:
    """Validate processing, projection, variability and simulation steps."""

    processing = user_input["processing_resolution"]
    projection = user_input["projection_resolution"]
    variability = user_input["variability_resolution"]
    simulation = user_input["simulation_resolution"]

    if input_resolution == "daily" and processing == "hourly":
        raise ValueError(
            "Hourly processing requires hourly historical data.")
    if projection == "hourly" and processing != "hourly":
        raise ValueError(
            "Hourly projection requires hourly historical processing.")
    if variability != projection:
        raise ValueError(
            "projection.variability.resolution must match "
            "projection.resolution.")
    if simulation == "daily" and projection == "hourly":
        raise ValueError(
            "A daily simulation cannot consume an hourly projection.")
    if (
        simulation == "hourly"
        and projection == "daily"
        and input_resolution != "hourly"
    ):
        profile = user_input.get("external_hourly_profile_file")
        if not profile:
            raise ValueError(
                "Hourly simulation from daily historical data requires "
                "simulation.hourly_profile_file.")
        profile_path = _validate_file(
            root_dir, profile, "Hourly profile")
        _validate_hourly_profile(profile_path)



def _validate_hourly_profile(path: Path) -> None:
    """Validate the structure of one external hourly-profile file."""

    if path.suffix.lower() == ".xlsx":
        frame = pd.read_excel(path, engine="openpyxl")
    elif path.suffix.lower() == ".csv":
        frame = pd.read_csv(path)
    else:
        raise ValueError(
            "Hourly profile file must be an XLSX or CSV file.")
    if len(frame) != 8760:
        raise ValueError(
            "Hourly profile file must contain exactly 8760 rows.")
    available = build_name_lookup(
        frame.columns, "hourly-profile columns")
    if "demand" not in available:
        raise ValueError(
            "Hourly profile file must contain a Demand column.")


def _validate_required_columns(
        user_input: dict, columns: Any
) -> None:
    """Validate Date, Demand and configured source columns."""

    available = build_name_lookup(columns, "historical-data columns")
    historical_sources = []
    for name, source_data in user_input["sources"].items():
        model = ""
        if isinstance(source_data, dict):
            model = str(
                source_data.get("model", "")
            ).strip().lower()
        if model != "custom":
            historical_sources.append(name)
    requested = [
        user_input["date_column"],
        "Demand",
        *historical_sources,]
    missing = [
        name for name in requested
        if str(name).strip().casefold() not in available]
    if missing:
        values = ", ".join(missing)
        raise ValueError(
            f"Historical data is missing required columns: {values}.")



def _validate_historical_values(
        user_input: dict, frame: pd.DataFrame
) -> None:
    """Reject negative values in configured generation and demand series."""

    required = ["Demand"]
    for name, source_input in user_input["sources"].items():
        model = ""
        if isinstance(source_input, dict):
            model = str(source_input.get("model", "")).strip().lower()
        if model != "custom":
            required.append(name)

    for column in required:
        values = pd.to_numeric(frame[column], errors="coerce")
        if not values.lt(0.0).any():
            continue
        first = int(values.index[values.lt(0.0)][0])
        raise ValueError(
            f"Historical column '{column}' contains a negative value "
            f"at row {first}. Generation and demand must be "
            "non-negative.")

def _validate_demand(user_input: dict) -> None:
    """Validate the demand projection and anchor."""

    demand = user_input.get("demand")
    if not isinstance(demand, dict):
        raise ValueError("demand must be a mapping.")
    target = _as_float(
        demand.get("target_production"),
        "demand.target_production",)
    if target <= 0.0:
        raise ValueError(
            "demand.target_production must be greater than zero.")
    balance = _as_float(demand.get("balance", 1.0), "demand.balance")
    if balance <= 0.0:
        raise ValueError("demand.balance must be greater than zero.")
    _validate_model(demand, "demand")
    _validate_anchor(demand.get("anchor"), "demand.anchor")


def _register_dispatch_priority(
        priorities, operation, source_name, priority, required=False):
    """Validate and register one source dispatch priority."""

    if priority is None and not required:
        return
    priority = _as_positive_int(
        priority, f"sources.{source_name}.dispatch_priority")
    previous = priorities[operation].get(priority)
    if previous is not None:
        raise ValueError(
            f"dispatch_priority {priority} is repeated for "
            f"'{previous}' and '{source_name}'.")
    priorities[operation][priority] = source_name


def _validate_sources(user_input: dict) -> None:
    """Validate source projections, capacities and dispatch priorities."""

    sources = user_input.get("sources")
    if not isinstance(sources, dict) or not sources:
        raise ValueError("sources must define at least one source.")
    build_name_lookup(sources, "source definitions")
    priorities = {
        "profile_preserving": {},
        "load_following": {},
        "dispatchable": {},}

    for source_name, source_input in sources.items():
        if not isinstance(source_input, dict):
            raise ValueError(
                f"Source '{source_name}' must contain a mapping.")
        unknown = sorted(set(source_input) - _ALLOWED_SOURCE_FIELDS)
        if unknown:
            values = ", ".join(unknown)
            raise ValueError(
                f"Unknown fields for source '{source_name}': {values}.")
        explicit_capacity = source_input.get("capacity_additions")
        share_value = source_input.get("share")
        if share_value is None and explicit_capacity is None:
            raise ValueError(
                f"sources.{source_name}.share is required unless "
                "capacity_additions is defined.")
        if share_value is not None:
            share = _as_float(
                share_value, f"sources.{source_name}.share")
            if share < 0.0:
                raise ValueError(
                    f"sources.{source_name}.share cannot be negative.")

        if "limit" in source_input:
            limit_value = _as_float(
                source_input["limit"],
                f"sources.{source_name}.limit",)
            if limit_value < 0.0:
                raise ValueError(
                    f"sources.{source_name}.limit cannot be negative.")

        if explicit_capacity is None:
            capacity_factor = _as_float(
                source_input.get("capacity_factor"),
                f"sources.{source_name}.capacity_factor",)
            if capacity_factor <= 0.0 or capacity_factor > 1.0:
                raise ValueError(
                    f"sources.{source_name}.capacity_factor must be in "
                    "the interval (0, 1].")
        else:
            _validate_explicit_capacity(source_name, source_input)

        if "emission_factor_co2" in source_input:
            emissions = _as_float(
                source_input["emission_factor_co2"],
                f"sources.{source_name}.emission_factor_co2",)
            if emissions < 0.0:
                raise ValueError(
                    f"sources.{source_name}.emission_factor_co2 "
                    "cannot be negative.")

        _validate_model(source_input, f"sources.{source_name}")
        _validate_custom_source(
            source_name, source_input, set(sources))
        _validate_anchor(
            source_input.get("anchor"),
            f"sources.{source_name}.anchor",)
        _validate_fuel_cycle(source_name, source_input)
        operation = source_input.get("hourly_operation")
        if operation is None:
            _validate_refueling(source_name, source_input)
            continue
        normalized = str(operation).strip().lower()
        if normalized not in _ALLOWED_HOURLY_OPERATIONS:
            allowed = (
                "profile_preserving, dispatchable, must_run, "
                "load_following")
            raise ValueError(
                f"Invalid hourly_operation for '{source_name}'. "
                f"Use: {allowed}.")
        priority = source_input.get("dispatch_priority")
        if normalized == "dispatchable":
            _register_dispatch_priority(
                priorities, normalized, source_name, priority, required=True)
        elif normalized in {"profile_preserving", "load_following"}:
            _register_dispatch_priority(
                priorities, normalized, source_name, priority)
            if normalized == "load_following":
                _validate_load_following(source_name, source_input)
        elif normalized == "must_run":
            _validate_must_run(source_name, source_input)
            _validate_refueling(source_name, source_input)
            if priority is not None:
                raise ValueError(
                    f"Source '{source_name}' defines dispatch_priority but "
                    "is not profile-preserving, load-following, "
                    "or dispatchable.")
        elif priority is not None:
            raise ValueError(
                f"Source '{source_name}' defines dispatch_priority but "
                "is not profile-preserving, load-following, "
                "or dispatchable.")



def _validate_explicit_capacity(
        source_name: str, source_input: dict) -> None:
    """Validate explicit capacity additions and reference generation."""

    additions = source_input.get("capacity_additions")
    label = f"sources.{source_name}.capacity_additions"
    if not isinstance(additions, dict) or not additions:
        raise ValueError(f"{label} must be a non-empty mapping.")
    if "capacity_factor" in source_input:
        raise ValueError(
            f"sources.{source_name} cannot define both capacity_factor "
            "and capacity_additions.")

    initial = _as_float(
        source_input.get("initial_capacity", 0.0),
        f"sources.{source_name}.initial_capacity",)
    if initial < 0.0:
        raise ValueError(
            f"sources.{source_name}.initial_capacity cannot be "
            "negative.")

    dated_changes = []
    for raw_date, raw_value in additions.items():
        try:
            date = pd.Timestamp(raw_date)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid date in {label}: {raw_date}.") from exc
        value = _as_float(raw_value, f"{label}[{raw_date}]")
        if abs(value) <= 1e-12:
            raise ValueError(
                f"{label}[{raw_date}] must be non-zero.")
        dated_changes.append((date, value))
    dates = [date for date, _ in dated_changes]
    if len(dates) != len(set(dates)):
        raise ValueError(f"{label} contains duplicate dates.")
    running_capacity = initial
    for date, change in sorted(dated_changes):
        running_capacity += change
        if running_capacity < -1e-9:
            raise ValueError(
                f"{label}[{date.date().isoformat()}] makes installed "
                "capacity negative.")

    model = str(source_input.get("model", "")).strip().lower()
    if model != "custom":
        raise ValueError(
            f"{label} currently requires model: custom.")
    refueling = source_input.get("refueling", {}) or {}
    refueling_enabled = (
        isinstance(refueling, dict) and bool(refueling))
    reference = source_input.get("reference_generation")

    if refueling_enabled:
        if reference is not None:
            raise ValueError(
                f"sources.{source_name}.reference_generation must be "
                "omitted when refueling is enabled. Reference generation "
                "is derived from refuelling availability.")
        return

    if not isinstance(reference, dict):
        raise ValueError(
            f"sources.{source_name}.reference_generation must be a "
            "mapping when refueling is not enabled.")
    capacity_factor = _as_float(
        reference.get("capacity_factor"),
        f"sources.{source_name}.reference_generation.capacity_factor",)
    if capacity_factor <= 0.0 or capacity_factor > 1.0:
        raise ValueError(
            f"sources.{source_name}.reference_generation.capacity_factor "
            "must be in the interval (0, 1].")


def _validate_must_run(source_name: str, source_input: dict) -> None:
    """Validate optional refuelling-aware must-run settings."""

    settings = source_input.get("must_run", {}) or {}
    if not isinstance(settings, dict):
        raise ValueError(
            f"sources.{source_name}.must_run must be a mapping.")
    if "power_fraction" not in settings:
        raise ValueError(
            f"sources.{source_name}.must_run.power_fraction is required.")
    fraction = _as_float(
        settings["power_fraction"],
        f"sources.{source_name}.must_run.power_fraction",)
    if fraction <= 0.0 or fraction > 1.0:
        raise ValueError(
            f"sources.{source_name}.must_run.power_fraction must be "
            "in (0, 1].")


def _validate_load_following(source_name: str, source_input: dict) -> None:
    """Validate one annual-energy-preserving load-following rule."""

    settings = source_input.get("load_following")
    if not isinstance(settings, dict):
        raise ValueError(
            f"sources.{source_name}.load_following must be a mapping.")
    required = {
        "energy_policy", "control_mode",
        "minimum_power_fraction", "maximum_power_fraction",
    }
    missing = sorted(required - set(settings))
    if missing:
        raise ValueError(
            f"sources.{source_name}.load_following requires: "
            f"{', '.join(missing)}.")

    policy = str(settings["energy_policy"]).strip().lower()
    allowed_policies = {"preserve_annual", "follow_residual"}
    if policy not in allowed_policies:
        raise ValueError(
            f"sources.{source_name}.load_following.energy_policy must be "
            "'preserve_annual' or 'follow_residual'.")

    refueling = source_input.get("refueling", {}) or {}
    fuel_cycle = source_input.get("fuel_cycle", {}) or {}
    burnup_driven = (
        bool(refueling)
        and refueling.get("operating_cycle") is None
        and fuel_cycle.get("target_burnup") is not None)
    if burnup_driven and policy != "follow_residual":
        raise ValueError(
            f"sources.{source_name}.load_following with EFPD refuelling "
            "requires energy_policy: follow_residual.")

    minimum = _as_float(
        settings["minimum_power_fraction"],
        f"sources.{source_name}.load_following."
        "minimum_power_fraction",)
    maximum = _as_float(
        settings["maximum_power_fraction"],
        f"sources.{source_name}.load_following."
        "maximum_power_fraction",)
    if minimum < 0.0 or minimum > 1.0:
        raise ValueError(
            f"sources.{source_name}.load_following."
            "minimum_power_fraction must be between 0 and 1.")
    if maximum <= 0.0 or maximum > 1.0:
        raise ValueError(
            f"sources.{source_name}.load_following."
            "maximum_power_fraction must be in (0, 1].")
    if minimum > maximum:
        raise ValueError(
            f"sources.{source_name}.load_following minimum power cannot "
            "exceed maximum power.")

    control_mode = str(settings["control_mode"]).strip().lower()
    allowed_modes = {"direct", "constrained_hourly"}
    if control_mode not in allowed_modes:
        raise ValueError(
            f"sources.{source_name}.load_following.control_mode must be "
            "'direct' or 'constrained_hourly'.")
    if control_mode == "constrained_hourly":
        if policy != "follow_residual":
            raise ValueError(
                f"sources.{source_name}.load_following.control_mode "
                "'constrained_hourly' requires energy_policy "
                "'follow_residual'.")
        _validate_constrained_load_following(
            source_name, settings, minimum, maximum)
    _validate_refueling(source_name, source_input)


def _validate_constrained_load_following(
        source_name: str, settings: dict, minimum: float, maximum: float
) -> None:
    """Validate simple causal limits for hourly load following."""

    prefix = f"sources.{source_name}.load_following"
    ramp_up = _as_float(
        settings.get("ramp_up_rate"),
        f"{prefix}.ramp_up_rate")
    ramp_down = _as_float(
        settings.get("ramp_down_rate"),
        f"{prefix}.ramp_down_rate")
    threshold = _as_float(
        settings.get("deep_reduction_threshold_fraction"),
        f"{prefix}.deep_reduction_threshold_fraction")
    persistence = _as_float(
        settings.get("deep_reduction_request_duration"),
        f"{prefix}.deep_reduction_request_duration")
    maximum_cycles = _as_float(
        settings.get("max_deep_reduction_cycles"),
        f"{prefix}.max_deep_reduction_cycles")

    for value, label in (
        (ramp_up, "ramp_up_rate"),
        (ramp_down, "ramp_down_rate"),
    ):
        if value <= 0.0 or value > 1.0:
            raise ValueError(
                f"{prefix}.{label} must be in (0, 1].")
    if threshold < minimum or threshold > maximum:
        raise ValueError(
            f"{prefix}.deep_reduction_threshold_fraction must be "
            "between minimum and maximum power fractions.")
    if persistence <= 0.0 or not float(persistence).is_integer():
        raise ValueError(
            f"{prefix}.deep_reduction_request_duration must be a "
            "positive integer.")
    if maximum_cycles < 0.0 or not float(maximum_cycles).is_integer():
        raise ValueError(
            f"{prefix}.max_deep_reduction_cycles must be a "
            "non-negative integer.")


def _validate_refueling(source_name: str, source_input: dict) -> None:
    """Validate optional calendar- or burnup-driven refuelling."""

    settings = source_input.get("refueling")
    if settings is None:
        return
    if not isinstance(settings, dict) or not settings:
        raise ValueError(
            f"sources.{source_name}.refueling must be a non-empty mapping.")

    prefix = f"sources.{source_name}"
    unit_capacity = _as_float(
        source_input.get("unit_capacity"), f"{prefix}.unit_capacity")
    if unit_capacity <= 0.0:
        raise ValueError(
            f"{prefix}.unit_capacity must be positive when refueling is "
            "defined.")

    mode = str(settings.get("mode", "offline")).strip().lower()
    if mode not in {"offline", "online"}:
        raise ValueError(
            f"{prefix}.refueling.mode must be 'offline' or 'online'.")

    schedule = str(settings.get("schedule", "auto")).strip().lower()
    if schedule not in {"auto", "staggered"}:
        raise ValueError(
            f"{prefix}.refueling.schedule must be 'auto' or 'staggered'.")

    batches = _as_positive_int(
        settings.get("fuel_batches"),
        f"{prefix}.refueling.fuel_batches")
    if batches < 1:
        raise ValueError(
            f"{prefix}.refueling.fuel_batches must be at least 1.")

    fuel_cycle = source_input.get("fuel_cycle", {}) or {}
    operating_cycle = settings.get("operating_cycle")
    target_burnup = fuel_cycle.get("target_burnup")
    has_calendar = operating_cycle is not None
    has_burnup = target_burnup is not None

    if has_calendar == has_burnup:
        raise ValueError(
            f"{prefix} must define exactly one refuelling basis: "
            "refueling.operating_cycle or fuel_cycle.target_burnup.")

    if mode == "online" and has_calendar:
        raise ValueError(
            f"{prefix}.refueling.operating_cycle is not used with "
            "mode: online.")

    if mode == "offline":
        outage = _as_float(
            settings.get("outage_duration"),
            f"{prefix}.refueling.outage_duration")
        if outage <= 0.0 or not float(outage).is_integer():
            raise ValueError(
                f"{prefix}.refueling.outage_duration must resolve to "
                "whole positive days.")

    if has_calendar:
        cycle = _as_float(
            operating_cycle, f"{prefix}.refueling.operating_cycle")
        if cycle <= 0.0:
            raise ValueError(
                f"{prefix}.refueling.operating_cycle must be positive.")
        thermal = _as_float(
            fuel_cycle.get("thermal_power"),
            f"{prefix}.fuel_cycle.thermal_power")
        mass = _as_float(
            fuel_cycle.get("core_fuel_mass"),
            f"{prefix}.fuel_cycle.core_fuel_mass")
        if thermal <= 0.0 or mass <= 0.0:
            raise ValueError(
                f"{prefix}.fuel_cycle must define positive thermal_power "
                "and core_fuel_mass for calendar-driven refuelling.")
    else:
        _validate_burnup_basis(source_name, fuel_cycle)


def _validate_burnup_basis(source_name: str, fuel_cycle: dict) -> None:
    """Validate the parameters needed for burnup-driven refuelling."""

    prefix = f"sources.{source_name}.fuel_cycle"
    burnup = _as_float(
        fuel_cycle.get("target_burnup"), f"{prefix}.target_burnup")
    thermal = _as_float(
        fuel_cycle.get("thermal_power"), f"{prefix}.thermal_power")
    mass = _as_float(
        fuel_cycle.get("core_fuel_mass"), f"{prefix}.core_fuel_mass")
    if burnup <= 0.0 or thermal <= 0.0 or mass <= 0.0:
        raise ValueError(
            f"{prefix} target burnup, thermal power and core fuel mass "
            "must be positive for burnup-driven refuelling.")


def _validate_fuel_cycle(source_name: str, source_input: dict) -> None:
    """Validate optional fleet-level fuel-cycle parameters."""

    settings = source_input.get("fuel_cycle")
    if settings is None:
        return
    if not isinstance(settings, dict):
        raise ValueError(
            f"sources.{source_name}.fuel_cycle must be a mapping.")
    burnup = settings.get("target_burnup")
    if burnup is not None:
        value = _as_float(
            burnup, f"sources.{source_name}.fuel_cycle.target_burnup")
        if value <= 0.0:
            raise ValueError(
                f"sources.{source_name}.fuel_cycle.target_burnup must be "
                "positive.")
    thermal = settings.get("thermal_power")
    mass = settings.get("core_fuel_mass")
    if (thermal is None) != (mass is None):
        raise ValueError(
            f"sources.{source_name}.fuel_cycle must define both "
            "thermal_power and core_fuel_mass when either is used.")
    if thermal is not None:
        thermal_value = _as_float(
            thermal, f"sources.{source_name}.fuel_cycle.thermal_power")
        mass_value = _as_float(
            mass, f"sources.{source_name}.fuel_cycle.core_fuel_mass")
        if thermal_value <= 0.0 or mass_value <= 0.0:
            raise ValueError(
                f"sources.{source_name}.fuel_cycle thermal power and "
                "core fuel mass must be positive.")


def _validate_custom_source(
        source_name: str, source_input: dict, defined_sources: set[str]
) -> None:
    """Validate current custom-source schedules and replacement groups."""

    model = str(source_input.get("model", "")).strip().lower()
    if model != "custom":
        return

    mode = str(source_input.get("custom_mode", "add")).strip().lower()
    if mode not in {"add", "replace"}:
        raise ValueError(
            f"sources.{source_name}.custom_mode must be 'add' or "
            "'replace'.")

    custom_data = source_input.get("custom_data")
    capacity = source_input.get("capacity_additions")
    if custom_data is not None and capacity is not None:
        raise ValueError(
            f"sources.{source_name} cannot define both custom_data and "
            "capacity_additions.")
    if capacity is not None:
        parsed_dates = []
    elif not isinstance(custom_data, dict) or not custom_data:
        raise ValueError(
            f"sources.{source_name}.custom_data must be a non-empty "
            "mapping of dates to values.")
    else:
        parsed_dates = []
        for raw_date, raw_value in custom_data.items():
            try:
                parsed_dates.append(pd.Timestamp(raw_date))
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid custom date for '{source_name}': "
                    f"{raw_date}.") from exc
            _as_float(
                raw_value,
                f"sources.{source_name}.custom_data[{raw_date}]",)

    if len(parsed_dates) != len(set(parsed_dates)):
        raise ValueError(
            f"sources.{source_name}.custom_data contains duplicate "
            "dates.")

    if mode != "replace":
        return

    groups = source_input.get("replaces")
    if not isinstance(groups, list) or not groups:
        raise ValueError(
            f"sources.{source_name}.replaces must be a non-empty list.")

    used = set()
    for group in groups:
        if isinstance(group, dict):
            allocation = str(
                group.get("allocation", "proportional")
            ).strip().lower()
            if allocation != "proportional":
                raise ValueError(
                    f"sources.{source_name}.replaces supports only "
                    "proportional allocation.")
            names = group.get("sources")
        else:
            names = group
        if not isinstance(names, list) or not names:
            raise ValueError(
                f"Each replacement group for '{source_name}' must "
                "contain at least one source.")
        for name in names:
            if name == source_name:
                raise ValueError(
                    f"Custom source '{source_name}' cannot replace "
                    "itself.")
            if name not in defined_sources:
                raise ValueError(
                    f"Replacement source '{name}' is not configured.")
            if name in used:
                raise ValueError(
                    f"Replacement source '{name}' appears more than "
                    "once.")
            used.add(name)

def _validate_model(config: dict, label: str) -> None:
    """Validate one trend model expressed through dated values."""

    model = str(config.get("model", "")).strip().lower()
    if model == "custom":
        return
    allowed = ("linear", "exponential")
    if model not in allowed:
        choices = ", ".join(allowed)
        raise ValueError(
            f"{label}.model must be one of: {choices}, or 'custom'.")
    values = config.get("values")
    if not isinstance(values, dict) or len(values) < 2:
        raise ValueError(
            f"{label}.values must contain at least two dated values.")

    parsed = []
    for raw_date, raw_value in values.items():
        try:
            date = pd.Timestamp(raw_date)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid date in {label}.values: {raw_date}.") from exc
        value = _as_float(raw_value, f"{label}.values[{raw_date}]")
        if value < 0.0:
            raise ValueError(
                f"{label}.values[{raw_date}] cannot be negative.")
        parsed.append(date)

    if len(parsed) != len(set(parsed)):
        raise ValueError(f"{label}.values contains duplicate dates.")


def _validate_anchor(anchor: object, label: str) -> None:
    """Validate an optional historical anchoring rule."""

    if anchor is None:
        return
    if not isinstance(anchor, dict):
        raise ValueError(f"{label} must be a mapping.")
    method = str(anchor.get("method", "auto")).strip().lower()
    if method not in {"auto", "mean", "median", "last", "max"}:
        raise ValueError(
            f"{label}.method must be auto, mean, median, last or max.")
    if method != "auto" or "window" in anchor:
        _as_positive_int(
            anchor.get("window", 1),
            f"{label}.window",)


def _validate_emission_configuration(user_input: dict) -> None:
    """Require explicit source factors whenever emissions are requested."""

    output = user_input.get("output", {}) or {}
    enabled = bool(output.get("emissions", False))
    if not enabled:
        return
    missing = [
        name for name, source in user_input.get("sources", {}).items()
        if isinstance(source, dict) and "emission_factor_co2" not in source
    ]
    if missing:
        raise ValueError(
            "output.emissions is true, but emission_factor_co2 is missing "
            "for sources: " + ", ".join(sorted(missing)) + ".")


def _validate_monte_carlo(user_input: dict) -> None:
    """Validate the unified stochastic simulation configuration."""

    monte_carlo = user_input.get("monte_carlo", {}) or {}
    if not isinstance(monte_carlo, dict):
        raise ValueError("simulation.monte_carlo must be a mapping.")

    simulations = _as_nonnegative_int(
        monte_carlo.get("simulations", 0),
        "simulation.monte_carlo.simulations",)
    _as_positive_int(
        monte_carlo.get("workers", 1),
        "simulation.monte_carlo.workers",)
    _as_nonnegative_int(
        monte_carlo.get("seed", 12345),
        "simulation.monte_carlo.seed",)

    confidence = _as_float(
        monte_carlo.get("confidence_level", 0.95),
        "simulation.monte_carlo.confidence_level",)
    if confidence <= 0.0 or confidence >= 1.0:
        raise ValueError(
            "simulation.monte_carlo.confidence_level must be between "
            "zero and one.")

    for key in (
            "resume", "keep_analysis_temp", "enabled",
            "technology_uncertainty", "preserve_annual_targets"):
        if key not in monte_carlo:
            continue
        if not isinstance(monte_carlo[key], bool):
            raise ValueError(
                f"simulation.monte_carlo.{key} must be boolean.")

    value = str(
        monte_carlo.get("temporary_output_format", "parquet")
    ).strip().lower()
    if value not in {"parquet", "pickle", "pkl"}:
        raise ValueError(
            "simulation.monte_carlo.temporary_output_format must be "
            "'parquet' or 'pickle'.")

    if simulations <= 0:
        return

    if not user_input.get("variability_enabled", False):
        raise ValueError(
            "projection.variability.enabled must be true when Monte Carlo "
            "simulations are requested.")

    selected = monte_carlo.get("sources", [])
    if not isinstance(selected, list) or not selected:
        raise ValueError(
            "Monte Carlo requires at least one eligible stochastic "
            "energy series.")
    if len(selected) != len(set(selected)):
        raise ValueError(
            "simulation.monte_carlo.sources contains repeated values.")
    defined = {*user_input["sources"], "Demand"}
    unknown = [name for name in selected if name not in defined]
    if unknown:
        values = ", ".join(unknown)
        raise ValueError(
            "simulation.monte_carlo.sources contains undefined sources: "
            f"{values}.")


def _validate_commodities_input(
        user_input: dict, root_dir: Path
) -> None:
    """Validate the optional commodity and flexibility settings."""

    commodities = user_input.get("commodities_input", {}) or {}
    if not isinstance(commodities, dict):
        raise ValueError("commodities_input must be a mapping.")
    if not commodities:
        return

    unknown = sorted(set(commodities) - _ALLOWED_COMMODITIES_FIELDS)
    if unknown:
        values = ", ".join(unknown)
        raise ValueError(
            "Unknown commodities_input fields are not supported: "
            f"{values}.")

    if not commodities.get("run_commodities", True):
        return

    database_path = commodities.get(
        "database_path", "data/Database.yaml")
    database_file = _validate_file(
        root_dir, database_path, "Commodities database")
    with Path(database_file).open(encoding="utf-8") as stream:
        database = yaml.safe_load(stream) or {}
    if not isinstance(database, dict):
        raise ValueError("Commodities database must contain a YAML mapping.")

    if "run_commodities" in commodities and not isinstance(
            commodities["run_commodities"], bool):
        raise ValueError(
            "commodities_input.run_commodities must be boolean.")

    _validate_bess_lifetime(commodities.get("BESS"))
    _validate_interconnections(commodities.get("Interconnections"))
    _validate_fuel_to_electricity(commodities.get("Fuel_to_Electricity"))
    _validate_dispatch_order(commodities.get("Dispatch_Order"))
    _validate_commodity_pathways(commodities, database)


def _validate_commodity_pathways(
        commodities: dict, database: dict) -> None:
    """Validate commodity pathways and require explicit storage physics."""

    production = commodities.get("Commodities_Production")
    if production is None:
        return
    if not isinstance(production, dict) or not production:
        raise ValueError(
            "commodities_input.Commodities_Production must be a "
            "non-empty mapping.")

    electricity = database.get("Electricity_Consumption", {}) or {}
    requirements = database.get("Production_Inputs", {}) or {}
    storage = commodities.get("Commodity_Storage")
    if not isinstance(storage, dict):
        raise ValueError(
            "commodities_input.Commodity_Storage is required when "
            "commodity production is enabled.")

    total_share = 0.0
    for commodity, settings in production.items():
        prefix = f"commodities_input.Commodities_Production.{commodity}"
        if not isinstance(settings, dict):
            raise ValueError(f"{prefix} must be a mapping.")
        unknown = sorted(set(settings) - {"share", "technology", "inputs"})
        if unknown:
            raise ValueError(
                f"Unknown {prefix} fields: {', '.join(unknown)}.")
        if "share" not in settings or "technology" not in settings:
            raise ValueError(f"{prefix} requires share and technology.")
        share = _as_float(settings["share"], f"{prefix}.share")
        if share < 0.0:
            raise ValueError(f"{prefix}.share cannot be negative.")
        total_share += share

        technology = str(settings["technology"]).strip()
        available = electricity.get(commodity, {}) or {}
        if technology not in available:
            raise ValueError(
                f"{prefix}.technology '{technology}' is not defined for "
                f"commodity '{commodity}' in Database.yaml.")

        expected = requirements.get(commodity, {}) or {}
        inputs = settings.get("inputs", {}) or {}
        if not isinstance(inputs, dict):
            raise ValueError(f"{prefix}.inputs must be a mapping.")
        if set(inputs) != set(expected):
            missing = sorted(set(expected) - set(inputs))
            extra = sorted(set(inputs) - set(expected))
            details = []
            if missing:
                details.append("missing: " + ", ".join(missing))
            if extra:
                details.append("unexpected: " + ", ".join(extra))
            raise ValueError(
                f"{prefix}.inputs do not match Database.yaml ("
                + "; ".join(details) + ").")
        for input_name, input_technology in inputs.items():
            options = electricity.get(input_name, {}) or {}
            if input_technology not in options:
                raise ValueError(
                    f"{prefix}.inputs.{input_name} technology "
                    f"'{input_technology}' is not defined for "
                    f"'{input_name}' in Database.yaml.")

        if commodity not in storage or not isinstance(
                storage[commodity], dict):
            raise ValueError(
                f"commodities_input.Commodity_Storage.{commodity} is "
                "required for each produced commodity.")
        storage_settings = storage[commodity]
        unknown_storage = sorted(
            set(storage_settings)
            - {"max_age", "max_storage_quantity", "max_storage_mass"})
        if unknown_storage:
            raise ValueError(
                f"Unknown Commodity_Storage.{commodity} fields: "
                f"{', '.join(unknown_storage)}.")
        if "max_age" not in storage_settings:
            raise ValueError(
                f"Commodity_Storage.{commodity}.max_age is required.")
        if not ({"max_storage_quantity", "max_storage_mass"}
                & set(storage_settings)):
            raise ValueError(
                f"Commodity_Storage.{commodity}.max_storage_quantity is "
                "required.")
        if _as_float(
                storage_settings["max_age"],
                f"Commodity_Storage.{commodity}.max_age") <= 0.0:
            raise ValueError(
                f"Commodity_Storage.{commodity}.max_age must be positive.")
        capacity = storage_settings.get(
            "max_storage_quantity", storage_settings.get("max_storage_mass"))
        if _as_float(
                capacity, f"Commodity_Storage.{commodity}.max_storage_quantity"
        ) < 0.0:
            raise ValueError(
                f"Commodity_Storage.{commodity}.max_storage_quantity cannot "
                "be negative.")

    if abs(total_share - 1.0) > 1e-9:
        raise ValueError(
            "Commodities_Production shares must sum to 1.")


def _validate_interconnections(config: object) -> None:
    """Require explicit directional interconnection assumptions."""

    if config is None:
        return
    if not isinstance(config, dict) or not config:
        raise ValueError(
            "commodities_input.Interconnections must be a non-empty mapping.")
    if "import" in config or "export" in config:
        items = [("Total", config)]
    else:
        items = list(config.items())
    for name, item in items:
        prefix = f"commodities_input.Interconnections.{name}"
        if not isinstance(item, dict):
            raise ValueError(f"{prefix} must be a mapping.")
        if "mode" not in item:
            raise ValueError(f"{prefix}.mode is required (MW or fraction).")
        mode = str(item["mode"]).strip().lower()
        if mode not in {"mw", "fraction"}:
            raise ValueError(f"{prefix}.mode must be MW or fraction.")
        for direction in ("import", "export"):
            if direction not in item:
                raise ValueError(f"{prefix}.{direction} is required.")
            branch = item[direction]
            if not isinstance(branch, dict):
                raise ValueError(f"{prefix}.{direction} must be a mapping.")
            model = str(branch.get("model", "")).strip().lower()
            if model not in {"constant", "step", "linear"}:
                raise ValueError(
                    f"{prefix}.{direction}.model must be constant, step or "
                    "linear.")
            if model == "constant" and "value" not in branch:
                raise ValueError(f"{prefix}.{direction}.value is required.")
            if model in {"step", "linear"} and not branch.get("values"):
                raise ValueError(f"{prefix}.{direction}.values is required.")
            if "availability" not in branch:
                raise ValueError(
                    f"{prefix}.{direction}.availability is required.")
            availability = branch["availability"]
            if isinstance(availability, dict):
                missing = [
                    month for month in range(1, 13)
                    if month not in availability
                    and str(month) not in availability]
                if missing:
                    raise ValueError(
                        f"{prefix}.{direction}.availability is missing "
                        f"months: {missing}.")
                values = [
                    availability.get(month, availability.get(str(month)))
                    for month in range(1, 13)]
            else:
                values = [availability]
            for value in values:
                fraction = _as_float(
                    value, f"{prefix}.{direction}.availability")
                if fraction < 0.0 or fraction > 1.0:
                    raise ValueError(
                        f"{prefix}.{direction}.availability must be "
                        "between 0 and 1.")

def _validate_fuel_to_electricity(config: object) -> None:
    """Validate explicit fuel shares and reconversion technologies."""

    if config is None:
        return
    if not isinstance(config, dict):
        raise ValueError(
            "commodities_input.Fuel_to_Electricity must be a mapping.")
    for commodity, setting in config.items():
        prefix = f"commodities_input.Fuel_to_Electricity.{commodity}"
        if isinstance(setting, dict):
            unknown = sorted(set(setting) - {"share", "technology"})
            if unknown:
                raise ValueError(
                    f"Unknown {prefix} fields: {', '.join(unknown)}.")
            if "share" not in setting or "technology" not in setting:
                raise ValueError(
                    f"{prefix} requires share and technology.")
            share = _as_float(setting["share"], f"{prefix}.share")
            if not str(setting["technology"]).strip():
                raise ValueError(f"{prefix}.technology cannot be empty.")
        else:
            # Legacy scalar shares remain readable. Runtime resolution allows
            # them only when the database contains a unique technology.
            share = _as_float(setting, prefix)
        if share < 0.0:
            raise ValueError(f"{prefix}.share cannot be negative.")


def _validate_dispatch_order(config: object) -> None:
    """Reject dispatch orders the current balance engine cannot execute."""

    if config is None:
        return
    if not isinstance(config, dict):
        raise ValueError("commodities_input.Dispatch_Order must be a mapping.")
    allowed = {
        "surplus": {"BESS", "Interconnections", "Commodities_Production"},
        "deficit": {"BESS", "Interconnections", "Fuel_to_Electricity"},
    }
    final_stage = {
        "surplus": "Commodities_Production",
        "deficit": "Fuel_to_Electricity",
    }
    for kind, order in config.items():
        if kind not in allowed:
            raise ValueError(
                f"commodities_input.Dispatch_Order.{kind} is not supported.")
        if not isinstance(order, list):
            raise ValueError(
                f"commodities_input.Dispatch_Order.{kind} must be a list.")
        normalized = [str(value).strip() for value in order]
        unknown = [value for value in normalized if value not in allowed[kind]]
        if unknown:
            raise ValueError(
                f"Unknown Dispatch_Order.{kind} items: "
                f"{', '.join(unknown)}.")
        if len(normalized) != len(set(normalized)):
            raise ValueError(
                f"Dispatch_Order.{kind} must not contain duplicates.")
        terminal = final_stage[kind]
        if terminal in normalized and normalized[-1] != terminal:
            raise ValueError(
                f"Dispatch_Order.{kind} requires {terminal} to be the "
                "final stage because commodity conversion is applied after "
                "BESS/interconnections in the current balance engine.")


def _validate_bess_lifetime(bess: object) -> None:
    """Validate optional simple BESS lifetime assumptions."""

    if bess is None:
        return
    if not isinstance(bess, dict):
        raise ValueError("commodities_input.BESS must be a mapping.")

    if _ALLOWED_BESS_FIELDS.intersection(bess):
        items = [("Total", bess)]
    else:
        items = list(bess.items())

    lifetime_values = []
    lifetime_count = 0
    for name, config in items:
        if not isinstance(config, dict):
            raise ValueError(
                f"BESS.{name} must be a mapping.")
        unknown = sorted(set(config) - _ALLOWED_BESS_FIELDS)
        if unknown:
            values = ", ".join(unknown)
            raise ValueError(
                f"Unknown BESS.{name} fields are not supported: {values}.")
        if "duration" not in config:
            raise ValueError(f"BESS.{name}.duration is required.")
        if "efficiency" not in config:
            raise ValueError(f"BESS.{name}.efficiency is required.")
        duration = _as_float(config["duration"], f"BESS.{name}.duration")
        efficiency = _as_float(
            config["efficiency"], f"BESS.{name}.efficiency")
        if duration <= 0.0:
            raise ValueError(f"BESS.{name}.duration must be > 0.")
        if efficiency <= 0.0 or efficiency > 1.0:
            raise ValueError(
                f"BESS.{name}.efficiency must be in (0, 1].")
        lifetime = config.get("lifetime")
        if lifetime is None:
            continue
        lifetime_count += 1
        if not isinstance(lifetime, dict):
            raise ValueError(
                f"BESS.{name}.lifetime must be a mapping.")
        if not lifetime:
            continue

        unknown = sorted(
            set(lifetime) - _ALLOWED_BESS_LIFETIME_FIELDS)
        if unknown:
            values = ", ".join(unknown)
            raise ValueError(
                f"Unknown BESS.{name}.lifetime fields are not supported: "
                f"{values}.")

        calendar = lifetime.get("calendar_years")
        cycle = lifetime.get("cycle_life_efc")
        if calendar is None and cycle is None:
            raise ValueError(
                f"BESS.{name}.lifetime requires calendar_years and/or "
                "cycle_life_efc.")
        if calendar is not None:
            value = _as_float(
                calendar, f"BESS.{name}.lifetime.calendar_years")
            if value <= 0.0:
                raise ValueError(
                    f"BESS.{name}.lifetime.calendar_years must be > 0.")
        if cycle is not None:
            value = _as_float(
                cycle, f"BESS.{name}.lifetime.cycle_life_efc")
            if value <= 0.0:
                raise ValueError(
                    f"BESS.{name}.lifetime.cycle_life_efc must be > 0.")
        lifetime_values.append((calendar, cycle))

    if 0 < lifetime_count < len(items):
        raise ValueError(
            "All named BESS assets must define lifetime assumptions when "
            "aggregate BESS replacements are enabled.")
    if len(set(lifetime_values)) > 1:
        raise ValueError(
            "All named BESS assets must use the same lifetime assumptions "
            "for the aggregate replacement estimate.")


def _validate_output(config: object) -> None:
    """Validate the compact public output block."""

    if config is None:
        return
    if not isinstance(config, dict):
        raise ValueError("output must be a mapping.")

    allowed_fields = {"level", "event_threshold", "emissions"}
    unknown = sorted(set(config) - allowed_fields)
    if unknown:
        values = ", ".join(unknown)
        raise ValueError(
            "Unknown output fields are not supported: " + values + ".")

    level = str(config.get("level", "analysis")).strip().lower()
    if level not in {"comparison", "analysis", "detailed"}:
        raise ValueError(
            "output.level must be comparison, analysis or detailed.")

    if "emissions" in config and not isinstance(config["emissions"], bool):
        raise ValueError("output.emissions must be boolean.")

    threshold = config.get("event_threshold", 1e-9)
    value = _as_float(threshold, "output.event_threshold")
    if value < 0.0:
        raise ValueError("output.event_threshold cannot be negative.")


def _validate_file(
        root_dir: Path, file_name: object, label: str
) -> Path:
    """Resolve and validate one required file."""

    file_path = Path(str(file_name))
    if not file_path.is_absolute():
        file_path = root_dir / file_path
    file_path = file_path.resolve()
    if not file_path.is_file():
        raise FileNotFoundError(
            f"{label} file not found: {file_path}")
    return file_path


def _as_float(value: Any, name: str) -> float:
    """Convert one value to a finite float."""

    try:
        converted = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be numeric.") from exc
    if not math.isfinite(converted):
        raise ValueError(f"{name} must be finite.")
    return converted


def _as_nonnegative_int(value: Any, name: str) -> int:
    """Convert one value to a finite non-negative integer."""

    if isinstance(value, bool):
        raise ValueError(f"{name} must be a non-negative integer.")
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            f"{name} must be a non-negative integer.") from exc
    if not math.isfinite(numeric) or numeric < 0.0:
        raise ValueError(f"{name} must be a non-negative integer.")
    if not numeric.is_integer():
        raise ValueError(f"{name} must be a non-negative integer.")
    return int(numeric)


def _as_positive_int(value: Any, name: str) -> int:
    """Convert one value to a positive integer."""

    converted = _as_nonnegative_int(value, name)
    if converted <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return converted
