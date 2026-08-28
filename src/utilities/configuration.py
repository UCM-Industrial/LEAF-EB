"""Load and normalize the current LEAF-EB configuration schema."""

from __future__ import annotations

from copy import deepcopy
import hashlib
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from src.utilities.name_resolution import normalize_user_input
from src.utilities.units import (
    canonical_energy_unit, parse_burnup, parse_calendar_months,
    parse_count_rate_per_day,
    parse_duration_days, parse_duration_hours,
    parse_emission_factor, parse_energy, parse_energy_rate,
    parse_fraction_rate_per_hour, parse_heavy_metal_mass,
    parse_percent, parse_power)


_REQUIRED_SECTIONS = {
    "scenario",
    "historical_data",
    "projection",
    "simulation",
    "demand",
    "sources",}
_OPTIONAL_SECTIONS = {
    "commodities_input",
    "output",}
_RESOLUTION_CODES = {
    "daily": "D",
    "hourly": "h",}


def load_config_file(path: str | Path) -> dict[str, Any]:
    """Load one YAML file and return the normalized runtime mapping."""

    config_path = Path(path).resolve()
    with config_path.open(encoding="utf-8") as stream:
        raw = yaml.safe_load(stream) or {}

    project_root = Path(__file__).resolve().parents[2]
    runtime = build_runtime_config(raw, project_root)
    runtime["_config_path"] = str(config_path)
    return runtime


def build_runtime_config(
        raw: dict[str, Any], root_dir: str | Path
) -> dict[str, Any]:
    """Validate the schema shape and build the canonical runtime mapping."""

    if not isinstance(raw, dict):
        raise TypeError("The input file must contain a YAML mapping.")

    unknown = sorted(
        set(raw).difference(_REQUIRED_SECTIONS | _OPTIONAL_SECTIONS))
    if unknown:
        values = ", ".join(unknown)
        raise ValueError(
            "Unknown top-level input fields are not supported: "
            f"{values}.")

    missing = sorted(_REQUIRED_SECTIONS.difference(raw))
    if missing:
        values = ", ".join(missing)
        raise ValueError(
            f"Missing required input sections: {values}.")

    config = deepcopy(raw)
    root = Path(root_dir).resolve()
    _apply_source_technology_templates(config, root)
    scenario = _mapping(config, "scenario")
    historical = _mapping(config, "historical_data")
    projection = _mapping(config, "projection")
    simulation = _mapping(config, "simulation")
    variability = projection.get("variability", {}) or {}

    if not isinstance(variability, dict):
        raise ValueError(
            "projection.variability must be a mapping.")

    projection_resolution = _resolution(
        projection.get("resolution"),
        "projection.resolution",)
    simulation_resolution = _resolution(
        simulation.get("resolution"),
        "simulation.resolution",)
    processing_resolution = _resolution(
        historical.get("processing_resolution"),
        "historical_data.processing_resolution",)
    variability_resolution = _resolution(
        variability.get("resolution", projection_resolution),
        "projection.variability.resolution",)

    config["scenario_folder"] = _required_text(
        scenario, "folder", "scenario.folder")
    config["scenario_subfolder"] = _required_text(
        scenario, "subfolder", "scenario.subfolder")
    start_value = _required_text(
        scenario, "start_date", "scenario.start_date")
    end_value = _required_text(
        scenario, "end_date", "scenario.end_date")
    start_date = pd.Timestamp(start_value)
    end_date = pd.Timestamp(end_value)
    if (
        projection_resolution == "hourly"
        and end_date == end_date.normalize()
    ):
        end_date += pd.Timedelta(hours=23)
    config["start_date"] = start_date.isoformat()
    config["end_date"] = end_date.isoformat()

    config["historical_data_file"] = _required_text(
        historical, "file", "historical_data.file")
    config["date_column"] = str(
        historical.get("date_column", "Date")).strip()
    config["energy_unit"] = canonical_energy_unit(
        _required_text(
            historical, "unit", "historical_data.unit"))
    config["processing_resolution"] = processing_resolution

    _normalize_physical_input_schema(
        config, projection_resolution=projection_resolution)

    demand = config.get("demand", {})
    if isinstance(demand, dict):
        demand.setdefault("balance", 1.0)

    config["projection_resolution"] = projection_resolution
    config["variability_resolution"] = variability_resolution
    config["simulation_resolution"] = simulation_resolution
    config["projection_frequency"] = _RESOLUTION_CODES[
        projection_resolution]
    config["projection_grouping"] = (
        "Day" if projection_resolution == "daily" else "Hour")
    config["pattern_file"] = str(
        projection.get("pattern_file", "Pattern.xlsx"))
    config["save_projection_plots"] = _boolean(
        projection.get("save_plots", False),
        "projection.save_plots",)
    config["variability_enabled"] = _boolean(
        variability.get("enabled", False),
        "projection.variability.enabled",)
    config["variability_mode"] = variability.get("mode", "global")
    config["plot_residuals"] = _boolean(
        variability.get("plot_residuals", False),
        "projection.variability.plot_residuals",)
    console_output = str(
        simulation.get("console_output", "standard")
    ).strip().lower()
    if console_output not in {"quiet", "standard", "detailed"}:
        raise ValueError(
            "simulation.console_output must be quiet, standard or detailed.")
    config["console_output"] = console_output
    config["hourly_simulation"] = {
        "enabled": simulation_resolution == "hourly",}

    output_dir = (
        root
        / config["scenario_folder"]
        / config["scenario_subfolder"])
    external_profile = simulation.get("hourly_profile_file")
    profile_columns = None
    if external_profile:
        profile_path = Path(str(external_profile))
        if not profile_path.is_absolute():
            profile_path = root / profile_path
        profile_path = profile_path.resolve()
        config["external_hourly_profile_file"] = str(profile_path)
        if profile_path.is_file():
            profile_columns = _profile_columns(profile_path)
    else:
        profile_path = output_dir / "Hourly_Pattern.xlsx"
        config["external_hourly_profile_file"] = None
    config["historical_hourly_pattern_file"] = str(profile_path)

    if (
        projection_resolution == "daily"
        and simulation_resolution == "hourly"
    ):
        demand = config.get("demand", {})
        demand_available = (
            profile_columns is None or "demand" in profile_columns)
        if isinstance(demand, dict) and demand_available:
            demand["hourly_patterns"] = str(profile_path)
        for source_name, source_data in config.get("sources", {}).items():
            if not isinstance(source_data, dict):
                continue
            model = str(
                source_data.get("model", "")
            ).strip().lower()
            operation = str(
                source_data.get("hourly_operation", "")
            ).strip().lower()
            if model == "custom" or operation in {
                "must_run", "load_following"
            }:
                continue
            source_available = (
                profile_columns is None
                or str(source_name).casefold() in profile_columns)
            if source_available:
                source_data["hourly_patterns"] = str(profile_path)

    commodities = config.get("commodities_input")
    if commodities is not None and not isinstance(commodities, dict):
        raise ValueError("commodities_input must be a mapping.")

    _apply_monte_carlo_settings(config)
    config = normalize_user_input(config)
    return config



def _normalize_physical_input_schema(
        config: dict[str, Any], *, projection_resolution: str) -> None:
    """Parse dimensional input values in place using canonical field names."""

    energy_unit = config["energy_unit"]
    demand = config.get("demand", {})
    if isinstance(demand, dict) and "target_production" in demand:
        demand["target_production"] = parse_energy_rate(
            demand["target_production"],
            field="demand.target_production",
            target_energy_unit=energy_unit,
            target_resolution=projection_resolution)

    simulation = config.get("simulation", {})
    if isinstance(simulation, dict):
        _normalize_percent_value(
            simulation, "capacity_tolerance",
            "simulation.capacity_tolerance")

    sources = config.get("sources", {})
    if isinstance(sources, dict):
        for source_name, source in sources.items():
            if not isinstance(source, dict):
                continue
            prefix = f"sources.{source_name}"
            _normalize_source_quantities(
                source, prefix=prefix, energy_unit=energy_unit)

    commodities = config.get("commodities_input")
    if isinstance(commodities, dict):
        _normalize_commodities_quantities(commodities)


def _normalize_source_quantities(
        source: dict[str, Any], *, prefix: str, energy_unit: str) -> None:
    """Normalize unit-aware source, operation, refuelling and fuel data."""

    _normalize_percent_value(
        source, "capacity_tolerance", f"{prefix}.capacity_tolerance")
    _normalize_scalar_value(
        source, "initial_capacity",
        lambda value: parse_power(
            value, field=f"{prefix}.initial_capacity"))
    _normalize_mapping_values(
        source, "capacity_additions",
        lambda value, date: parse_power(
            value, field=f"{prefix}.capacity_additions[{date}]"))

    custom_data = source.get("custom_data")
    if isinstance(custom_data, dict):
        for date, value in list(custom_data.items()):
            if isinstance(value, str):
                custom_data[date] = parse_energy(
                    value, field=f"{prefix}.custom_data[{date}]",
                    to_unit=energy_unit)

    emission = source.get("emission_factor_co2")
    if isinstance(emission, str):
        source["emission_factor_co2"] = parse_emission_factor(
            emission, field=f"{prefix}.emission_factor_co2")

    load_following = source.get("load_following")
    if isinstance(load_following, dict):
        _normalize_scalar_value(
            load_following, "ramp_up_rate",
            lambda value: parse_fraction_rate_per_hour(
                value, field=f"{prefix}.load_following.ramp_up_rate"))
        _normalize_scalar_value(
            load_following, "ramp_down_rate",
            lambda value: parse_fraction_rate_per_hour(
                value, field=f"{prefix}.load_following.ramp_down_rate"))
        _normalize_scalar_value(
            load_following, "deep_reduction_request_duration",
            lambda value: parse_duration_hours(
                value,
                field=(f"{prefix}.load_following."
                       "deep_reduction_request_duration")))
        _normalize_scalar_value(
            load_following, "max_deep_reduction_cycles",
            lambda value: parse_count_rate_per_day(
                value,
                field=(f"{prefix}.load_following."
                       "max_deep_reduction_cycles")))

    _normalize_scalar_value(
        source, "unit_capacity",
        lambda value: parse_power(
            value, field=f"{prefix}.unit_capacity"))

    refueling = source.get("refueling")
    if isinstance(refueling, dict):
        _normalize_scalar_value(
            refueling, "outage_duration",
            lambda value: _whole_days(
                parse_duration_days(
                    value, field=f"{prefix}.refueling.outage_duration"),
                f"{prefix}.refueling.outage_duration"))
        _normalize_scalar_value(
            refueling, "operating_cycle",
            lambda value: parse_calendar_months(
                value, field=f"{prefix}.refueling.operating_cycle"))

    fuel_cycle = source.get("fuel_cycle")
    if isinstance(fuel_cycle, dict):
        _normalize_scalar_value(
            fuel_cycle, "reference_net_power",
            lambda value: parse_power(
                value, field=f"{prefix}.fuel_cycle.reference_net_power"))
        _normalize_scalar_value(
            fuel_cycle, "reference_thermal_power",
            lambda value: parse_power(
                value, field=f"{prefix}.fuel_cycle.reference_thermal_power"))
        _normalize_scalar_value(
            fuel_cycle, "reference_core_fuel_mass",
            lambda value: parse_heavy_metal_mass(
                value,
                field=f"{prefix}.fuel_cycle.reference_core_fuel_mass"))
        _normalize_scalar_value(
            fuel_cycle, "thermal_power",
            lambda value: parse_power(
                value, field=f"{prefix}.fuel_cycle.thermal_power"))
        _normalize_scalar_value(
            fuel_cycle, "core_fuel_mass",
            lambda value: parse_heavy_metal_mass(
                value, field=f"{prefix}.fuel_cycle.core_fuel_mass"))
        _normalize_scalar_value(
            fuel_cycle, "target_burnup",
            lambda value: parse_burnup(
                value, field=f"{prefix}.fuel_cycle.target_burnup"))


def _normalize_commodities_quantities(commodities: dict[str, Any]) -> None:
    """Normalize the unit-aware commodity/BESS public input fields."""

    storage = commodities.get("Commodity_Storage")
    if isinstance(storage, dict):
        for name, settings in storage.items():
            if not isinstance(settings, dict):
                continue
            _normalize_scalar_value(
                settings, "max_age",
                lambda value, label=f"Commodity_Storage.{name}.max_age":
                parse_calendar_months(value, field=label) / 12.0)

    bess = commodities.get("BESS")
    if isinstance(bess, dict):
        for name, settings in bess.items():
            if not isinstance(settings, dict):
                continue
            prefix = f"commodities_input.BESS.{name}"
            _normalize_scalar_value(
                settings, "value",
                lambda value: parse_power(
                    value, field=f"{prefix}.value"))
            values = settings.get("values")
            if isinstance(values, dict):
                for date, value in list(values.items()):
                    if isinstance(value, str):
                        values[date] = parse_power(
                            value, field=f"{prefix}.values[{date}]")
            _normalize_scalar_value(
                settings, "duration",
                lambda value: parse_duration_hours(
                    value, field=f"{prefix}.duration"))
            if "efficiency" in settings:
                settings["efficiency"] = float(settings["efficiency"])

    _normalize_interconnection_quantities(commodities)


def _normalize_interconnection_quantities(
        commodities: dict[str, Any]) -> None:
    """Normalize absolute interconnection schedules to MW.

    Interconnections default to absolute power.  Public inputs may therefore
    use explicit power strings such as ``"5000 MW"`` or ``"8.5 GW"``.
    ``mode: fraction`` remains dimensionless and is left unchanged.
    """

    interconnections = commodities.get("Interconnections")
    if not isinstance(interconnections, dict):
        return

    if any(key in interconnections for key in ("import", "export")):
        items = [("Total", interconnections)]
    else:
        items = [
            (str(name), settings)
            for name, settings in interconnections.items()
            if isinstance(settings, dict)
        ]

    for name, settings in items:
        mode = str(settings.get("mode", "MW")).strip().lower()
        if mode == "fraction":
            continue
        for direction in ("import", "export"):
            branch = settings.get(direction)
            field = (
                f"commodities_input.Interconnections.{name}.{direction}")
            if isinstance(branch, str):
                settings[direction] = parse_power(branch, field=field)
                continue
            if not isinstance(branch, dict):
                continue
            for key in ("value", "initial", "final"):
                if isinstance(branch.get(key), str):
                    branch[key] = parse_power(
                        branch[key], field=f"{field}.{key}")
            values = branch.get("values")
            if isinstance(values, dict):
                for date, value in list(values.items()):
                    if isinstance(value, str):
                        values[date] = parse_power(
                            value, field=f"{field}.values[{date}]")


def _normalize_percent_value(
        mapping: dict[str, Any], key: str, field: str) -> None:
    """Parse one explicit percentage while retaining its canonical key."""

    _normalize_scalar_value(
        mapping, key,
        lambda value: parse_percent(value, field=field))


def _normalize_scalar_value(
        mapping: dict[str, Any], key: str, parser) -> None:
    """Parse one unit-aware scalar in place."""

    if key not in mapping:
        return
    mapping[key] = parser(mapping[key])


def _normalize_mapping_values(
        mapping: dict[str, Any], key: str, parser) -> None:
    """Parse one date-to-quantity mapping in place."""

    if key not in mapping:
        return
    raw = mapping[key]
    if not isinstance(raw, dict):
        raise ValueError(f"{key} must be a mapping of dates to values.")
    mapping[key] = {
        date: parser(value, date) for date, value in raw.items()}


def _whole_days(value: float, field: str) -> int:
    """Require an outage duration that maps exactly to whole model days."""

    rounded = int(round(value))
    if abs(value - rounded) > 1e-9:
        raise ValueError(
            f"{field} must resolve to a whole number of days in the current "
            "refuelling model.")
    return rounded


def _apply_monte_carlo_settings(config: dict[str, Any]) -> None:
    """Normalize the current global Monte Carlo configuration."""

    simulation = config.get("simulation", {})
    commodities = config.get("commodities_input", {}) or {}
    raw_mc = simulation.get("monte_carlo", {}) or {}
    if not isinstance(raw_mc, dict):
        raise ValueError("simulation.monte_carlo must be a mapping.")

    simulations = raw_mc.get("simulations", 0)
    workers = raw_mc.get("workers", 1)
    seed = raw_mc.get("seed", 12345)
    confidence = raw_mc.get("confidence_level", 0.95)
    resume = raw_mc.get("resume", False)
    temp_format = raw_mc.get("temporary_output_format", "parquet")
    keep_temp = raw_mc.get("keep_analysis_temp", False)
    technology_uncertainty = raw_mc.get("technology_uncertainty", False)
    if not isinstance(technology_uncertainty, bool):
        raise ValueError(
            "simulation.monte_carlo.technology_uncertainty must be boolean.")
    preserve_annual_targets = raw_mc.get(
        "preserve_annual_targets", False)
    if not isinstance(preserve_annual_targets, bool):
        raise ValueError(
            "simulation.monte_carlo.preserve_annual_targets must be boolean.")
    save_forecasts = raw_mc.get(
        "save_perturbed_forecasts", {
            "first_monte_carlo": True,
            "all_monte_carlo": False})
    requested_sources = raw_mc.get("sources")
    stochastic = _positive_simulation_count(simulations)

    if requested_sources is not None and not isinstance(
            requested_sources, list):
        raise ValueError(
            "simulation.monte_carlo.sources must be a list when supplied.")
    if not stochastic:
        requested_sources = []
    elif requested_sources is None:
        raise ValueError(
            "simulation.monte_carlo.sources must be supplied when "
            "Monte Carlo simulations are requested. Explicit selection "
            "prevents unintended stochastic treatment of energy series.")

    runtime = {
        "simulations": simulations, "workers": workers, "seed": seed,
        "confidence_level": confidence, "resume": resume,
        "temporary_output_format": temp_format,
        "keep_analysis_temp": keep_temp,
        "technology_uncertainty": technology_uncertainty,
        "preserve_annual_targets": preserve_annual_targets,
        "save_perturbed_forecasts": save_forecasts,
        "sources": list(requested_sources), "enabled": stochastic}
    config["monte_carlo"] = runtime

    if commodities:
        commodities.setdefault("database_path", "data/Database.yaml")

def _positive_simulation_count(value: object) -> bool:
    """Return whether a Monte Carlo count requests stochastic cases."""

    try:
        return int(value) > 0
    except (TypeError, ValueError):
        return False



def _mapping(config: dict[str, Any], key: str) -> dict[str, Any]:
    """Return one required mapping from the configuration."""

    value = config.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"{key} must be a mapping.")
    return value


def _required_text(
        mapping: dict[str, Any], key: str, label: str
) -> str:
    """Return one required non-empty text value."""

    value = mapping.get(key)
    if value is None or not str(value).strip():
        raise ValueError(f"{label} is required.")
    return str(value).strip()


def _resolution(value: object, label: str) -> str:
    """Normalize one daily or hourly resolution value."""

    resolution = str(value or "").strip().lower()
    if resolution not in _RESOLUTION_CODES:
        raise ValueError(
            f"{label} must be 'daily' or 'hourly'.")
    return resolution



def _boolean(value: object, label: str) -> bool:
    """Return one strict YAML boolean value."""

    if not isinstance(value, bool):
        raise ValueError(f"{label} must be boolean.")
    return value

def _profile_columns(path: Path) -> set[str]:
    """Return normalized column names from one hourly-profile file."""

    if path.suffix.lower() == ".xlsx":
        columns = pd.read_excel(
            path,
            engine="openpyxl",
            nrows=0,
        ).columns
    elif path.suffix.lower() == ".csv":
        columns = pd.read_csv(path, nrows=0).columns
    else:
        raise ValueError(
            "Hourly profile file must be an XLSX or CSV file.")
    return {str(column).strip().casefold() for column in columns}

def _apply_source_technology_templates(
        config: dict[str, Any], root_dir: Path) -> None:
    """Merge reusable technology templates into configured sources."""

    sources = config.get("sources", {})
    if not isinstance(sources, dict):
        return
    for source_name, source_input in sources.items():
        if not isinstance(source_input, dict):
            continue
        reference = source_input.get("technology_template")
        if reference is None:
            continue
        template = _load_source_technology_template(
            reference, root_dir, source_name)
        overrides = deepcopy(source_input)
        merged = _deep_merge_mapping(template, overrides)
        sources[source_name] = merged


def _load_source_technology_template(
        reference: object, root_dir: Path, source_name: str
) -> dict[str, Any]:
    """Load one named source template from a project YAML file."""

    if not isinstance(reference, dict):
        raise ValueError(
            f"sources.{source_name}.technology_template must be a mapping.")
    file_value = reference.get("file")
    name_value = reference.get("name")
    if file_value is None or not str(file_value).strip():
        raise ValueError(
            f"sources.{source_name}.technology_template.file is required.")
    if name_value is None or not str(name_value).strip():
        raise ValueError(
            f"sources.{source_name}.technology_template.name is required.")

    path = Path(str(file_value).strip())
    if not path.is_absolute():
        path = root_dir / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(
            f"Technology template file not found for '{source_name}': "
            f"{path}.")

    with path.open(encoding="utf-8") as stream:
        document = yaml.safe_load(stream) or {}
    if not isinstance(document, dict):
        raise ValueError(
            f"Technology template file must contain a YAML mapping: {path}.")
    templates = document.get("templates", {})
    if not isinstance(templates, dict):
        raise ValueError(
            f"Technology template file requires a templates mapping: {path}.")

    name = str(name_value).strip()
    selected = templates.get(name)
    if not isinstance(selected, dict):
        raise ValueError(
            f"Technology template '{name}' was not found in {path}.")
    source_defaults = selected.get("source", {})
    if not isinstance(source_defaults, dict):
        raise ValueError(
            f"Technology template '{name}' requires a source mapping.")
    return deepcopy(source_defaults)


def _deep_merge_mapping(
        defaults: dict[str, Any], overrides: dict[str, Any]
) -> dict[str, Any]:
    """Return recursively merged mappings with explicit overrides winning."""

    merged = deepcopy(defaults)
    for key, value in overrides.items():
        existing = merged.get(key)
        if isinstance(existing, dict) and isinstance(value, dict):
            merged[key] = _deep_merge_mapping(existing, value)
        else:
            merged[key] = deepcopy(value)
    return merged



_RESOLVED_INPUT_SECTIONS = (
    "scenario", "historical_data", "projection", "simulation", "demand",
    "sources", "commodities_input", "output")
_RESOLVED_RUNTIME_FIELDS = (
    "scenario_folder", "scenario_subfolder", "start_date", "end_date",
    "historical_data_file", "date_column", "energy_unit",
    "processing_resolution", "projection_resolution",
    "variability_resolution", "simulation_resolution",
    "projection_frequency", "projection_grouping", "pattern_file",
    "save_projection_plots", "variability_enabled", "variability_mode",
    "plot_residuals", "console_output", "hourly_simulation",
    "external_hourly_profile_file", "historical_hourly_pattern_file",
    "monte_carlo")


def _yaml_safe(value: Any) -> Any:
    """Convert runtime values to deterministic YAML-safe Python objects."""

    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): _yaml_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_yaml_safe(item) for item in value]
    if hasattr(value, "item") and callable(value.item):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    return value


def _file_sha256(path: str | Path | None) -> str | None:
    """Return a SHA-256 digest when a referenced file exists."""

    if path is None:
        return None
    candidate = Path(str(path))
    if not candidate.is_file():
        return None
    digest = hashlib.sha256()
    with candidate.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _merge_resolved(base: dict[str, Any], update: dict[str, Any]) -> None:
    """Merge nested resolved metadata without discarding prior stages."""

    for key, value in update.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _merge_resolved(base[key], value)
        else:
            base[key] = deepcopy(value)


def write_resolved_config(
        config: dict[str, Any],
        scenario_dir: str | Path,
        *,
        resolved: dict[str, Any] | None = None,
        run_id: str | None = None,
        database_path: str | Path | None = None,
) -> Path:
    """Write the effective LEAF configuration after defaults and templates."""

    output_dir = Path(scenario_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "Resolved_Config.yaml"
    prior_resolved: dict[str, Any] = {}
    if output_path.is_file():
        with output_path.open(encoding="utf-8") as stream:
            prior = yaml.safe_load(stream) or {}
        if isinstance(prior, dict) and isinstance(prior.get("resolved"), dict):
            prior_resolved = prior["resolved"]
    if resolved:
        _merge_resolved(prior_resolved, resolved)

    source_path = config.get("_config_path")
    runtime = {
        field: deepcopy(config[field])
        for field in _RESOLVED_RUNTIME_FIELDS
        if field in config}
    effective_input = {
        section: deepcopy(config[section])
        for section in _RESOLVED_INPUT_SECTIONS
        if section in config}
    provenance = {
        "source_config": source_path,
        "source_config_sha256": _file_sha256(source_path)}
    if database_path is not None:
        database = Path(database_path).resolve()
        provenance["database"] = str(database)
        provenance["database_sha256"] = _file_sha256(database)
    if run_id:
        provenance["run_id"] = str(run_id)

    document = {
        "schema": "LEAF-EB Resolved Configuration v1",
        "effective_input": effective_input,
        "runtime": runtime,
        "resolved": prior_resolved,
        "provenance": provenance}
    with output_path.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(
            _yaml_safe(document), stream, sort_keys=False,
            allow_unicode=True, width=100)
    return output_path
