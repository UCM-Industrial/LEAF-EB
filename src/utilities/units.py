"""Centralize physical-unit parsing and conversions used by LEAF-EB.

Public YAML inputs may carry units directly with the value, for example
``"300 MW"`` or ``"60 GWd/tHM"``.  Model calculations continue to use a
small set of canonical internal units so downstream modules never need to
interpret the user's original unit spelling.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


_ENERGY_TO_MWH = {
    "KWH": 1.0e-3,
    "MWH": 1.0,
    "GWH": 1.0e3,
    "TWH": 1.0e6,}
_ENERGY_LABELS = {
    "KWH": "kWh",
    "MWH": "MWh",
    "GWH": "GWh",
    "TWH": "TWh",}
_POWER_TO_MW = {
    "KW": 1.0e-3,
    "MW": 1.0,
    "GW": 1.0e3,}
_QUANTITY_UNIT_ALIASES = {
    "l": "L",
    "lt": "L",
    "liter": "L",
    "litre": "L",
    "liters": "L",
    "litres": "L",
    "kg": "kg",
    "kilogram": "kg",
    "kilograms": "kg",}

_NUMBER_UNIT_RE = re.compile(
    r"^\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)"
    r"\s*(.*?)\s*$")

_TIME_TO_HOURS = {
    "H": 1.0,
    "HR": 1.0,
    "HRS": 1.0,
    "HOUR": 1.0,
    "HOURS": 1.0,
    "D": 24.0,
    "DAY": 24.0,
    "DAYS": 24.0,}
_ENERGY_RATE_PERIOD_HOURS = {
    "H": 1.0,
    "HR": 1.0,
    "HOUR": 1.0,
    "DAY": 24.0,
    "D": 24.0,
    "YEAR": 8760.0,
    "YR": 8760.0,
    "Y": 8760.0,}
_CALENDAR_TO_MONTHS = {
    "MONTH": 1.0,
    "MONTHS": 1.0,
    "MO": 1.0,
    "YEAR": 12.0,
    "YEARS": 12.0,
    "YR": 12.0,
    "Y": 12.0,}
_HEAVY_METAL_TO_THM = {
    "KGHM": 1.0e-3,
    "THM": 1.0,}
_BURNUP_TO_GWD_THM = {
    "MWD/THM": 1.0e-3,
    "GWD/THM": 1.0,
    "MWD/KGHM": 1.0,
    "GWD/KGHM": 1.0e3,}
_CO2_MASS_TO_KG = {
    "GCO2E": 1.0e-3,
    "KGCO2E": 1.0,
    "TCO2E": 1.0e3,}


@dataclass(frozen=True)
class ParsedQuantity:
    """One parsed scalar physical quantity before canonical conversion."""

    value: float
    unit: str


def _split_quantity(value, field: str) -> ParsedQuantity:
    """Split ``'<number> <unit>'`` and reject unitless dimensional values."""

    if not isinstance(value, str):
        raise ValueError(
            f"{field} requires a value with an explicit unit, for example "
            '"300 MW".')
    match = _NUMBER_UNIT_RE.match(value)
    if not match or not match.group(2).strip():
        raise ValueError(
            f"{field} requires a value with an explicit unit, for example "
            '"300 MW".')
    return ParsedQuantity(float(match.group(1)), match.group(2).strip())


def _compact_unit(unit: str) -> str:
    """Normalize case and insignificant spaces in a compound unit."""

    return re.sub(r"\s+", "", str(unit)).upper()


def normalize_energy_unit(unit) -> str:
    """Return a canonical energy-unit label or raise for invalid input."""

    normalized = str(unit).strip().upper()
    if normalized not in _ENERGY_TO_MWH:
        valid = ", ".join(_ENERGY_TO_MWH)
        raise ValueError(
            f"Unsupported energy unit '{unit}'. Use {valid}.")
    return normalized


def canonical_energy_unit(unit) -> str:
    """Return the canonical display spelling of an energy unit."""

    return _ENERGY_LABELS[normalize_energy_unit(unit)]


def normalize_power_unit(unit) -> str:
    """Return a canonical power-unit label or raise for invalid input."""

    normalized = str(unit).strip().upper()
    # Electrical/thermal qualifiers are accepted unit spellings.
    # Field semantics, rather than the unit token, distinguish the quantity.
    for suffix in ("TH", "E"):
        if normalized.endswith(suffix):
            candidate = normalized[:-len(suffix)]
            if candidate in _POWER_TO_MW:
                normalized = candidate
                break
    if normalized not in _POWER_TO_MW:
        valid = ", ".join(_POWER_TO_MW)
        raise ValueError(
            f"Unsupported power unit '{unit}'. Use {valid}.")
    return normalized


def is_energy_unit(unit) -> bool:
    """Return whether a value names a supported energy unit."""

    return str(unit).strip().upper() in _ENERGY_TO_MWH


def is_power_unit(unit) -> bool:
    """Return whether a value names a supported power unit."""

    try:
        normalize_power_unit(unit)
    except ValueError:
        return False
    return True


def energy_to_mwh_factor(unit) -> float:
    """Return the multiplier converting one energy unit to MWh."""

    return _ENERGY_TO_MWH[normalize_energy_unit(unit)]


def energy_from_mwh_factor(unit) -> float:
    """Return the multiplier converting one MWh to an energy unit."""

    return 1.0 / energy_to_mwh_factor(unit)


def energy_conversion_factor(from_unit, to_unit) -> float:
    """Return the multiplier converting between supported energy units."""

    return energy_to_mwh_factor(from_unit) * energy_from_mwh_factor(to_unit)


def convert_energy(value, from_unit, to_unit):
    """Convert scalar or array-like energy values between energy units."""

    return value * energy_conversion_factor(from_unit, to_unit)


def power_to_mw_factor(unit) -> float:
    """Return the multiplier converting one power unit to MW."""

    return _POWER_TO_MW[normalize_power_unit(unit)]


def power_from_mw_factor(unit) -> float:
    """Return the multiplier converting one MW to a power unit."""

    return 1.0 / power_to_mw_factor(unit)


def power_conversion_factor(from_unit, to_unit) -> float:
    """Return the multiplier converting between supported power units."""

    return power_to_mw_factor(from_unit) * power_from_mw_factor(to_unit)


def convert_power(value, from_unit, to_unit):
    """Convert scalar or array-like power values between power units."""

    return value * power_conversion_factor(from_unit, to_unit)


def parse_energy(value, *, field: str, to_unit: str = "MWh") -> float:
    """Parse an explicit energy quantity and convert it to ``to_unit``."""

    parsed = _split_quantity(value, field)
    return float(convert_energy(parsed.value, parsed.unit, to_unit))


def parse_power(value, *, field: str, to_unit: str = "MW") -> float:
    """Parse an explicit power quantity and convert it to ``to_unit``."""

    parsed = _split_quantity(value, field)
    return float(convert_power(parsed.value, parsed.unit, to_unit))


def parse_energy_rate(
        value, *, field: str, target_energy_unit: str,
        target_resolution: str) -> float:
    """Parse energy per time and return energy per projection time step.

    A year is treated as 365 days for this scalar conversion.  Calendar-year
    patterning remains the responsibility of the forecast model.  The explicit
    rate syntax prevents accidental use of an annual value as a daily target.
    """

    parsed = _split_quantity(value, field)
    raw_unit = parsed.unit.strip()
    if "/" not in raw_unit:
        raise ValueError(
            f"{field} requires an energy-per-period value, for example "
            '"500 TWh/year" or "550 GWh/day".')
    energy_unit, period_unit = raw_unit.rsplit("/", 1)
    period_key = period_unit.strip().upper()
    if period_key not in _ENERGY_RATE_PERIOD_HOURS:
        raise ValueError(
            f"Unsupported time denominator '{period_unit}' in {field}. "
            "Use hour, day or year.")
    resolution = str(target_resolution).strip().lower()
    if resolution not in {"daily", "hourly"}:
        raise ValueError(
            f"Unsupported target resolution '{target_resolution}' for {field}.")
    step_hours = 24.0 if resolution == "daily" else 1.0
    numerator = convert_energy(
        parsed.value, energy_unit.strip(), target_energy_unit)
    period_hours = _ENERGY_RATE_PERIOD_HOURS[period_key]
    return float(numerator) * step_hours / period_hours


def parse_duration_hours(value, *, field: str) -> float:
    """Parse an elapsed-time quantity and return hours."""

    parsed = _split_quantity(value, field)
    unit = parsed.unit.strip().upper()
    if unit not in _TIME_TO_HOURS:
        raise ValueError(
            f"Unsupported duration unit '{parsed.unit}' in {field}. "
            "Use h/hour or day.")
    return parsed.value * _TIME_TO_HOURS[unit]


def parse_duration_days(value, *, field: str) -> float:
    """Parse an elapsed-time quantity and return days."""

    return parse_duration_hours(value, field=field) / 24.0


def parse_calendar_months(value, *, field: str) -> float:
    """Parse a calendar duration expressed in months or years."""

    parsed = _split_quantity(value, field)
    unit = parsed.unit.strip().upper()
    if unit not in _CALENDAR_TO_MONTHS:
        raise ValueError(
            f"Unsupported calendar duration unit '{parsed.unit}' in {field}. "
            "Use month or year.")
    return parsed.value * _CALENDAR_TO_MONTHS[unit]


def parse_heavy_metal_mass(value, *, field: str) -> float:
    """Parse heavy-metal mass and return tonnes of heavy metal (tHM)."""

    parsed = _split_quantity(value, field)
    unit = _compact_unit(parsed.unit)
    if unit not in _HEAVY_METAL_TO_THM:
        raise ValueError(
            f"Unsupported heavy-metal mass unit '{parsed.unit}' in {field}. "
            "Use kgHM or tHM.")
    return parsed.value * _HEAVY_METAL_TO_THM[unit]


def parse_burnup(value, *, field: str) -> float:
    """Parse burnup and return GWd/tHM."""

    parsed = _split_quantity(value, field)
    unit = _compact_unit(parsed.unit)
    if unit not in _BURNUP_TO_GWD_THM:
        raise ValueError(
            f"Unsupported burnup unit '{parsed.unit}' in {field}. "
            "Use MWd/tHM, GWd/tHM, MWd/kgHM or GWd/kgHM.")
    return parsed.value * _BURNUP_TO_GWD_THM[unit]


def parse_efpd(value, *, field: str) -> float:
    """Parse fuel residence expressed in equivalent full-power days."""

    parsed = _split_quantity(value, field)
    if _compact_unit(parsed.unit) != "EFPD":
        raise ValueError(
            f"{field} must be expressed in EFPD, for example "
            '"1200 EFPD".')
    return parsed.value


def parse_percent(value, *, field: str) -> float:
    """Parse an explicit percentage and return percentage points."""

    parsed = _split_quantity(value, field)
    if _compact_unit(parsed.unit) not in {"%", "PERCENT", "PCT"}:
        raise ValueError(
            f"{field} must be expressed as a percentage, for example "
            '"2 %".')
    return parsed.value


def parse_fraction_rate_per_hour(value, *, field: str) -> float:
    """Parse a dimensionless fractional rate and return fraction per hour."""

    parsed = _split_quantity(value, field)
    unit = _compact_unit(parsed.unit)
    factors = {
        "/H": 1.0,
        "1/H": 1.0,
        "/HOUR": 1.0,
        "1/HOUR": 1.0,
        "/DAY": 1.0 / 24.0,
        "1/DAY": 1.0 / 24.0,}
    if unit not in factors:
        raise ValueError(
            f"{field} must be a fractional rate such as \"0.25 /h\".")
    return parsed.value * factors[unit]


def parse_count_rate_per_day(value, *, field: str) -> float:
    """Parse an event-count rate and return events per day."""

    parsed = _split_quantity(value, field)
    unit = _compact_unit(parsed.unit)
    factors = {
        "/DAY": 1.0,
        "1/DAY": 1.0,
        "/H": 24.0,
        "1/H": 24.0,
        "/HOUR": 24.0,
        "1/HOUR": 24.0,}
    if unit not in factors:
        raise ValueError(
            f"{field} must be an event rate such as \"2 /day\".")
    return parsed.value * factors[unit]


def parse_emission_factor(value, *, field: str) -> float:
    """Parse a CO2-equivalent mass-per-energy factor to kgCO2e/MWh."""

    parsed = _split_quantity(value, field)
    raw_unit = parsed.unit.strip()
    if "/" not in raw_unit:
        raise ValueError(
            f"{field} requires mass of CO2e per energy, for example "
            '"11 gCO2e/kWh".')
    mass_unit, energy_unit = raw_unit.rsplit("/", 1)
    mass_key = _compact_unit(mass_unit)
    if mass_key not in _CO2_MASS_TO_KG:
        raise ValueError(
            f"Unsupported emissions mass unit '{mass_unit}' in {field}. "
            "Use gCO2e, kgCO2e or tCO2e.")
    kg_value = parsed.value * _CO2_MASS_TO_KG[mass_key]
    energy_denominator = energy_to_mwh_factor(energy_unit.strip())
    return kg_value / energy_denominator


def mw_period_energy_factor(hours, energy_unit) -> float:
    """Convert MW sustained for ``hours`` to the configured energy unit."""

    return float(hours) * energy_from_mwh_factor(energy_unit)


def kwh_per_quantity_to_energy_factor(energy_unit) -> float:
    """Convert kWh per quantity unit to active energy per quantity unit."""

    return energy_conversion_factor("kWh", energy_unit)


def normalize_quantity_unit(unit) -> str:
    """Normalize physical commodity quantity-unit labels."""

    value = str(unit).strip()
    return _QUANTITY_UNIT_ALIASES.get(value.lower(), value)


def commodity_quantity_unit(database, commodity, default="kg") -> str:
    """Return the configured physical quantity unit for one commodity."""

    units = database.get("Commodity_Units", {}) or {}
    return normalize_quantity_unit(units.get(commodity, default))
