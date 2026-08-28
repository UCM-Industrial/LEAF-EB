"""Calculate source, commodity, and net emissions for LEAF-EB outputs."""

import numpy as np
import pandas as pd

from src.utilities.units import energy_to_mwh_factor


def _numeric_column(frame, column):
    """Return a numeric array or zeros when a column is absent."""

    if column not in frame.columns:
        return np.zeros(len(frame), dtype=float)

    values = pd.to_numeric(
        frame[column], errors="coerce")
    return values.fillna(0.0).to_numpy(dtype=float)


def calculate_emissions(frame, generation, user, database):
    """Add explicitly requested electricity and synthetic-CH4 carbon accounting."""

    output = frame
    output_config = user.get("output", {}) or {}
    if not bool(output_config.get("emissions", False)):
        return output

    energy_factor = energy_to_mwh_factor(
        user.get("energy_unit", "MWh"))
    carbon = database.get("Carbon_Accounting", {}).get("CH4", {})
    required = {
        "gross_co2_feed_kg_per_kg",
        "synthesis_co2_release_kg_per_kg",
        "co2_content_kg_per_kg",}
    missing = sorted(required.difference(carbon))
    if missing:
        raise ValueError(
            "Database.yaml is missing CH4 carbon-accounting values: "
            + ", ".join(missing))

    gross_feed_factor = float(
        carbon["gross_co2_feed_kg_per_kg"])
    synthesis_release_factor = float(
        carbon["synthesis_co2_release_kg_per_kg"])
    ch4_co2_factor = float(carbon["co2_content_kg_per_kg"])
    if not np.isclose(
            gross_feed_factor - synthesis_release_factor,
            ch4_co2_factor, rtol=0.0, atol=1.0e-9):
        raise ValueError(
            "CH4 carbon accounting must satisfy gross feed minus synthesis "
            "release equals CO2 content.")

    electricity = np.zeros(len(output), dtype=float)
    for source, config in user.get("sources", {}).items():
        if source not in generation.columns:
            continue
        if "emission_factor_co2" not in config:
            raise ValueError(
                f"Missing emission_factor_co2 for source '{source}' while "
                "output.emissions is enabled.")
        factor = float(config["emission_factor_co2"])
        values = _numeric_column(generation, source)
        emissions = values * energy_factor * factor
        output[f"Emissions_{source} (kgCO2e)"] = emissions
        electricity += emissions

    produced_ch4 = _numeric_column(output, "CH4")
    burned_ch4 = _numeric_column(
        output, "Burned_CH4 (kg)")
    sold_ch4 = _numeric_column(
        output, "Sold_CH4 (kg)")
    expired_ch4 = _numeric_column(
        output, "Expired_CH4 (kg)")
    inventory_ch4 = _numeric_column(
        output, "Inventory_CH4 (kg)")

    gross_capture = produced_ch4 * gross_feed_factor
    synthesis_release = produced_ch4 * synthesis_release_factor
    captured = produced_ch4 * ch4_co2_factor
    released = burned_ch4 * ch4_co2_factor
    sold_carbon = sold_ch4 * ch4_co2_factor
    expired_carbon = expired_ch4 * ch4_co2_factor
    inventory_carbon = inventory_ch4 * ch4_co2_factor
    retained = captured - released - sold_carbon - expired_carbon

    output["Electricity_Emissions (kgCO2e)"] = electricity
    output["Atmospheric_CO2_Gross_Capture (kg)"] = gross_capture
    output["CO2_Released_from_CH4_Synthesis (kg)"] = synthesis_release
    output["Atmospheric_CO2_Captured (kg)"] = captured
    output["CO2_Released_from_CH4 (kg)"] = released
    output["CO2_in_Sold_CH4 (kg)"] = sold_carbon
    output["CO2_in_Expired_CH4 (kg)"] = expired_carbon
    output["CO2_in_CH4_Inventory (kg)"] = inventory_carbon
    output["Retained_Atmospheric_CO2 (kg)"] = retained

    gross = electricity + synthesis_release + released
    net = (
        electricity + synthesis_release + released + sold_carbon
        + expired_carbon - gross_capture)

    output["Gross_Emissions (kgCO2e)"] = gross
    output["Net_Emissions (kgCO2e)"] = net

    return output
