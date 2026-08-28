#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Convert LEAF-EB nuclear results to the annual ANICCA input format.

The ANICCA interface contains one worksheet per nuclear fleet with:

    Year | Installed_Power | Burnup | Load_Factor |
    Reactors_In | Reactors_Out

Two nuclear fuel-management formulations are treated explicitly:

1. Fixed burnup
   ``fuel_cycle.target_burnup`` is passed directly to ANICCA for every
   active year.

2. Fixed refueling calendar
   LEAF calculates physical batch discharge burnup event by event, but
   ANICCA requires one burnup value for every annual time step. The
   converter therefore calculates an annual-equivalent burnup that keeps
   the fixed-calendar reload rate consistent with the annual LEAF load
   factor:

       BU_eq,y = LF_y * P_th * N_batches * T_calendar
                 / (M_core * 1000)

   where T_calendar is the start-to-start refueling interval in days:
   the LEAF operating cycle, rounded exactly as in LEAF, plus the outage.

This annual-equivalent burnup is an interface quantity for ANICCA. It is
not a replacement for the physical discharge burnup stored in LEAF.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


DAYS_PER_YEAR = 365.2425

STATISTIC_COLUMNS = {
    "deterministic": (
        "Deterministic_Load_Factor",
        "Deterministic_Generation",
    ),
    "mean": (
        "MC_Load_Factor_Mean",
        "MC_Mean",
    ),
    "p025": (
        "MC_Load_Factor_P025",
        "MC_P025",
    ),
    "p50": (
        "MC_Load_Factor_P50",
        "MC_P50",
    ),
    "p975": (
        "MC_Load_Factor_P975",
        "MC_P975",
    ),
}

ANICCA_COLUMNS = [
    "Year",
    "Installed_Power",
    "Burnup",
    "Load_Factor",
    "Reactors_In",
    "Reactors_Out",
]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Convert LEAF-EB nuclear output to ANICCA_Input.xlsx."
        )
    )
    parser.add_argument(
        "leaf_output",
        type=Path,
        help="LEAF scenario Output directory.",
    )
    parser.add_argument(
        "leaf_input",
        type=Path,
        help="YAML input used for the LEAF scenario.",
    )
    parser.add_argument(
        "--statistic",
        choices=tuple(STATISTIC_COLUMNS),
        default="deterministic",
        help=(
            "Annual LEAF series passed to ANICCA. "
            "Default: deterministic."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Output workbook. Default: "
            "<leaf_output>/ANICCA_Input_from_LEAF.xlsx."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build and print the tables without writing Excel.",
    )
    return parser.parse_args()


def project_root_from_script() -> Path:
    """Return the LEAF project root containing src/, Inputs/, and data/."""

    return Path(__file__).resolve().parents[1]


def load_leaf_config(input_path: Path) -> dict:
    """Load the normalized LEAF input, including technology templates."""

    root = project_root_from_script()
    root_text = str(root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)

    from src.utilities.configuration import load_config_file

    return load_config_file(input_path)


def nuclear_sources(config: dict) -> dict[str, dict]:
    """Return configured nuclear fleets with fuel/refueling information."""

    fleets = {}
    for source, source_config in config.get("sources", {}).items():
        if not isinstance(source_config, dict):
            continue
        refueling = source_config.get("refueling", {}) or {}
        fuel_cycle = source_config.get("fuel_cycle", {}) or {}
        if refueling or fuel_cycle:
            fleets[source] = source_config
    return fleets


def read_nuclear_results(output_dir: Path) -> pd.DataFrame:
    """Read annual nuclear generation/load-factor statistics from LEAF."""

    results_path = output_dir / "Results.xlsx"
    if results_path.is_file():
        return pd.read_excel(
            results_path,
            sheet_name="Nuclear",
            engine="openpyxl",
        )

    generation_path = output_dir / "Nuclear_Generation.xlsx"
    if generation_path.is_file():
        return pd.read_excel(
            generation_path,
            sheet_name="Annual",
            engine="openpyxl",
        )

    raise FileNotFoundError(
        "Could not find Results.xlsx or Nuclear_Generation.xlsx in "
        f"{output_dir}."
    )


def read_anicca_skeleton(output_dir: Path) -> dict[str, pd.DataFrame]:
    """Read LEAF's structural ANICCA export used for fleet changes."""

    path = output_dir / "ANICCA_Input.xlsx"
    if not path.is_file():
        raise FileNotFoundError(
            "ANICCA_Input.xlsx was not found in the LEAF Output folder. "
            "Run LEAF with analysis/detailed nuclear output first."
        )

    with pd.ExcelFile(path, engine="openpyxl") as workbook:
        return {
            sheet: pd.read_excel(workbook, sheet_name=sheet)
            for sheet in workbook.sheet_names
        }


def safe_sheet_name(name: str) -> str:
    """Return the sheet-name transformation used by LEAF."""

    invalid = set('[]:*?/\\')
    cleaned = "".join(
        "_" if char in invalid else char
        for char in str(name)
    )
    return cleaned[:31] or "Fleet"


def calendar_cycle_days(source_config: dict) -> int:
    """Return operating days using the same rounding rule as LEAF."""

    refueling = source_config.get("refueling", {}) or {}
    months = refueling.get("operating_cycle")
    if months is None:
        raise ValueError(
            "Fixed-calendar conversion requires refueling.operating_cycle."
        )

    rounded_months = int(round(float(months)))
    if abs(float(months) - rounded_months) < 1e-9:
        # Integer-month LEAF cycles use calendar months. For the annual
        # equivalent interface, use the same mean calendar duration.
        return int(round(DAYS_PER_YEAR * rounded_months / 12.0))

    return int(round(DAYS_PER_YEAR * float(months) / 12.0))


def calendar_burnup_coefficient(source_config: dict) -> float:
    """Return GWd/tHM per unit annual load factor for fixed calendar."""

    refueling = source_config.get("refueling", {}) or {}
    fuel_cycle = source_config.get("fuel_cycle", {}) or {}

    thermal_power = float(fuel_cycle.get("thermal_power", 0.0))
    core_mass = float(fuel_cycle.get("core_fuel_mass", 0.0))
    fuel_batches = int(refueling.get("fuel_batches", 0) or 0)
    outage_days = int(refueling.get("outage_duration", 0) or 0)

    if thermal_power <= 0.0:
        raise ValueError(
            "Fixed-calendar ANICCA conversion requires positive "
            "fuel_cycle.thermal_power."
        )
    if core_mass <= 0.0:
        raise ValueError(
            "Fixed-calendar ANICCA conversion requires positive "
            "fuel_cycle.core_fuel_mass."
        )
    if fuel_batches <= 0:
        raise ValueError(
            "Fixed-calendar ANICCA conversion requires positive "
            "refueling.fuel_batches."
        )

    operating_days = calendar_cycle_days(source_config)
    interval_days = operating_days + outage_days

    return (
        thermal_power
        * fuel_batches
        * interval_days
        / (core_mass * 1000.0)
    )


def burnup_series(
        load_factor: pd.Series,
        installed_power: pd.Series,
        source_config: dict) -> tuple[pd.Series, str]:
    """Return the ANICCA burnup history and its interpretation."""

    refueling = source_config.get("refueling", {}) or {}
    fuel_cycle = source_config.get("fuel_cycle", {}) or {}
    target = fuel_cycle.get("target_burnup")
    operating_cycle = refueling.get("operating_cycle")

    active = pd.to_numeric(
        installed_power,
        errors="coerce",
    ).fillna(0.0) > 0.0

    burnup = pd.Series(float("nan"), index=load_factor.index)

    if target is not None and operating_cycle is None:
        burnup.loc[active] = float(target)
        return burnup, "fixed burnup"

    if operating_cycle is not None and target is None:
        coefficient = calendar_burnup_coefficient(source_config)
        lf = pd.to_numeric(load_factor, errors="coerce")
        valid = active & lf.notna() & lf.gt(0.0)
        burnup.loc[valid] = lf.loc[valid] * coefficient
        return burnup, (
            "fixed refueling calendar; annual-equivalent burnup"
        )

    if target is not None and operating_cycle is not None:
        raise ValueError(
            "A nuclear fleet cannot define both target_burnup and "
            "operating_cycle for this converter."
        )

    raise ValueError(
        "Nuclear fleet requires either fuel_cycle.target_burnup or "
        "refueling.operating_cycle."
    )


def build_anicca_tables(
        output_dir: Path,
        config: dict,
        statistic: str) -> dict[str, pd.DataFrame]:
    """Build one ANICCA table per nuclear fleet."""

    annual = read_nuclear_results(output_dir)
    skeletons = read_anicca_skeleton(output_dir)
    fleets = nuclear_sources(config)
    load_factor_column, generation_column = STATISTIC_COLUMNS[statistic]

    missing = [
        column
        for column in ("Year", "Source", load_factor_column)
        if column not in annual.columns
    ]
    if missing:
        raise ValueError(
            "Nuclear annual results are missing columns: "
            + ", ".join(missing)
        )

    tables = {}
    for source, source_config in fleets.items():
        sheet_name = safe_sheet_name(source)
        if sheet_name not in skeletons:
            raise ValueError(
                f"ANICCA skeleton has no worksheet for '{source}'."
            )

        base = skeletons[sheet_name].copy()
        required = {
            "Year",
            "Installed_Power",
            "Reactors_In",
            "Reactors_Out",
        }
        missing_base = sorted(required.difference(base.columns))
        if missing_base:
            raise ValueError(
                f"ANICCA skeleton for '{source}' is missing: "
                + ", ".join(missing_base)
            )

        source_annual = annual.loc[
            annual["Source"].astype(str).eq(source),
            ["Year", load_factor_column, generation_column],
        ].copy()
        source_annual["Year"] = source_annual["Year"].astype(int)
        source_annual = source_annual.rename(
            columns={
                load_factor_column: "Load_Factor_Selected",
                generation_column: "Generation_Selected",
            }
        )

        base["Year"] = base["Year"].astype(int)
        merged = base.merge(
            source_annual,
            on="Year",
            how="left",
            validate="one_to_one",
        )

        merged["Load_Factor"] = pd.to_numeric(
            merged["Load_Factor_Selected"],
            errors="coerce",
        ).fillna(0.0)

        burnup, interpretation = burnup_series(
            merged["Load_Factor"],
            merged["Installed_Power"],
            source_config,
        )
        merged["Burnup"] = burnup

        table = merged[ANICCA_COLUMNS].copy()
        table.attrs["interpretation"] = interpretation
        table.attrs["statistic"] = statistic
        tables[sheet_name] = table

    return tables


def write_workbook(
        tables: dict[str, pd.DataFrame],
        output_path: Path) -> None:
    """Write the exact six-column ANICCA workbook."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        for sheet_name, table in tables.items():
            table.to_excel(
                writer,
                sheet_name=sheet_name,
                index=False,
            )


def print_summary(
        tables: dict[str, pd.DataFrame],
        statistic: str) -> None:
    """Print compact conversion diagnostics."""

    print(f"Statistic: {statistic}")
    for sheet_name, table in tables.items():
        active = table[table["Installed_Power"] > 0.0]
        values = pd.to_numeric(
            active["Burnup"],
            errors="coerce",
        ).dropna()
        print(f"\n{sheet_name}")
        print(f"  Mapping: {table.attrs.get('interpretation')}")
        print(f"  Active years: {len(active)}")
        if values.empty:
            print("  Burnup: no active values")
        else:
            print(
                "  Burnup range: "
                f"{values.min():.4f} to {values.max():.4f} GWd/tHM"
            )
        print(
            table.loc[
                table["Installed_Power"] > 0.0,
                ANICCA_COLUMNS,
            ].head(5).to_string(index=False)
        )


def main() -> int:
    """Run the LEAF-to-ANICCA conversion."""

    args = parse_args()
    output_dir = args.leaf_output.resolve()
    input_path = args.leaf_input.resolve()

    if not output_dir.is_dir():
        raise NotADirectoryError(
            f"LEAF Output directory not found: {output_dir}"
        )
    if not input_path.is_file():
        raise FileNotFoundError(
            f"LEAF input YAML not found: {input_path}"
        )

    config = load_leaf_config(input_path)
    tables = build_anicca_tables(
        output_dir,
        config,
        args.statistic,
    )
    print_summary(tables, args.statistic)

    if args.dry_run:
        print("\nDry run: no workbook written.")
        return 0

    output_path = args.output
    if output_path is None:
        output_path = output_dir / "ANICCA_Input_from_LEAF.xlsx"
    output_path = output_path.resolve()

    write_workbook(tables, output_path)
    print(f"\nWritten: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
