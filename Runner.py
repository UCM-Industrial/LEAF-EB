# -*- coding: utf-8 -*-
"""Run the LEAF-EB forecasting and electricity-balance pipeline."""

from __future__ import annotations

import os
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from src.utilities.configuration import load_config_file
from src.utilities.input_validation import validate_input


ROOT = Path(__file__).resolve().parent
INPUTS_DIR = ROOT / "Inputs"
BANNER = """
============================================================
LEAF-EB — Long-term Energy Analysis Framework
Electricity Balance
============================================================
"""
HELP_TEXT = """Usage:
  python Runner.py
  python Runner.py --list
  python Runner.py <input name or partial name>

Inputs are always read from Inputs/. The .yml extension is optional.
"""


def discover_inputs() -> list[Path]:
    """Return user scenario YAML files available under Inputs/."""
    if not INPUTS_DIR.is_dir():
        return []
    available = [
        path for path in INPUTS_DIR.glob("*.yml")
        if path.is_file() and not path.name.startswith(".")]
    return sorted(
        available, key=lambda path: path.stem.casefold())


def _normalized_query(value: str) -> str:
    """Return a case-insensitive scenario query without .yml."""
    name = Path(value.strip()).name
    if name.casefold().endswith(".yml"):
        name = name[:-4]
    return name.casefold()


def find_input_matches(
        query: str, inputs: list[Path] | None = None) -> list[Path]:
    """Find exact or partial input-name matches inside Inputs/."""
    available = discover_inputs() if inputs is None else inputs
    normalized = _normalized_query(query)
    if not normalized:
        return []

    exact = [
        path for path in available
        if path.stem.casefold() == normalized]
    if exact:
        return exact
    return [
        path for path in available
        if normalized in path.stem.casefold()]


def print_input_list(
        inputs: list[Path], heading: str = "Available LEAF-EB inputs:") -> None:
    """Print a numbered scenario list without exposing full paths."""
    print(heading)
    if not inputs:
        print("  No .yml inputs were found in Inputs/.")
        return
    for number, path in enumerate(inputs, 1):
        print(f"  {number}. {path.stem}")


def choose_input(
        inputs: list[Path],
        prompt: Callable[[str], str] = input) -> Path | None:
    """Ask the user to choose one input from a numbered list."""
    if not inputs:
        return None
    while True:
        try:
            answer = prompt(
                f"Select scenario [1-{len(inputs)}] or q to cancel: ")
        except (EOFError, KeyboardInterrupt):
            print("\nSelection cancelled.")
            return None
        value = answer.strip()
        if value.casefold() in {"q", "quit", "exit"}:
            return None
        try:
            number = int(value)
        except ValueError:
            print("Enter a scenario number or q to cancel.")
            continue
        if 1 <= number <= len(inputs):
            return inputs[number - 1]
        print(f"Enter a number between 1 and {len(inputs)}.")


def resolve_input(
        query: str | None,
        prompt: Callable[[str], str] = input) -> Path | None:
    """Resolve a scenario interactively or from an exact/partial name."""
    available = discover_inputs()
    if not available:
        print("ERROR: no .yml input files were found in Inputs/.")
        return None

    if query is None:
        print_input_list(available)
        return choose_input(available, prompt)

    matches = find_input_matches(query, available)
    if not matches:
        print(f"ERROR: no input in Inputs/ matches {query!r}.")
        print("Use 'python Runner.py --list' to see available inputs.")
        return None
    if len(matches) == 1:
        matched = matches[0]
        if matched.stem.casefold() != _normalized_query(query):
            print(f"Matched input: {matched.stem}")
        return matched

    print_input_list(
        matches,
        heading=f"Multiple inputs match {query!r}:")
    return choose_input(matches, prompt)


def load_input(input_name: str) -> tuple[dict[str, Any], Path]:
    """Load one current-schema YAML input from the Inputs directory."""
    input_stem = Path(input_name).stem
    input_path = INPUTS_DIR / f"{input_stem}.yml"
    if not input_path.is_file():
        raise FileNotFoundError(
            f"Input file not found: {input_path}")
    user_input = load_config_file(input_path)
    return user_input, input_path


def _run_module_step(module_name: str, input_stem: str) -> None:
    """Run one file-producing pipeline phase in an isolated process."""

    command = [
        sys.executable, "-u", "-m", module_name, input_stem]
    environment = os.environ.copy()
    environment["LEAF_PIPELINE_CHILD"] = "1"
    subprocess.run(
        command, cwd=ROOT, check=True, env=environment)


def run_pipeline(input_name: str) -> int:
    """Run forecasting and the configured electricity-balance model."""

    input_stem = Path(input_name).stem
    try:
        user_input, input_path = load_input(input_stem)
    except (FileNotFoundError, TypeError, ValueError) as exc:
        print(f"ERROR loading input: {exc}")
        return 1

    print(f"\nSelected input: {input_path.name}")
    try:
        validate_input(user_input, ROOT)
    except (FileNotFoundError, TypeError, ValueError) as exc:
        print(f"ERROR in input: {exc}")
        return 1

    start_time = time.time()
    print("Preparing temporal patterns...")
    try:
        _run_module_step("src.forecasting.Patterns", input_stem)
    except subprocess.CalledProcessError as exc:
        print(
            "ERROR generating temporal patterns: "
            f"process exited with code {exc.returncode}.")
        return 1

    print("Generating scenario series...")
    try:
        _run_module_step("src.forecasting.Predictor", input_stem)
    except subprocess.CalledProcessError as exc:
        print(
            "ERROR generating scenario series: "
            f"process exited with code {exc.returncode}.")
        return 1

    commodities_input = user_input.get("commodities_input", {})
    run_system = bool(commodities_input) and bool(
        commodities_input.get("run_commodities", True))
    variability_required = bool(user_input.get("variability_enabled", False))

    if run_system:
        monte_carlo = user_input.get("simulation", {}).get(
            "monte_carlo", {}) or {}
        simulations = int(monte_carlo.get("simulations", 0) or 0)
        if simulations > 0 and not variability_required:
            print(
                "WARNING: Monte Carlo simulations are requested while "
                "variability is disabled; temporal perturbations cannot "
                "be generated.")

        print("Running electricity balance...")
        try:
            from src.core.simulation import LEAFSimulator

            LEAFSimulator(str(input_path)).run()
        except Exception as exc:
            print(f"ERROR in electricity-balance simulation: {exc}")
            return 1
    else:
        print("Electricity-balance simulation is disabled in the input.")

    total_time = time.time() - start_time
    print(f"Completed in {total_time:.2f} seconds.")
    return 0

def main() -> int:
    """Run the command-line entry point."""
    print(BANNER)
    arguments = sys.argv[1:]
    if arguments and arguments[0] in {"-h", "--help"}:
        print(HELP_TEXT)
        return 0
    if arguments and arguments[0] == "--list":
        print_input_list(discover_inputs())
        return 0
    if len(arguments) > 1:
        print(HELP_TEXT)
        return 2

    query = arguments[0] if arguments else None
    selected = resolve_input(query)
    if selected is None:
        return 2
    return run_pipeline(selected.stem)


if __name__ == "__main__":
    sys.exit(main())
