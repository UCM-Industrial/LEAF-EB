"""Resolve configured source and column names without letter-case dependence."""

from __future__ import annotations

from copy import deepcopy
from typing import Iterable

import pandas as pd


_RESERVED_NAMES = ("Date", "Demand", "Total")


def name_key(value: object) -> str:
    """Return the normalized comparison key for one configured name."""

    return str(value).strip().casefold()


def build_name_lookup(
        names: Iterable[object], context: str = "configured names") -> dict:
    """Build a case-insensitive lookup and reject ambiguous names."""

    lookup = {}
    for raw_name in names:
        name = str(raw_name).strip()
        key = name_key(name)
        if not key:
            raise ValueError(f"Empty name found in {context}.")
        previous = lookup.get(key)
        if previous is not None and previous != name:
            raise ValueError(
                f"Ambiguous names in {context}: '{previous}' and "
                f"'{name}' differ only by letter case.")
        lookup[key] = name
    return lookup


def resolve_configured_name(
        value: object, names: Iterable[object], context: str) -> str:
    """Resolve one value against configured names without case sensitivity."""

    lookup = build_name_lookup(names, context)
    key = name_key(value)
    if key not in lookup:
        choices = ", ".join(sorted(lookup.values()))
        raise ValueError(
            f"Unknown name '{value}' in {context}. Available names: "
            f"{choices}.")
    return lookup[key]


def normalize_user_input(user_input: dict) -> dict:
    """Normalize source references while preserving configured source labels."""

    if not isinstance(user_input, dict):
        return user_input

    normalized = deepcopy(user_input)
    sources = normalized.get("sources", {})
    if not isinstance(sources, dict):
        return normalized

    source_names = list(sources)
    build_name_lookup(source_names, "source definitions")
    build_name_lookup(
        [*source_names, *_RESERVED_NAMES],
        "source definitions and reserved names")
    date_name = str(normalized.get("date_column", "Date")).strip()
    if name_key(date_name) == name_key("Date"):
        normalized["date_column"] = "Date"
    allowed_names = [*source_names, "Demand"]

    for source_name, source_data in sources.items():
        if not isinstance(source_data, dict):
            continue
        groups = source_data.get("replaces")
        if groups is not None:
            source_data["replaces"] = _normalize_replacement_groups(
                groups, source_names, source_name)

    _normalize_monte_carlo_sources(normalized, allowed_names)

    return normalized


def normalize_frame_columns(
        frame: pd.DataFrame, expected_names: Iterable[object],
        context: str) -> pd.DataFrame:
    """Rename matching frame columns to configured names case-insensitively."""

    expected = [str(name).strip() for name in expected_names]
    build_name_lookup(expected, f"expected columns for {context}")
    actual_lookup = build_name_lookup(frame.columns, f"columns in {context}")
    rename = {}

    for expected_name in expected:
        actual_name = actual_lookup.get(name_key(expected_name))
        if actual_name is not None and actual_name != expected_name:
            rename[actual_name] = expected_name

    if not rename:
        return frame.copy()
    return frame.rename(columns=rename).copy()


def normalize_input_frame(
        frame: pd.DataFrame, user_input: dict, context: str) -> pd.DataFrame:
    """Normalize Date, Demand, sources and derived source columns in a frame."""

    date_name = str(user_input.get("date_column", "Date")).strip()
    source_names = list(user_input.get("sources", {}))
    expected = [date_name, "Date", "Demand", "Total", *source_names]

    for source_name in source_names:
        expected.extend([
            f"Installed_Capacity_{source_name}",
            f"{source_name}_per",
            f"{source_name}_week_per",
        ])

    expected.extend(["Demand_per", "Demand_week_per"])
    return normalize_frame_columns(frame, expected, context)


def normalize_source_axis(
        frame: pd.DataFrame, source_names: Iterable[object],
        context: str) -> pd.DataFrame:
    """Normalize source labels on both axes of a square data table."""

    canonical = [*source_names, "Demand"]
    lookup = build_name_lookup(canonical, f"source names for {context}")
    result = frame.copy()
    row_labels = _normalized_labels(result.index, lookup)
    column_labels = _normalized_labels(result.columns, lookup)
    result.index = row_labels
    result.columns = column_labels
    _reject_duplicate_labels(result.index, f"rows in {context}")
    _reject_duplicate_labels(result.columns, f"columns in {context}")
    return result


def normalize_source_values(
        values: pd.Series, source_names: Iterable[object],
        context: str) -> pd.Series:
    """Normalize source names stored as values in a table column."""

    canonical = [*source_names, "Demand"]
    lookup = build_name_lookup(canonical, f"source names for {context}")

    def resolve(value: object) -> object:
        """Resolve one source value and leave missing values unchanged."""

        if pd.isna(value):
            return value
        return lookup.get(name_key(value), str(value).strip())

    return values.map(resolve)


def _normalize_monte_carlo_sources(
        config: dict, allowed_names: list
) -> None:
    """Normalize optional simulation.monte_carlo source overrides."""

    monte_carlo = config.get("monte_carlo")
    if not isinstance(monte_carlo, dict):
        return
    values = monte_carlo.get("sources")
    if values is None or not isinstance(values, list):
        return
    normalized = [
        resolve_configured_name(
            value, allowed_names, "simulation.monte_carlo.sources")
        for value in values]
    monte_carlo["sources"] = normalized


def _normalize_replacement_groups(
        groups: object, source_names: list[str],
        custom_name: str) -> object:
    """Normalize source labels inside custom replacement groups."""

    if not isinstance(groups, list):
        return groups

    normalized = []
    for group in groups:
        if isinstance(group, list):
            names = []
            for value in group:
                names.append(resolve_configured_name(
                    value, source_names,
                    f"replaces for '{custom_name}'"))
            normalized.append(names)
            continue
        if isinstance(group, dict):
            copied = dict(group)
            values = copied.get("sources")
            if isinstance(values, list):
                names = []
                for value in values:
                    names.append(resolve_configured_name(
                        value, source_names,
                        f"replaces for '{custom_name}'"))
                copied["sources"] = names
            normalized.append(copied)
            continue
        normalized.append(group)
    return normalized


def _normalized_labels(labels: Iterable[object], lookup: dict) -> list:
    """Return labels with configured source names restored where possible."""

    normalized = []
    for value in labels:
        key = name_key(value)
        normalized.append(lookup.get(key, value))
    return normalized


def _reject_duplicate_labels(labels: Iterable[object], context: str) -> None:
    """Reject labels that become duplicated after case normalization."""

    seen = set()
    duplicates = set()
    for value in labels:
        if value in seen:
            duplicates.add(str(value))
        seen.add(value)
    if duplicates:
        values = ", ".join(sorted(duplicates))
        raise ValueError(
            f"Duplicate labels after case normalization in {context}: "
            f"{values}.")
