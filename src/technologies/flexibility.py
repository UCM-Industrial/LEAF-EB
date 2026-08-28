"""Model interconnections, batteries, commodities, and flexible dispatch."""

from collections import deque

import numpy as np
import pandas as pd

from src.utilities.units import (
    energy_from_mwh_factor, normalize_energy_unit)


User = {}


def configure_flexibility(user_dict):
    """Set the active flexibility configuration."""

    global User
    User = user_dict


def _infer_period_hours(dates: pd.Series) -> pd.Series:
    """Estimate the number of hours represented by each forecast row."""
    dt = pd.to_datetime(dates).reset_index(drop=True)

    if len(dt) == 0:
        return pd.Series(dtype=float)

    if len(dt) == 1:
        return pd.Series([24.0], index=dates.index)

    deltas_h = dt.diff().dt.total_seconds().div(3600.0)
    next_deltas_h = dt.shift(-1).sub(dt).dt.total_seconds().div(3600.0)

    # Use the distance to the next row; reuse the previous distance at end.
    hours = next_deltas_h.copy()
    hours.iloc[-1] = deltas_h.iloc[-1]

    # Replace invalid intervals with the median valid interval.
    fallback = float(hours.replace([np.inf, -np.inf], np.nan).dropna().median())
    if not np.isfinite(fallback) or fallback <= 0:
        fallback = 24.0

    hours = hours.replace([np.inf, -np.inf], np.nan).fillna(fallback)
    hours = hours.clip(lower=1e-9)

    return pd.Series(hours.to_numpy(dtype=np.float32), index=dates.index)


def _as_float_fraction(value, name="Interconnections") -> float:
    """
    Converts an interconnection fraction to float and validates it.
    Values are expected as fractions, e.g. 0.15 for 15%.
    """
    try:
        out = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"Invalid {name} value: {value}") from exc

    if not np.isfinite(out):
        raise ValueError(f"Invalid {name} value: {value}")
    if out < 0.0:
        raise ValueError(f"{name} cannot be negative: {value}")

    return out


def _parse_dated_events(events, name, value_parser):
    """Parse and de-duplicate dated scalar values in chronological order."""

    if not isinstance(events, dict) or not events:
        return []
    parsed = {}
    for key, raw_value in events.items():
        date = pd.to_datetime(key, errors="coerce")
        if pd.isna(date):
            raise ValueError(f"Invalid {name} date: {key}")
        parsed[pd.Timestamp(date)] = value_parser(raw_value)
    return sorted(parsed.items(), key=lambda item: item[0])


def _parse_interconnection_events(events) -> list[tuple[pd.Timestamp, float]]:
    """
    Parses a date -> fraction mapping used by Interconnections.

    Accepted YAML shape:
      values:
        2025-01-01: 0.10
        2030-01-01: 0.15
        2040-01-01: 0.20

    YAML may parse date keys as strings, dates or timestamps; all are accepted.
    """
    return _parse_dated_events(
        events, "Interconnections", _as_float_fraction)


def _piecewise_linear_values(
        dt: pd.Series,
        events_sorted: list[tuple[pd.Timestamp, float]]) -> np.ndarray:
    """
    Piecewise-linear interpolation for any number of dated values.
    Before the first date, the first value is held constant.
    After the last date, the last value is held constant.
    """
    if len(events_sorted) == 0:
        return np.zeros(len(dt), dtype=float)
    if len(events_sorted) == 1:
        return np.full(len(dt), events_sorted[0][1], dtype=float)

    x = np.array([d.value for d, _ in events_sorted], dtype="float64")
    y = np.array([v for _, v in events_sorted], dtype="float64")
    xi = dt.astype("int64").to_numpy(dtype="float64")

    return np.interp(xi, x, y, left=y[0], right=y[-1])


def _step_values(
        dt: pd.Series,
        events_sorted: list[tuple[pd.Timestamp, float]],
        initial=None) -> np.ndarray:
    """
    Stepwise schedule for any number of dated values.
    Before the first date, it uses `initial` if provided; otherwise the first
    value.
    """
    if len(events_sorted) == 0:
        value = _as_float_fraction(initial or 0.0)
        return np.full(len(dt), value, dtype=float)

    first_value = events_sorted[0][1]
    if initial is None:
        initial_value = first_value
    else:
        initial_value = _as_float_fraction(initial)

    values = np.full(len(dt), initial_value, dtype=float)
    for event_date, event_value in events_sorted:
        values[dt >= event_date] = event_value

    return values



def _safe_column_token(value) -> str:
    """Converts an interconnection name into a safe column suffix."""
    token = (
        str(value)
        .strip()
        .replace(' ', '_')
        .replace('-', '_')
        .replace('/', '_'))
    token = "".join(ch if (ch.isalnum() or ch == "_") else "_" for ch in token)
    token = "_".join(part for part in token.split("_") if part)
    return token or "Interconnection"


def _has_directional_interconnection_config(cfg) -> bool:
    """Returns True when a config has explicit import/export branches."""
    return (
        isinstance(cfg, dict) and
        any((k in cfg for k in ('import', 'export'))))


def _single_interconnection_value_series(
        dates: pd.Series,
        cfg,
        name="Interconnections") -> pd.Series:
    """
    Builds one time series for interconnection values.

    The numerical meaning depends on the parent mode:
      - mode omitted or mode: MW       -> values are MW
      - mode: fraction                 -> values are fractions, e.g. 0.15

    Accepted models are ``constant``, ``step``, and ``linear``.
    """
    return _single_interconnection_schedule_series(dates, cfg, name=name).clip(
        lower=0.0)


def _single_interconnection_schedule_series(
        dates: pd.Series,
        cfg,
        name="Interconnections") -> pd.Series:
    """
    Build a numerical schedule for one interconnection.

    The parent function determines whether values represent absolute power or
    a fraction of installed capacity.
    """
    dt = pd.to_datetime(dates).reset_index(drop=True)

    if len(dt) == 0:
        return pd.Series(dtype=float, index=dates.index)

    # Simple scalar value.
    if not isinstance(cfg, dict):
        value = _as_float_fraction(cfg or 0.0, name=name)
        return pd.Series(
            np.full(len(dt), value, dtype=float),
            index=dates.index)

    model = str(cfg.get("model", "constant")).strip().lower()

    if model == "constant":
        value = _as_float_fraction(
            cfg.get('value', 0.0),
            name=name)
        return pd.Series(
            np.full(len(dt), value, dtype=float),
            index=dates.index)

    if model == "step":
        events = cfg.get("values", {})
        events_sorted = _parse_interconnection_events(events)
        values = _step_values(
            dt,
            events_sorted,
            initial=cfg.get('initial', None))
        return pd.Series(values, index=dates.index)

    if model == "linear":
        events = cfg.get("values", None)

        # Preferred format: values with any number of dated points.
        if isinstance(events, dict) and len(events) > 0:
            events_sorted = _parse_interconnection_events(events)
            values = _piecewise_linear_values(dt, events_sorted)
            return pd.Series(values, index=dates.index)

        # Two-point linear schedule.
        start_date = pd.to_datetime(cfg.get("start_date", dt.iloc[0]))
        end_date = pd.to_datetime(cfg.get("end_date", dt.iloc[-1]))
        initial = _as_float_fraction(
            cfg.get('initial', 0.0),
            name=name)
        final = _as_float_fraction(
            cfg.get('final', 0.15),
            name=name)

        total_seconds = max((end_date - start_date).total_seconds(), 1.0)
        elapsed = (
            (dt - start_date).dt
            .total_seconds()
            .to_numpy(dtype=np.float32))
        alpha = np.clip(elapsed / total_seconds, 0.0, 1.0)
        values = initial + alpha * (final - initial)
        return pd.Series(values, index=dates.index)

    raise ValueError(
        f"Invalid Interconnections model for '{name}'. "
        "Use a numeric value, 'constant', 'step', or 'linear'.")


def _interconnection_directional_items(cfg):
    """
    Normalizes Interconnections into named directional items.

    Named directional format:
      Interconnections:
        EU:
          # mode: fraction   # optional; default is MW
          import: {...}
          export: {...}

    Aggregate directional format is also accepted:
      Interconnections:
        mode: fraction
        import: {...}
        export: {...}
    """
    if _has_directional_interconnection_config(cfg):
        return [("Total", cfg)]

    if isinstance(cfg, dict):
        items = []
        for name, item_cfg in cfg.items():
            if _has_directional_interconnection_config(item_cfg):
                items.append((str(name), item_cfg))
            else:
                return None
        return items

    return None


def _mode_is_fraction(mode_value) -> bool:
    """Returns True when an interconnection mode means fraction of installed
    capacity."""
    mode = str(mode_value or "MW").strip().lower()
    return mode == "fraction"


def _direction_cfg(item_cfg: dict, direction: str):
    """Returns import/export schedule, falling back to the other direction if
    missing."""
    if direction == "import":
        cfg = item_cfg.get("import")
        other = item_cfg.get("export")
    else:
        cfg = item_cfg.get("export")
        other = item_cfg.get("import")

    if cfg is None:
        cfg = other
    if cfg is None:
        cfg = 0.0
    return cfg


def _interconnection_availability(cfg, months, direction):
    """Return clipped scalar or monthly availability for one direction."""

    availability = (
        cfg.get("availability", 1.0)
        if isinstance(cfg, dict) else 1.0)
    if not isinstance(availability, dict):
        return np.full(len(months), float(availability), dtype=float).clip(
            0.0, 1.0)

    missing = [
        month for month in range(1, 13)
        if month not in availability and str(month) not in availability]
    if missing:
        raise ValueError(
            f"Missing {direction} Availability values for months: "
            f"{missing}")
    values = [
        float(availability.get(month, availability.get(str(month))))
        for month in months]
    return np.clip(np.asarray(values, dtype=np.float32), 0.0, 1.0)


def _interconnection_capacity_fraction(
        values, installed_capacity, fraction_mode, has_capacity):
    """Convert one directional schedule to MW capacity and fraction."""

    values = values.clip(lower=0.0)
    if fraction_mode:
        if not has_capacity:
            raise ValueError(
                "Forecast.xlsx must include 'Installed_Capacity_Total' "
                "to apply fraction-based interconnection limits.")
        return values * installed_capacity, values

    capacity = values
    denominator = installed_capacity.replace(0.0, np.nan)
    fraction = (
        capacity / denominator
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0))
    return capacity, fraction


def _interconnection_limit_series(perturbed_df_energies_data: pd.DataFrame):
    """
    Calculates import and export interconnection limits.

    Preferred baseline format, with absolute MW values:
      Interconnections:
        EU:
          import:
            model: linear
            values:
              2025-01-01: 5000
              2030-01-01: 8500
          export:
            model: linear
            values:
              2025-01-01: 5678
              2030-01-01: 9200

    Optional fraction mode for sensitivity cases:
      Interconnections:
        EU_15:
          mode: fraction
          import:
            model: linear
            values:
              2025-01-01: 0.036
              2030-01-01: 0.150
          export:
            model: linear
            values:
              2025-01-01: 0.036
              2030-01-01: 0.150

    Returns:
      import_limit, export_limit,
      import_power_capacity, export_power_capacity,
      import_fraction, export_fraction,
      import_limit_df, export_limit_df,
      import_capacity_df, export_capacity_df,
      import_fraction_df, export_fraction_df
    """
    dates = pd.to_datetime(perturbed_df_energies_data["Date"])
    hours = _infer_period_hours(dates).reset_index(drop=True)
    unit_factor = energy_from_mwh_factor(
        User.get("energy_unit", "MWh"))

    if "Installed_Capacity_Total" in perturbed_df_energies_data.columns:
        installed_capacity = pd.to_numeric(
            perturbed_df_energies_data["Installed_Capacity_Total"],
            errors="coerce"
        ).fillna(0.0).clip(lower=0.0).reset_index(drop=True)
    else:
        installed_capacity = pd.Series(
            np.nan,
            index=range(len(dates)),
            dtype=float)

    cfg = User.get("Interconnections")
    capacity_column = "Installed_Capacity_Total"
    if not cfg:
        zero = pd.Series(0.0, index=range(len(dates)), dtype=float)
        zero_frame = pd.DataFrame({"Total": zero})
        return (
            zero, zero, zero, zero, zero, zero,
            zero_frame, zero_frame, zero_frame, zero_frame,
            zero_frame, zero_frame,)

    directional_items = _interconnection_directional_items(cfg)
    if not directional_items:
        raise ValueError(
            "Interconnections must use explicit import/export branches.")

    import_capacity_dict = {}
    export_capacity_dict = {}
    import_fraction_dict = {}
    export_fraction_dict = {}
    import_availability_dict = {}
    export_availability_dict = {}

    for name, item_cfg in directional_items:
        if not isinstance(item_cfg, dict):
            raise ValueError(f"Interconnection '{name}' must be a dictionary.")

        mode = item_cfg.get("mode", "MW")
        is_fraction = _mode_is_fraction(mode)

        imp_cfg = _direction_cfg(item_cfg, "import")
        exp_cfg = _direction_cfg(item_cfg, "export")

        imp_values = _single_interconnection_value_series(
            dates, imp_cfg, name=f"{name} import"
        ).reset_index(drop=True)
        exp_values = _single_interconnection_value_series(
            dates, exp_cfg, name=f"{name} export"
        ).reset_index(drop=True)

        month_index = dates.reset_index(drop=True).dt.month
        imp_availability = _interconnection_availability(
            imp_cfg, month_index, "import")
        exp_availability = _interconnection_availability(
            exp_cfg, month_index, "export")

        has_capacity = capacity_column in perturbed_df_energies_data.columns
        imp_capacity, imp_fraction = _interconnection_capacity_fraction(
            imp_values, installed_capacity, is_fraction, has_capacity)
        exp_capacity, exp_fraction = _interconnection_capacity_fraction(
            exp_values, installed_capacity, is_fraction, has_capacity)

        import_capacity_dict[str(name)] = imp_capacity.to_numpy(
            dtype=np.float32)
        export_capacity_dict[str(name)] = exp_capacity.to_numpy(
            dtype=np.float32)
        import_fraction_dict[str(name)] = imp_fraction.to_numpy(
            dtype=np.float32)
        export_fraction_dict[str(name)] = exp_fraction.to_numpy(
            dtype=np.float32)
        import_availability_dict[str(name)] = imp_availability
        export_availability_dict[str(name)] = exp_availability

    import_capacity_df = pd.DataFrame(import_capacity_dict)
    export_capacity_df = pd.DataFrame(export_capacity_dict)
    import_fraction_df = pd.DataFrame(import_fraction_dict)
    export_fraction_df = pd.DataFrame(export_fraction_dict)

    import_availability_df = pd.DataFrame(import_availability_dict)
    export_availability_df = pd.DataFrame(export_availability_dict)

    import_limit_df = (
        import_capacity_df
        .mul(import_availability_df)
        .mul(hours * unit_factor, axis=0))
    export_limit_df = (
        export_capacity_df
        .mul(export_availability_df)
        .mul(hours * unit_factor, axis=0))

    import_capacity_total = import_capacity_df.sum(axis=1)
    export_capacity_total = export_capacity_df.sum(axis=1)
    import_limit_total = import_limit_df.sum(axis=1)
    export_limit_total = export_limit_df.sum(axis=1)
    import_fraction_total = import_fraction_df.sum(axis=1)
    export_fraction_total = export_fraction_df.sum(axis=1)

    return (
        import_limit_total, export_limit_total,
        import_capacity_total, export_capacity_total,
        import_fraction_total, export_fraction_total,
        import_limit_df, export_limit_df,
        import_capacity_df, export_capacity_df,
        import_fraction_df, export_fraction_df,)

def _normalize_dispatch_item(item: str) -> str:
    """Return the canonical Dispatch_Order item."""

    normalized = str(item).strip().lower()
    canonical = {
        "bess": "BESS",
        "interconnections": "Interconnections",
        "commodities_production": "Commodities_Production",
        "fuel_to_electricity": "Fuel_to_Electricity",}
    if normalized not in canonical:
        raise ValueError(f"Unknown dispatch item: {item}")
    return canonical[normalized]


def _get_dispatch_order(kind: str) -> list[str]:
    """
    Reads Dispatch_Order from the YAML input.

    Defaults:
      surplus: Interconnections -> BESS -> Commodities_Production
      deficit: Interconnections -> BESS -> Fuel_to_Electricity
    """
    defaults = {
        "surplus": ["Interconnections", "BESS", "Commodities_Production"],
        "deficit": ["Interconnections", "BESS", "Fuel_to_Electricity"],}

    cfg = User.get("Dispatch_Order", {})
    if not isinstance(cfg, dict):
        return defaults[kind]

    order = cfg.get(kind, defaults[kind])
    if not isinstance(order, list):
        raise ValueError(f"Dispatch_Order.{kind} must be a list.")

    normalized = []
    for item in order:
        normalized_item = _normalize_dispatch_item(item)
        if normalized_item not in normalized:
            normalized.append(normalized_item)

    return normalized


def surplus(perturbed_df_energies_data):
    """Apply the configured balance-stage dispatch order.

    The returned frame contains raw and post-dispatch electricity balances,
    imports, and exports. Commodity production and fuel reconversion are
    applied later by the commodity-conversion functions.
    """

    df = pd.DataFrame()
    df['Date'] = pd.to_datetime(perturbed_df_energies_data['Date'])

    energy_label = normalize_energy_unit(User.get("energy_unit", "MWh"))
    balance_col = (
        "Electricity_Balance_After_Storage_and_Interconnections "
        f"({energy_label})")

    row_count = len(df)
    raw_list = np.empty(row_count, dtype=np.float32)
    adjusted = np.empty(row_count, dtype=np.float32)
    exports = np.zeros(row_count, dtype=np.float32)
    imports = np.zeros(row_count, dtype=np.float32)
    after_inter = np.empty(row_count, dtype=np.float32)
    after_bess = np.empty(row_count, dtype=np.float32)

    (inter_import_limit,
        inter_export_limit,
        inter_import_power_capacity,
        inter_export_power_capacity,
        inter_import_fraction,
        inter_export_fraction,
        inter_import_limit_df,
        inter_export_limit_df,
        inter_import_capacity_df,
        inter_export_capacity_df,
        inter_import_fraction_df,
        inter_export_fraction_df,
    ) = _interconnection_limit_series(perturbed_df_energies_data)

    # BESS is optional. If configured, it starts empty by definition.
    batteries = _bess_config_table(df["Date"]) if User.get("BESS", None) else []
    unit_factor = (
        energy_from_mwh_factor(User.get("energy_unit", "MWh"))
        if batteries else 1.0)
    period_hours = (
        _infer_period_hours(df['Date']).to_numpy(dtype=np.float64)
        if batteries
        else None)
    import_limits = inter_import_limit.to_numpy(
        dtype=np.float64, copy=False)
    export_limits = inter_export_limit.to_numpy(
        dtype=np.float64, copy=False)
    surplus_order = _get_dispatch_order("surplus")
    deficit_order = _get_dispatch_order("deficit")

    bess_soc_list = np.zeros(row_count, dtype=np.float32)
    bess_charge_list = np.zeros(row_count, dtype=np.float32)
    bess_energy_withdrawn = np.zeros(row_count, dtype=np.float32)
    bess_discharge_to_system = np.zeros(row_count, dtype=np.float32)
    bess_power_capacity = np.zeros(row_count, dtype=np.float32)
    bess_energy_capacity_list = np.zeros(
        row_count, dtype=np.float32)

    production = perturbed_df_energies_data[
        "Total"].to_numpy(dtype=float, copy=False)
    demand_values = perturbed_df_energies_data[
        "Demand"].to_numpy(dtype=float, copy=False)

    for pos in range(len(df)):

        prod_total = float(production[pos])
        demand = float(demand_values[pos])

        diff_raw = prod_total - demand
        diff = diff_raw

        export = 0.0
        import_ = 0.0
        diff_after_inter = diff_raw
        diff_after_bess = diff_raw

        total_charge = 0.0
        total_discharge = 0.0
        total_delivered = 0.0
        total_power_capacity = 0.0
        total_energy_capacity = 0.0

        if batteries:
            hours = float(period_hours[pos])
            total_power_capacity, total_energy_capacity = (
                _prepare_battery_step(
                    batteries, pos, hours, unit_factor,
                    initialize_soc=True))

        if diff > 0:
            dispatch_order = surplus_order
        elif diff < 0:
            dispatch_order = deficit_order
        else:
            dispatch_order = ()

        for asset in dispatch_order:

            if abs(diff) <= 1e-12:
                diff = 0.0
                break

            if asset == "BESS" and batteries:
                (diff, charge, discharge, delivered) = (
                    _dispatch_batteries(diff, batteries))
                total_charge += charge
                total_discharge += discharge
                total_delivered += delivered
                diff_after_bess = diff

            elif asset == "Interconnections":

                if diff > 0:
                    max_export = float(export_limits[pos])
                    exported = min(diff, max_export)
                    diff -= exported
                    export += exported

                elif diff < 0:
                    max_import = float(import_limits[pos])
                    imported = min(-diff, max_import)
                    diff += imported
                    import_ += imported

                diff_after_inter = diff

            elif asset in ("Commodities_Production", "Fuel_to_Electricity"):
                # Handled later by final_production() and
                # to_electricity_with_storage().
                continue

        raw_list[pos] = diff_raw
        adjusted[pos] = diff
        exports[pos] = export
        imports[pos] = import_
        after_inter[pos] = diff_after_inter
        after_bess[pos] = diff_after_bess

        if batteries:
            bess_soc_list[pos] = sum(
                float(batt["soc"]) for batt in batteries)
            bess_charge_list[pos] = total_charge
            bess_energy_withdrawn[pos] = total_discharge
            bess_discharge_to_system[pos] = total_delivered
            bess_power_capacity[pos] = total_power_capacity
            bess_energy_capacity_list[pos] = total_energy_capacity

    df[f'Raw_Electricity_Balance ({energy_label})'] = raw_list
    df[balance_col] = adjusted
    df[f'Exports ({energy_label})'] = exports
    df[f'Imports ({energy_label})'] = imports
    after_inter_col = (
        f"Electricity_Balance_After_Interconnections ({energy_label})")
    df[after_inter_col] = after_inter

    # Aggregate interconnection traceability.
    # Aggregate import-side values are retained because deficits are covered
    # through imports; explicit import/export columns avoid ambiguity.
    df['Interconnection_Fraction'] = inter_import_fraction.to_numpy(
        dtype=np.float32)
    df['Interconnection_Power_Capacity'] = inter_import_power_capacity.to_numpy(
        dtype=np.float32)
    df[f'Interconnection_Limit ({energy_label})'] = inter_import_limit.to_numpy(
        dtype=np.float32)

    df['Interconnection_Import_Fraction'] = inter_import_fraction.to_numpy(
        dtype=np.float32)
    df['Interconnection_Export_Fraction'] = inter_export_fraction.to_numpy(
        dtype=np.float32)
    import_capacity_column = "Interconnection_Import_Power_Capacity"
    export_capacity_column = "Interconnection_Export_Power_Capacity"
    import_limit_column = (
        f"Interconnection_Import_Limit ({energy_label})")
    export_limit_column = (
        f"Interconnection_Export_Limit ({energy_label})")
    df[import_capacity_column] = inter_import_power_capacity.to_numpy(
        dtype=np.float32)
    df[export_capacity_column] = inter_export_power_capacity.to_numpy(
        dtype=np.float32)
    df[import_limit_column] = inter_import_limit.to_numpy(
        dtype=np.float32)
    df[export_limit_column] = inter_export_limit.to_numpy(
        dtype=np.float32)

    # Per-interconnection traceability when named interconnections are used.
    named_columns = list(inter_import_fraction_df.columns)
    show_named = not (len(named_columns) == 1 and named_columns[0] == "Total")
    if show_named:
        for name in named_columns:
            suffix = _safe_column_token(name)
            import_fraction_column = (
                f"Interconnection_Import_Fraction_{suffix}")
            export_fraction_column = (
                f"Interconnection_Export_Fraction_{suffix}")
            import_capacity_column = (
                f"Interconnection_Import_Power_Capacity_{suffix}")
            export_capacity_column = (
                f"Interconnection_Export_Power_Capacity_{suffix}")
            import_limit_column = (
                f"Interconnection_Import_Limit_{suffix} "
                f"({energy_label})")
            export_limit_column = (
                f"Interconnection_Export_Limit_{suffix} "
                f"({energy_label})")
            df[import_fraction_column] = (
                inter_import_fraction_df[name]
                .to_numpy(dtype=np.float32))
            df[export_fraction_column] = (
                inter_export_fraction_df[name]
                .to_numpy(dtype=np.float32))
            df[import_capacity_column] = (
                inter_import_capacity_df[name]
                .to_numpy(dtype=np.float32))
            df[export_capacity_column] = (
                inter_export_capacity_df[name]
                .to_numpy(dtype=np.float32))
            df[import_limit_column] = (
                inter_import_limit_df[name]
                .to_numpy(dtype=np.float32))
            df[export_limit_column] = (
                inter_export_limit_df[name]
                .to_numpy(dtype=np.float32))

    if not batteries:
        after_bess = adjusted.copy()

    df["BESS_Power_Capacity"] = bess_power_capacity
    df[f"BESS_Energy_Capacity ({energy_label})"] = (
        bess_energy_capacity_list)
    df[f"BESS_SOC ({energy_label})"] = bess_soc_list
    df[f"BESS_Charge ({energy_label})"] = bess_charge_list
    df[f"BESS_Energy_Withdrawn ({energy_label})"] = bess_energy_withdrawn
    df[f"BESS_Discharge_to_System ({energy_label})"] = bess_discharge_to_system
    df[f"Electricity_Balance_After_BESS ({energy_label})"] = after_bess
    return df


def _as_nonnegative_float(value, name="value") -> float:
    """Converts a value to a non-negative float."""
    try:
        out = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"Invalid {name}: {value}") from exc

    if not np.isfinite(out) or out < 0.0:
        raise ValueError(
            f'{name} must be a non-negative finite number: {value}')

    return out


def _parse_bess_events(events, name="BESS") -> list[tuple[pd.Timestamp, float]]:
    """
    Parses BESS dated capacity values.

    Expected format:
      values:
        2025-01-01: 11875
        2030-01-01: 22500

    Values are interpreted as installed BESS power capacity in MW.
    """
    def parse_capacity(value):
        """Parse one non-negative installed BESS capacity."""

        return _as_nonnegative_float(value, name=f"{name} capacity")

    return _parse_dated_events(events, name, parse_capacity)


def _bess_power_capacity_series(
        dates: pd.Series,
        cfg: dict,
        name="BESS") -> pd.Series:
    """
    Builds a BESS installed power capacity series in MW.

    Accepted models are ``constant``, ``step``, and ``linear``.

    Values are always installed power capacity in MW.
    """
    if not isinstance(cfg, dict):
        value = _as_nonnegative_float(cfg, name=f"{name} capacity")
        return pd.Series(
            np.full(len(dates), value, dtype=float),
            index=dates.index)

    dt = pd.to_datetime(dates).reset_index(drop=True)
    model = str(cfg.get("model", "constant")).strip().lower()

    if model == "constant":
        if "value" in cfg:
            value = _as_nonnegative_float(cfg["value"], name=f"{name} capacity")
        else:
            events = _parse_bess_events(cfg.get("values", {}), name=name)
            value = events[0][1] if events else 0.0
        return pd.Series(
            np.full(len(dt), value, dtype=float),
            index=dates.index)

    events = _parse_bess_events(cfg.get("values", {}), name=name)
    if not events:
        raise ValueError(
            f"BESS '{name}' requires dated values for model '{model}'.")

    if model == "step":
        values = _step_values(dt, events, initial=cfg.get("initial", None))
        return pd.Series(values, index=dates.index)

    if model == "linear":
        values = _piecewise_linear_values(dt, events)
        return pd.Series(values, index=dates.index)

    raise ValueError(
        f"Invalid BESS model for '{name}'. Use 'constant', "
        "'step' or 'linear'.")


def _is_single_bess_config(cfg: dict) -> bool:
    """Detects whether BESS is an aggregate schedule or a dict of named
    schedules."""
    if not isinstance(cfg, dict):
        return True
    single_keys = {"model", "value", "values", "duration", "efficiency"}
    return any(key in cfg for key in single_keys)


def _initial_conditions() -> dict:
    """Return optional operational initial conditions."""

    simulation = User.get("simulation", {})
    if not isinstance(simulation, dict):
        return {}
    conditions = simulation.get("initial_conditions", {})
    return conditions if isinstance(conditions, dict) else {}


def _initialize_battery_soc(battery: dict, energy_capacity: float) -> None:
    """Initialize one battery state of charge once per simulation."""

    if battery.get("soc_initialized", False):
        return
    fraction = float(battery.get("initial_soc_fraction", 0.0))
    battery["soc"] = energy_capacity * fraction
    battery["soc_initialized"] = True


def _prepare_battery_step(
        batteries, position, hours, unit_factor, initialize_soc=False):
    """Update battery limits for one time step and return fleet capacities."""

    total_power_capacity = 0.0
    total_energy_capacity = 0.0
    for battery in batteries:
        power_capacity = float(battery["power_capacity_values"][position])
        energy_capacity = (
            power_capacity * battery["duration"] * unit_factor)
        battery["current_e_max"] = energy_capacity
        battery["current_power_period_limit"] = (
            power_capacity * hours * unit_factor)
        if initialize_soc:
            _initialize_battery_soc(battery, energy_capacity)
        battery["soc"] = min(
            max(float(battery["soc"]), 0.0), energy_capacity)
        total_power_capacity += power_capacity
        total_energy_capacity += energy_capacity
    return total_power_capacity, total_energy_capacity


def _dispatch_batteries(diff, batteries):
    """Charge or discharge batteries in configured order for one balance."""

    total_charge = 0.0
    total_discharge = 0.0
    total_delivered = 0.0
    if diff > 0.0:
        remaining = diff
        for battery in batteries:
            if remaining <= 1e-12:
                break
            free = max(
                battery["current_e_max"] - battery["soc"], 0.0)
            charge = min(
                remaining, free, battery["current_power_period_limit"])
            if charge > 0.0:
                battery["soc"] += charge
                remaining -= charge
                total_charge += charge
        diff = remaining
    elif diff < 0.0:
        remaining_need = -diff
        for battery in batteries:
            if remaining_need <= 1e-12:
                break
            efficiency = battery["efficiency"]
            if efficiency <= 0.0:
                continue
            max_deliver = min(
                battery["soc"] * efficiency,
                battery["current_power_period_limit"])
            delivered = min(remaining_need, max_deliver)
            if delivered > 0.0:
                discharge = delivered / efficiency
                battery["soc"] -= discharge
                remaining_need -= delivered
                total_delivered += delivered
                total_discharge += discharge
        diff = -remaining_need
    return diff, total_charge, total_discharge, total_delivered


def _bess_config_table(dates: pd.Series):
    """
    Read the canonical BESS format.

    Expected format:
      BESS:
        Identifier:
          model: linear
          values:
            2025-01-01: 11875
            2030-01-01: 22500
          duration: 4
          efficiency: 0.81

    The values are installed BESS power capacity in MW.
    """
    bess_cfg = User.get("BESS", None)
    if not bess_cfg:
        return []

    if _is_single_bess_config(bess_cfg):
        items = [("Total", bess_cfg)]
    else:
        items = list(bess_cfg.items())

    batteries = []
    for name, cfg in items:
        if not isinstance(cfg, dict):
            cfg = {"model": "constant", "value": cfg}

        power_capacity = _bess_power_capacity_series(dates, cfg, name=str(name))
        duration = _as_nonnegative_float(
            cfg.get('duration'),
            name=f'{name} duration')
        if duration <= 0.0:
            raise ValueError(f"BESS '{name}' requires duration > 0.")

        efficiency = _as_nonnegative_float(
            cfg.get('efficiency'),
            name=f'{name} efficiency')
        if efficiency > 1.0:
            raise ValueError(f"BESS '{name}' efficiency must be in [0, 1].")

        conditions = _initial_conditions()
        initial_fraction = float(
            conditions.get("bess_state_of_charge", 0.0))
        power_capacity = power_capacity.astype(float)
        batteries.append({
            "name": str(name),
            "power_capacity": power_capacity,
            "power_capacity_values": power_capacity.to_numpy(
                dtype=np.float64, copy=False),
            "duration": duration,
            "efficiency": efficiency,
            "soc": 0.0,
            "initial_soc_fraction": initial_fraction,
            "soc_initialized": False,
        })

    return batteries


class FIFOStorage:
    """Store commodity quantities and withdraw the oldest batch first."""

    def __init__(self, max_age_days=np.inf,
                 max_storage_quantity=np.inf):
        """Create an empty FIFO inventory with age and quantity limits."""

        self.max_age_days = max_age_days
        self.max_storage_quantity = max_storage_quantity
        self.batches = deque()
        self.current_quantity = 0.0

    @property
    def total_quantity(self):
        """Return the commodity quantity currently stored."""

        return self.current_quantity

    def add(self, date, quantity):
        """Store available quantity and return stored and sold amounts."""

        quantity = float(quantity)
        if quantity <= 0:
            return 0.0, 0.0

        free_capacity = max(
            self.max_storage_quantity - self.current_quantity, 0.0)
        stored = min(quantity, free_capacity)
        sold = max(quantity - stored, 0.0)

        if stored > 0:
            if not np.isinf(self.max_age_days):
                batch = {
                    "date": pd.Timestamp(date),
                    "quantity": stored}
                self.batches.append(batch)
            self.current_quantity += stored

        return stored, sold

    def expire(self, current_date):
        """Remove batches older than the configured maximum age."""

        if np.isinf(self.max_age_days):
            return 0.0

        cutoff = pd.Timestamp(current_date)
        cutoff -= pd.Timedelta(days=self.max_age_days)
        expired_quantity = 0.0

        while self.batches:
            batch = self.batches[0]
            if batch["date"] >= cutoff:
                break

            expired_quantity += batch["quantity"]
            self.current_quantity -= batch["quantity"]
            self.batches.popleft()

        if self.current_quantity <= 1e-12:
            self.current_quantity = 0.0

        return expired_quantity

    def withdraw(self, needed_quantity):
        """Withdraw up to the requested quantity, oldest batches first."""

        remaining_need = float(needed_quantity)
        if remaining_need <= 0:
            return 0.0

        if np.isinf(self.max_age_days):
            withdrawn = min(self.current_quantity, remaining_need)
            self.current_quantity -= withdrawn
            if self.current_quantity <= 1e-12:
                self.current_quantity = 0.0
            return withdrawn

        withdrawn = 0.0

        while self.batches and remaining_need > 1e-12:
            batch = self.batches[0]
            taken = min(batch["quantity"], remaining_need)
            batch["quantity"] -= taken
            remaining_need -= taken
            withdrawn += taken
            self.current_quantity -= taken

            if batch["quantity"] <= 1e-12:
                self.batches.popleft()

        if self.current_quantity <= 1e-12:
            self.current_quantity = 0.0

        return withdrawn


def build_storage_config(commodities, user):
    """Build one FIFO storage object for each commodity."""

    config = user.get("Commodity_Storage", {})
    storages = {}

    for commodity in commodities:
        commodity_cfg = config.get(commodity, {})

        if "max_age_days" in commodity_cfg:
            max_age_days = float(
                commodity_cfg["max_age_days"])
        elif "max_age" in commodity_cfg:
            max_age_days = float(
                commodity_cfg["max_age"]) * 365.0
        else:
            max_age_days = np.inf

        max_quantity = float(commodity_cfg.get(
            "max_storage_quantity",
            commodity_cfg.get("max_storage_mass", np.inf)))
        storages[commodity] = FIFOStorage(
            max_age_days=max_age_days,
            max_storage_quantity=max_quantity)

    return storages
