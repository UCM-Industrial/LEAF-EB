"""Generate correlated stochastic series from historical residuals."""

import re
from functools import lru_cache
from statistics import NormalDist

import numpy as np
import pandas as pd
from src.utilities.bootstrap_tools import (
    ANNUAL_VARIABILITY_SEED_OFFSET,
    apply_annual_variability,
    seasonal_stationary_indices,)
from src.utilities.constants import MC_PROFILE_DATE_COLUMN


_STANDARD_NORMAL = NormalDist()


def _params_df_to_dict(df: pd.DataFrame) -> dict:
    """Convert a parameter table into a dictionary by source."""

    if "column" not in df.columns:
        raise ValueError(
            "Residual params must have a 'column' column.")

    output = {}

    for row in df.to_dict("records"):
        column = row["column"]
        output[column] = {
            key: value for key, value in row.items()
            if pd.notna(value)}

    return output


def residual_params_dict(params_file: str):
    """Load global, monthly, slow and fallback residual parameters."""

    params_file = str(params_file)
    if params_file.lower().endswith(".csv"):
        return _params_df_to_dict(pd.read_csv(params_file))

    with pd.ExcelFile(params_file) as workbook:
        if len(workbook.sheet_names) == 1:
            frame = pd.read_excel(
                workbook, sheet_name=workbook.sheet_names[0])
            return _params_df_to_dict(frame)

        output = {}
        for sheet in workbook.sheet_names:
            frame = pd.read_excel(workbook, sheet_name=sheet)
            normalized = sheet.strip().lower()
            month_match = re.fullmatch(
                r"m(0[1-9]|1[0-2])", normalized)

            if month_match:
                output[int(month_match.group(1))] = (
                    _params_df_to_dict(frame))
            elif normalized == "global":
                output["global"] = _params_df_to_dict(frame)
            elif normalized == "slow":
                output["slow"] = _params_df_to_dict(frame)

    if not output:
        raise ValueError(
            "Residual workbook has no Global, Slow or M01..M12 sheets.")
    return output


def load_cov_matrix(cov_file: str):
    """Load global, monthly, slow and fallback covariance matrices."""

    with pd.ExcelFile(cov_file) as workbook:
        if len(workbook.sheet_names) == 1:
            return pd.read_excel(
                workbook, sheet_name=workbook.sheet_names[0],
                index_col=0)

        output = {}
        for sheet in workbook.sheet_names:
            normalized = sheet.strip().lower()
            month_match = re.fullmatch(
                r"m(0[1-9]|1[0-2])", normalized)

            if month_match:
                key = int(month_match.group(1))
            elif normalized == "global":
                key = "global"
            elif normalized == "slow":
                key = "slow"
            else:
                continue

            output[key] = pd.read_excel(
                workbook, sheet_name=sheet, index_col=0)

    if not output:
        raise ValueError(
            "Covariance workbook has no Global, Slow or M01..M12 sheets.")
    return output


def _is_monthly_mapping(data):
    """Return True when a dictionary contains month-number keys."""

    return isinstance(data, dict) and any(
        isinstance(key, int) for key in data)


def _fast_mapping(data):
    """Return the short-term global or monthly parameter mapping."""

    if not isinstance(data, dict):
        return data

    months = {
        key: value for key, value in data.items()
        if isinstance(key, int)}

    if months:
        return months

    if "global" in data:
        return data["global"]

    return data


def _slow_mapping(data):
    """Return optional monthly slow-component parameters."""

    if isinstance(data, dict):
        return data.get("slow")

    return None


def _bootstrap_mapping(data):
    """Return optional fast-residual history for stationary bootstrap."""

    if isinstance(data, dict):
        return data.get("bootstrap")

    return None


def _build_correlation(covariance, technologies):
    """Convert a covariance matrix into a correlation matrix."""

    covariance = covariance.loc[
        technologies, technologies
    ].to_numpy(dtype=float)
    standard = np.sqrt(
        np.clip(np.diag(covariance), 1e-12, None))
    denominator = np.outer(standard, standard)
    correlation = covariance / denominator
    correlation = np.clip(correlation, -0.9999, 0.9999)
    np.fill_diagonal(correlation, 1.0)

    return correlation


def _autocorrelation_vector(parameters, technologies):
    """Return bounded AR(1) coefficients for one period."""

    coefficients = np.zeros(len(technologies), dtype=float)

    for index, technology in enumerate(technologies):
        values = parameters.get(technology, {})
        raw_value = values.get("autocorr_lag1", 0.0)

        try:
            coefficient = float(raw_value)
        except (TypeError, ValueError):
            coefficient = 0.0

        if not np.isfinite(coefficient):
            coefficient = 0.0

        coefficients[index] = np.clip(
            coefficient, -0.98, 0.98)

    return coefficients


def _positive_semidefinite_covariance(matrix, target_diagonal):
    """Project a symmetric matrix to a valid covariance matrix."""

    symmetric = 0.5 * (matrix + matrix.T)
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    eigenvalues = np.clip(eigenvalues, 1e-12, None)
    projected = (eigenvectors * eigenvalues) @ eigenvectors.T

    current_diagonal = np.clip(
        np.diag(projected), 1e-12, None)
    target_diagonal = np.clip(
        np.asarray(target_diagonal, dtype=float), 1e-12, None)
    scale = np.sqrt(target_diagonal / current_diagonal)
    projected *= np.outer(scale, scale)

    return 0.5 * (projected + projected.T)


def _innovation_covariance(correlation, coefficients):
    """Build VAR(1) innovations for a target covariance."""

    coefficient_products = np.outer(
        coefficients, coefficients)
    covariance = correlation * (1.0 - coefficient_products)
    target_diagonal = 1.0 - coefficients**2

    return _positive_semidefinite_covariance(
        covariance, target_diagonal)


def _period_data(data, month, monthly):
    """Return global or month-specific parameter data."""

    if monthly:
        return data.get(month)

    return data


def _temporal_gaussian_scores(
        row_count, month_values, technologies, residual_params,
        cov_matrix, params_monthly, covariance_monthly):
    """Generate correlated Gaussian scores with AR(1) persistence."""

    technology_count = len(technologies)
    scores = np.zeros(
        (row_count, technology_count), dtype=float)
    innovations = np.zeros_like(scores)
    coefficients_by_row = np.zeros_like(scores)
    active = np.zeros(row_count, dtype=bool)
    correlations = {}

    if params_monthly or covariance_monthly:
        periods = sorted(set(month_values))
    else:
        periods = [None]

    zero_mean = np.zeros(technology_count, dtype=float)

    for month in periods:
        positions = _period_positions(month_values, month)

        if positions.size == 0:
            continue

        parameters = _period_data(
            residual_params, month, params_monthly)
        covariance = _period_data(
            cov_matrix, month, covariance_monthly)

        if parameters is None or covariance is None:
            continue

        correlation = _build_correlation(
            covariance, technologies)
        coefficients = _autocorrelation_vector(
            parameters, technologies)
        innovation_covariance = _innovation_covariance(
            correlation, coefficients)

        innovations[positions] = np.random.multivariate_normal(
            zero_mean, innovation_covariance,
            size=positions.size, check_valid="ignore")
        coefficients_by_row[positions] = coefficients
        active[positions] = True
        correlations[month] = correlation

    for row in range(row_count):
        if not active[row]:
            continue

        previous_is_active = row > 0 and active[row - 1]

        if not previous_is_active:
            month = month_values[row]
            period = month if (
                params_monthly or covariance_monthly) else None
            scores[row] = np.random.multivariate_normal(
                zero_mean, correlations[period],
                check_valid="ignore")
            continue

        scores[row] = (
            coefficients_by_row[row] * scores[row - 1]
            + innovations[row])

    return scores


def _stationary_fast_shocks(
        dates, technologies, bootstrap_data
) -> tuple[np.ndarray, set[str], str, np.ndarray | None]:
    """Sample one shared historical residual sequence for all sources.

    The same bootstrap indices are applied to every available stochastic
    series. The selected historical dates are returned as metadata so hourly
    expansion can reuse the observed intraday profile from those same days.
    """

    output = np.zeros(
        (len(dates), len(technologies)), dtype=float)

    if not isinstance(bootstrap_data, dict):
        return output, set(), "global", None

    history = bootstrap_data.get("history")
    mean_length = bootstrap_data.get("mean_block_length", 1.0)

    if not isinstance(history, pd.DataFrame) or history.empty:
        return output, set(), "global", None

    date_column = bootstrap_data.get("date_column", "Date")

    if date_column not in history.columns:
        return output, set(), "global", None

    available = [
        technology for technology in technologies
        if technology in history.columns]

    if not available:
        return output, set(), "global", None

    history_dates = pd.to_datetime(history[date_column])
    history_months = history_dates.dt.month.to_numpy(dtype=int)
    target_months = pd.DatetimeIndex(dates).month.to_numpy(dtype=int)
    history_values = np.column_stack([
        pd.to_numeric(
            history[technology], errors="coerce"
        ).to_numpy(dtype=float)
        for technology in available
    ])
    if not np.isfinite(history_values).all():
        raise ValueError(
            "Residual history contains missing or non-finite bootstrap "
            "values.")
    seed = int(np.random.randint(
        0, np.iinfo(np.uint32).max, dtype=np.int64))
    generator = np.random.default_rng(seed)
    regime_generator = np.random.default_rng(
        (seed + ANNUAL_VARIABILITY_SEED_OFFSET)
        % np.iinfo(np.uint32).max)
    indices = seasonal_stationary_indices(
        history_months,
        target_months,
        float(mean_length),
        generator,
        history_dates=history_dates.to_numpy(),
        target_dates=np.asarray(dates),
        history_values=history_values,)
    technology_index = {
        technology: index
        for index, technology in enumerate(technologies)}

    sampled_shocks = history_values[indices]
    annual_scales = bootstrap_data.get(
        "annual_variability_scales")
    if annual_scales is not None:
        sampled_shocks = apply_annual_variability(
            sampled_shocks, dates, annual_scales, regime_generator)

    for source_index, technology in enumerate(available):
        output[:, technology_index[technology]] = (
            sampled_shocks[:, source_index])

    basis = str(bootstrap_data.get(
        "relative_basis", "global")).strip().lower()
    if basis not in {"global", "local"}:
        raise ValueError(
            "Bootstrap relative_basis must be 'global' or 'local'.")
    sampled_dates = history_dates.to_numpy(dtype="datetime64[ns]")[indices]
    return output, set(available), basis, sampled_dates


def _limit_array(parameters, dates, row_count):
    """Build the applicable upper limit for every simulated row."""

    base_limit = parameters.get("limit")
    if base_limit is None:
        return None

    try:
        base_value = float(base_limit)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            f"Invalid stochastic upper limit: {base_limit}.") from exc
    if not np.isfinite(base_value) or base_value < 0.0:
        raise ValueError(
            "Stochastic upper limits must be finite and non-negative.")
    limits = np.full(row_count, base_value, dtype=float)

    changes = parameters.get("custom_data", {})
    if changes is None:
        return limits
    if not isinstance(changes, dict):
        raise ValueError(
            "Stochastic limit custom_data must be a mapping.")

    for date_text, raw_change in changes.items():
        try:
            threshold = pd.Timestamp(date_text)
            change = float(raw_change)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                f"Invalid stochastic limit change at {date_text}: "
                f"{raw_change}.") from exc
        if not np.isfinite(change):
            raise ValueError(
                f"Stochastic limit change at {date_text} must be finite.")
        limits[dates >= threshold] += change

    return np.maximum(limits, 0.0)


def _period_positions(month_values, month):
    """Return row positions belonging to one monthly period."""

    if month is None:
        return np.arange(len(month_values), dtype=int)

    return np.flatnonzero(month_values == month)


@lru_cache(maxsize=16)
def _smooth_clip_shape(z_limit: float) -> float:
    """Match winsorized variance with a smooth bounded transform."""

    limit = float(z_limit)

    if not np.isfinite(limit) or limit <= 0.0:
        return 1.0

    tail = 1.0 - _STANDARD_NORMAL.cdf(limit)
    target_variance = (
        1.0
        - 2.0 * limit * _STANDARD_NORMAL.pdf(limit)
        + 2.0 * (limit**2 - 1.0) * tail)
    target_std = np.sqrt(max(target_variance, 1e-12))
    nodes, weights = np.polynomial.hermite.hermgauss(48)
    normal_nodes = np.sqrt(2.0) * nodes
    normal_weights = weights / np.sqrt(np.pi)

    def transformed_std(shape: float) -> float:
        """Return the standard deviation after smooth clipping."""

        values = limit * np.tanh(normal_nodes / shape)
        variance = np.sum(normal_weights * values**2)
        return float(np.sqrt(max(variance, 0.0)))

    lower = 1e-3
    upper = 100.0

    for _ in range(60):
        middle = 0.5 * (lower + upper)

        if transformed_std(middle) > target_std:
            lower = middle
        else:
            upper = middle

    return 0.5 * (lower + upper)


def _smooth_confidence_scores(scores, z_limit):
    """Bound Gaussian scores continuously without boundary masses."""

    if not np.isfinite(z_limit) or z_limit <= 0.0:
        return scores

    shape = _smooth_clip_shape(float(z_limit))

    return float(z_limit) * np.tanh(scores / shape)


def _slow_relative_shocks(
        dates, technologies, slow_params, slow_covariance, z_limit
):
    """Generate one correlated persistent anomaly per calendar month."""

    row_count = len(dates)
    output = np.zeros(
        (row_count, len(technologies)), dtype=float)

    if not slow_params or slow_covariance is None:
        return output

    active_technologies = [
        technology for technology in technologies
        if technology in slow_params
        and technology in slow_covariance.index
        and technology in slow_covariance.columns]

    if not active_technologies:
        return output

    periods = pd.DatetimeIndex(dates).to_period("M")
    month_codes, unique_months = pd.factorize(periods, sort=True)
    dummy_period = np.zeros(len(unique_months), dtype=int)
    scores = _temporal_gaussian_scores(
        row_count=len(unique_months),
        month_values=dummy_period,
        technologies=active_technologies,
        residual_params=slow_params,
        cov_matrix=slow_covariance,
        params_monthly=False,
        covariance_monthly=False,)
    scores = _smooth_confidence_scores(scores, z_limit)
    technology_index = {
        technology: index
        for index, technology in enumerate(technologies)}

    for local_index, technology in enumerate(active_technologies):
        values = slow_params[technology]
        original_mean = float(values.get("original_mean", 0.0))

        if abs(original_mean) <= 1e-12:
            continue

        relative_scale = float(values.get("scale", 0.0))
        relative_scale /= abs(original_mean)
        relative_scale = min(max(relative_scale, 0.0), 1.0)
        target_index = technology_index[technology]
        output[:, target_index] = (
            relative_scale * scores[month_codes, local_index])

    return output


def _bounded_rescale(values, target, limits=None):
    """Scale one series proportionally without redistributing energy."""

    result = np.maximum(np.asarray(values, dtype=float), 0.0)
    tolerance = max(1e-9, abs(float(target)) * 1e-10)
    if target <= tolerance:
        return np.zeros_like(result)

    current = float(result.sum())
    if current <= tolerance:
        return result

    result *= float(target) / current
    if limits is None:
        return result

    upper = np.maximum(np.asarray(limits, dtype=float), 0.0)
    return np.minimum(result, upper)


def _rescale_annual_totals(
        output, baseline, technologies, date_col, limit_arrays=None
):
    """Scale annual targets while preserving their temporal shape."""

    years = pd.to_datetime(output[date_col]).dt.year.to_numpy()
    limit_arrays = limit_arrays or {}

    for technology in technologies:
        if technology not in baseline.columns:
            continue

        base_values = pd.to_numeric(
            baseline[technology], errors="coerce"
        ).fillna(0.0).to_numpy(dtype=float)
        new_values = pd.to_numeric(
            output[technology], errors="coerce"
        ).fillna(0.0).clip(lower=0.0).to_numpy(
            dtype=float, copy=True)
        limits = limit_arrays.get(technology)

        for year in np.unique(years):
            positions = years == year
            target = float(base_values[positions].sum())
            year_limits = None
            if limits is not None:
                year_limits = limits[positions]
            new_values[positions] = _bounded_rescale(
                new_values[positions], target, year_limits)

        output[technology] = new_values

    return output


def _technology_limit_arrays(
        fast_params, params_monthly, dates, month_values, technologies
):
    """Build complete limit arrays for every constrained technology."""

    arrays = {}
    row_count = len(dates)
    periods = sorted(set(month_values)) if params_monthly else [None]
    for technology in technologies:
        combined = np.full(row_count, np.inf, dtype=float)
        found = False
        for month in periods:
            positions = _period_positions(month_values, month)
            parameters = (
                fast_params.get(month)
                if params_monthly else fast_params)
            if not isinstance(parameters, dict):
                continue
            values = parameters.get(technology)
            if not isinstance(values, dict):
                continue
            limits = _limit_array(
                values, dates[positions], positions.size)
            if limits is None:
                continue
            combined[positions] = limits
            found = True
        if found:
            arrays[technology] = combined
    return arrays


def _stochastic_technologies(output, fast_params, params_monthly):
    """Return forecast columns that have stochastic residual parameters."""

    if params_monthly:
        available = {
            technology
            for month_data in fast_params.values()
            for technology in month_data}
        return [
            column for column in output.columns
            if column in available]

    return [
        column for column in output.columns
        if column in fast_params]


def _gaussian_fallback_scores(
        row_count, month_values, technologies, fast_params,
        fast_covariance, params_monthly, covariance_monthly,
        fallback_sources, z_limit):
    """Generate Gaussian fallback scores when bootstrap data are absent."""

    if not fallback_sources:
        return np.zeros((row_count, len(technologies)), dtype=float)

    scores = _temporal_gaussian_scores(
        row_count=row_count,
        month_values=month_values,
        technologies=technologies,
        residual_params=fast_params,
        cov_matrix=fast_covariance,
        params_monthly=params_monthly,
        covariance_monthly=covariance_monthly,)
    return _smooth_confidence_scores(scores, z_limit)


def _transition_weights(dates):
    """Return full stochastic amplitude from the first forecast step.

    The empirical stationary bootstrap already supplies a valid historical
    residual at the forecast boundary. Dampening that residual to zero would
    force every Monte Carlo realization to start on the deterministic
    forecast and would artificially suppress variability at the beginning of
    both future simulations and blind holdout tests.

    No boundary ramp is applied.
    """
    index = pd.DatetimeIndex(pd.to_datetime(dates))
    return np.ones(len(index), dtype=float)


def _period_parameters(fast_params, month, params_monthly):
    """Return residual parameters for one global or monthly period."""

    if params_monthly:
        return fast_params.get(month)
    return fast_params


def _fallback_relative_shock(values, scores):
    """Convert Gaussian fallback scores to relative residual shocks."""

    original_mean = float(values.get("original_mean", 0.0))
    if abs(original_mean) <= 1e-12:
        return None

    relative_scale = float(values.get("scale", 0.0))
    relative_scale /= abs(original_mean)
    relative_scale = min(max(relative_scale, 0.0), 1.0)
    return relative_scale * scores


def _reconstruct_stochastic_values(
        base, combined_shock, uses_local_bootstrap):
    """Reconstruct one stochastic period from relative residual shocks."""

    if uses_local_bootstrap:
        return base * (1.0 + combined_shock)

    period_scale = max(abs(float(np.mean(base))), 1e-12)
    return base + period_scale * combined_shock


def _apply_stochastic_shocks(
        output, dates, month_values, technologies, fast_params,
        params_monthly, bootstrap_shocks, bootstrap_sources,
        bootstrap_basis, gaussian_scores, slow_shocks, transition):
    """Apply prepared stochastic shocks to all requested forecast columns."""

    technology_index = {
        technology: index
        for index, technology in enumerate(technologies)}
    periods = (
        sorted(set(month_values))
        if params_monthly else [None])

    for month in periods:
        positions = _period_positions(month_values, month)
        if positions.size == 0:
            continue

        parameters = _period_parameters(
            fast_params, month, params_monthly)
        if parameters is None:
            continue

        for technology in technologies:
            values = parameters.get(technology)
            if values is None:
                continue

            index = technology_index[technology]
            base = output[technology].to_numpy(
                dtype=float, copy=False
            )[positions]
            slow = slow_shocks[positions, index]

            if technology in bootstrap_sources:
                fast = bootstrap_shocks[positions, index]
            else:
                fast = _fallback_relative_shock(
                    values, gaussian_scores[positions, index])
                if fast is None:
                    continue

            combined_shock = slow + fast
            combined_shock *= transition[positions]
            uses_local_bootstrap = (
                technology in bootstrap_sources
                and bootstrap_basis == "local")
            new_values = _reconstruct_stochastic_values(
                base, combined_shock, uses_local_bootstrap)
            limits = _limit_array(
                values, dates[positions], positions.size)
            if limits is not None:
                new_values = np.minimum(new_values, limits)

            output.loc[
                output.index[positions], technology
            ] = np.maximum(new_values, 0.0)

    return output


def _restore_annual_energy_if_requested(
        output, baseline, preserve_annual_targets, fast_params,
        params_monthly, dates, month_values, technologies, date_col):
    """Optionally restore deterministic annual energy after perturbation.

    When ``preserve_annual_targets`` is false, sampled historical residuals
    are allowed to change annual energy as well as its timing.  When true,
    every selected stochastic series is rescaled within each calendar year
    to the deterministic annual total.  Explicit capacity limits still bind.
    """

    if not preserve_annual_targets:
        return output

    limit_arrays = _technology_limit_arrays(
        fast_params,
        params_monthly,
        dates,
        month_values,
        technologies,)
    return _rescale_annual_totals(
        output,
        baseline,
        technologies,
        date_col,
        limit_arrays,)


def _update_generation_total(output, baseline, technologies):
    """Update Total using stochastic changes in generation, not demand."""

    if "Total" not in baseline.columns:
        return output

    total = pd.to_numeric(
        baseline["Total"], errors="coerce"
    ).fillna(0.0).to_numpy(dtype=float)

    for technology in technologies:
        if technology == "Demand":
            continue
        change = output[technology].to_numpy(dtype=float, copy=True)
        change -= baseline[technology].to_numpy(dtype=float)
        total += change

    output["Total"] = total
    return output


def perturb_df_energies(
        forecast_df: pd.DataFrame, residual_params,
        confidence_level: float, cov_matrix,
        date_col: str = "Date", preserve_annual_targets: bool = False):
    """Apply stochastic residuals to a deterministic energy forecast.

    Current simulations use empirical local-relative residuals drawn with a
    shared stationary bootstrap. Gaussian parameters provide the fallback
    when a source has no bootstrap history. The selected
    historical dates are attached as temporary metadata so hourly expansion
    can reuse the intraday profiles from the same sampled days.  By default,
    annual energy is also allowed to vary.  Set ``preserve_annual_targets``
    to true to restore each deterministic annual total after perturbation.
    """

    output = forecast_df.copy()
    output[date_col] = pd.to_datetime(output[date_col])
    fast_params = _fast_mapping(residual_params)
    fast_covariance = _fast_mapping(cov_matrix)
    slow_params = _slow_mapping(residual_params)
    slow_covariance = _slow_mapping(cov_matrix)
    bootstrap_data = _bootstrap_mapping(residual_params)
    params_monthly = _is_monthly_mapping(fast_params)
    covariance_monthly = _is_monthly_mapping(fast_covariance)
    dates = output[date_col].to_numpy()
    month_values = output[date_col].dt.month.to_numpy()
    confidence = float(confidence_level)
    z_limit = _STANDARD_NORMAL.inv_cdf(0.5 + confidence / 2.0)

    technologies = _stochastic_technologies(
        output, fast_params, params_monthly)
    if not technologies:
        return forecast_df

    bootstrap_result = _stationary_fast_shocks(
        dates, technologies, bootstrap_data)
    bootstrap_shocks = bootstrap_result[0]
    bootstrap_sources = bootstrap_result[1]
    bootstrap_basis = bootstrap_result[2]
    bootstrap_profile_dates = bootstrap_result[3]
    fallback_sources = set(technologies) - bootstrap_sources

    gaussian_scores = _gaussian_fallback_scores(
        len(output),
        month_values,
        technologies,
        fast_params,
        fast_covariance,
        params_monthly,
        covariance_monthly,
        fallback_sources,
        z_limit,)
    slow_shocks = _slow_relative_shocks(
        dates,
        technologies,
        slow_params,
        slow_covariance,
        z_limit,)
    transition = _transition_weights(dates)

    output = _apply_stochastic_shocks(
        output,
        dates,
        month_values,
        technologies,
        fast_params,
        params_monthly,
        bootstrap_shocks,
        bootstrap_sources,
        bootstrap_basis,
        gaussian_scores,
        slow_shocks,
        transition,)
    output = _restore_annual_energy_if_requested(
        output,
        forecast_df,
        preserve_annual_targets,
        fast_params,
        params_monthly,
        dates,
        month_values,
        technologies,
        date_col,)

    if bootstrap_profile_dates is not None and bootstrap_sources:
        output[MC_PROFILE_DATE_COLUMN] = pd.to_datetime(
            bootstrap_profile_dates)

    return _update_generation_total(
        output, forecast_df, technologies)
