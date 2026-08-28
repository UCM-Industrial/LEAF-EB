"""Utilities for multivariate seasonal stationary-bootstrap sampling.

The module has three responsibilities:

1. estimate a data-driven mean block length for each residual series;
2. draw one shared sequence of historical indices for all stochastic series;
3. preserve observed year-to-year changes in residual variability.

Sharing bootstrap indices and annual variability regimes across sources
preserves contemporaneous dependence without adding technology-specific
rules. Restart pools preserve calendar compatibility.
"""

from __future__ import annotations

import numpy as np


COMMON_BLOCK_LENGTH_QUANTILE = 0.75
ANNUAL_VARIABILITY_SEED_OFFSET = 88
_RESTART_JUMP_QUANTILE = 0.995
_COMPLETE_YEAR_FRACTION = 0.99
_MIN_VARIANCE = 1e-12
_MIN_DENOMINATOR = 1e-24


def optimal_stationary_block_length(
        values: np.ndarray,
        observation_step_days: float = 1.0) -> float:
    """Estimate the mean stationary-bootstrap block length for one series.

    The implementation follows the automatic plug-in selector of Politis and
    White, with the corrected variance constant for the stationary bootstrap.
    Invalid values are removed before estimation. Constant or very short
    series fall back to a one-step block.

    Parameters
    ----------
    values:
        One-dimensional historical residual series.

    Returns
    -------
    float
        Estimated mean block length in observations, bounded to a sensible
        sample-size-dependent range.
    """

    array = np.asarray(values, dtype=float).reshape(-1)
    array = array[np.isfinite(array)]
    n_obs = int(array.size)

    if n_obs < 10 or float(np.std(array)) <= _MIN_VARIANCE:
        return 1.0

    step_days = float(observation_step_days)
    if not np.isfinite(step_days) or step_days <= 0.0:
        raise ValueError(
            "observation_step_days must be finite and positive.")

    centered = array - float(np.mean(array))
    physical_scale = max(1.0 / step_days, 1.0)
    block_max = float(np.ceil(min(
        3.0 * np.sqrt(n_obs * physical_scale),
        n_obs / 3.0)))
    consecutive = max(5, int(np.log10(n_obs)))
    lag_max = int(np.ceil(
        np.sqrt(n_obs * physical_scale))) + consecutive
    lag_max = min(lag_max, n_obs - 2)
    critical = 2.0 * np.sqrt(np.log10(n_obs) / n_obs)
    autocovariance = np.zeros(lag_max + 1, dtype=float)
    abs_autocorrelation = np.zeros(lag_max + 1, dtype=float)
    selected_lag = None

    for lag in range(lag_max + 1):
        current = centered[lag:]
        previous = centered[:n_obs - lag]
        cross_product = current @ previous
        autocovariance[lag] = cross_product / n_obs

        denominator = np.sqrt(
            (current @ current) * (previous @ previous))
        if denominator > _MIN_VARIANCE:
            abs_autocorrelation[lag] = abs(cross_product) / denominator

        if lag >= consecutive and selected_lag is None:
            start = lag - consecutive
            window = abs_autocorrelation[start:lag]
            if np.all(window < critical):
                selected_lag = start

    if selected_lag is None:
        truncation = lag_max
    else:
        truncation = 2 * max(int(selected_lag), 1)
        truncation = min(truncation, lag_max)

    weighted_lag_sum = 0.0
    long_run_variance = float(autocovariance[0])

    for lag in range(1, truncation + 1):
        ratio = lag / truncation
        weight = 1.0 if ratio <= 0.5 else 2.0 * (1.0 - ratio)
        weighted_lag_sum += 2.0 * weight * lag * autocovariance[lag]
        long_run_variance += 2.0 * weight * autocovariance[lag]

    denominator = 2.0 * long_run_variance**2
    if denominator <= _MIN_DENOMINATOR:
        return 1.0

    numerator = 2.0 * weighted_lag_sum**2
    block_length = (numerator / denominator) ** (1.0 / 3.0)
    block_length *= n_obs ** (1.0 / 3.0)

    if not np.isfinite(block_length):
        return 1.0

    return float(np.clip(block_length, 1.0, block_max))


def aggregate_stationary_block_lengths(
        estimates: np.ndarray,
        quantile: float = COMMON_BLOCK_LENGTH_QUANTILE) -> float:
    """Combine per-series block lengths into one shared robust length.

    A common length is needed because all stochastic series use the same
    historical index sequence. The upper quartile gives persistent series
    adequate representation without simply adopting the maximum estimate.
    """

    values = np.asarray(estimates, dtype=float).reshape(-1)
    valid = values[np.isfinite(values) & (values >= 1.0)]
    if valid.size == 0:
        return 1.0

    quantile = float(quantile)
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("Block-length quantile must be between 0 and 1.")

    return max(float(np.quantile(valid, quantile)), 1.0)


def common_stationary_block_length(
        values: np.ndarray,
        observation_step_days: float = 1.0) -> tuple[float, np.ndarray]:
    """Estimate per-series lengths and their shared multivariate length."""

    matrix = np.asarray(values, dtype=float)
    if matrix.ndim == 1:
        matrix = matrix[:, None]
    if matrix.ndim != 2:
        raise ValueError("Residual history must be one- or two-dimensional.")

    estimates = np.asarray([
        optimal_stationary_block_length(
            matrix[:, column],
            observation_step_days=observation_step_days)
        for column in range(matrix.shape[1])
    ], dtype=float)
    common = aggregate_stationary_block_lengths(estimates)
    return common, estimates


def annual_variability_scales(
        values: np.ndarray,
        history_dates: np.ndarray) -> np.ndarray:
    """Return within-year residual standard deviations for complete years.

    The returned rows are historical annual variability regimes.  They are
    absolute residual scales, not multiplicative factors around a pooled
    standard deviation.  This distinction matters because the stationary
    bootstrap already samples residuals with their historical amplitude.
    Applying an additional scale factor to those raw shocks would count the
    same interannual variation twice and can systematically shrink or inflate
    the simulated variability.

    Partial calendar years are excluded.  If fewer than two complete years
    are available, a single all-one row is returned and annual scale
    modulation is disabled by :func:`apply_annual_variability`.
    """

    matrix = np.asarray(values, dtype=float)
    if matrix.ndim == 1:
        matrix = matrix[:, None]
    if matrix.ndim != 2 or matrix.shape[0] == 0:
        raise ValueError(
            "Residual history must be a non-empty one- or two-dimensional "
            "array.")
    if not np.isfinite(matrix).all():
        raise ValueError(
            "Residual history used for annual variability must be finite.")

    dates = np.asarray(history_dates, dtype="datetime64[ns]")
    if dates.size != matrix.shape[0]:
        raise ValueError(
            "Residual values and residual dates must have equal length.")

    date_values = np.sort(np.unique(dates.astype("int64")))
    differences = np.diff(date_values)
    positive = differences[differences > 0]
    if positive.size == 0:
        return np.ones((1, matrix.shape[1]), dtype=float)
    step = float(np.median(positive))

    years = dates.astype("datetime64[Y]").astype("int64") + 1970
    complete = []
    for year in np.unique(years):
        start = np.datetime64(f"{int(year):04d}-01-01", "ns")
        end = np.datetime64(f"{int(year) + 1:04d}-01-01", "ns")
        expected = int(round(
            (end.astype("int64") - start.astype("int64")) / step))
        observed = int(np.count_nonzero(years == year))
        if expected > 0 and observed >= _COMPLETE_YEAR_FRACTION * expected:
            complete.append(int(year))

    if len(complete) < 2:
        return np.ones((1, matrix.shape[1]), dtype=float)

    scales = np.ones((len(complete), matrix.shape[1]), dtype=float)
    for row, year in enumerate(complete):
        annual = matrix[years == year]
        annual_scale = np.std(annual, axis=0, ddof=1)
        valid = np.isfinite(annual_scale) & (annual_scale > _MIN_VARIANCE)
        scales[row, valid] = annual_scale[valid]

    scales[~np.isfinite(scales)] = 1.0
    return np.maximum(scales, 0.0)



def apply_annual_variability(
        shocks: np.ndarray,
        target_dates: np.ndarray,
        scales: np.ndarray,
        rng: np.random.Generator) -> np.ndarray:
    """Match each simulated year's residual spread to a historical regime.

    The bootstrap sequence is first sampled normally.  For each target year,
    one complete historical scale vector is selected jointly across sources.
    The sampled shocks are then rescaled around their own annual mean so their
    within-year standard deviation matches that historical regime.  Keeping
    the sampled mean unchanged separates annual variability in *scale* from
    any annual residual-level anomaly already present in the bootstrap path.

    This avoids multiplying raw historical shocks by a second variability
    factor, which would double-count their historical amplitude.
    """

    values = np.asarray(shocks, dtype=float)
    if values.ndim == 1:
        values = values[:, None]
    if values.ndim != 2:
        raise ValueError(
            "Bootstrap shocks must be one- or two-dimensional.")

    scale_matrix = np.asarray(scales, dtype=float)
    if scale_matrix.ndim == 1:
        scale_matrix = scale_matrix[None, :]
    if scale_matrix.ndim != 2:
        raise ValueError(
            "Annual variability scales must be one- or two-dimensional.")
    if scale_matrix.shape[1] != values.shape[1]:
        raise ValueError(
            "Annual variability scales must match bootstrap source count.")
    if scale_matrix.shape[0] <= 1:
        return values.copy()

    dates = np.asarray(target_dates, dtype="datetime64[ns]")
    if dates.size != values.shape[0]:
        raise ValueError(
            "Bootstrap shocks and target dates must have equal length.")

    years = dates.astype("datetime64[Y]").astype("int64") + 1970
    output = values.copy()
    for year in np.unique(years):
        mask = years == year
        annual = output[mask]
        if annual.shape[0] < 2:
            continue
        regime = int(rng.integers(scale_matrix.shape[0]))
        desired = scale_matrix[regime]
        mean = np.mean(annual, axis=0)
        centered = annual - mean
        current = np.std(centered, axis=0, ddof=1)
        valid = (
            np.isfinite(current) & (current > _MIN_VARIANCE)
            & np.isfinite(desired) & (desired > _MIN_VARIANCE))
        adjusted = annual.copy()
        adjusted[:, valid] = (
            mean[valid]
            + centered[:, valid] * (desired[valid] / current[valid]))
        output[mask] = adjusted
    return output


def _continuation_mask(
        history_dates: np.ndarray | None,
        history_size: int) -> np.ndarray:
    """Mark historical rows that directly follow the preceding row."""

    valid = np.ones(history_size, dtype=bool)
    valid[0] = False
    if history_dates is None:
        return valid

    dates = np.asarray(history_dates, dtype="datetime64[ns]")
    if dates.size != history_size:
        raise ValueError(
            "Residual history dates and months must have equal length.")

    differences = np.diff(dates).astype("timedelta64[ns]")
    positive = differences[differences > np.timedelta64(0, "ns")]
    if positive.size == 0:
        return valid

    expected = int(np.median(positive.astype("int64")))
    tolerance = max(int(expected * 0.05), 1)
    actual = differences.astype("int64")
    valid[1:] = np.abs(actual - expected) <= tolerance
    return valid


def _hourly_restart_candidates(
        history_dates: np.ndarray | None,
        history_months: np.ndarray
) -> tuple[bool, dict[tuple[int, int], np.ndarray]]:
    """Build month-hour restart pools when the history is hourly."""

    if history_dates is None:
        return False, {}

    dates = np.asarray(history_dates, dtype="datetime64[ns]")
    if dates.size != history_months.size:
        raise ValueError(
            "Residual history dates and months must have equal length.")

    differences = np.diff(dates).astype("timedelta64[m]")
    positive = differences[differences > np.timedelta64(0, "m")]
    is_hourly = bool(
        positive.size and np.median(positive.astype("int64")) <= 120)
    if not is_hourly:
        return False, {}

    hours = dates.astype("datetime64[h]").astype("int64") % 24
    pools: dict[tuple[int, int], list[int]] = {}
    for position, month in enumerate(history_months):
        key = int(month), int(hours[position])
        pools.setdefault(key, []).append(position)

    candidates = {
        key: np.asarray(positions, dtype=int)
        for key, positions in pools.items()}
    return True, candidates


def _daily_restart_candidates(
        history_dates: np.ndarray | None,
        history_months: np.ndarray
) -> tuple[np.ndarray | None, dict[tuple[int, int], np.ndarray]]:
    """Build month-weekday restart pools when dates are available."""

    if history_dates is None:
        return None, {}

    history_days = np.asarray(history_dates, dtype="datetime64[D]")
    history_weekdays = (history_days.astype("int64") + 3) % 7
    pools = {}

    for month in range(1, 13):
        for weekday in range(7):
            mask = (
                (history_months == month)
                & (history_weekdays == weekday))
            pools[(month, weekday)] = np.flatnonzero(mask)

    return history_weekdays, pools


def _target_weekdays(
        target_dates: np.ndarray | None,
        target_size: int) -> np.ndarray | None:
    """Return Monday-zero weekday numbers for target dates."""

    if target_dates is None:
        return None

    dates = np.asarray(target_dates, dtype="datetime64[D]")
    if dates.size != target_size:
        raise ValueError(
            "Target dates and months must have equal length.")
    return (dates.astype("int64") + 3) % 7


def _restart_jump_limits(
        history_values: np.ndarray | None,
        continuation: np.ndarray,
        history_size: int
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Return empirical one-step limits for bootstrap block joins.

    The limits are calculated from observed consecutive residual changes.
    They are used only when two sampled blocks are joined, so extreme events
    already present inside historical blocks remain untouched.
    """

    if history_values is None:
        return None, None

    values = np.asarray(history_values, dtype=float)
    if values.ndim == 1:
        values = values[:, None]
    if values.ndim != 2 or values.shape[0] != history_size:
        raise ValueError(
            "Residual values and residual dates must have equal length.")
    if not np.isfinite(values).all():
        raise ValueError(
            "Residual values used for bootstrap joins must be finite.")

    changes = np.abs(np.diff(values, axis=0))
    observed = changes[continuation[1:]]
    if observed.size == 0:
        return values, None

    limits = np.quantile(
        observed, _RESTART_JUMP_QUANTILE, axis=0)
    return values, np.asarray(limits, dtype=float)


def _compatible_restart_pool(
        pool: np.ndarray,
        previous: int,
        history_values: np.ndarray | None,
        jump_limits: np.ndarray | None) -> np.ndarray:
    """Prefer restart positions whose seam resembles observed changes."""

    if (
            history_values is None
            or jump_limits is None
            or pool.size == 0):
        return pool

    changes = np.abs(history_values[pool] - history_values[previous])
    compatible = np.all(
        changes <= jump_limits + _MIN_VARIANCE, axis=1)
    if np.any(compatible):
        return pool[compatible]

    scales = np.maximum(jump_limits, _MIN_VARIANCE)
    score = np.max(changes / scales, axis=1)
    best = float(np.min(score))
    return pool[np.isclose(score, best)]


def seasonal_stationary_indices(
        history_months: np.ndarray,
        target_months: np.ndarray,
        mean_block_length: float,
        rng: np.random.Generator,
        history_dates: np.ndarray | None = None,
        target_dates: np.ndarray | None = None,
        history_values: np.ndarray | None = None) -> np.ndarray:
    """Draw one calendar-compatible stationary-bootstrap index sequence.

    Restarts use month and weekday for daily histories and month and hour for
    hourly histories. Once a block starts, consecutive historical rows are
    retained until the geometric restart rule fires or the historical time
    axis is discontinuous. When residual values are supplied, a restart also
    avoids artificial joins beyond the empirical 99.5th percentile of
    observed one-step residual changes. Values inside blocks are untouched.
    The returned indices can be applied to every stochastic series to
    preserve their observed joint behavior.
    """

    history_months = np.asarray(history_months, dtype=int)
    target_months = np.asarray(target_months, dtype=int)
    history_size = int(history_months.size)
    target_size = int(target_months.size)

    if history_size == 0:
        raise ValueError("Stationary bootstrap requires residual history.")
    if not np.isfinite(mean_block_length) or mean_block_length <= 0.0:
        raise ValueError("Mean block length must be finite and positive.")

    target_dates_array = None
    if target_dates is not None:
        target_dates_array = np.asarray(
            target_dates, dtype="datetime64[ns]")
        if target_dates_array.size != target_size:
            raise ValueError(
                "Target dates and months must have equal length.")

    probability = 1.0 / max(float(mean_block_length), 1.0)
    all_positions = np.arange(history_size, dtype=int)
    monthly_candidates = {
        month: np.flatnonzero(history_months == month)
        for month in range(1, 13)}
    continuation = _continuation_mask(history_dates, history_size)
    restart_values, jump_limits = _restart_jump_limits(
        history_values, continuation, history_size)

    is_hourly, hourly_candidates = _hourly_restart_candidates(
        history_dates, history_months)
    _, daily_candidates = _daily_restart_candidates(
        history_dates, history_months)
    target_weekdays = _target_weekdays(target_dates, target_size)

    history_hours = None
    target_hours = None
    if (is_hourly and history_dates is not None
            and target_dates_array is not None):
        history_hours = (
            np.asarray(history_dates, dtype="datetime64[h]")
            .astype("int64") % 24)
        target_hours = (
            target_dates_array.astype("datetime64[h]")
            .astype("int64") % 24)

    output = np.empty(target_size, dtype=int)
    previous = 0

    for row, month_value in enumerate(target_months):
        next_position = previous + 1
        can_continue = (
            row > 0
            and next_position < history_size
            and continuation[next_position])

        if can_continue and history_hours is not None:
            can_continue = (
                history_hours[next_position] == target_hours[row])

        restart = not can_continue or rng.random() < probability
        if restart:
            pool = None

            if is_hourly and target_hours is not None:
                key = int(month_value), int(target_hours[row])
                pool = hourly_candidates.get(key)
            elif target_weekdays is not None:
                key = int(month_value), int(target_weekdays[row])
                pool = daily_candidates.get(key)

            if pool is None or pool.size == 0:
                pool = monthly_candidates.get(int(month_value))
            if pool is None or pool.size == 0:
                pool = all_positions
            if row > 0:
                pool = _compatible_restart_pool(
                    pool, previous, restart_values, jump_limits)

            previous = int(pool[rng.integers(pool.size)])
        else:
            previous = next_position

        output[row] = previous

    return output
