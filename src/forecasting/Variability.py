"""Persist residual histories and fallback covariance information.

The current LEAF-EB stochastic workflow samples empirical relative residuals
with a multivariate stationary bootstrap. ``PatternGenerator`` computes the
residuals; this module stores those histories, estimates automatic block
lengths, and writes Gaussian moments only as a fallback for runs where an
empirical residual history is unavailable.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.utilities.bootstrap_tools import common_stationary_block_length


_MIN_SCALE = 1e-12
_MIN_FIT_OBSERVATIONS = 10


def lag1_autocorrelation(data: pd.Series) -> float:
    """Return lag-1 correlation without joining gaps in the time axis."""

    series = pd.to_numeric(data, errors="coerce").dropna()
    if len(series) < 3 or series.nunique() < 2:
        return 0.0

    previous = series.shift(1)
    valid = previous.notna()

    if isinstance(series.index, pd.DatetimeIndex):
        differences = series.index.to_series().diff()
        positive = differences[differences > pd.Timedelta(0)]
        if not positive.empty:
            expected_step = positive.median()
            valid &= differences <= 1.5 * expected_step

    current_values = series.loc[valid]
    previous_values = previous.loc[valid]
    if len(current_values) < 3:
        return 0.0

    correlation = current_values.corr(previous_values)
    if not np.isfinite(correlation):
        return 0.0
    return float(np.clip(correlation, -0.98, 0.98))


def _fit_gaussian_fallback(data: pd.Series) -> dict[str, float] | None:
    """Estimate Gaussian fallback location and scale for one residual."""

    values = pd.to_numeric(
        data, errors="coerce"
    ).dropna().to_numpy(dtype=float)
    if values.size < _MIN_FIT_OBSERVATIONS:
        return None

    location = float(np.mean(values))
    scale = float(np.std(values, ddof=1))
    if not np.isfinite(location) or not np.isfinite(scale):
        return None
    return {"loc": location, "scale": max(scale, _MIN_SCALE)}


def _distribution_row(
        data: pd.Series,
        mean_value: float) -> dict[str, float] | None:
    """Build fallback distribution parameters and simple diagnostics."""

    clean = pd.to_numeric(data, errors="coerce").dropna()
    if len(clean) < _MIN_FIT_OBSERVATIONS:
        return None

    fitted = _fit_gaussian_fallback(clean)
    if fitted is None:
        return None

    denominator = mean_value if abs(mean_value) > _MIN_SCALE else 1.0
    row = {
        "original_mean": mean_value,
        "autocorr_lag1": lag1_autocorrelation(clean),
        "q05_rel": float(clean.quantile(0.05) / denominator),
        "q95_rel": float(clean.quantile(0.95) / denominator)}
    row.update(fitted)
    return row


def _write_period_parameters(
        residuals: pd.DataFrame,
        mean_values: dict[str, float],
        covariance_writer: pd.ExcelWriter,
        distribution_writer: pd.ExcelWriter,
        sheet_name: str) -> None:
    """Write covariance and fallback distribution data for one period."""

    if residuals.empty:
        return

    residuals.cov().to_excel(
        covariance_writer, sheet_name=sheet_name)
    rows = []
    for column in residuals.columns:
        row = _distribution_row(
            residuals[column],
            float(mean_values.get(column, 1.0)),)
        if row is None:
            continue
        row["column"] = column
        rows.append(row)

    if rows:
        pd.DataFrame(rows).to_excel(
            distribution_writer,
            index=False,
            sheet_name=sheet_name,)


def _save_fallback_parameters(
        residuals: pd.DataFrame,
        mean_values: dict[str, float],
        output_dir: Path,
        variability_mode: str) -> None:
    """Write monthly or global covariance and Gaussian fallback files."""

    distribution_path = output_dir / "Residual_distribution.xlsx"
    covariance_path = output_dir / "Cov_matrix.xlsx"
    monthly = "m" in variability_mode or "monthly" in variability_mode

    with pd.ExcelWriter(
            covariance_path, engine="openpyxl"
    ) as covariance_writer, pd.ExcelWriter(
            distribution_path, engine="openpyxl"
    ) as distribution_writer:
        if monthly:
            for month in range(1, 13):
                subset = residuals.loc[residuals.index.month == month]
                _write_period_parameters(
                    subset,
                    mean_values,
                    covariance_writer,
                    distribution_writer,
                    f"M{month:02d}",)
        else:
            _write_period_parameters(
                residuals,
                mean_values,
                covariance_writer,
                distribution_writer,
                "Global",)

    del monthly


def _global_relative_history(
        residuals: pd.DataFrame,
        mean_values: dict[str, float]) -> pd.DataFrame:
    """Convert absolute residuals to global-relative fallback residuals."""

    relative = pd.DataFrame(index=residuals.index)
    for column in residuals.columns:
        original_mean = float(mean_values.get(column, 0.0))
        if abs(original_mean) <= _MIN_SCALE:
            continue
        relative[column] = residuals[column] / abs(original_mean)
    return relative


def _prepare_local_relative_history(
        local_relative: dict[str, pd.Series] | None,
        reference: pd.DataFrame) -> pd.DataFrame | None:
    """Align local-relative residuals to the stored residual history."""

    if local_relative is None:
        return None

    frame = pd.DataFrame(local_relative).reindex(reference.index)
    available = [
        column for column in reference.columns
        if column in frame.columns]
    if not available:
        return None

    frame = frame[available]
    frame = frame.replace([np.inf, -np.inf], np.nan)
    if frame.isna().any().any():
        missing = frame.isna().sum()
        details = ", ".join(
            f"{column}={int(count)}"
            for column, count in missing.items()
            if count > 0)
        raise ValueError(
            "Local-relative residual history contains missing values: "
            f"{details}.")
    return frame


def _save_residual_history(
        residuals: pd.DataFrame,
        mean_values: dict[str, float],
        local_relative: dict[str, pd.Series] | None,
        output_dir: Path) -> None:
    """Save empirical residual histories and stationary block metadata."""

    global_relative = _global_relative_history(
        residuals, mean_values)
    if global_relative.empty:
        raise ValueError(
            "No residual series has a non-zero historical mean.")

    local_frame = _prepare_local_relative_history(
        local_relative, global_relative)
    bootstrap_frame = (
        local_frame if local_frame is not None else global_relative)
    if isinstance(bootstrap_frame.index, pd.DatetimeIndex):
        differences = bootstrap_frame.index.to_series().diff().dropna()
        positive = differences[differences > pd.Timedelta(0)]
        if positive.empty:
            observation_step_days = 1.0
        else:
            observation_step_days = (
                positive.median().total_seconds() / 86400.0)
    else:
        observation_step_days = 1.0

    common, estimates = common_stationary_block_length(
        bootstrap_frame.to_numpy(dtype=float),
        observation_step_days=observation_step_days)

    metadata = pd.DataFrame({
        "column": bootstrap_frame.columns,
        "stationary_block_length": estimates,
        "common_block_length": common,
        "observation_step_days": observation_step_days,
    })

    date_name = residuals.index.name or "Date"
    output_path = output_dir / "Residual_history.xlsx"
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        global_history = global_relative.reset_index()
        global_history = global_history.rename(
            columns={global_history.columns[0]: date_name})
        global_history.to_excel(
            writer, index=False, sheet_name="FastRelative")

        if local_frame is not None:
            local_history = local_frame.reset_index()
            local_history = local_history.rename(
                columns={local_history.columns[0]: date_name})
            local_history.to_excel(
                writer, index=False, sheet_name="LocalRelative")

        metadata.to_excel(
            writer, index=False, sheet_name="Bootstrap")



def process_pre_calculated_residuals(
        combined_resid: dict[str, pd.Series],
        mean_values: dict[str, float],
        output_dir: str,
        local_relative: dict[str, pd.Series] | None = None,
        variability_mode: str = "global") -> pd.DataFrame:
    """Persist residuals calculated by ``PatternGenerator``.

    Parameters
    ----------
    combined_resid:
        Absolute residual series keyed by source or demand name.
    mean_values:
        Historical means used only by the Gaussian fallback and the
        global-relative fallback residual sheet.
    output_dir:
        Scenario directory where residual and covariance workbooks are saved.
    local_relative:
        Preferred residual history normalized by each time step's fitted local
        level. When supplied, this is the history used by the stationary
        bootstrap.
    variability_mode:
        ``"M"``/``"monthly"`` writes monthly fallback parameter sheets;
        other values write one global sheet.
    Returns
    -------
    pandas.DataFrame
        Aligned absolute residual history.
    """

    residuals = pd.DataFrame(combined_resid)
    if residuals.empty:
        raise ValueError("No residuals were provided for variability analysis.")

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    mode = str(variability_mode).strip().lower()

    _save_residual_history(
        residuals,
        mean_values,
        local_relative,
        destination,)
    _save_fallback_parameters(
        residuals,
        mean_values,
        destination,
        mode,)
    return residuals
