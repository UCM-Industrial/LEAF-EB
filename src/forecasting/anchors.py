"""Anchor forecast projections to recent historical observations."""

import numpy as np
import pandas as pd


def _automatic_anchor(df: pd.DataFrame) -> float:
    """Return the mean level over the latest annual data cycle."""

    values = pd.to_numeric(df["y"], errors="coerce")
    finite = values.notna()
    if not finite.any():
        return np.nan

    if "ds" not in df.columns:
        return float(values[finite].mean())

    dates = pd.to_datetime(df["ds"], errors="coerce")
    valid = finite & dates.notna()
    if not valid.any():
        return float(values[finite].mean())

    end = dates[valid].max()
    start = end - pd.DateOffset(years=1)
    recent = values[valid & dates.gt(start) & dates.le(end)]
    if recent.empty:
        recent = values[valid]
    return float(recent.mean())


def get_anchor_value(
        df: pd.DataFrame,
        method: str = "auto",
        window: int = 30,
) -> float:
    """Return one historical level used to start a forecast path."""

    if df is None or df.empty or "y" not in df.columns:
        return np.nan

    normalized_method = str(method or "auto").strip().lower()
    if normalized_method == "auto":
        return _automatic_anchor(df)

    values = pd.to_numeric(df["y"], errors="coerce").dropna()
    if values.empty:
        return np.nan
    if normalized_method == "last":
        return float(values.iloc[-1])
    if normalized_method == "mean":
        return float(values.tail(window).mean())
    if normalized_method == "median":
        return float(values.tail(window).median())
    if normalized_method == "max":
        return float(values.tail(window).max())
    raise ValueError(f"Unknown anchor method: {method}")


def apply_anchor(
        df_forecast: pd.DataFrame,
        df_hist: pd.DataFrame,
        year: int | None = None,
        method: str = "auto",
        window: int = 30,
) -> pd.DataFrame:
    """Adjust a forecast path to its historical starting level."""

    if (
        df_hist is None or
        len(df_hist) == 0 or
        df_forecast is None or
        len(df_forecast) == 0
    ):
        return df_forecast

    anchor = get_anchor_value(
        df_hist,
        method=method,
        window=window,)
    if not np.isfinite(anchor):
        return df_forecast

    if year is None:
        first_idx = df_forecast.index[0]
        y0 = float(df_forecast.loc[first_idx, "y"])
        if np.isfinite(y0) and abs(y0) > 1e-12:
            df_forecast["y"] *= anchor / y0
        return df_forecast

    cut = pd.Timestamp(year=year, month=1, day=1)
    mask = df_forecast["ds"] < cut
    if not mask.any():
        return df_forecast

    idxs = df_forecast.index[mask]
    yseg = df_forecast.loc[idxs, "y"].to_numpy(dtype=float)
    y0 = yseg[0] if len(yseg) > 0 else np.nan
    if not np.isfinite(y0) or abs(y0) < 1e-12:
        return df_forecast

    factor0 = anchor / y0
    scales = np.linspace(factor0, 1.0, len(yseg))
    df_forecast.loc[idxs, "y"] = yseg * scales
    return df_forecast
