"""
Single source of truth for building the monthly series the models consume.

SILSO daily files (and the Hathaway/Upton sunspot-area file) encode missing days
as ``-1.0``. Those sentinels must be masked to NaN at the *native (daily)
resolution*, before any monthly aggregation — averaging them into monthly means
drags early-record months toward or below zero. Every data path (univariate,
multivariate, plotting, backtesting) goes through :func:`build_monthly_series`
so the masking happens exactly once, in one place.
"""

from __future__ import annotations

from typing import Iterable, Optional, Tuple

import numpy as np
import pandas as pd

# Physically non-negative columns where a negative value can only be a sentinel.
NON_NEGATIVE_COLS = ('sunspot_number', 'sunspot_area', 'std_dev', 'ap_avg', 'kp_sum')


def mask_sentinels(df: pd.DataFrame, columns: Optional[Iterable[str]] = None) -> pd.DataFrame:
    """Return a copy with negative sentinel values replaced by NaN.

    Applies to ``columns`` if given, else to every known non-negative column
    present in the frame (sunspot number/area, std dev, geomagnetic indices).
    """
    df = df.copy()
    cols = list(columns) if columns is not None else [
        c for c in df.columns if c in NON_NEGATIVE_COLS or c.startswith('f107')
    ]
    for col in cols:
        values = pd.to_numeric(df[col], errors='coerce')
        df[col] = values.where(values >= 0.0)
    return df


def build_monthly_series(df: pd.DataFrame, target_col: str = 'sunspot_number',
                         start_year: int = 1749,
                         interpolate_limit: int = 3) -> pd.Series:
    """Monthly-mean target series with sentinels masked at native resolution.

    Works for daily or already-monthly input (monthly input passes through
    ``resample('ME').mean()`` unchanged). Short gaps (<= ``interpolate_limit``
    months) are linearly interpolated; remaining edge NaNs are dropped.
    """
    d = df.copy()
    d['date'] = pd.to_datetime(d['date'])
    d = d.set_index('date').sort_index()
    d = d[d.index.year >= start_year]

    d = mask_sentinels(d, columns=[target_col])
    monthly = d[target_col].resample('ME').mean()
    monthly = monthly.interpolate('linear', limit=interpolate_limit)
    monthly = monthly.dropna()

    if (monthly < 0).any():
        raise ValueError(
            f"Negative monthly means remain in '{target_col}' after sentinel masking - "
            "the input data contains non-sentinel negative values."
        )
    return monthly


def build_monthly_frame(df: pd.DataFrame, target_col: str = 'sunspot_number',
                        precursor_cols: Iterable[str] = (),
                        start_year: int = 1749) -> Tuple[pd.Series, pd.DataFrame]:
    """Monthly target plus monthly precursor channels aligned to the target index.

    Returns ``(target: Series, precursors: DataFrame)``; precursor months with no
    data stay NaN (the caller decides fill/mask semantics).
    """
    target = build_monthly_series(df, target_col=target_col, start_year=start_year)

    precursor_cols = list(precursor_cols)
    if not precursor_cols:
        return target, pd.DataFrame(index=target.index)

    d = df.copy()
    d['date'] = pd.to_datetime(d['date'])
    d = d.set_index('date').sort_index()
    d = d[d.index.year >= start_year]
    d = mask_sentinels(d, columns=precursor_cols)

    precursors = pd.DataFrame(index=target.index)
    for col in precursor_cols:
        precursors[col] = d[col].resample('ME').mean().reindex(target.index)
    return target, precursors


def smooth_13m(values: np.ndarray) -> np.ndarray:
    """Standard SIDC 13-month smoothing (12-month boxcar, half-weight endpoints).

    Centered filter: uses +-6 months around each point, so the first/last 6
    entries are computed from a truncated (renormalized) kernel. Intended for
    *evaluation and reporting* (the standard smoothed-SSN scale) - do not feed
    the smoothed series back into model inputs, where the centered window would
    leak future observations.
    """
    values = np.asarray(values, dtype=float)
    weights = np.ones(13)
    weights[0] = weights[-1] = 0.5
    kernel = weights / weights.sum()

    padded = np.pad(values, 6, mode='edge')
    smoothed = np.convolve(padded, kernel, mode='valid')
    return smoothed
