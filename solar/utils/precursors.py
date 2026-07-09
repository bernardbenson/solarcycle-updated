"""
Solar-cycle precursors.

Provides the terminator/cycle-length precursor: the length of the most recently
completed solar cycle (minimum-to-minimum), which anti-correlates with the
amplitude of the following cycle. This is a robust proxy for the Hale-cycle
"terminator separation" of McIntosh et al. (2023), derivable from the sunspot
record alone.

Two minima detectors exist:

- :func:`causal_cycle_minima` - strictly causal (trailing smoothing + a
  confirmation delay). A minimum at month ``m`` is only *known* at month
  ``m + confirm_months``. Use this for anything that feeds model inputs or
  conditioning (no look-ahead).
- :func:`detect_cycle_minima` - hindsight version (centered smoothing over the
  full series). Use only for defining evaluation folds, backtest panel origins,
  and plots, where knowing where past cycles started is legitimate.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks


def detect_cycle_minima(raw_monthly: np.ndarray, horizon: int) -> np.ndarray:
    """Indices of solar-cycle minima in a monthly series (HINDSIGHT - see module doc).

    Troughs of the centrally-smoothed series, spaced at least ~0.7 of a horizon
    apart so at most one is found per ~11-year cycle. Uses future data around each
    trough; do not use for causal features.
    """
    smooth = uniform_filter1d(np.asarray(raw_monthly, dtype=float), size=24)
    troughs, _ = find_peaks(-smooth, distance=int(horizon * 0.7))
    return troughs


def trailing_smooth(raw_monthly: np.ndarray, window: int = 13) -> np.ndarray:
    """Trailing (one-sided) moving average: value at t uses months (t-window, t]."""
    raw_monthly = np.asarray(raw_monthly, dtype=float)
    out = np.empty_like(raw_monthly)
    csum = np.cumsum(np.insert(raw_monthly, 0, 0.0))
    for t in range(len(raw_monthly)):
        lo = max(0, t - window + 1)
        out[t] = (csum[t + 1] - csum[lo]) / (t + 1 - lo)
    return out


def causal_cycle_minima(raw_monthly: np.ndarray,
                        confirm_months: int = 18,
                        min_spacing: int = 80,
                        smooth_window: int = 13) -> Tuple[np.ndarray, np.ndarray]:
    """Strictly causal cycle-minima detection.

    Walks the trailing-smoothed series keeping a running candidate minimum; the
    candidate at month ``m`` is confirmed once ``confirm_months`` pass with no
    lower smoothed value (and it is at least ``min_spacing`` months after the
    previous confirmed minimum). Everything at month ``t`` depends only on
    months ``<= t``, so features built from these minima have no look-ahead.

    Returns ``(minima_idx, confirmed_at_idx)`` - the minimum's position and the
    month at which it became known.
    """
    smooth = trailing_smooth(raw_monthly, smooth_window)
    minima, confirmed_at = [], []

    last_confirmed = -min_spacing  # allow a minimum near the series start
    cand_idx, cand_val = None, np.inf

    for t in range(len(smooth)):
        if cand_idx is None or smooth[t] < cand_val:
            if t - last_confirmed >= 1:  # candidates only after the previous minimum
                cand_idx, cand_val = t, smooth[t]
            continue
        if cand_idx is not None and (t - cand_idx) >= confirm_months \
                and (cand_idx - last_confirmed) >= min_spacing:
            minima.append(cand_idx)
            confirmed_at.append(t)
            last_confirmed = cand_idx
            cand_idx, cand_val = None, np.inf

    return np.asarray(minima, dtype=int), np.asarray(confirmed_at, dtype=int)


def cycle_length_series(raw_monthly: np.ndarray, horizon: int,
                        default: float = 132.0) -> np.ndarray:
    """Per-month length (in months) of the most recently completed cycle (CAUSAL).

    The min-to-min length of a cycle only enters the series from the month its
    closing minimum is *confirmed* (not from the minimum itself), so the value
    at month t is computable from data up to t. Months before the first
    confirmed completed cycle get ``default``. Returns raw, unscaled (T,).
    """
    raw_monthly = np.asarray(raw_monthly, dtype=float)
    out = np.full(len(raw_monthly), float(default))
    minima, confirmed_at = causal_cycle_minima(raw_monthly)
    for prev, cur, known_at in zip(minima[:-1], minima[1:], confirmed_at[1:]):
        out[known_at:] = float(cur - prev)
    return out


def cycle_length_series_hindsight(raw_monthly: np.ndarray, horizon: int,
                                  default: float = 132.0) -> np.ndarray:
    """Legacy non-causal cycle-length series (kept for plots/diagnostics only)."""
    raw_monthly = np.asarray(raw_monthly, dtype=float)
    out = np.full(len(raw_monthly), float(default))
    troughs = detect_cycle_minima(raw_monthly, horizon)
    for start, end in zip(troughs[:-1], troughs[1:]):
        out[end:] = float(end - start)
    return out
