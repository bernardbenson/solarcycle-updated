"""
Solar-cycle precursors.

Currently provides the terminator/cycle-length precursor: the length of the most
recently completed solar cycle (minimum-to-minimum), which anti-correlates with the
amplitude of the following cycle. This is a robust proxy for the Hale-cycle
"terminator separation" of McIntosh et al. (2023), derivable from the sunspot
record alone (a Hilbert-transform terminator identification is a possible future
refinement). Cycle minima are detected the same way as the hindcast backtest, so
the two share one implementation.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks


def detect_cycle_minima(raw_monthly: np.ndarray, horizon: int) -> np.ndarray:
    """Indices of solar-cycle minima in a monthly series.

    Troughs of the smoothed series, spaced at least ~0.7 of a horizon apart so at
    most one is found per ~11-year cycle.
    """
    smooth = uniform_filter1d(np.asarray(raw_monthly, dtype=float), size=24)
    troughs, _ = find_peaks(-smooth, distance=int(horizon * 0.7))
    return troughs


def cycle_length_series(raw_monthly: np.ndarray, horizon: int,
                        default: float = 132.0) -> np.ndarray:
    """Per-month length (in months) of the most recently completed cycle.

    Step-holds the min-to-min length from each detected minimum until the next.
    Months before the second detected minimum (no completed cycle yet) get
    ``default``. Returns a raw, unscaled array of shape (T,).
    """
    raw_monthly = np.asarray(raw_monthly, dtype=float)
    out = np.full(len(raw_monthly), float(default))
    troughs = detect_cycle_minima(raw_monthly, horizon)
    for start, end in zip(troughs[:-1], troughs[1:]):
        out[end:] = float(end - start)
    return out
