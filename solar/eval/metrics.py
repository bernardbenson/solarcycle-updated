"""
Raw-unit forecast metrics.

Everything here is computed in physical target units (sunspot number / area)
AFTER inverse-transforming model output - never in the scaled training space.
Peak metrics are computed on the standard 13-month smoothed series, the scale
solar-cycle amplitudes are quoted on.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from ..data.monthly import smooth_13m


def forecast_metrics(actual: np.ndarray, pred: np.ndarray,
                     q10: Optional[np.ndarray] = None,
                     q90: Optional[np.ndarray] = None) -> Dict[str, float]:
    """Metrics for one forecast window (all arrays shape (horizon,), raw units).

    Returns RMSE/MAE on the raw monthly series, peak amplitude/timing errors on
    the 13-month smoothed series (pred - actual, so positive = over-prediction),
    and 80% interval coverage/width when q10/q90 are given.
    """
    actual = np.asarray(actual, dtype=float)
    pred = np.asarray(pred, dtype=float)

    out: Dict[str, float] = {
        'rmse': float(np.sqrt(np.mean((pred - actual) ** 2))),
        'mae': float(np.mean(np.abs(pred - actual))),
    }

    actual_s, pred_s = smooth_13m(actual), smooth_13m(pred)
    out['peak_amp_err'] = float(np.max(pred_s) - np.max(actual_s))
    out['peak_timing_err'] = float(int(np.argmax(pred_s)) - int(np.argmax(actual_s)))
    out['rmse_smoothed'] = float(np.sqrt(np.mean((pred_s - actual_s) ** 2)))

    if q10 is not None and q90 is not None:
        q10 = np.asarray(q10, dtype=float)
        q90 = np.asarray(q90, dtype=float)
        inside = (actual >= q10) & (actual <= q90)
        out['coverage_80'] = float(np.mean(inside))
        out['interval_width'] = float(np.mean(q90 - q10))

    return out


def aggregate_metrics(per_window: List[Dict[str, float]]) -> Dict[str, float]:
    """Mean of each metric across windows, plus MAE-style absolute peak errors."""
    if not per_window:
        return {}
    keys = set().union(*(m.keys() for m in per_window))
    agg = {k: float(np.mean([m[k] for m in per_window if k in m])) for k in sorted(keys)}
    agg['peak_amp_mae'] = float(np.mean([abs(m['peak_amp_err'])
                                         for m in per_window if 'peak_amp_err' in m]))
    agg['peak_timing_mae'] = float(np.mean([abs(m['peak_timing_err'])
                                            for m in per_window if 'peak_timing_err' in m]))
    return agg
