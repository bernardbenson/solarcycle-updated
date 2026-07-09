"""
Split-conformal calibration of quantile forecast intervals (CQR).

The trained q10/q90 head gives a nominal 80% band; empirically it under-covers
out-of-sample (the audited run's validation coverage was 0.49). Conformalized
Quantile Regression (Romano et al. 2019) fixes that: on held-out folds, compute
the score s = max(q10 - y, y - q90) per month, take its finite-sample-corrected
(1 - alpha) quantile, and widen the band by that amount.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np


class ConformalCalibrator:
    """CQR calibrator with optional per-horizon-month resolution.

    With few calibration folds (LOCO gives ~10), per-month quantiles are noisy;
    ``per_month=False`` (default) pools scores across the horizon, which is the
    right bias/variance trade-off at this sample size.
    """

    def __init__(self, alpha: float = 0.2, per_month: bool = False):
        self.alpha = alpha
        self.per_month = per_month
        self.qhat: Optional[np.ndarray] = None   # scalar array () or (horizon,)

    def fit(self, actuals: List[np.ndarray], q10s: List[np.ndarray],
            q90s: List[np.ndarray]) -> 'ConformalCalibrator':
        """Fit from held-out forecasts (lists of (horizon,) arrays, raw units)."""
        scores = np.stack([
            np.maximum(np.asarray(q10) - np.asarray(y), np.asarray(y) - np.asarray(q90))
            for y, q10, q90 in zip(actuals, q10s, q90s)
        ])  # (n_folds, horizon)

        n = scores.shape[0]
        # Finite-sample corrected quantile level, capped at 1.
        level = min(1.0, np.ceil((n + 1) * (1 - self.alpha)) / n)

        if self.per_month:
            self.qhat = np.quantile(scores, level, axis=0)
        else:
            self.qhat = np.quantile(scores.ravel(), level)
        return self

    def apply(self, q10: np.ndarray, q90: np.ndarray,
              clip_min: float = 0.0) -> Dict[str, np.ndarray]:
        """Widen a nominal band to the calibrated one (lower clipped at clip_min)."""
        if self.qhat is None:
            raise ValueError("Calibrator not fitted.")
        lower = np.clip(np.asarray(q10) - self.qhat, clip_min, None)
        upper = np.asarray(q90) + self.qhat
        return {'q10': lower, 'q90': upper}

    def to_dict(self) -> Dict:
        return {'alpha': self.alpha, 'per_month': self.per_month,
                'qhat': np.asarray(self.qhat).tolist()}

    @classmethod
    def from_dict(cls, d: Dict) -> 'ConformalCalibrator':
        obj = cls(alpha=d['alpha'], per_month=d['per_month'])
        qhat = np.asarray(d['qhat'])
        obj.qhat = qhat if qhat.ndim else float(qhat)
        return obj
