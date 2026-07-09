"""
Non-neural baseline forecasters - the bar every deep model must clear.

All baselines share one interface: ``forecast(history, horizon) -> dict`` with
'mean' (horizon,) and optionally 'q10'/'q90'. ``history`` is the raw monthly
target series ending at the forecast origin (a cycle minimum); nothing after
the origin is ever seen, so these are honest per-origin forecasts.

- ClimatologyForecaster: the paper's naive baseline - all completed historical
  cycles rescaled to the horizon length and averaged; quantiles from the
  empirical spread across cycles.
- PersistenceForecaster: repeat the previous cycle.
- HathawayPrecursorForecaster: the published-SOTA-shaped method - regress the
  next cycle's smoothed peak amplitude from causal precursors (previous-cycle
  amplitude, cycle length, activity near minimum), then decode the full cycle
  shape through the Hathaway (1994) function
      f(t) = a (t - t0)^3 / [exp((t - t0)^2 / b^2) - c],  c = 0.71,
  with b tied to a ("cycles of the same amplitude have the same shape").
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from ..data.monthly import smooth_13m
from ..utils.precursors import detect_cycle_minima


# --------------------------------------------------------------------------- #
# Cycle bookkeeping
# --------------------------------------------------------------------------- #

def historical_cycles(history: np.ndarray, horizon: int) -> List[np.ndarray]:
    """Completed min-to-min cycles inside ``history`` (hindsight minima are fine
    here: the whole array is the past relative to the forecast origin)."""
    minima = detect_cycle_minima(history, horizon)
    return [history[a:b] for a, b in zip(minima[:-1], minima[1:]) if b - a >= 60]


def _rescale(cycle: np.ndarray, horizon: int) -> np.ndarray:
    """Resample one cycle to exactly ``horizon`` months."""
    src = np.linspace(0.0, 1.0, len(cycle))
    dst = np.linspace(0.0, 1.0, horizon)
    return np.interp(dst, src, cycle)


class ClimatologyForecaster:
    """Mean cycle shape across all completed historical cycles."""

    name = 'climatology'

    def forecast(self, history: np.ndarray, horizon: int = 132) -> Dict[str, np.ndarray]:
        cycles = historical_cycles(history, horizon)
        if not cycles:
            level = float(np.mean(history))
            flat = np.full(horizon, level)
            return {'mean': flat, 'q10': flat * 0.5, 'q90': flat * 1.5}
        stacked = np.stack([_rescale(c, horizon) for c in cycles])
        return {
            'mean': stacked.mean(axis=0),
            'q10': np.percentile(stacked, 10, axis=0),
            'q90': np.percentile(stacked, 90, axis=0),
        }


class PersistenceForecaster:
    """Repeat the most recently completed cycle."""

    name = 'persistence'

    def forecast(self, history: np.ndarray, horizon: int = 132) -> Dict[str, np.ndarray]:
        cycles = historical_cycles(history, horizon)
        if not cycles:
            return ClimatologyForecaster().forecast(history, horizon)
        return {'mean': _rescale(cycles[-1], horizon)}


# --------------------------------------------------------------------------- #
# Hathaway curve
# --------------------------------------------------------------------------- #

_HATHAWAY_C = 0.71
# b(a) from Hathaway, Wilson & Reichmann (1994), calibrated on version-1 sunspot
# numbers; V2 values are ~1.67x higher, so rescale `a` before applying it.
_V2_FACTOR = 1.67


def hathaway_curve(horizon: int, a: float, t0: float = 0.0) -> np.ndarray:
    """f(t) = a (t-t0)^3 / [exp((t-t0)^2/b^2) - c] with the b(a) shape tie."""
    a_v1 = max(a / _V2_FACTOR, 1e-8)
    b = 27.12 + 25.15 / (a_v1 * 1e3) ** 0.25
    t = np.arange(horizon, dtype=float) - t0
    t = np.clip(t, 0.0, None)
    denom = np.exp(np.clip((t / b) ** 2, None, 50.0)) - _HATHAWAY_C
    curve = a * t ** 3 / denom
    return np.clip(curve, 0.0, None)


def _amplitude_to_a(peak: float, horizon: int = 200) -> float:
    """Invert peak amplitude -> Hathaway `a` parameter by bisection."""
    lo, hi = 1e-6, 1.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if hathaway_curve(horizon, mid).max() < peak:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def hathaway_curve_for_peak(horizon: int, peak: float, t0: float = 0.0) -> np.ndarray:
    """Hathaway curve whose maximum equals ``peak`` (smoothed-SSN units)."""
    return hathaway_curve(horizon, _amplitude_to_a(peak), t0)


# --------------------------------------------------------------------------- #
# Shared causal precursor features (Hathaway + learned models use identical
# extraction so they stay directly comparable on the leaderboard).
# --------------------------------------------------------------------------- #

def _cycle_features(history: np.ndarray, minima: np.ndarray, k: int,
                    smoothed: np.ndarray, extended: bool = False) -> List[float]:
    """Causal features available at minimum ``k`` (the start of a new cycle).

    Base (4): previous-cycle smoothed peak, previous-cycle length, mean raw
    activity in the 36 months before the minimum, and the smoothed minimum
    level. With ``extended`` two further causal features are appended: the
    slope of the smoothed series over the 24 months into the minimum (Waldmeier
    rise proxy) and the minimum level relative to the previous peak.
    """
    prev_start, start = minima[k - 1], minima[k]
    prev_amp = float(smoothed[prev_start:start].max())
    prev_len = float(start - prev_start)
    tail_mean = float(history[max(0, start - 36):start].mean())
    min_level = float(smoothed[max(0, start - 12):start + 1].min())
    feats = [prev_amp, prev_len, tail_mean, min_level]
    if extended:
        lo = max(0, start - 24)
        seg = smoothed[lo:start + 1]
        tail_slope = float(seg[-1] - seg[0]) / max(1, len(seg) - 1)
        min_over_prev = float(min_level / prev_amp) if prev_amp > 1e-6 else 0.0
        feats += [tail_slope, min_over_prev]
    return feats


def precursor_feature_matrix(history: np.ndarray, horizon: int,
                             extended: bool = False):
    """Build causal (features -> next-cycle smoothed peak) pairs from history.

    Returns ``(X, y, x_now)`` where ``X`` is ``(n_cycles, F)`` features at each
    interior minimum, ``y`` is the following cycle's smoothed peak amplitude,
    and ``x_now`` is the ``(F,)`` feature vector at the forecast origin (the end
    of ``history``, itself a cycle minimum in the LOCO harness). Returns
    ``None`` when there are too few completed cycles to regress on - callers
    should fall back to climatology.

    Uses hindsight ``detect_cycle_minima`` on ``history`` only, which is
    legitimate: everything in ``history`` is the past relative to the origin.
    """
    history = np.asarray(history, dtype=float)
    smoothed = smooth_13m(history)
    minima = detect_cycle_minima(history, horizon)
    if len(minima) < 5:
        return None

    X, y = [], []
    for k in range(1, len(minima) - 1):
        X.append(_cycle_features(history, minima, k, smoothed, extended))
        end = minima[k + 1]
        y.append(float(smoothed[minima[k]:end].max()))

    # Features at the forecast origin. The detector often re-finds the origin
    # itself a few months before the series end, so drop any minimum within 60
    # months of the end so the "previous" cycle can't degenerate.
    origin_idx = len(history) - 1
    past_minima = minima[minima < origin_idx - 60]
    if len(past_minima) < 1:
        return None
    minima_plus_origin = np.append(past_minima, origin_idx)
    x_now = _cycle_features(history, minima_plus_origin,
                            len(minima_plus_origin) - 1, smoothed, extended)
    return np.asarray(X), np.asarray(y), np.asarray(x_now)


class HathawayPrecursorForecaster:
    """Ridge-regressed amplitude precursor decoded through the Hathaway shape.

    Per-origin: build (features -> next-cycle smoothed peak) pairs from all
    completed cycles inside ``history``, fit a small ridge regression, predict
    the upcoming cycle's peak, and emit the matching Hathaway curve. q10/q90
    come from the leave-one-out residual spread of the amplitude fit.
    """

    name = 'hathaway_precursor'

    def __init__(self, ridge_alpha: float = 5.0):
        self.ridge_alpha = ridge_alpha

    def forecast(self, history: np.ndarray, horizon: int = 132) -> Dict[str, np.ndarray]:
        from sklearn.linear_model import Ridge

        history = np.asarray(history, dtype=float)
        # Need >= 4 completed cycles to regress on; else fall back to climatology.
        built = precursor_feature_matrix(history, horizon, extended=False)
        if built is None:
            return ClimatologyForecaster().forecast(history, horizon)
        X, y, x_now = built

        mu, sd = X.mean(axis=0), np.maximum(X.std(axis=0), 1e-9)
        model = Ridge(alpha=self.ridge_alpha).fit((X - mu) / sd, y)

        # Leave-one-out residual spread for the interval.
        residuals = []
        if len(y) >= 5:
            for i in range(len(y)):
                keep = np.arange(len(y)) != i
                m = Ridge(alpha=self.ridge_alpha).fit((X[keep] - mu) / sd, y[keep])
                residuals.append(y[i] - m.predict(((X[i] - mu) / sd)[None])[0])
        sigma = float(np.std(residuals)) if residuals else float(np.std(y))

        peak = float(model.predict(((x_now - mu) / sd)[None])[0])
        peak = float(np.clip(peak, 20.0, 400.0))

        return {
            'mean': hathaway_curve_for_peak(horizon, peak),
            'q10': hathaway_curve_for_peak(horizon, max(20.0, peak - 1.28 * sigma)),
            'q90': hathaway_curve_for_peak(horizon, peak + 1.28 * sigma),
        }


# The learned precursor deep ensemble (models/precursor_ensemble.py) depends on
# the helpers above and is appended to this list in solar/models/__init__.py, so
# it is evaluated on the same LOCO leaderboard as these baselines.
ALL_BASELINES = [ClimatologyForecaster, PersistenceForecaster, HathawayPrecursorForecaster]
