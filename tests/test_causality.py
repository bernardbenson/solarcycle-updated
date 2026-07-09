"""Causality tests: precursor features must be invariant under truncation."""

import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).parent.parent))

from solar.utils.precursors import (
    causal_cycle_minima, cycle_length_series, trailing_smooth,
)


def _synthetic_cycles(n_months=1200, period=130, seed=0):
    """Sunspot-like series with curved (non-flat) minima, as in the real record."""
    rng = np.random.default_rng(seed)
    t = np.arange(n_months)
    base = 60 * (0.5 + 0.5 * np.sin(2 * np.pi * t / period)) ** 1.7 + 2
    return np.maximum(0, base + rng.normal(0, 6, n_months))


class TestTrailingSmooth:
    def test_is_causal(self):
        series = _synthetic_cycles()
        full = trailing_smooth(series)
        for T in (200, 500, 900):
            truncated = trailing_smooth(series[:T])
            np.testing.assert_allclose(truncated, full[:T])


class TestCausalMinima:
    def test_minima_confirmed_only_from_past(self):
        """A minimum confirmed by month t must be identical when the series is cut at t."""
        series = _synthetic_cycles()
        minima, confirmed = causal_cycle_minima(series)
        assert len(minima) >= 5
        for T in (400, 700, 1000):
            m_t, c_t = causal_cycle_minima(series[:T])
            known = confirmed <= T - 1
            np.testing.assert_array_equal(m_t, minima[known])
            np.testing.assert_array_equal(c_t, confirmed[known])

    def test_confirmation_lag(self):
        series = _synthetic_cycles()
        minima, confirmed = causal_cycle_minima(series, confirm_months=18)
        assert ((confirmed - minima) >= 18).all()

    def test_spacing(self):
        series = _synthetic_cycles()
        minima, _ = causal_cycle_minima(series, min_spacing=80)
        assert (np.diff(minima) >= 80).all()


class TestCycleLengthSeries:
    def test_no_lookahead(self):
        """cycle_length_series value at month t must not change when future data is removed."""
        series = _synthetic_cycles()
        full = cycle_length_series(series, horizon=132)
        for T in (400, 700, 1000):
            truncated = cycle_length_series(series[:T], horizon=132)
            np.testing.assert_allclose(truncated, full[:T])

    def test_lengths_match_period(self):
        series = _synthetic_cycles(period=130)
        lengths = cycle_length_series(series, horizon=132)
        # Once cycles are confirmed, the step-held length should be ~the true period.
        settled = lengths[600:]
        assert np.all(np.abs(settled - 130) < 20)
