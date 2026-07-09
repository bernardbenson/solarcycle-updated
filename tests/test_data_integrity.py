"""Data-integrity tests: sentinel masking, scaler correctness, split leakage."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.append(str(Path(__file__).parent.parent))

from solar.data.monthly import build_monthly_series, mask_sentinels, smooth_13m
from solar.utils.normalization import RobustScaler, prepare_multivariate_monthly_data
from solar.utils.splits import time_axis_split, assert_no_target_overlap


def _daily_df_with_sentinels():
    """3 years of daily data where half the days in early months are -1 sentinels."""
    dates = pd.date_range("1900-01-01", periods=3 * 365, freq="D")
    rng = np.random.default_rng(0)
    values = rng.uniform(20, 80, len(dates))
    values[: 200] = np.where(np.arange(200) % 2 == 0, -1.0, values[:200])
    return pd.DataFrame({"date": dates, "sunspot_number": values})


class TestSentinelMasking:
    def test_no_negative_monthly_means(self):
        monthly = build_monthly_series(_daily_df_with_sentinels(), start_year=1900)
        assert (monthly >= 0).all()

    def test_sentinels_do_not_bias_monthly_mean(self):
        # A month of alternating -1/-known values must average to the known mean,
        # not be dragged down by the sentinels.
        dates = pd.date_range("1900-01-01", "1900-01-31", freq="D")
        values = np.where(np.arange(len(dates)) % 2 == 0, -1.0, 50.0)
        df = pd.DataFrame({"date": dates, "sunspot_number": values})
        monthly = build_monthly_series(df, start_year=1900)
        assert monthly.iloc[0] == pytest.approx(50.0)

    def test_mask_sentinels_leaves_valid_zero(self):
        df = pd.DataFrame({"sunspot_number": [0.0, -1.0, 5.0]})
        out = mask_sentinels(df)
        assert out["sunspot_number"].iloc[0] == 0.0
        assert np.isnan(out["sunspot_number"].iloc[1])

    def test_multivariate_prep_has_zero_sqrt_shift(self):
        # The audited run's smoking gun: sqrt_shift=0.4 can only appear when a
        # monthly mean went negative. With daily-level masking it must be 0.
        _, X_raw, scaler, _ = prepare_multivariate_monthly_data(
            _daily_df_with_sentinels(), start_year=1900,
            scaler_config={'transform': 'sqrt'})
        assert (X_raw[:, 0] >= 0).all()
        assert scaler.target_scaler.transform_params.get('sqrt_shift', 0.0) == 0.0


class TestScaler:
    def test_transform_never_mutates_state(self):
        scaler = RobustScaler(method="robust", transform="sqrt")
        scaler.fit(np.array([1.0, 4.0, 9.0, 16.0]))
        params_before = dict(scaler.transform_params)
        scaler.transform(np.array([-5.0, 100.0]))  # value below fit-time min
        assert scaler.transform_params == params_before

    @pytest.mark.parametrize("method,transform", [
        ("robust", "sqrt"), ("robust", "log1p"), ("standard", "identity"),
        ("minmax", "identity"), ("standard", "asinh"), ("minmax", "asinh"),
    ])
    def test_round_trip(self, method, transform):
        rng = np.random.default_rng(1)
        data = rng.uniform(0, 250, 500)
        scaler = RobustScaler(method=method, transform=transform)
        scaled = scaler.fit_transform(data)
        recovered = scaler.inverse_transform(scaled)
        np.testing.assert_allclose(recovered, data, rtol=1e-8, atol=1e-8)

    def test_round_trip_on_unseen_larger_values(self):
        scaler = RobustScaler(method="minmax", transform="identity")
        scaler.fit(np.linspace(0, 100, 200))
        unseen = np.array([150.0, 300.0])  # beyond the fit range
        np.testing.assert_allclose(
            scaler.inverse_transform(scaler.transform(unseen)), unseen, rtol=1e-10)

    def test_serialization_round_trip(self, tmp_path):
        scaler = RobustScaler(method="minmax", transform="identity")
        data = np.linspace(0, 250, 300)
        scaled = scaler.fit_transform(data)
        scaler.save_params(tmp_path / "s.json")
        loaded = RobustScaler.load_params(tmp_path / "s.json")
        np.testing.assert_allclose(loaded.transform(data), scaled, rtol=1e-10)
        np.testing.assert_allclose(loaded.inverse_transform(scaled), data, rtol=1e-8)


class TestSplits:
    def test_no_target_overlap(self):
        train, val, t_split = time_axis_split(3000, 528, 132, val_ratio=0.2)
        assert_no_target_overlap(train, val, 132)  # raises on leakage

    def test_train_targets_end_before_split(self):
        train, val, t_split = time_axis_split(3000, 528, 132, val_ratio=0.2)
        assert max(train) + 132 <= t_split
        assert min(val) >= t_split

    def test_overlap_is_detected(self):
        with pytest.raises(AssertionError):
            assert_no_target_overlap([600], [650], 132)

    def test_too_short_series_raises(self):
        with pytest.raises(ValueError):
            time_axis_split(600, 528, 132, val_ratio=0.2)


class TestSmoothing:
    def test_13m_smoothing_preserves_constant(self):
        np.testing.assert_allclose(smooth_13m(np.full(100, 42.0)), np.full(100, 42.0))

    def test_13m_smoothing_shape(self):
        assert smooth_13m(np.arange(50, dtype=float)).shape == (50,)
