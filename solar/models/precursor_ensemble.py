"""
Precursor-scalar deep ensemble - a learned, calibrated generalization of the
Hathaway precursor baseline.

The neural sequence models (WaveNet+LSTM, seq2seq) emit 132 free monthly values
from hundreds of thousands of parameters and lose to a ~5-parameter ridge on the
honest leave-one-cycle-out leaderboard, because there are only ~20 independent
cycle examples. This forecaster instead does what the winning baseline does -
regress a handful of causal precursor scalars to a SINGLE scalar (the next
cycle's smoothed peak amplitude) and render the 132-month curve through the
fixed Hathaway (1994) shape - but replaces the linear ridge with a small deep
ensemble of MLPs:

- richer causal features (:func:`precursor_feature_matrix` extended set),
- a nonlinear feature -> amplitude map,
- principled predictive uncertainty from the ensemble (each member outputs a
  Gaussian over amplitude; the mixture variance combines aleatoric + epistemic),
  which fixes the q10-collapse and interval under-coverage of the seq models.

It exposes the same ``forecast(history, horizon) -> {'mean','q10','q90'}``
interface as the baselines and retrains per-origin from ``history`` alone, so it
slots straight into the LOCO harness as a direct competitor to Hathaway.
"""

from __future__ import annotations

from typing import Dict

import numpy as np

from ..data.monthly import smooth_13m
from .baselines import (
    ClimatologyForecaster,
    HathawayPrecursorForecaster,
    hathaway_curve_for_peak,
    precursor_feature_matrix,
)

# Amplitude is clipped to the same physical range Hathaway uses.
_AMP_LO, _AMP_HI = 20.0, 400.0
_Z80 = 1.2816  # 10th/90th-percentile z-score for the 80% interval

# Minimum number of (feature, target) pairs before the ensemble is worth
# fitting; below this we defer to the linear precursor baseline.
_MIN_PAIRS = 6


class PrecursorEnsembleForecaster:
    """Deep ensemble over causal precursor features, decoded via Hathaway.

    Per-origin: build (features -> next-cycle smoothed peak) pairs from all
    completed cycles in ``history``, fit ``n_members`` small MLPs (each on a
    bootstrap resample, Gaussian-NLL loss), predict the upcoming cycle's peak as
    a Gaussian mixture, and emit Hathaway curves for the mean/q10/q90 amplitudes.
    """

    name = 'precursor_ensemble'

    def __init__(self, n_members: int = 10, hidden: int = 16,
                 epochs: int = 300, lr: float = 1e-2, weight_decay: float = 1e-2,
                 dropout: float = 0.1, extended_features: bool = True,
                 seed: int = 42):
        self.n_members = n_members
        self.hidden = hidden
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.dropout = dropout
        self.extended_features = extended_features
        self.seed = seed

    # ------------------------------------------------------------------ #
    def _train_member(self, Xs, ys, member: int):
        """Train one MLP on a bootstrap resample; return it in eval mode."""
        import torch
        import torch.nn as nn

        torch.manual_seed(self.seed + member)
        rng = np.random.default_rng(self.seed + member)
        # Member 0 sees the full set (an unbiased anchor); others bootstrap.
        idx = (np.arange(len(ys)) if member == 0
               else rng.integers(0, len(ys), size=len(ys)))
        xb = torch.from_numpy(Xs[idx]).float()
        yb = torch.from_numpy(ys[idx]).float().unsqueeze(1)

        model = nn.Sequential(
            nn.Linear(Xs.shape[1], self.hidden),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden, 2),  # (mean, raw_scale)
        )
        opt = torch.optim.Adam(model.parameters(), lr=self.lr,
                               weight_decay=self.weight_decay)
        nll = nn.GaussianNLLLoss(eps=1e-6)
        model.train()
        for _ in range(self.epochs):
            opt.zero_grad()
            out = model(xb)
            mean = out[:, :1]
            var = torch.nn.functional.softplus(out[:, 1:]) + 1e-3
            loss = nll(mean, yb, var)
            loss.backward()
            opt.step()
        model.eval()
        return model

    def forecast(self, history: np.ndarray, horizon: int = 132) -> Dict[str, np.ndarray]:
        import torch

        history = np.asarray(history, dtype=float)
        built = precursor_feature_matrix(history, horizon,
                                         extended=self.extended_features)
        if built is None:
            return ClimatologyForecaster().forecast(history, horizon)
        X, y, x_now = built
        # Too few cycles for a stable ensemble: the linear precursor is safer.
        if len(y) < _MIN_PAIRS:
            return HathawayPrecursorForecaster().forecast(history, horizon)

        # Standardize features and target (stabilizes the NLL fit); predictions
        # are un-standardized back to smoothed-SSN amplitude units.
        x_mu, x_sd = X.mean(axis=0), np.maximum(X.std(axis=0), 1e-9)
        y_mu, y_sd = float(y.mean()), float(max(y.std(), 1e-6))
        Xs = ((X - x_mu) / x_sd).astype(np.float64)
        ys = ((y - y_mu) / y_sd).astype(np.float64)
        xs_now = ((x_now - x_mu) / x_sd).astype(np.float32)

        # Train every ensemble member (grad enabled), then predict under no_grad.
        members = [self._train_member(Xs, ys, m) for m in range(self.n_members)]

        means, variances = [], []
        xt_all = torch.from_numpy(Xs.astype(np.float32))
        insample = np.zeros(len(ys))
        with torch.no_grad():
            xt = torch.from_numpy(xs_now).float().unsqueeze(0)
            for model in members:
                out = model(xt)
                mu_std = float(out[0, 0])
                var_std = float(torch.nn.functional.softplus(out[0, 1]) + 1e-3)
                means.append(mu_std * y_sd + y_mu)
                variances.append(var_std * y_sd * y_sd)
                # In-sample mean prediction (un-standardized) for a residual floor.
                insample += (model(xt_all)[:, 0].numpy() * y_sd + y_mu) / self.n_members

        means = np.asarray(means)
        variances = np.asarray(variances)
        # Gaussian-mixture moments: aleatoric (mean var) + epistemic (var of means).
        peak = float(means.mean())
        total_var = float((variances + means ** 2).mean() - peak ** 2)
        sigma_amp = float(np.sqrt(max(total_var, 1e-6)))

        # The ensemble is overconfident on ~20 points, so floor the amplitude
        # sigma with the in-sample residual spread of the ensemble mean.
        resid_std = float(np.std(y - insample))
        sigma_amp = max(sigma_amp, resid_std)

        # Raw monthly observations scatter around the smooth cycle shape; the
        # 80% band must cover that too (this is why climatology's empirical
        # percentile band reaches 0.79 coverage while a bare shifted curve does
        # not). Estimate it from the raw-minus-smooth residual over history.
        sigma_scatter = float(np.std(history - smooth_13m(history)))

        peak = float(np.clip(peak, _AMP_LO, _AMP_HI))
        peak_lo = float(np.clip(peak - _Z80 * sigma_amp, _AMP_LO, _AMP_HI))
        peak_hi = float(np.clip(peak + _Z80 * sigma_amp, _AMP_LO, _AMP_HI))

        mean_curve = hathaway_curve_for_peak(horizon, peak)
        amp_lo_curve = hathaway_curve_for_peak(horizon, peak_lo)
        amp_hi_curve = hathaway_curve_for_peak(horizon, peak_hi)

        # Combine amplitude-driven spread (phase-dependent, widest near peak)
        # with the raw scatter in quadrature.
        amp_half = 0.5 * (amp_hi_curve - amp_lo_curve)
        total_half = np.sqrt(amp_half ** 2 + (_Z80 * sigma_scatter) ** 2)

        return {
            'mean': mean_curve,
            'q10': np.clip(mean_curve - total_half, 0.0, None),
            'q90': mean_curve + total_half,
        }
