"""
Leave-one-cycle-out (LOCO) evaluation - the honest headline number.

Each fold holds out one full solar cycle: the model (or baseline) sees ONLY
data strictly before that cycle's onset minimum, is (re)trained on it, and
forecasts the following horizon. This is a true expanding-window hindcast -
per-origin retraining, train-era-only scalers, no window overlap with the
evaluated cycle - so fold metrics are genuine out-of-sample errors.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ..utils.precursors import detect_cycle_minima
from .metrics import forecast_metrics


def loco_origins(raw: np.ndarray, input_window: int, horizon: int,
                 val_ratio: float = 0.2, max_folds: Optional[int] = None) -> List[int]:
    """Cycle-minimum forecast origins with enough preceding data to train on.

    An origin o is usable when the truncated series [0, o) still admits a
    leak-free train/val split (0.8*o >= input_window + horizon) and the full
    actual horizon [o, o+horizon) is observed.
    """
    minima = detect_cycle_minima(raw, horizon)
    min_origin = int(np.ceil((input_window + horizon) / max(1e-9, 1.0 - val_ratio)))
    usable = [int(m) for m in minima
              if m >= min_origin and m + horizon <= len(raw)]
    if max_folds is not None:
        usable = usable[-max_folds:]
    return usable


def run_baseline_loco(baseline, raw: np.ndarray, origins: List[int],
                      horizon: int) -> List[Dict]:
    """Evaluate one baseline across all folds. Returns per-fold records."""
    records = []
    for origin in origins:
        fc = baseline.forecast(raw[:origin], horizon)
        actual = raw[origin:origin + horizon]
        m = forecast_metrics(actual, fc['mean'], q10=fc.get('q10'), q90=fc.get('q90'))
        records.append({'origin': origin, 'metrics': m, 'forecast': fc,
                        'actual': actual})
    return records


def run_model_loco(config, df: pd.DataFrame, raw: np.ndarray, dates,
                   origins: List[int], horizon: int, device: str = 'auto',
                   epochs: Optional[int] = None,
                   work_dir: Path = Path('data/experiments/cv')) -> List[Dict]:
    """Retrain the configured torch model per fold and forecast the held-out cycle.

    ``df`` is the full raw dataframe; each fold truncates it at the fold origin,
    trains from scratch (fresh trainer, train-era-only scalers), and forecasts
    the horizon from the truncated end. Intervals come from the trained
    quantile head (q10/q90); the point forecast is the deterministic median.
    """
    from ..trainers.seq2seq_trainer import Seq2SeqTrainer

    date_col = pd.to_datetime(df['date'])
    records = []
    for origin in origins:
        fold_config = copy.deepcopy(config)
        fold_config.plot_training = False
        fold_config.plot_predictions = False
        if epochs is not None:
            fold_config.training.epochs = epochs

        cutoff = dates[origin - 1]
        df_fold = df[date_col <= cutoff].copy()

        fold_dir = work_dir / fold_config.experiment_name / f"fold_{dates[origin].strftime('%Y%m')}"
        trainer = Seq2SeqTrainer(fold_config, device=device)
        trainer.train(df_fold, output_dir=fold_dir)

        raw_t, X_raw, cond_raw, dates_t = trainer._monthly_arrays(df_fold)
        fc = trainer._forecast_from(raw_t, X_raw, cond_raw, len(raw_t),
                                    n_mc=fold_config.model.mc_dropout_samples)

        actual = raw[origin:origin + horizon]
        point = fc.get('q50', fc['mean'])
        m = forecast_metrics(actual, point, q10=fc.get('q10'), q90=fc.get('q90'))
        records.append({'origin': origin, 'metrics': m, 'forecast': fc,
                        'actual': actual, 'run_dir': str(fold_dir)})
        print(f"  fold {dates[origin].strftime('%Y-%m')}: "
              f"RMSE {m['rmse']:.1f}, peak amp err {m['peak_amp_err']:+.1f}, "
              f"peak timing err {m['peak_timing_err']:+.0f} mo")
    return records
