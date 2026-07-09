#!/usr/bin/env python3
"""
Solar Cycle 25 spot-check: the calibration test the 2020 paper failed.

Trains the configured model ONLY on data before the SC25 onset minimum
(2019), forecasts the following 132 months, and scores it against the months
observed since - including whether the (conformally calibrated) 80% interval
covers the actual smoothed peak (~156 in late 2024). Baselines are scored the
same way. The paper's own SC25 prediction was 106 +/- 19.75, which missed.

  uv run python scripts/hindcast_sc25.py --config solar/configs/wavenet_lstm_v2.yaml
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')  # headless: write PNGs without a display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).parent.parent))

from solar.data import load_solar_data
from solar.data.monthly import build_monthly_series, smooth_13m
from solar.eval.conformal import ConformalCalibrator
from solar.models.baselines import ALL_BASELINES
from solar.trainers.seq2seq_trainer import Seq2SeqTrainer
from solar.utils.config import load_config
from solar.utils.precursors import detect_cycle_minima


def plot_sc25(forecasts, actual, out_path):
    """Overlay each model's SC25 forecast against the observed partial cycle.

    ``forecasts`` maps model name -> dict(mean, q10, q90). The band is drawn for
    any model that supplies q10/q90.
    """
    n = len(actual)
    months = np.arange(n)
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(months, actual, color='#1f2933', linewidth=3.0, zorder=6,
            label='Observed SC25 (so far)')
    cmap = plt.get_cmap('tab10')
    for i, (name, fc) in enumerate(forecasts.items()):
        color = cmap(i % 10)
        ax.plot(months, np.asarray(fc['mean'])[:n], linewidth=2.0, color=color,
                label=f"{name} (peak {np.max(fc['mean']):.0f})")
        if fc.get('q10') is not None and fc.get('q90') is not None:
            ax.fill_between(months, np.asarray(fc['q10'])[:n],
                            np.asarray(fc['q90'])[:n], color=color, alpha=0.12)
    ax.axhline(float(smooth_13m(actual).max()), color='crimson', linestyle=':',
               linewidth=1.5, label=f'observed smoothed peak {smooth_13m(actual).max():.0f}')
    ax.set_title('Solar Cycle 25 hindcast: forecasts vs. observed', fontweight='bold')
    ax.set_xlabel('Months into cycle')
    ax.set_ylabel('Sunspot Number')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def score(name, pred, actual, q10=None, q90=None):
    n = len(actual)
    pred_s, actual_s = smooth_13m(pred[:n]), smooth_13m(actual)
    line = {
        'name': name,
        'rmse': float(np.sqrt(np.mean((pred[:n] - actual) ** 2))),
        'pred_peak_smoothed': float(pred_s.max()),
        'actual_peak_smoothed': float(actual_s.max()),
        'peak_covered': None,
    }
    if q10 is not None and q90 is not None:
        # Peak coverage: does the band around the predicted-peak month contain
        # the actual smoothed peak value?
        q10_s, q90_s = smooth_13m(q10[:n]), smooth_13m(q90[:n])
        line['peak_covered'] = bool(q10_s.max() <= actual_s.max() <= q90_s.max())
        line['band_at_peak'] = [round(float(q10_s.max()), 1), round(float(q90_s.max()), 1)]
    return line


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config', default='solar/configs/wavenet_lstm_v2.yaml')
    parser.add_argument('--device', default='auto')
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--skip-model', action='store_true', help='Baselines only')
    args = parser.parse_args()

    config = load_config(args.config)
    df = load_solar_data(dataset=config.data.dataset,
                         need_precursors=bool(config.data.precursor_cols))
    monthly = build_monthly_series(df, target_col=config.data.target_col,
                                   start_year=config.data.start_year)
    raw, dates = monthly.values, monthly.index
    horizon = config.data.prediction_horizon

    # SC25 onset = last detected minimum with >= 4 observed years after it.
    minima = detect_cycle_minima(raw, horizon)
    origin = int([m for m in minima if len(raw) - m >= 48][-1])
    actual = raw[origin:origin + horizon]  # partial cycle observed so far
    print(f"SC25 hindcast origin: {dates[origin].strftime('%Y-%m')} "
          f"({len(actual)} observed months to score against)\n")

    rows = []
    forecasts = {}
    history = raw[:origin]
    for cls in ALL_BASELINES:
        fc = cls().forecast(history, horizon)
        forecasts[cls.name] = fc
        rows.append(score(cls.name, fc['mean'], actual, fc.get('q10'), fc.get('q90')))

    if not args.skip_model:
        import copy
        fold_config = copy.deepcopy(config)
        fold_config.plot_training = False
        fold_config.plot_predictions = False
        if args.epochs:
            fold_config.training.epochs = args.epochs

        date_col = pd.to_datetime(df['date'])
        df_fold = df[date_col <= dates[origin - 1]].copy()
        out_dir = Path('data/experiments/cv') / config.experiment_name / 'fold_sc25'
        trainer = Seq2SeqTrainer(fold_config, device=args.device)
        trainer.train(df_fold, output_dir=out_dir)

        raw_t, X_raw, cond_raw, _ = trainer._monthly_arrays(df_fold)
        fc = trainer._forecast_from(raw_t, X_raw, cond_raw, len(raw_t),
                                    n_mc=fold_config.model.mc_dropout_samples)
        q10, q90 = fc['q10'], fc['q90']
        cal_path = Path('data/experiments/cv') / config.experiment_name / 'conformal.json'
        if cal_path.exists():
            with open(cal_path) as f:
                cal = ConformalCalibrator.from_dict(json.load(f))
            band = cal.apply(q10, q90)
            q10, q90 = band['q10'], band['q90']
        rows.append(score(config.experiment_name + ' (calibrated)',
                          fc['q50'], actual, q10, q90))
        forecasts[config.experiment_name + ' (calibrated)'] = {
            'mean': fc['q50'], 'q10': q10, 'q90': q90}

    print(f"{'model':<32}{'RMSE':>7}{'predPeak':>10}{'actualPeak':>11}{'80% band':>18}{'covered':>9}")
    for r in rows:
        band = str(r.get('band_at_peak', '-'))
        cov = {True: 'YES', False: 'no', None: '-'}[r['peak_covered']]
        print(f"{r['name']:<32}{r['rmse']:>7.1f}{r['pred_peak_smoothed']:>10.1f}"
              f"{r['actual_peak_smoothed']:>11.1f}{band:>18}{cov:>9}")

    with open('data/experiments/cv/sc25_hindcast.json', 'w') as f:
        json.dump(rows, f, indent=2)
    plot_sc25(forecasts, actual, Path('data/experiments/cv/plots/sc25_hindcast.png'))
    print("\nSaved data/experiments/cv/sc25_hindcast.json")
    print("Saved data/experiments/cv/plots/sc25_hindcast.png")
    print("Reference: Benson et al. (2020) predicted SC25 peak 106 +/- 19.75 (missed ~156).")


if __name__ == "__main__":
    main()
