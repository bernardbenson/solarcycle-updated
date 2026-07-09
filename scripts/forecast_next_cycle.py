#!/usr/bin/env python3
"""
Forecast the next solar cycle from the latest data, with calibrated intervals.

Loads a trained run (default: most recent for the config), forecasts the next
132 months from the latest 528-month window, and - when a conformal calibrator
from a LOCO CV run is available (scripts/run_cv.py writes
data/experiments/cv/<experiment>/conformal.json) - widens the quantile-head
band to the empirically calibrated 80% interval. Prints peak amplitude/timing
(on the standard 13-month smoothed scale) and saves a JSON + plot.

Example:
  uv run python scripts/run_cv.py --configs solar/configs/wavenet_lstm_v2.yaml --folds 12
  uv run python scripts/train_seq2seq.py --config solar/configs/wavenet_lstm_v2.yaml
  uv run python scripts/forecast_next_cycle.py --config solar/configs/wavenet_lstm_v2.yaml
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).parent.parent))

from solar.data import load_solar_data
from solar.data.monthly import smooth_13m
from solar.eval.conformal import ConformalCalibrator
from solar.trainers.seq2seq_trainer import Seq2SeqTrainer
from solar.utils.config import load_config
from solar.utils.plotting import SolarCyclePlotter


def find_latest_run(output_dir, experiment_name):
    runs = sorted(Path(output_dir).glob(f"run_{experiment_name}_*"),
                  key=lambda p: p.stat().st_mtime)
    return runs[-1] if runs else None


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config', default='solar/configs/wavenet_lstm_v2.yaml')
    parser.add_argument('--run', default=None, help='Trained run dir (default: most recent)')
    parser.add_argument('--conformal', default=None,
                        help='Conformal calibrator JSON (default: data/experiments/cv/<experiment>/conformal.json)')
    parser.add_argument('--device', default='auto')
    args = parser.parse_args()

    config = load_config(args.config)
    run_dir = Path(args.run) if args.run else \
        find_latest_run(config.output_dir, config.experiment_name)
    if run_dir is None or not run_dir.exists():
        raise SystemExit("No trained run found - train first with scripts/train_seq2seq.py.")

    trainer = Seq2SeqTrainer(config, device=args.device)
    trainer.load_trained(run_dir)

    df = load_solar_data(dataset=config.data.dataset,
                         need_precursors=bool(config.data.precursor_cols))
    raw, X_raw, cond_raw, dates = trainer._monthly_arrays(df)
    horizon = config.data.prediction_horizon

    fc = trainer._forecast_from(raw, X_raw, cond_raw, len(raw),
                                n_mc=config.model.mc_dropout_samples)
    q10, q50, q90 = fc['q10'], fc['q50'], fc['q90']

    calibrated = False
    cal_path = Path(args.conformal) if args.conformal else \
        Path('data/experiments/cv') / config.experiment_name / 'conformal.json'
    if cal_path.exists():
        with open(cal_path) as f:
            cal = ConformalCalibrator.from_dict(json.load(f))
        band = cal.apply(q10, q90)
        q10, q90 = band['q10'], band['q90']
        calibrated = True
        print(f"Applied conformal calibration from {cal_path} (qhat={cal.qhat})")
    else:
        print(f"No conformal calibrator at {cal_path}; intervals are the raw "
              f"quantile-head band (expect under-coverage).")

    forecast_dates = pd.date_range(dates[-1] + pd.offsets.MonthEnd(1),
                                   periods=horizon, freq='ME')
    q50_s = smooth_13m(q50)
    peak_idx = int(np.argmax(q50_s))
    result = {
        'origin': str(dates[-1].date()),
        'run_dir': str(run_dir),
        'conformal_calibrated': calibrated,
        'peak_amplitude_smoothed': round(float(q50_s.max()), 1),
        'peak_amplitude_q10': round(float(smooth_13m(q10).max()), 1),
        'peak_amplitude_q90': round(float(smooth_13m(q90).max()), 1),
        'peak_date': str(forecast_dates[peak_idx].date()),
        'horizon_months': horizon,
    }

    print(f"\nNext-cycle forecast from {result['origin']} "
          f"({config.data.dataset.upper()} model, {run_dir.name}):")
    print(f"  peak amplitude (13-mo smoothed): {result['peak_amplitude_smoothed']}"
          f"  [{result['peak_amplitude_q10']}, {result['peak_amplitude_q90']}]"
          f" ({'calibrated 80% interval' if calibrated else 'raw quantile band'})")
    print(f"  peak date: {result['peak_date']}")

    out_json = run_dir / 'next_cycle_forecast.json'
    with open(out_json, 'w') as f:
        json.dump({**result,
                   'dates': [str(d.date()) for d in forecast_dates],
                   'q10': q10.tolist(), 'q50': q50.tolist(), 'q90': q90.tolist()}, f, indent=2)

    plotter = SolarCyclePlotter(style='publication')
    out_png = run_dir / 'plots' / 'next_cycle_forecast_calibrated.png'
    out_png.parent.mkdir(exist_ok=True)
    plotter.plot_forecast_continuation(
        history=raw, forecast_mean=q50, forecast_lower=q10, forecast_upper=q90,
        title=f"Next Solar Cycle Forecast ({'calibrated' if calibrated else 'raw'} 80% band)",
        save_path=out_png)
    print(f"\nSaved {out_json} and {out_png}")


if __name__ == "__main__":
    main()
