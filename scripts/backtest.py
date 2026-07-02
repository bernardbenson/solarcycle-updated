#!/usr/bin/env python3
"""
Hindcast backtest: validate a trained model against past solar cycles.

Loads a trained experiment run (no retraining), forecasts several historical
cycles from their onset, and plots each forecast (with its MC-Dropout interval)
against the actual observed cycle. Also prints per-cycle error metrics.
"""

import sys
import argparse
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).parent.parent))

from solar.utils.config import load_config
from solar.trainers.seq2seq_trainer import Seq2SeqTrainer
from solar.utils.plotting import SolarCyclePlotter
from solar.data import load_solar_data


def find_latest_run(output_dir: str):
    runs = sorted(Path(output_dir).glob("run_*"), key=lambda p: p.stat().st_mtime)
    return runs[-1] if runs else None


def run_backtest(config, run_dir, device='auto', n_panels=4, raw_units=False, df=None):
    """Load a trained run, backtest it against past cycles, print metrics, save the plot.

    Shared by the CLI and by the training script's post-run hook. Returns the path to
    the saved figure. ``df`` may be passed to reuse an already-loaded dataframe.
    """
    run_dir = Path(run_dir)
    print(f"Loading model from: {run_dir}")
    trainer = Seq2SeqTrainer(config, device=device)
    trainer.load_trained(run_dir)

    if df is None:
        df = load_solar_data()
    panels = trainer.backtest_cycles(df, n_panels=n_panels)
    print(f"Built {len(panels)} backtest panels.\n")

    # Per-cycle validation metrics (mean forecast vs actual).
    print(f"{'Origin':<10}{'RMSE':>8}{'PeakErr(mo)':>13}{'PeakErr(SN)':>13}")
    for p in panels:
        actual = np.asarray(p['actual'])
        pred = np.asarray(p['pred_mean'])
        rmse = float(np.sqrt(np.mean((actual - pred) ** 2)))
        peak_err_months = int(np.argmax(pred) - np.argmax(actual))
        peak_err_sn = float(np.max(pred) - np.max(actual))
        origin = p['origin_date'].strftime('%Y-%m')
        print(f"{origin:<10}{rmse:>8.1f}{peak_err_months:>13d}{peak_err_sn:>13.1f}")

    plotter = SolarCyclePlotter(style='publication')
    out_path = run_dir / "plots" / "cycle_backtest.png"
    out_path.parent.mkdir(exist_ok=True)
    plotter.plot_cycle_backtest(panels, normalize=not raw_units, save_path=out_path)
    print(f"\nSaved backtest figure to: {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Backtest a trained solar model against past cycles")
    parser.add_argument('--config', default='solar/configs/seq2seq_quantile.yaml')
    parser.add_argument('--run', default=None, help='Experiment run dir (default: most recent)')
    parser.add_argument('--n-panels', type=int, default=4, help='Number of past cycles to validate')
    parser.add_argument('--device', default='auto')
    parser.add_argument('--raw-units', action='store_true',
                        help='Plot raw sunspot numbers instead of 0-1 normalized')
    args = parser.parse_args()

    config = load_config(args.config)
    run_dir = Path(args.run) if args.run else find_latest_run(config.output_dir)
    if run_dir is None or not run_dir.exists():
        raise SystemExit("No trained run found. Train a model first with scripts/train_seq2seq.py.")

    run_backtest(config, run_dir, device=args.device,
                 n_panels=args.n_panels, raw_units=args.raw_units)


if __name__ == "__main__":
    main()
