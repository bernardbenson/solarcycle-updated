#!/usr/bin/env python3
"""
Leave-one-cycle-out CV leaderboard: baselines vs deep models, raw units.

This is the number that decides model choices. Each fold retrains from scratch
on data strictly before a held-out cycle's onset and forecasts that cycle -
per-origin retraining, no leakage. Metrics: RMSE/MAE on the raw monthly target,
peak amplitude/timing errors on the 13-month smoothed series, and 80% interval
coverage (pre- and post-conformal calibration for quantile models).

Examples:
  uv run python scripts/run_cv.py --configs solar/configs/wavenet_lstm_v2.yaml --folds 8
  uv run python scripts/run_cv.py --baselines-only --folds 12
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')  # headless: write PNGs without a display
import matplotlib.pyplot as plt
import numpy as np

sys.path.append(str(Path(__file__).parent.parent))

from solar.data import load_solar_data
from solar.data.monthly import build_monthly_series
from solar.eval.conformal import ConformalCalibrator
from solar.eval.loco import loco_origins, run_baseline_loco, run_model_loco
from solar.eval.metrics import aggregate_metrics, forecast_metrics
from solar.models.baselines import ALL_BASELINES
from solar.utils.config import load_config
from solar.utils.plotting import SolarCyclePlotter


def plot_loco_panels(name, records, raw, dates, horizon, out_dir, history_months=264):
    """Per-fold hindcast panels (history + forecast band + actual) for one model."""
    panels = []
    for r in sorted(records, key=lambda r: r['origin']):
        o = r['origin']
        fc = r['forecast']
        lo = max(0, o - history_months)
        mean = np.asarray(fc.get('q50', fc['mean']))
        q10 = np.asarray(fc.get('q10', mean))
        q90 = np.asarray(fc.get('q90', mean))
        panels.append({
            'history_dates': dates[lo:o], 'history_values': raw[lo:o],
            'forecast_dates': dates[o:o + horizon],
            'pred_mean': mean, 'pred_lower': q10, 'pred_upper': q90,
            'actual': r['actual'], 'origin_date': dates[o],
            'label': f"{dates[o].strftime('%Y-%m')}  RMSE {r['metrics']['rmse']:.1f}",
        })
    if not panels:
        return
    plots_dir = out_dir / 'plots'
    plots_dir.mkdir(parents=True, exist_ok=True)
    plotter = SolarCyclePlotter(style='publication')
    fig = plotter.plot_cycle_backtest(
        panels, title=f"LOCO backtest: {name}",
        save_path=plots_dir / 'cycle_backtest.png')
    plt.close(fig)


def plot_leaderboard(rows, out_dir):
    """Bar charts comparing RMSE and 80% coverage across evaluated models."""
    ranked = sorted(rows, key=lambda r: r['rmse'])
    names = [r['name'].split(' (')[0].split('+')[0][:24] for r in ranked]
    rmse = [r['rmse'] for r in ranked]
    # Show intrinsic (pre-conformal) coverage - that's where models differ;
    # conformal calibration lifts everything to ~nominal afterwards.
    cov = [r.get('coverage_80') for r in ranked]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    ax1.barh(names, rmse, color='#3a7ca5')
    ax1.invert_yaxis()
    ax1.set_xlabel('RMSE (raw SSN)')
    ax1.set_title('LOCO RMSE (lower is better)', fontweight='bold')
    for i, v in enumerate(rmse):
        ax1.text(v, i, f' {v:.1f}', va='center')

    cov_vals = [(c if c is not None else 0.0) for c in cov]
    ax2.barh(names, cov_vals, color='#f4a261')
    ax2.invert_yaxis()
    ax2.axvline(0.80, color='crimson', linestyle='--', label='nominal 0.80')
    ax2.set_xlabel('80% interval coverage')
    ax2.set_title('Calibration (closer to 0.80 is better)', fontweight='bold')
    ax2.legend()
    for i, c in enumerate(cov):
        ax2.text(cov_vals[i], i, f" {c:.2f}" if c is not None else ' n/a', va='center')

    plots_dir = out_dir / 'plots'
    plots_dir.mkdir(parents=True, exist_ok=True)
    fig.suptitle('LOCO Leaderboard', fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(plots_dir / 'leaderboard.png', dpi=200, bbox_inches='tight',
                facecolor='white')
    plt.close(fig)


def summarize(name, records):
    agg = aggregate_metrics([r['metrics'] for r in records])
    return {
        'name': name,
        'folds': len(records),
        'rmse': agg.get('rmse'),
        'mae': agg.get('mae'),
        'rmse_smoothed': agg.get('rmse_smoothed'),
        'peak_amp_mae': agg.get('peak_amp_mae'),
        'peak_timing_mae': agg.get('peak_timing_mae'),
        'coverage_80': agg.get('coverage_80'),
    }


def conformal_pass(records, alpha=0.2):
    """Fit CQR on fold residuals, return (calibrator, post-calibration coverage)."""
    usable = [r for r in records if r['forecast'].get('q10') is not None]
    if len(usable) < 3:
        return None, None
    cal = ConformalCalibrator(alpha=alpha).fit(
        [r['actual'] for r in usable],
        [r['forecast']['q10'] for r in usable],
        [r['forecast']['q90'] for r in usable])
    coverages = []
    for r in usable:
        band = cal.apply(r['forecast']['q10'], r['forecast']['q90'])
        inside = (r['actual'] >= band['q10']) & (r['actual'] <= band['q90'])
        coverages.append(float(np.mean(inside)))
    return cal, float(np.mean(coverages))


def ensemble_row(model_records, hathaway_records, horizon):
    """Convex blend of model median and Hathaway curve; weight fit on the folds.

    NOTE: the blend weight is chosen on these same folds, so this row is mildly
    optimistic - treat it as an upper bound on the ensemble's value.
    """
    by_origin = {r['origin']: r for r in hathaway_records}
    common = [r for r in model_records if r['origin'] in by_origin]
    if len(common) < 3:
        return None, None

    best_w, best_rmse = 1.0, np.inf
    for w in np.linspace(0.0, 1.0, 21):
        errs = []
        for r in common:
            point = r['forecast'].get('q50', r['forecast']['mean'])
            blend = w * point + (1 - w) * by_origin[r['origin']]['forecast']['mean']
            errs.append(np.sqrt(np.mean((blend - r['actual']) ** 2)))
        rmse = float(np.mean(errs))
        if rmse < best_rmse:
            best_w, best_rmse = float(w), rmse

    records = []
    for r in common:
        point = r['forecast'].get('q50', r['forecast']['mean'])
        blend = best_w * point + (1 - best_w) * by_origin[r['origin']]['forecast']['mean']
        records.append({'origin': r['origin'],
                        'metrics': forecast_metrics(r['actual'], blend),
                        'forecast': {'mean': blend}, 'actual': r['actual']})
    return best_w, records


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--configs', nargs='*', default=[],
                        help='Model config YAMLs to evaluate (each retrains per fold)')
    parser.add_argument('--baselines-only', action='store_true')
    parser.add_argument('--folds', type=int, default=8,
                        help='Most recent N cycles to hold out (default 8)')
    parser.add_argument('--epochs', type=int, default=None, help='Override epochs per fold')
    parser.add_argument('--device', default='auto')
    parser.add_argument('--dataset', default=None,
                        help="Override dataset (ssn|area); default from first config or ssn")
    parser.add_argument('--out', default=None, help='Output directory for leaderboard files')
    args = parser.parse_args()

    configs = [load_config(p) for p in args.configs]
    dataset = args.dataset or (configs[0].data.dataset if configs else 'ssn')
    target_col = configs[0].data.target_col if configs else \
        ('sunspot_area' if dataset == 'area' else 'sunspot_number')
    start_year = configs[0].data.start_year if configs else (1874 if dataset == 'area' else 1749)
    input_window = configs[0].data.input_window if configs else 528
    horizon = configs[0].data.prediction_horizon if configs else 132

    df = load_solar_data(dataset=dataset,
                         need_precursors=bool(configs and configs[0].data.precursor_cols))
    monthly = build_monthly_series(df, target_col=target_col, start_year=start_year)
    raw, dates = monthly.values, monthly.index

    origins = loco_origins(raw, input_window, horizon, max_folds=args.folds)
    print(f"LOCO folds ({len(origins)} held-out cycles): "
          + ", ".join(dates[o].strftime('%Y-%m') for o in origins))

    out_dir = Path(args.out or 'data/experiments/cv')
    all_records, rows = {}, []

    for cls in ALL_BASELINES:
        baseline = cls()
        records = run_baseline_loco(baseline, raw, origins, horizon)
        all_records[baseline.name] = records
        row = summarize(baseline.name, records)
        # Baselines with intervals (climatology, precursor_ensemble) also get a
        # conformal-calibrated coverage column, mirroring the neural-model path.
        cal, post_cov = conformal_pass(records)
        if post_cov is not None:
            row['coverage_80_conformal'] = post_cov
            base_dir = out_dir / baseline.name
            base_dir.mkdir(parents=True, exist_ok=True)
            with open(base_dir / 'conformal.json', 'w') as f:
                json.dump(cal.to_dict(), f, indent=2)
        rows.append(row)

    if not args.baselines_only:
        for path, config in zip(args.configs, configs):
            name = config.experiment_name
            print(f"\nTraining {name} per fold ({len(origins)} folds)...")
            records = run_model_loco(config, df, raw, dates, origins, horizon,
                                     device=args.device, epochs=args.epochs)
            all_records[name] = records
            row = summarize(name, records)

            cal, post_cov = conformal_pass(records)
            row['coverage_80_conformal'] = post_cov
            rows.append(row)

            out_dir = Path(args.out or 'data/experiments/cv') / name
            out_dir.mkdir(parents=True, exist_ok=True)
            if cal is not None:
                with open(out_dir / 'conformal.json', 'w') as f:
                    json.dump(cal.to_dict(), f, indent=2)

            w, ens_records = ensemble_row(records, all_records['hathaway_precursor'], horizon)
            if ens_records:
                ens = summarize(f"ensemble({name}+hathaway, w={w:.2f})", ens_records)
                rows.append(ens)

    # ---------------- leaderboard ----------------
    out_dir.mkdir(parents=True, exist_ok=True)

    header = (f"{'model':<40}{'folds':>6}{'RMSE':>8}{'MAE':>8}{'RMSEsm':>8}"
              f"{'|pkAmp|':>9}{'|pkMo|':>8}{'cov80':>7}{'cov80cal':>9}")
    lines = [header, '-' * len(header)]
    for r in sorted(rows, key=lambda r: r['rmse']):
        lines.append(
            f"{r['name']:<40}{r['folds']:>6}{r['rmse']:>8.1f}{r['mae']:>8.1f}"
            f"{r['rmse_smoothed']:>8.1f}{r['peak_amp_mae']:>9.1f}{r['peak_timing_mae']:>8.1f}"
            + (f"{r['coverage_80']:>7.2f}" if r.get('coverage_80') is not None else f"{'-':>7}")
            + (f"{r['coverage_80_conformal']:>9.2f}"
               if r.get('coverage_80_conformal') is not None else f"{'-':>9}"))
    table = "\n".join(lines)
    print("\n" + table)

    with open(out_dir / 'leaderboard.txt', 'w') as f:
        f.write(table + "\n")
    import csv
    fieldnames = list(dict.fromkeys(k for r in rows for k in r))
    with open(out_dir / 'leaderboard.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, restval='')
        writer.writeheader()
        writer.writerows(rows)

    # Per-fold detail for later analysis (ensembles, plots, calibration).
    detail = {
        name: [{'origin_date': dates[r['origin']].strftime('%Y-%m'),
                'metrics': r['metrics']} for r in records]
        for name, records in all_records.items()
    }
    with open(out_dir / 'per_fold_metrics.json', 'w') as f:
        json.dump(detail, f, indent=2)

    # ---------------- plots ----------------
    for name, records in all_records.items():
        plot_loco_panels(name, records, raw, dates, horizon, out_dir / name)
    plot_leaderboard(rows, out_dir)
    print(f"Saved per-model backtest panels + leaderboard.png under {out_dir}")

    print(f"\nSaved leaderboard + per-fold metrics to {out_dir}")


if __name__ == "__main__":
    main()
