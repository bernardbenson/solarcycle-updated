# 🌞 Solar Cycle Prediction

Forecasting a full solar cycle (132 months) from four cycles of history (528
months) — a modernized, **honestly evaluated** rebuild of Benson et al. (2020),
*Forecasting Solar Cycle 25 Using Deep Neural Networks* (Solar Physics 295:65).

The headline lesson of this rebuild: **under leakage-free evaluation, big
history-driven deep networks lose to a tiny precursor model.** There are only
~25 solar cycles on record, so a model that emits 132 free monthly values from
hundreds of thousands of parameters overfits and regresses toward the mean cycle.
The production forecaster here is therefore the **Precursor Ensemble** — a small
model that predicts a *single scalar* (next-cycle peak amplitude) from causal
precursors and renders the cycle shape through a fixed physics curve. It matches
the strongest physics baseline on accuracy, is better on peak amplitude, and is
the only accurate forecaster that is also **well-calibrated** — with no GPU.

The WaveNet+LSTM deep model is retained as a research/comparison vehicle.

---

## 🚀 Key Features

- **Precursor Ensemble (production forecaster)** — a deep ensemble of 10 tiny
  MLPs (~1,460 total parameters) maps a handful of causal precursor scalars to
  the next cycle's peak amplitude, decoded through the Hathaway (1994) shape.
  Best peak-amplitude error and best-calibrated intervals on honest CV;
  CPU-only, retrains in seconds per fold. See
  [Precursor Ensemble](#precursor-ensemble-production-modelname-precursor_ensemble).
- **WaveNetLSTM (deep model, research vehicle)** — the paper's architecture,
  corrected: dilations double continuously 1→512 (receptive field 1024 months ≥
  the full 4-cycle input), weight-norm residual blocks (no BatchNorm on the
  residual stream), single LSTM summary, one-shot 132-month quantile forecast.
  ~217K parameters (11× smaller than the legacy seq2seq).
- **Honest evaluation first** — leave-one-cycle-out (LOCO) CV with per-origin
  retraining, train-era-only scalers, and a time-axis train/val split with
  provably disjoint targets. All headline metrics are in raw sunspot-number units.
- **Strong baselines** — climatology (average cycle), persistence, and a
  Hathaway-precursor ridge regression: the bar every model must clear.
- **Calibrated uncertainty, two ways** — the ensemble's intrinsic mixture
  variance + monthly-scatter band, and the deep model's trained quantile head +
  split-conformal (CQR) calibration from LOCO residuals.
- **Curated data** — official SILSO monthly-mean SSN (1749–present), total
  sunspot area (1874–present), and daily multivariate (SSN + Kp/Ap) with
  missing-day `-1` sentinels masked at native resolution *before* any monthly
  aggregation.
- **Exogenous precursors (optional, deep model)** — geomagnetic Ap/Kp input
  channels with availability mask, plus a strictly causal terminator /
  cycle-length conditioning signal (confirmed-minima only, no look-ahead).
- **Automatic plots** — every training run, LOCO leaderboard, hindcast, and
  forecast writes publication-style figures.

---

## 📋 Requirements

This project uses `uv` for fast dependency management and execution.

### Install uv (if not already installed)
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Dependencies
All dependencies are managed by `uv`:
- PyTorch (CPU is sufficient for the precursor ensemble and baselines; MPS/CUDA used by the deep model)
- Pandas, NumPy (data processing)
- Matplotlib, Seaborn (publication-quality plots)
- Scikit-learn (ridge baseline, preprocessing)
- Pydantic (configuration validation)
- PyYAML (configuration files)

---

## ⚡ Quick Start

### 1. Clone and Setup
```bash
git clone [your-repo-url]
cd solarcycle-updated
uv sync                       # install core dependencies
```

### 2. Get Data
Fetch the curated datasets (SILSO monthly SSN, monthly total sunspot area, and
the daily multivariate table for precursor runs):
```bash
uv run python -m solar.data.collection
# writes data/silso_monthly.csv, data/sunspot_area_monthly.csv, data/raw_multivariate_data.csv
```
If no CSV is found, training falls back to a synthetic series (smoke tests only).

### 3. The Fast Path: Honest Leaderboard (no GPU)
This is the recommended first command. It runs every baseline **and the
Precursor Ensemble** through leave-one-cycle-out CV — each fold retrains from
scratch on data strictly before a held-out cycle's onset and forecasts that
cycle (per-origin retraining, no leakage):
```bash
uv run python scripts/run_cv.py --baselines-only --folds 12
```
Writes to `data/experiments/cv/`:
- `leaderboard.{txt,csv}` — RMSE / MAE / peak errors / coverage per model
- `per_fold_metrics.json` — per-held-out-cycle detail
- `plots/leaderboard.png` — RMSE and calibration bar charts
- `<model>/plots/cycle_backtest.png` — 12-panel forecast-vs-actual grid per model
- `<model>/conformal.json` — CQR calibrator for the forecast script

### 4. Train the Deep Model (optional, GPU helps)
```bash
# Paper-faithful WaveNet+LSTM (4-cycle input -> 1-cycle quantile forecast)
uv run python scripts/train_seq2seq.py --config solar/configs/wavenet_lstm_v2.yaml

# Include it on the leaderboard (retrains per fold — slow):
uv run python scripts/run_cv.py --configs solar/configs/wavenet_lstm_v2.yaml --folds 12

# Other variants:
uv run python scripts/train_seq2seq.py --config solar/configs/wavenet_lstm_tiny.yaml      # ~70K params
uv run python scripts/train_seq2seq.py --config solar/configs/wavenet_lstm_v2_area.yaml   # sunspot area
uv run python scripts/train_seq2seq.py --config solar/configs/wavenet_lstm_v2_term.yaml   # + terminator
uv run python scripts/train_seq2seq.py --config solar/configs/wavenet_lstm_v2.yaml --epochs 2  # smoke test
```
A training run writes a timestamped `data/experiments/run_<name>_.../` directory
with checkpoints, metrics, a summary report, and plots (including a post-training
hindcast backtest).

> Legacy configs (`seq2seq_precursors.yaml`, `seq2seq_quantile.yaml`, `base.yaml`,
> `ablation_tcn.yaml`) still run for comparison; the trainer selects the
> architecture from `model.name`. `input_dim`/`cond_dim` are derived automatically
> from the data config (`resolve_derived_dims`).

### 5. Spot-Checks and the Next-Cycle Forecast
```bash
# SC25 calibration test: forecast from pre-2019 data only, score against the
# observed cycle (actual smoothed peak ~156; the 2020 paper predicted 106 +/- 19.75).
# Baselines + Precursor Ensemble run automatically; --skip-model avoids retraining
# the deep model. Writes data/experiments/cv/plots/sc25_hindcast.png.
uv run python scripts/hindcast_sc25.py --skip-model
uv run python scripts/hindcast_sc25.py --config solar/configs/wavenet_lstm_v2.yaml  # + deep model

# Cycle 26 forecast from a trained deep run, with conformally calibrated 80% band
uv run python scripts/forecast_next_cycle.py --config solar/configs/wavenet_lstm_v2.yaml

# Quick (partly in-sample) visual backtest of a trained run — panels overlapping
# the training era are labeled [IN-SAMPLE]; run_cv.py is the honest version.
uv run python scripts/backtest.py --config solar/configs/wavenet_lstm_v2.yaml --n-panels 4
```

---

## 📊 Results (honest leave-one-cycle-out, 12 held-out cycles 1889–2008)

All numbers in raw sunspot-number units; peak errors on the standard 13-month
smoothed series. Each fold retrains on data strictly before the held-out cycle.
`cov80` is the empirical 80% interval coverage (raw → after conformal calibration).

| Model | RMSE | MAE | peak-amp MAE | peak-timing MAE (mo) | cov80 (raw → cal) |
|---|---|---|---|---|---|
| **Precursor Ensemble** | 31.7 | 23.7 | **30.9** | 7.5 | **0.76 → 0.92** |
| Hathaway-precursor (ridge) | **31.3** | **23.3** | 31.1 | **7.3** | 0.37 → 0.92 |
| Climatology (average cycle) | 37.8 | 28.4 | 46.4 | 8.2 | 0.79 |
| Persistence (repeat last cycle) | 46.0 | 34.4 | 49.5 | 10.5 | — |
| WaveNetLSTM v2 (217K, deep) | 61.4 | 44.6 | 83.1 | 19.4 | 0.46 → 0.92 |
| WaveNetLSTM tiny (70K) | 61.7 | 45.2 | 86.5 | 17.1 | 0.48 → 0.92 |
| WaveNetLSTM + terminator | 64.5 | 50.6 | 77.1 | 21.6 | 0.50 → 0.92 |

**The honest headline:** a pure history-driven deep network — at any size tried —
systematically under-predicts strong-cycle peaks (it regresses toward the average
cycle of its training era) and does not beat the precursor baselines. Cycle
*amplitude* information lives in **precursors** (previous-cycle length and
depth-of-minimum activity), not in the shape of the raw SSN history. The
**Precursor Ensemble** ties the physics baseline on RMSE, is best on peak
amplitude (30.9), and is the only accurate forecaster whose intervals are
well-calibrated *without* post-hoc conformal (0.76 vs Hathaway's 0.37) — while
costing ~1,460 parameters and no GPU. It is the recommended production forecaster.

> The 2020 paper's RMSE 2.93 is not reproducible under embargoed splits — that
> number reflected a train/validation window overlap, not forecasting skill. Treat
> any figure below climatology's error (~38) with suspicion unless the split is
> described.

---

## 🏗️ Architecture Overview

### Precursor Ensemble (production, `model.name: precursor_ensemble`)

`solar/models/precursor_ensemble.py`. A three-part design — only the middle
part is learned:

```
sunspot history (up to forecast origin, strictly causal)
        │
        ▼  1. FEATURE EXTRACTOR  (deterministic; precursor_feature_matrix)
   6 causal scalars per cycle:
     • previous cycle peak amplitude      • minimum depth
     • previous cycle length (min→min)    • rise slope into minimum
     • mean activity, 36 mo before min    • min level ÷ previous peak
        │  x ∈ ℝ⁶  (z-scored on training pairs)
        ▼  2. DEEP ENSEMBLE  (learned; 10 independent MLPs)
   member i:  Linear(6→16) → ReLU → Dropout(0.1) → Linear(16→2) = (μ, s)
              σ = softplus(s) + 1e-3     → Gaussian N(μ, σ²) over PEAK AMPLITUDE
   trained on a bootstrap resample, Gaussian-NLL loss, 300 epochs, Adam
        │  mixture of 10 Gaussians → peak, σ_amp
        ▼  3. SHAPE DECODER  (fixed physics; hathaway_curve_for_peak)
   f(t) = a·t³ / (exp((t/b)²) − 0.71)    → 132-month curve for peak, peak±z·σ
        │
        ▼
   mean / q10 / q90   (132-month forecast + 80% band)
```

- **Why it wins:** predicting *one* number (peak amplitude) from ~20 cycles is
  learnable; predicting 132 free values is not. The cycle *shape* is supplied by
  the Hathaway curve rather than learned from data that doesn't exist.
- **Size:** each member is a 2-layer MLP (~146 params); 10 members ≈ **1,460
  learned parameters** — ~150× smaller than the WaveNetLSTM it matches.
- **Calibrated intervals:** the 80% band combines, in quadrature, (a) epistemic
  spread across the 10 members, (b) each member's learned aleatoric variance —
  floored by the in-sample residual spread, since the ensemble is overconfident
  on ~20 points, and (c) the raw-monthly scatter around the smooth cycle shape
  (`history − smooth_13m`). This third term is what lifted coverage 0.24 → 0.76.
- **Fallbacks:** too few completed cycles to fit an ensemble → defer to the
  Hathaway baseline; almost none → climatology.
- **Interface:** `forecast(history, horizon) -> {'mean','q10','q90'}`, identical
  to the baselines, so it is registered in `ALL_BASELINES` and evaluated by
  `run_cv.py` and `hindcast_sc25.py` automatically. No config file or GPU needed.

### WaveNetLSTM (deep model, `model.name: WaveNetLSTM`)

`solar/models/wavenet_lstm_direct.py`. The Benson et al. (2020) architecture,
corrected and slimmed:

1. **WaveNet stem** — 10 gated residual blocks, kernel 2, dilations doubling
   **continuously** 1,2,4,…,512 → receptive field 1 + (2¹⁰−1) = **1024 months**,
   covering the entire 528-month input (the legacy per-stack dilation reset gave
   only 91 months — less than one cycle). 32 channels, weight-norm convs, **no
   BatchNorm on the residual stream** (it strips the amplitude being predicted).
2. **LSTM summary** — a single unidirectional LSTM (hidden 128) over the skip-sum
   features; its final state summarizes the window.
3. **Direct one-shot heads** — linear point head + linear quantile head
   (132 × {0.1, 0.5, 0.9}, sorted for monotonicity). Non-autoregressive: the whole
   cycle is emitted at once — no feedback loop, no error accumulation. ~217K
   parameters total.

### Non-neural baselines

`solar/models/baselines.py` — the bar every model must clear:
- **Climatology** — mean of all completed historical cycles rescaled to the horizon.
- **Persistence** — repeat the previous cycle.
- **Hathaway-precursor** — ridge regression of next-cycle peak amplitude from the
  shared `precursor_feature_matrix` features (base 4 scalars), decoded through the
  Hathaway curve; interval from leave-one-out residual spread. The Precursor
  Ensemble is the learned, ensembled, better-calibrated generalization of this.

### Comparison-only models

`WaveNetAttnSeq2Seq` (WaveNet + BiLSTM encoder, transformer-query decoder, 2.46M
params), `TCNOnly`, and `NBEATSx` remain in the registry as ablations. The
seq2seq's transformer decoder alone costs ~790K parameters and its WaveNet
receptive field (91 months) is shorter than one solar cycle — kept for reference,
not recommended.

### Exogenous Precursors & Conditioning (deep model)

Beyond the sunspot channel, the deep model can ingest physically-motivated inputs:

- **Geomagnetic input channels** — `ap_avg` / `kp_sum` concatenated as extra
  channels. They exist only from 1932, so pre-1932 months are zero-filled and
  flagged by a **binary availability-mask channel** the model learns to gate on.
- **Terminator / cycle-length conditioning** — the length of the most recently
  completed cycle (min-to-min), step-held per month and fed as a conditioning
  vector (`cond_dim`). Cycle length anti-correlates with the amplitude of the
  *following* cycle — a sunspot-derivable proxy for the Hale-cycle "terminator
  separation" of McIntosh et al. (2023). The series is **strictly causal**: minima
  are detected with trailing smoothing and only enter the series once confirmed by
  18 subsequent months (no look-ahead; verified by truncation tests). A hindsight
  detector exists separately for defining evaluation folds and plots.

Channel counts (`input_dim`, `cond_dim`) are derived from the data config by
`resolve_derived_dims`; an empty `precursor_cols` keeps the deep model fully
univariate.

### Prediction Heads (deep model)

- **MSE Head** — point predictions.
- **Quantile Head** — pinball loss over {0.1, 0.5, 0.9}, sorted for monotonicity.
- **Combined Head** — both MSE and quantile outputs.

### Uncertainty Quantification

- **Precursor Ensemble (intrinsic)** — Gaussian mixture over amplitude (epistemic
  member spread + aleatoric variance, floored by in-sample residuals) combined in
  quadrature with raw-monthly scatter. Well-calibrated out of the box (cov80 0.76).
- **Deep model quantile head** — the trained q10/q90 interval (pinball loss).
- **Conformal calibration (CQR)** — LOCO fold residuals widen any q10/q90 band to
  empirically valid 80% coverage (`solar/eval/conformal.py`; fitted by `run_cv.py`,
  applied by `forecast_next_cycle.py` and `hindcast_sc25.py`).
- **MC-Dropout** — 30 forward passes with dropout enabled — epistemic diagnostic
  for the deep model only, never the reported interval.

---

## 📊 Output Structure

### LOCO leaderboard (`scripts/run_cv.py`)
```
data/experiments/cv/
├── leaderboard.txt / leaderboard.csv         # ranked models, all metrics
├── per_fold_metrics.json                     # per-held-out-cycle detail
├── plots/
│   └── leaderboard.png                       # RMSE + calibration bar charts
├── precursor_ensemble/
│   ├── plots/cycle_backtest.png              # 12-panel forecast-vs-actual grid
│   └── conformal.json                        # CQR calibrator
├── hathaway_precursor/ …                      # (one dir per evaluated model)
└── <deep-config-name>/ …                      # when --configs is passed
```

### Deep training run (`scripts/train_seq2seq.py`)
```
data/experiments/run_wavenet_lstm_v2_<timestamp>_<hash>/
├── best_model.pt / final_model.pt            # checkpoints
├── scaler.json                               # normalization parameters
├── training_metrics.json                     # full training history
├── summary_report.md                         # automated summary
└── plots/
    ├── training_history.png
    ├── recent_predictions_with_uncertainty.png
    ├── mc_dropout_uncertainty.png
    ├── peak_distribution.png
    ├── cycle_backtest.png                     # post-training hindcast panels
    └── next_cycle_forecast_calibrated.png     # after forecast_next_cycle.py
```

### SC25 hindcast (`scripts/hindcast_sc25.py`)
```
data/experiments/cv/
├── sc25_hindcast.json                        # per-model scores
└── plots/sc25_hindcast.png                   # all forecasts vs observed SC25
```

`data/` is gitignored — everything above regenerates from the scripts.

---

## 🎯 Model Configurations

### The Precursor Ensemble needs no config

It is a baseline-style forecaster with sensible constructor defaults
(`n_members=10, hidden=16, epochs=300, extended_features=True`) and is evaluated
automatically by `run_cv.py` / `hindcast_sc25.py`. To use it directly:

```python
from solar.models.precursor_ensemble import PrecursorEnsembleForecaster
from solar.data import load_solar_data
from solar.data.monthly import build_monthly_series

df = load_solar_data(dataset="ssn")
raw = build_monthly_series(df, target_col="sunspot_number", start_year=1749).values
fc = PrecursorEnsembleForecaster().forecast(raw, horizon=132)
# fc['mean'], fc['q10'], fc['q90']  -> each shape (132,)
```

### Deep model configs (`solar/configs/`)

1. **`wavenet_lstm_v2.yaml`** — recommended deep config: univariate SILSO monthly
   SSN (1749–present), min-max scaling (paper parity; no peak-compressing sqrt),
   quantile head, warmup→cosine, weight EMA, early stop on raw-unit val RMSE.
2. **`wavenet_lstm_tiny.yaml`** — ~70K-param ablation (LOCO shows it matches v2).
3. **`wavenet_lstm_v2_area.yaml`** — same model on total sunspot area (1874+),
   the paper's cross-consistency dataset.
4. **`wavenet_lstm_v2_term.yaml`** — v2 + causal terminator/cycle-length conditioning.
5. Legacy: **`seq2seq_precursors.yaml`** (2.46M-param seq2seq + Ap/Kp channels),
   **`seq2seq_quantile.yaml`**, **`base.yaml`**, **`ablation_tcn.yaml`** — comparison
   rows, superseded by the configs above.

### Custom deep config

Copy and modify a YAML file:

```yaml
experiment_name: my_experiment
seed: 42
device: auto  # auto, cpu, cuda, mps

data:
  normalization:
    method: minmax        # minmax, robust, standard, none
    transform: identity   # identity, asinh, sqrt, log1p

model:
  name: WaveNetLSTM       # or WaveNetAttnSeq2Seq, TCNOnly, NBEATSx
  head: quantile          # mse, quantile, combined
  quantiles: [0.1, 0.5, 0.9]
  decoder_lstm_hidden: 128
  wavenet:
    channels: 32
    kernel_size: 2
    num_layers: 10        # dilations 1..2^(n-1), RF = 1+(k-1)(2^n - 1)
    norm: weight          # weight, group, none

training:
  epochs: 100
  batch_size: 32
  lr: 5.0e-4
  early_stop_metric: val_rmse_raw   # select on raw-unit RMSE, not scaled loss
```

---

## 🔬 Advanced Features

### Leak-Free Evaluation Machinery
- **Time-axis split** (`solar/utils/splits.py`): train windows must have their
  entire target before the boundary; validation origins start at it. Disjoint
  targets are asserted at loader construction.
- **Leave-one-cycle-out CV** (`solar/eval/loco.py`): per-origin retraining with a
  fresh trainer and train-era-only scalers per fold.
- **Raw-unit metrics** (`solar/eval/metrics.py`): RMSE/MAE on raw monthly values,
  peak amplitude/timing on the 13-month smoothed series, interval coverage/width.
- **Conformal calibration** (`solar/eval/conformal.py`): CQR with finite-sample
  correction; pooled across the horizon.

### Shared Causal Features
`precursor_feature_matrix(history, horizon, extended=…)` in
`solar/models/baselines.py` is the single source of the causal precursor features.
The Hathaway baseline uses the base 4 scalars; the Precursor Ensemble uses the
extended 6. Identical extraction keeps the two directly comparable, and all
features are computed from `history` up to the forecast origin only (no look-ahead).

### Data Hygiene
- **Sentinel masking** (`solar/data/monthly.py`): SILSO/area files mark missing
  days as `-1.0`; these are masked to NaN at native resolution *before* monthly
  averaging (averaging them in drags early-record months negative — the old
  pipeline's `sqrt_shift=0.4` scaler artifact was exactly this bug).
- **Frozen scalers**: transform shifts are computed at `fit()` only; `transform()`
  never mutates state, so forward/inverse always agree. All scalers fit on the
  training era only. Inverse-transformed predictions are clipped at 0.

### Device Support
- Automatic CUDA / Apple-Silicon MPS / CPU selection for the deep model.
- The Precursor Ensemble and all baselines run comfortably on CPU.

---

## 🛠️ Advanced Usage

### Programmatic training

```python
from solar.trainers.seq2seq_trainer import Seq2SeqTrainer
from solar.utils.config import load_config
from solar.data import load_solar_data

config = load_config("solar/configs/wavenet_lstm_v2.yaml")
config.training.epochs = 100
trainer = Seq2SeqTrainer(config, device='auto')
results = trainer.train(load_solar_data())
```

### Custom plotting

```python
from solar.utils.plotting import SolarCyclePlotter

plotter = SolarCyclePlotter(style='publication')
fig = plotter.plot_single_cycle_with_uncertainty(
    actual=actual_data,
    prediction=predictions,
    uncertainty={'q10': q10, 'q50': q50, 'q90': q90},
    title="Solar Cycle 25 Prediction",
    save_path="my_prediction.png",
)
```

---

## 🏃 Quick Testing

```bash
# Fastest end-to-end check: baselines + Precursor Ensemble, fewer folds
uv run python scripts/run_cv.py --baselines-only --folds 4

# 2-epoch deep-model smoke test
uv run python scripts/train_seq2seq.py --config solar/configs/wavenet_lstm_v2.yaml --epochs 2 --batch-size 4

# CPU-only deep training
uv run python scripts/train_seq2seq.py --config solar/configs/wavenet_lstm_v2.yaml --epochs 5 --device cpu

# Run the test suite
uv run pytest tests/
```

---

## 🔧 Troubleshooting

1. **MPS errors on Apple Silicon** — force CPU:
   ```bash
   uv run python scripts/train_seq2seq.py --config your_config.yaml --device cpu
   ```
2. **Out of memory** — reduce batch size:
   ```bash
   uv run python scripts/train_seq2seq.py --config your_config.yaml --batch-size 8
   ```
3. **Deep training too slow** — use the tiny config or fewer epochs, or just rely
   on the Precursor Ensemble (`run_cv.py --baselines-only`), which needs no GPU.
4. **YAML config errors** — check indentation; ensure enum values are strings; use
   the provided configs as templates.

---

## 📚 Project Structure

```
solarcycle-updated/
├── solar/                          # Main package
│   ├── data/                       # Data collection and loading
│   │   ├── collection.py           # Fetch SILSO monthly + daily, sunspot area, Kp/Ap, F10.7
│   │   ├── monthly.py              # Sentinel masking + monthly series (single source of truth)
│   │   └── loading.py              # Dataset selection (ssn | area | multivariate)
│   ├── models/                     # Architectures + baselines
│   │   ├── precursor_ensemble.py   # PRODUCTION: precursor deep ensemble (best on LOCO)
│   │   ├── baselines.py            # Climatology, persistence, Hathaway; shared precursor_feature_matrix
│   │   ├── wavenet_lstm_direct.py  # Deep model: corrected WaveNet+LSTM direct forecaster
│   │   ├── wavenet_attn_seq2seq.py # Legacy seq2seq (comparison only)
│   │   ├── tcn_only.py             # TCN ablation
│   │   ├── nbeatsx.py              # N-BEATS ablation
│   │   └── heads.py                # Prediction heads (MSE, Quantile, Combined)
│   ├── trainers/                   # Training logic
│   │   ├── seq2seq_trainer.py      # Trainer: registry, leak-free loaders, raw-unit val metrics
│   │   └── mixins.py               # Early stopping, scheduling, AMP, EMA, checkpoints
│   ├── eval/                       # Honest evaluation
│   │   ├── loco.py                 # Leave-one-cycle-out CV (per-origin retraining)
│   │   ├── metrics.py              # Raw-unit RMSE/MAE + smoothed peak metrics
│   │   └── conformal.py            # CQR interval calibration
│   ├── utils/                      # Utilities
│   │   ├── config.py               # Pydantic + YAML configuration
│   │   ├── splits.py               # Time-axis splits with disjoint-target guarantee
│   │   ├── normalization.py        # Frozen train-only scalers (minmax/robust/asinh/sqrt)
│   │   ├── precursors.py           # CAUSAL minima detection + cycle-length precursor
│   │   ├── plotting.py             # Visualization utilities (SolarCyclePlotter)
│   │   └── peak_metrics.py         # Peak detection
│   └── configs/                    # Deep-model YAML configs (see Model Configurations)
├── scripts/
│   ├── run_cv.py                   # LOCO leaderboard + plots (the honest numbers) — START HERE
│   ├── train_seq2seq.py            # Deep-model training entry point
│   ├── hindcast_sc25.py            # SC25 calibration spot-check + plot
│   ├── forecast_next_cycle.py      # Cycle 26 forecast with conformal intervals + plot
│   └── backtest.py                 # Visual backtest of a trained run (in-sample panels labeled)
├── tests/
│   ├── test_data_integrity.py      # Sentinels, scaler round-trips, split leakage
│   ├── test_causality.py           # Precursor truncation-invariance
│   └── test_device_compatibility.py
├── legacy/                         # Superseded first-generation scripts (kept for reference)
├── data/                           # Data + experiment outputs (gitignored)
└── README.md                       # This file
```

> **Note:** `legacy/` holds the original, superseded scripts (multiple standalone
> model/trainer variants). They are not part of the active pipeline and require the
> optional dependencies: `uv sync --extra legacy`.

---

## 🔬 Research Applications

- **Solar Physics** — solar cycle dynamics and the limits of amplitude prediction.
- **Space Weather** — cycle-amplitude forecasts with calibrated uncertainty.
- **Time-Series Forecasting** — scarce-data regimes where a strong prior + tiny
  model beats a large free-form network.
- **Uncertainty Quantification** — deep ensembles and conformal calibration on
  time series.

---

## 📖 Key References

- **Benson et al. (2020)** — *Forecasting Solar Cycle 25 Using Deep Neural
  Networks*, Solar Physics 295:65 (the work this repo rebuilds and re-evaluates).
- **Hathaway, Wilson & Reichmann (1994)** — the parametric solar-cycle shape
  function used by the decoder.
- **Lakshminarayanan et al. (2017)** — deep ensembles for predictive uncertainty.
- **WaveNet** — Van den Oord et al. (2016), dilated causal convolutions.
- **Quantile Regression** — Koenker & Bassett (1978).
- **Conformal Prediction / CQR** — Vovk et al. (2005); Romano et al. (2019).
- **Terminator / Hale-cycle precursors** — McIntosh et al. (2023), cycle-length
  terminator separation as an amplitude precursor.

---

## 🏆 Citation

```bibtex
@software{solar_cycle_prediction_2026,
  title={Solar Cycle Prediction: Precursor Ensembles and Honest Evaluation},
  author={Bernard Benson},
  year={2026},
  url={https://github.com/bernardbenson/solarcycle-updated},
  note={Precursor deep ensemble + WaveNet-LSTM with leave-one-cycle-out evaluation and calibrated uncertainty}
}
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Add tests for new functionality
4. Commit changes (`git commit -m 'Add amazing feature'`)
5. Push to the branch and open a pull request

## 📄 License

MIT License — see the LICENSE file for details.

---

**🌞 Happy Solar Forecasting! 🔮**
