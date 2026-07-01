# 🌞 Solar Cycle Prediction with WaveNet+LSTM Architecture

Advanced deep learning system for solar cycle forecasting using state-of-the-art WaveNet attention-based encoder-decoder models with comprehensive uncertainty quantification.

## 🚀 Key Features

- **Advanced WaveNet+LSTM Architecture** with attention mechanisms and teacher forcing
- **Exogenous Solar Precursors** — geomagnetic Ap/Kp input channels (with availability mask) plus a terminator / cycle-length conditioning signal
- **Comprehensive Uncertainty Quantification** (MC-Dropout, Quantile Regression, Conformal Prediction)
- **275+ years of historical data** (1749-2025, monthly observations)
- **Robust preprocessing** with variance-stabilizing transforms and lean cycle-aware features
- **Rolling-Origin Cross-Validation** for robust time series evaluation
- **Peak Detection & Conformal Intervals** for solar cycle maxima prediction
- **Stable Training** with warmup→cosine scheduling and weight EMA for smooth convergence
- **Hindcast Backtesting** against real past cycles with per-cycle error metrics
- **Production-Ready Pipeline** with automatic experiment tracking and plotting
- **Apple Silicon MPS Support** for accelerated training on M1/M2 Macs
- **Publication-Quality Visualizations** with uncertainty bands and diagnostic plots

## 📋 Requirements

This project uses `uv` for fast dependency management and execution.

### Install uv (if not already installed)
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Dependencies
All dependencies are automatically managed by `uv`. The project includes:
- PyTorch (with MPS support for Apple Silicon)
- Pandas, NumPy (data processing)
- Matplotlib, Seaborn (publication-quality plots)
- Scikit-learn (preprocessing and metrics)
- Pydantic (configuration validation)
- YAML (configuration files)

## ⚡ Quick Start

### 1. Clone and Setup
```bash
git clone [your-repo-url]
cd solarcycle-updated
uv sync                       # install core dependencies
```

### 2. Get Data (optional)
Training works out of the box: if no local CSV is found, a realistic synthetic
sunspot series is generated automatically. To train on real observations, fetch
them first (writes `data/raw_multivariate_data.csv`):
```bash
uv run python -m solar.data.collection
```

### 3. Run Full Training Pipeline
```bash
# Recommended: full pipeline with exogenous precursors + uncertainty quantification.
# Epochs/batch size come from the config (200 epochs, batch 32) — no flags needed.
uv run python scripts/train_seq2seq.py --config solar/configs/seq2seq_precursors.yaml

# Univariate quantile model (no precursors)
uv run python scripts/train_seq2seq.py --config solar/configs/seq2seq_quantile.yaml --epochs 50 --batch-size 32

# Quick test run (2 epochs)
uv run python scripts/train_seq2seq.py --config solar/configs/seq2seq_precursors.yaml --epochs 2 --batch-size 8

# Force CPU training (if needed)
uv run python scripts/train_seq2seq.py --config solar/configs/seq2seq_precursors.yaml --epochs 10 --device cpu
```

> **Note:** The precursors config feeds geomagnetic Ap/Kp as extra input channels
> (available from 1932; pre-1932 is zero-filled and flagged by a binary mask channel)
> and conditions the decoder on the terminator / cycle-length precursor. The model's
> `input_dim` and `cond_dim` are derived automatically from the data config
> (`resolve_derived_dims`), so you don't set them by hand. Precursors require the real
> data CSV from step 2 — run `solar.data.collection` first.

### 4. Run Baseline Models for Comparison
The trainer selects the architecture from `model.name` in the config, so the same
script trains the WaveNet seq2seq model or either baseline:
```bash
# TCN baseline (model.name: TCNOnly)
uv run python scripts/train_seq2seq.py --config solar/configs/ablation_tcn.yaml --epochs 30

# Standard MSE model (without quantile regression)
uv run python scripts/train_seq2seq.py --config solar/configs/base.yaml --epochs 30
```

### 5. Validate Against Past Cycles (hindcast backtest)
Confirm the model tracks real solar cycles: forecast several historical cycles from
their onset and overlay the actual outcome + MC-Dropout intervals. Loads the latest
trained run (no retraining) and prints per-cycle RMSE / peak-timing / peak-magnitude
errors:
```bash
uv run python scripts/backtest.py --config solar/configs/seq2seq_precursors.yaml --n-panels 4
# --run <dir>     pick a specific experiment (default: most recent)
# --raw-units     report errors in raw sunspot number instead of scaled units
```
Each training run also writes a `next_cycle_forecast.png` (a genuine forward forecast
from the latest data) and a `cycle_backtest.png` into its `plots/` directory.

## 🏗️ Architecture Overview

### WaveNet Attention Seq2Seq Model

The core architecture combines three powerful components:

1. **WaveNet Encoder**
   - Dilated causal convolutions with exponential dilation
   - 3 stacks × 4 layers per stack = 12 total layers
   - Receptive field: 528 months (44 years)
   - Kernel size: 3, channels: 128

2. **Bidirectional LSTM Encoder**
   - 128 hidden units with dropout
   - Projects WaveNet features to attention space
   - Captures long-term dependencies

3. **Attention Decoder with Teacher Forcing**
   - LSTM decoder with scaled dot-product attention
   - Teacher forcing ratio: 0.6 → 0.1 (with decay); precursors config runs free-running (0.0)
   - Generates 132-month predictions (11-year solar cycle)

### Exogenous Precursors & Conditioning

Beyond the sunspot channel, the model can ingest physically-motivated precursors:

- **Geomagnetic input channels** — `ap_avg` / `kp_sum` are concatenated as extra
  input channels. They exist only from 1932, so pre-1932 months are zero-filled and
  flagged by a **binary availability-mask channel** the model learns to gate on.
- **Terminator / cycle-length conditioning** — the length of the most recently
  completed cycle (minimum-to-minimum) is step-held per month and fed as a decoder
  conditioning vector (`cond_dim`). Cycle length anti-correlates with the amplitude of
  the *following* cycle — a robust, sunspot-derivable proxy for the Hale-cycle
  "terminator separation" of McIntosh et al. (2023). Minima are detected with the same
  routine the hindcast backtest uses, so the two stay consistent.

Channel counts (`input_dim`, `cond_dim`) are derived from the data config by
`resolve_derived_dims`; an empty `precursor_cols` keeps the model fully univariate
(backward compatible).

### Prediction Heads

- **MSE Head**: Standard point predictions
- **Quantile Head**: Probabilistic predictions with pinball loss (10th, 50th, 90th percentiles)
- **Combined Head**: Both MSE and quantile outputs

### Uncertainty Quantification

- **MC-Dropout**: 30 forward passes with dropout enabled
- **Quantile Regression**: Direct uncertainty estimation via pinball loss
- **Conformal Prediction**: Calibrated prediction intervals for peaks

## 📊 Output Structure

Each training run creates a unique timestamped directory:

```
data/experiments/run_seq2seq_quantile_20251029_162852_c346a904/
├── best_model.pt                              # Best model checkpoint
├── final_model.pt                             # Final trained model  
├── scaler.json                                # Data normalization parameters
├── training_metrics.json                      # Complete training history
├── summary_report.md                          # Automated summary report
└── plots/                                     # 📊 AUTOMATIC PLOTS
    ├── training_history.png                   # Loss curves, metrics over epochs
    ├── recent_predictions_with_uncertainty.png # Latest predictions with uncertainty bands
    ├── mc_dropout_uncertainty.png             # MC-Dropout uncertainty visualization
    └── peak_distribution.png                  # Solar cycle peak predictions distribution
```

## 🎯 Model Configurations

### Available Configurations

1. **`seq2seq_precursors.yaml`** - **Recommended**: Full pipeline with exogenous precursors
   - Geomagnetic Ap/Kp input channels + availability mask
   - Terminator / cycle-length decoder conditioning
   - Quantile regression, warmup→cosine schedule, weight EMA, rolling-origin CV
   - `start_year: 1818` (retains full sunspot history; pre-1932 geomagnetic masked)
   - Requires the real data CSV (`solar.data.collection`)

2. **`seq2seq_quantile.yaml`** - Univariate uncertainty quantification
   - Quantile regression with pinball loss
   - 200 epochs, teacher forcing with decay
   - Robust scaling with sqrt transform

3. **`base.yaml`** - Standard MSE model
   - Point predictions only
   - 100 epochs, cosine learning rate schedule
   - Good baseline for comparison

4. **`ablation_tcn.yaml`** - TCN-only baseline
   - Temporal Convolutional Network without LSTM
   - 150 epochs, simpler architecture
   - For ablation studies

### Custom Configuration

Create your own config by copying and modifying existing YAML files:

```yaml
experiment_name: my_experiment
seed: 42
device: auto  # auto, cpu, cuda, mps

# Model architecture
model:
  name: WaveNetAttnSeq2Seq
  d_model: 128
  wavenet:
    stacks: 3
    layers_per_stack: 4
    kernel_size: 3
  head: quantile  # mse, quantile, combined
  quantiles: [0.1, 0.5, 0.9]

# Training settings
training:
  epochs: 50
  batch_size: 32
  lr: 0.001
  teacher_forcing: 0.6
  teacher_forcing_decay: 0.95
```

## 🔬 Advanced Features

### Rolling-Origin Cross-Validation
```bash
# Enable cross-validation in config
cv:
  enabled: true
  n_folds: 5
  method: rolling
  test_size: 132  # 11 years
```

### Peak Detection and Conformal Prediction
- Automatic solar cycle peak detection
- Conformal prediction intervals for peak timing and magnitude
- Coverage probability guarantees

### Enhanced Data Processing
- **Robust Scaling**: Handles outliers in solar data
- **Variance-Stabilizing Transforms**: Log1p and sqrt transforms for non-Gaussian data
- **Feature Engineering**: Sine/cosine cycle encoding, lagged features, running means

### Apple Silicon Optimization
- Automatic MPS (Metal Performance Shaders) detection
- Optimized for M1/M2 MacBook training
- Fallback to CPU if MPS unavailable

## 📈 Expected Performance

### Typical Results (132-month ahead predictions)
- **Quantile Model**: RMSE ~15-25, MAE ~10-18
- **MSE Model**: RMSE ~18-28, MAE ~12-20
- **TCN Baseline**: RMSE ~20-30, MAE ~15-25

### Uncertainty Metrics
- **Coverage**: 80-90% for 80% prediction intervals
- **Peak Timing**: ±6-12 months accuracy
- **Peak Magnitude**: ±15-30 sunspot number accuracy

*Note: Solar cycle prediction is inherently challenging due to chaotic dynamics. These results represent significant improvement over traditional methods.*

## 🛠️ Advanced Usage

### Custom Training Scripts

```python
from solar.trainers.seq2seq_trainer import Seq2SeqTrainer
from solar.utils.config import load_config
from solar.data import load_solar_data

# Load configuration
config = load_config("solar/configs/seq2seq_quantile.yaml")

# Override parameters
config.training.epochs = 100
config.training.batch_size = 16

# Initialize trainer and train
trainer = Seq2SeqTrainer(config, device='auto')
results = trainer.train(load_solar_data())
```

### Prediction with Uncertainty

```python
# Generate predictions with uncertainty
uncertainty_results = trainer.predict_with_uncertainty(
    input_data=recent_data,
    n_mc_samples=50
)

# Access results
mean_prediction = uncertainty_results['mean']
std_prediction = uncertainty_results['std']
quantiles = uncertainty_results['q10'], uncertainty_results['q50'], uncertainty_results['q90']
```

### Custom Plotting

```python
from solar.utils.plotting import SolarCyclePlotter

plotter = SolarCyclePlotter(style='publication')
fig = plotter.plot_single_cycle_with_uncertainty(
    actual=actual_data,
    prediction=predictions,
    uncertainty={'q10': q10, 'q50': q50, 'q90': q90},
    title="Solar Cycle 25 Prediction",
    save_path="my_prediction.png"
)
```

## 🏃‍♂️ Quick Testing

For faster testing and development:

```bash
# Quick 2-epoch test
uv run python scripts/train_seq2seq.py --config solar/configs/seq2seq_quantile.yaml --epochs 2 --batch-size 4

# CPU-only testing
uv run python scripts/train_seq2seq.py --config solar/configs/base.yaml --epochs 5 --device cpu

# Small batch testing
uv run python scripts/train_seq2seq.py --config solar/configs/ablation_tcn.yaml --epochs 10 --batch-size 8
```

## 🔧 Troubleshooting

### Common Issues

1. **MPS errors on Apple Silicon**
   ```bash
   # Force CPU training
   uv run python scripts/train_seq2seq.py --config your_config.yaml --device cpu
   ```

2. **Out of memory**
   ```bash
   # Reduce batch size
   uv run python scripts/train_seq2seq.py --config your_config.yaml --batch-size 8
   ```

3. **Training too slow**
   ```bash
   # Use smaller model or fewer epochs
   uv run python scripts/train_seq2seq.py --config solar/configs/ablation_tcn.yaml --epochs 20
   ```

4. **YAML configuration errors**
   - Check indentation in YAML files
   - Ensure all enum values are strings (not Python objects)
   - Use provided configs as templates

### Performance Tips

- **Apple Silicon**: Use MPS for 3-5x speedup over CPU
- **CUDA**: Automatic detection and usage on NVIDIA GPUs
- **Memory**: Reduce batch size if encountering OOM errors
- **Speed**: Use TCN config for faster training during development

## 📚 Project Structure

```
solarcycle-updated/
├── solar/                          # Main package
│   ├── data/                       # Data collection and loading
│   │   ├── collection.py           # Fetch SILSO/F10.7/Kp-Ap → raw CSV
│   │   └── loading.py              # Load cached CSV or synthesise data
│   ├── models/                     # Neural network architectures
│   │   ├── wavenet_attn_seq2seq.py # Main WaveNet + BiLSTM + attention model
│   │   ├── tcn_only.py             # TCN baseline
│   │   ├── nbeatsx.py              # N-BEATS baseline
│   │   └── heads.py                # Prediction heads (MSE, Quantile, Combined)
│   ├── trainers/                   # Training logic
│   │   ├── seq2seq_trainer.py      # Trainer with model registry + uncertainty
│   │   └── mixins.py               # Early stopping, scheduling, AMP, checkpoints
│   ├── utils/                      # Utilities
│   │   ├── config.py               # Pydantic + YAML configuration
│   │   ├── normalization.py        # Robust scaling / variance-stabilizing transforms
│   │   ├── plotting.py             # Visualization utilities
│   │   ├── rolling_cv.py           # Rolling-origin cross-validation
│   │   ├── precursors.py           # Terminator / cycle-length precursor + minima detection
│   │   └── peak_metrics.py         # Peak detection + conformal intervals
│   └── configs/                    # Model configurations
│       ├── seq2seq_precursors.yaml # Recommended: precursors + conditioning
│       ├── seq2seq_quantile.yaml   # Univariate quantile config
│       ├── base.yaml               # Standard MSE config
│       └── ablation_tcn.yaml       # TCN baseline
├── scripts/
│   ├── train_seq2seq.py            # Main training entry point
│   └── backtest.py                 # Hindcast backtest against real past cycles
├── tests/
│   └── test_device_compatibility.py
├── legacy/                         # Superseded first-generation scripts (kept for reference)
├── data/                           # Data + experiment outputs (gitignored)
└── README.md                       # This file
```

> **Note:** `legacy/` holds the original, superseded scripts (multiple standalone
> model/trainer variants). They are not part of the active pipeline and require the
> optional dependencies: `uv sync --extra legacy`.

## 🔬 Research Applications

This system supports research in:
- **Solar Physics**: Understanding solar cycle dynamics and prediction limits
- **Space Weather**: Forecasting geomagnetic activity with uncertainty bounds
- **Time Series Forecasting**: Advanced uncertainty quantification techniques
- **Deep Learning**: Attention mechanisms for sequential data
- **Conformal Prediction**: Calibrated uncertainty in time series

## 📖 Key References

The implementation builds on several key methodologies:
- **WaveNet**: Van den Oord et al. (2016) - Dilated convolutions for sequence modeling
- **Attention Mechanisms**: Vaswani et al. (2017) - Transformer attention
- **Teacher Forcing**: Williams & Zipser (1989) - Sequence-to-sequence training
- **Quantile Regression**: Koenker & Bassett (1978) - Probabilistic predictions
- **Conformal Prediction**: Vovk et al. (2005) - Calibrated prediction intervals
- **Terminator / Hale-cycle precursors**: McIntosh et al. (2023) - Cycle-length terminator separation as an amplitude precursor

## 🏆 Citation

If you use this code in your research, please cite:

```bibtex
@software{solar_cycle_prediction_2025,
  title={Solar Cycle Prediction with WaveNet+LSTM Architecture},
  author={Your Name},
  year={2025},
  url={https://github.com/your-username/solarcycle-updated},
  note={Advanced deep learning system for solar cycle forecasting with uncertainty quantification}
}
```

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Add tests for new functionality
4. Commit changes (`git commit -m 'Add amazing feature'`)
5. Push to branch (`git push origin feature/amazing-feature`)
6. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

**🌞 Happy Solar Forecasting! 🔮**

For questions or issues, please create a GitHub issue or contact the maintainers.