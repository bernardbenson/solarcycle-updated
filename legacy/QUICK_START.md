# 🚀 Quick Start Guide - Enhanced Solar Cycle Prediction

## ⚡ **1-Minute Setup with uv**

```bash
# Navigate to project
cd solarcycle-updated

# Install dependencies (auto-creates .venv)
uv sync

# Test device compatibility  
uv run python test_device_compatibility.py

# Quick 5-epoch test run
uv run python scripts/train_seq2seq.py \
    --config solar/configs/seq2seq_quantile.yaml \
    --epochs 5 \
    --batch-size 8
```

## 🎯 **What You Get**

After the quick test run, check `data/experiments/run_*/`:

- **📊 Uncertainty predictions**: P10/P50/P90 quantiles
- **📈 Training plots**: Loss curves with coverage statistics  
- **🎨 Cycle visualizations**: Overlapping predictions with uncertainty bands
- **🔬 Peak analysis**: Distribution of predicted solar maximum
- **💾 Model checkpoints**: Best model saved automatically

## 🖥️ **Device Compatibility**

The system automatically detects and uses the best available device:

- **✅ Apple Silicon** (M1/M2/M3): Uses MPS acceleration
- **✅ NVIDIA GPU**: Uses CUDA with optional AMP
- **✅ CPU fallback**: Works on any system

## 📚 **Next Steps**

### Full Training Run
```bash
uv run python scripts/train_seq2seq.py \
    --config solar/configs/seq2seq_quantile.yaml \
    --epochs 100
```

### Compare Models
```bash
# TCN baseline
uv run python scripts/train_seq2seq.py --config solar/configs/ablation_tcn.yaml

# Original TensorFlow-style (legacy)
uv run python train_tensorflow_style.py
```

### Custom Configuration
```bash
# Override specific parameters
uv run python scripts/train_seq2seq.py \
    --config solar/configs/seq2seq_quantile.yaml \
    --epochs 50 \
    --batch-size 16 \
    --device mps
```

## 🔧 **Key Features**

- **🎲 Probabilistic forecasting** with quantile regression
- **🔄 Teacher forcing** for stable sequence generation  
- **🎯 Attention mechanisms** for long-range dependencies
- **📊 Rolling-origin CV** for realistic evaluation
- **🎨 Enhanced visualization** with uncertainty bands
- **⚙️ Production-ready** with experiment tracking

## 📁 **Output Structure**

Each run creates a unique directory:
```
data/experiments/run_20241029_143022_abc123ef/
├── model_config.yaml          # Exact configuration
├── training_metrics.json      # Loss curves & statistics  
├── best_model.pt              # Model checkpoint
├── scaler.json                # Data preprocessing params
├── predictions/               # Numerical predictions
│   ├── cycle_25_p10.npy      # 10th percentile
│   ├── cycle_25_p50.npy      # Median (50th percentile)
│   └── cycle_25_p90.npy      # 90th percentile
└── plots/                     # Visualizations
    ├── training_history.png
    ├── overlapping_cycles_enhanced.png
    └── uncertainty_bands.png
```

## 🎛️ **Configuration Examples**

### Probabilistic Setup (Recommended)
```yaml
# solar/configs/seq2seq_quantile.yaml
model:
  head: "quantile"
  quantiles: [0.1, 0.5, 0.9]
  mc_dropout_samples: 30
training:
  teacher_forcing: 0.6
  early_stop_patience: 20
```

### Fast Baseline
```yaml
# solar/configs/ablation_tcn.yaml  
model:
  name: "TCNOnly"
  head: "mse"
training:
  epochs: 50
  teacher_forcing: 0.0  # Not used for TCN
```

## 🚨 **Troubleshooting**

### Import Errors
```bash
# Ensure virtual environment is activated
source .venv/bin/activate
# OR use uv run prefix
uv run python scripts/train_seq2seq.py
```

### Memory Issues
```bash
# Reduce batch size
uv run python scripts/train_seq2seq.py --batch-size 4

# Or use CPU
uv run python scripts/train_seq2seq.py --device cpu
```

### No Real Data
The system automatically creates synthetic solar cycle data if no real data is found, so you can test immediately.

## 💡 **Tips**

1. **Start small**: Use `--epochs 5` for quick tests
2. **Monitor device**: Check `test_device_compatibility.py` output
3. **Compare baselines**: Run both seq2seq and TCN configs
4. **Check outputs**: Visualizations are in `data/experiments/run_*/plots/`
5. **Use unique runs**: Each run gets a timestamp + UUID for isolation

For complete documentation, see `ARCHITECTURE_UPGRADE.md`.