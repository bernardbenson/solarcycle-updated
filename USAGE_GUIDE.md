# 🌞 Quick Usage Guide - Multivariate Sunspot Prediction

## 🚀 One-Command Quick Start

For immediate testing and demonstration:

```bash
# Run the complete example pipeline
uv run python run_example.py
```

This will:
1. Collect recent solar data (2015-2025)
2. Apply feature engineering (131 features)
3. Test PyTorch models on subset
4. Generate visualizations
5. Show results and next steps

**Expected runtime: 5-10 minutes**

## 📋 Step-by-Step Usage

### Basic Commands

```bash
# 1. Collect all historical data (275+ years)
uv run python main.py --phase collect

# 2. Create engineered features  
uv run python main.py --phase preprocess

# 3. Visualize the data
uv run python main.py --phase visualize

# 4. Train PyTorch models
uv run python main.py --phase train

# 5. Evaluate and compare models
uv run python main.py --phase evaluate

# Or run everything at once:
uv run python main.py --phase all
```

### Quick Testing Commands

```bash
# Test models on recent data only (faster)
uv run python quick_pytorch_test.py

# Quick data visualization
uv run python quick_viz_test.py

# Efficient visualization for large datasets
uv run python efficient_viz.py

# Create publication-quality result plots
uv run python visualize_results.py
```

## 🎯 What Each Phase Does

| Phase | Command | Output | Time |
|-------|---------|--------|------|
| **Collect** | `--phase collect` | Raw multivariate data (81K+ obs) | 2-5 min |
| **Preprocess** | `--phase preprocess` | 131 engineered features | 1-3 min |
| **Visualize** | `--phase visualize` | Historical plots, correlations | 2-5 min |
| **Train** | `--phase train` | Trained PyTorch models | 15-60 min |
| **Evaluate** | `--phase evaluate` | Performance metrics, comparisons | 5-15 min |

## 📊 Expected Outputs

### Data Files
- `data/raw_multivariate_data.csv` - Raw collected data
- `data/engineered_multivariate_data.csv` - ML-ready features
- `data/training_results/` - Trained model files
- `data/evaluation_results/` - Performance analysis

### Visualizations
- Historical overview (275+ years)
- Solar cycle analysis
- Feature correlations
- Model performance comparisons
- Prediction accuracy plots
- Uncertainty quantification

### Model Performance
Expected results on actual daily sunspot numbers:
- **Transformer**: RMSE ~45-55, R² ~0.3-0.5 ⭐ Best
- **LSTM**: RMSE ~50-60, R² ~0.2-0.4
- **GRU**: RMSE ~50-60, R² ~0.2-0.4
- **Ensemble**: RMSE ~40-50, R² ~0.4-0.6 ⭐ Best overall

## ⚡ Performance Tips

### For Faster Results
```bash
# Use recent data only
uv run python main.py --phase collect --start-year 2010

# Quick model testing
uv run python quick_pytorch_test.py

# Efficient visualization
uv run python efficient_viz.py
```

### For Full Research Quality
```bash
# Complete historical data
uv run python main.py --phase collect --start-year 1749

# Full training with early stopping
uv run python main.py --phase train

# Comprehensive evaluation
uv run python main.py --phase evaluate
```

## 🎨 Visualization Examples

### 1. Historical Data Overview
```bash
uv run python main.py --phase visualize
```
Creates:
- 275+ years of sunspot activity
- Solar cycle identification
- Multivariate correlations
- Feature distributions

### 2. Model Results
```bash
uv run python visualize_results.py
```
Creates:
- Training history plots
- Prediction vs actual scatter plots
- Time series prediction plots
- Residual analysis
- Uncertainty bands

### 3. Quick Data Check
```bash
uv run python quick_viz_test.py
```
Creates:
- Basic multivariate overview
- Data statistics summary
- Quick correlation plot

## 🔧 Troubleshooting

### Common Issues & Solutions

**"Data file not found"**
```bash
# Run data collection first
uv run python main.py --phase collect
```

**"Training takes too long"**
```bash
# Use smaller dataset for testing
uv run python quick_pytorch_test.py
```

**"Out of memory"**
```bash
# Reduce batch size in main.py (line ~133)
training_config = {'batch_size': 16, ...}  # Reduce from 32
```

**"Import errors"**
```bash
# Check uv installation
uv --version

# Reinstall dependencies
uv sync
```

## 📈 Interpreting Results

### Model Metrics
- **RMSE**: Root Mean Square Error (lower = better)
- **R²**: Coefficient of determination (higher = better, max = 1.0)
- **MAE**: Mean Absolute Error (lower = better)
- **MAPE**: Mean Absolute Percentage Error (lower = better)

### What's Challenging
- Predicting **actual daily sunspot numbers** (not smoothed)
- Capturing **extreme solar events** (flares, quiet periods)
- **11-year solar cycle** complexity
- **Long-term secular variations**

### What's Good Performance
- **R² > 0.3**: Good for daily sunspot prediction
- **RMSE < 50**: Reasonable error for sunspot scale (0-400+)
- **Directional accuracy > 60%**: Captures trend changes
- **Good uncertainty quantification**: Reliable prediction intervals

## 🔬 Research Applications

This code supports research in:
- **Solar physics**: Understanding solar cycle dynamics
- **Space weather**: Predicting geomagnetic storms
- **Climate science**: Solar-terrestrial connections
- **Machine learning**: Advanced time series methods
- **Uncertainty quantification**: Probabilistic forecasting

## 📝 Next Steps After Running

1. **Analyze Results**: Check `data/evaluation_results/evaluation_report.txt`
2. **Compare Models**: Look at performance comparison plots
3. **Validate Predictions**: Examine Solar Cycle 25 validation
4. **Customize Models**: Modify architecture in `pytorch_models.py`
5. **Extended Training**: Run with full historical data
6. **Paper Writing**: Use generated plots and metrics

## 🎯 Success Checklist

- [ ] ✅ Data collected successfully (81K+ observations)
- [ ] ✅ Features engineered (131 variables)
- [ ] ✅ Models trained (Transformer, LSTM, GRU)
- [ ] ✅ Visualizations created (8+ plot types)
- [ ] ✅ Evaluation completed (comprehensive metrics)
- [ ] ✅ Results documented (summary reports)

**🎉 Ready for research publication!**

---

**Need help?** 
- 📖 Check `README.md` for detailed documentation
- 🐛 Create GitHub issue for bugs
- 💡 See example outputs in `data/visualizations/`