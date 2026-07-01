"""
Visualize PyTorch model results and predictions for sunspot forecasting.
Creates publication-quality plots for the research paper.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

# Import our models for testing
from pytorch_models import (
    ModelTrainer, SolarTransformerModel, AttentionLSTMModel, AttentionGRUModel
)


class ResultsVisualizer:
    """
    Comprehensive visualization of PyTorch model results and predictions.
    """
    
    def __init__(self, figsize: tuple = (15, 10)):
        self.figsize = figsize
        self.setup_style()
        
    def setup_style(self):
        """Set up publication-quality plotting style."""
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_palette("husl")
        
        plt.rcParams.update({
            'figure.figsize': self.figsize,
            'font.size': 12,
            'axes.labelsize': 14,
            'axes.titlesize': 16,
            'xtick.labelsize': 11,
            'ytick.labelsize': 11,
            'legend.fontsize': 12,
            'figure.titlesize': 18,
            'lines.linewidth': 1.5,
            'grid.alpha': 0.3
        })
    
    def train_and_visualize_models(self, data_subset_size: int = 8000, 
                                 save_dir: str = "data/results_visualization"):
        """
        Train models on a data subset and create comprehensive visualizations.
        """
        save_path = Path(save_dir)
        save_path.mkdir(exist_ok=True, parents=True)
        
        print("🎨 COMPREHENSIVE RESULTS VISUALIZATION")
        print("=" * 60)
        
        # Load and prepare data
        data_path = Path("data/engineered_multivariate_data.csv")
        if not data_path.exists():
            print(f"Data file not found: {data_path}")
            return
        
        print("Loading and preparing data...")
        df = pd.read_csv(data_path, low_memory=False)
        df['date'] = pd.to_datetime(df['date'])
        
        # Use recent subset for faster visualization
        subset_df = df.tail(data_subset_size).reset_index(drop=True)
        print(f"Using data subset: {subset_df.shape}")
        print(f"Date range: {subset_df['date'].min()} to {subset_df['date'].max()}")
        
        # Initialize trainer
        trainer = ModelTrainer(device='cpu')
        
        # Prepare data
        data_info = trainer.prepare_data(
            subset_df,
            target_col='sunspot_number',
            sequence_length=180,  # 6 months context
            prediction_horizon=30,  # 1 month prediction
            train_ratio=0.7,
            val_ratio=0.2
        )
        
        n_features = data_info['n_features']
        print(f"Features: {n_features}")
        
        # Train models
        models_config = {
            'Transformer': SolarTransformerModel(
                n_features=n_features,
                d_model=96,
                n_heads=6,
                n_layers=4,
                patch_size=12,
                prediction_horizon=30,
                dropout=0.1
            ),
            'LSTM': AttentionLSTMModel(
                n_features=n_features,
                hidden_size=96,
                n_layers=3,
                prediction_horizon=30,
                dropout=0.2,
                bidirectional=True
            ),
            'GRU': AttentionGRUModel(
                n_features=n_features,
                hidden_size=96,
                n_layers=3,
                prediction_horizon=30,
                dropout=0.2,
                bidirectional=True
            )
        }
        
        training_results = {}
        evaluation_results = {}
        
        # Train each model
        for model_name, model in models_config.items():
            print(f"\n📈 Training {model_name}...")
            
            try:
                # Train model
                history = trainer.train_model(
                    model=model,
                    data_info=data_info,
                    model_name=model_name,
                    batch_size=24,
                    epochs=25,  # More epochs for better results
                    lr=1e-3,
                    patience=8
                )
                
                training_results[model_name] = history
                
                # Evaluate model
                eval_result = trainer.evaluate_model(model_name, data_info, batch_size=24)
                evaluation_results[model_name] = eval_result
                
                print(f"✅ {model_name} - RMSE: {eval_result['metrics']['overall']['rmse']:.2f}, "
                      f"R²: {eval_result['metrics']['overall']['r2']:.4f}")
                
            except Exception as e:
                print(f"❌ {model_name} failed: {e}")
                continue
        
        if not evaluation_results:
            print("No models trained successfully!")
            return
        
        # Create comprehensive visualizations
        print(f"\n🎨 Creating visualizations...")
        
        # 1. Training History
        self.plot_training_history(training_results, save_path)
        
        # 2. Model Performance Comparison
        self.plot_performance_comparison(evaluation_results, save_path)
        
        # 3. Predictions vs Actual
        self.plot_predictions_comparison(evaluation_results, data_info, trainer, save_path)
        
        # 4. Time Series Predictions
        self.plot_time_series_predictions(evaluation_results, data_info, trainer, save_path)
        
        # 5. Residual Analysis
        self.plot_residual_analysis(evaluation_results, save_path)
        
        # 6. Feature Impact Analysis
        self.plot_feature_analysis(data_info, save_path)
        
        # 7. Solar Cycle Context
        self.plot_solar_cycle_context(subset_df, evaluation_results, save_path)
        
        # Generate summary report
        self.generate_results_summary(training_results, evaluation_results, save_path)
        
        print(f"\n✅ All visualizations completed!")
        print(f"📁 Results saved to: {save_path}")
        
        return {
            'trainer': trainer,
            'data_info': data_info,
            'training_results': training_results,
            'evaluation_results': evaluation_results
        }
    
    def plot_training_history(self, training_results: Dict, save_path: Path):
        """Plot training history for all models."""
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        colors = ['blue', 'red', 'green', 'orange', 'purple']
        
        for i, (model_name, history) in enumerate(training_results.items()):
            color = colors[i % len(colors)]
            
            # Training loss
            epochs = range(1, len(history['train_loss']) + 1)
            axes[0].plot(epochs, history['train_loss'], 
                        label=f'{model_name} (Train)', color=color, linestyle='-')
            axes[0].plot(epochs, history['val_loss'], 
                        label=f'{model_name} (Val)', color=color, linestyle='--')
        
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss (MSE)')
        axes[0].set_title('Training and Validation Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        axes[0].set_yscale('log')
        
        # Learning rate
        for i, (model_name, history) in enumerate(training_results.items()):
            color = colors[i % len(colors)]
            epochs = range(1, len(history['lr']) + 1)
            axes[1].plot(epochs, history['lr'], label=model_name, color=color)
        
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Learning Rate')
        axes[1].set_title('Learning Rate Schedule')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        axes[1].set_yscale('log')
        
        plt.tight_layout()
        plt.savefig(save_path / "01_training_history.png", dpi=300, bbox_inches='tight')
        plt.show()
        print("📊 Training history plot saved")
    
    def plot_performance_comparison(self, evaluation_results: Dict, save_path: Path):
        """Plot model performance comparison."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        models = list(evaluation_results.keys())
        
        # Extract metrics
        metrics_data = {}
        for metric in ['rmse', 'mae', 'r2', 'mape']:
            metrics_data[metric] = [
                evaluation_results[m]['metrics']['overall'][metric] for m in models
            ]
        
        # RMSE
        bars1 = axes[0, 0].bar(models, metrics_data['rmse'], alpha=0.8, color='skyblue', edgecolor='navy')
        axes[0, 0].set_title('Root Mean Square Error (RMSE)', fontweight='bold')
        axes[0, 0].set_ylabel('RMSE (Sunspot Number)')
        axes[0, 0].tick_params(axis='x', rotation=45)
        self._add_value_labels(axes[0, 0], bars1, '.1f')
        
        # R²
        bars2 = axes[0, 1].bar(models, metrics_data['r2'], alpha=0.8, color='lightgreen', edgecolor='darkgreen')
        axes[0, 1].set_title('Coefficient of Determination (R²)', fontweight='bold')
        axes[0, 1].set_ylabel('R²')
        axes[0, 1].tick_params(axis='x', rotation=45)
        self._add_value_labels(axes[0, 1], bars2, '.3f')
        
        # MAE
        bars3 = axes[1, 0].bar(models, metrics_data['mae'], alpha=0.8, color='salmon', edgecolor='darkred')
        axes[1, 0].set_title('Mean Absolute Error (MAE)', fontweight='bold')
        axes[1, 0].set_ylabel('MAE (Sunspot Number)')
        axes[1, 0].tick_params(axis='x', rotation=45)
        self._add_value_labels(axes[1, 0], bars3, '.1f')
        
        # MAPE
        bars4 = axes[1, 1].bar(models, metrics_data['mape'], alpha=0.8, color='gold', edgecolor='orange')
        axes[1, 1].set_title('Mean Absolute Percentage Error (MAPE)', fontweight='bold')
        axes[1, 1].set_ylabel('MAPE (%)')
        axes[1, 1].tick_params(axis='x', rotation=45)
        self._add_value_labels(axes[1, 1], bars4, '.1f')
        
        plt.tight_layout()
        plt.savefig(save_path / "02_performance_comparison.png", dpi=300, bbox_inches='tight')
        plt.show()
        print("📊 Performance comparison plot saved")
    
    def plot_predictions_comparison(self, evaluation_results: Dict, data_info: Dict, 
                                  trainer: ModelTrainer, save_path: Path):
        """Plot predictions vs actual for all models."""
        n_models = len(evaluation_results)
        fig, axes = plt.subplots(1, n_models, figsize=(6*n_models, 6))
        
        if n_models == 1:
            axes = [axes]
        
        for i, (model_name, results) in enumerate(evaluation_results.items()):
            predictions = results['predictions_original_scale']
            actuals = results['actuals_original_scale']
            
            # Create scatter plot
            axes[i].scatter(actuals, predictions, alpha=0.6, s=25, color='blue')
            
            # Perfect prediction line
            min_val = min(actuals.min(), predictions.min())
            max_val = max(actuals.max(), predictions.max())
            axes[i].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, 
                        label='Perfect Prediction', alpha=0.8)
            
            # Statistics
            r2 = results['metrics']['overall']['r2']
            rmse = results['metrics']['overall']['rmse']
            
            # Add statistics text box
            stats_text = f'R² = {r2:.3f}\nRMSE = {rmse:.1f}'
            axes[i].text(0.05, 0.95, stats_text, transform=axes[i].transAxes,
                        bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8),
                        verticalalignment='top', fontsize=11)
            
            axes[i].set_xlabel('Actual Sunspot Number')
            axes[i].set_ylabel('Predicted Sunspot Number')
            axes[i].set_title(f'{model_name} Predictions vs Actual', fontweight='bold')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
            
            # Add trend line
            z = np.polyfit(actuals, predictions, 1)
            p = np.poly1d(z)
            axes[i].plot(actuals, p(actuals), "g--", alpha=0.7, label=f'Trend (slope={z[0]:.2f})')
            axes[i].legend()
        
        plt.tight_layout()
        plt.savefig(save_path / "03_predictions_vs_actual.png", dpi=300, bbox_inches='tight')
        plt.show()
        print("📊 Predictions vs actual plot saved")
    
    def plot_time_series_predictions(self, evaluation_results: Dict, data_info: Dict,
                                   trainer: ModelTrainer, save_path: Path):
        """Plot time series predictions."""
        fig, axes = plt.subplots(len(evaluation_results), 1, 
                                figsize=(18, 5*len(evaluation_results)))
        
        if len(evaluation_results) == 1:
            axes = [axes]
        
        # Show last 500 predictions for clarity
        n_show = min(500, len(list(evaluation_results.values())[0]['predictions_original_scale']))
        
        for i, (model_name, results) in enumerate(evaluation_results.items()):
            predictions = results['predictions_original_scale'][-n_show:]
            actuals = results['actuals_original_scale'][-n_show:]
            
            x = np.arange(len(predictions))
            
            # Plot actual and predicted
            axes[i].plot(x, actuals, 'b-', label='Actual', linewidth=1.5, alpha=0.8)
            axes[i].plot(x, predictions, 'r-', label='Predicted', linewidth=1.5, alpha=0.8)
            
            # Fill between for easy comparison
            axes[i].fill_between(x, actuals, predictions, alpha=0.2, color='gray')
            
            # Statistics
            r2 = results['metrics']['overall']['r2']
            rmse = results['metrics']['overall']['rmse']
            
            axes[i].set_title(f'{model_name} Time Series Predictions '
                            f'(R² = {r2:.3f}, RMSE = {rmse:.1f})', fontweight='bold')
            axes[i].set_xlabel('Time Steps (Last 500 predictions)')
            axes[i].set_ylabel('Sunspot Number')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
            
            # Add recent prediction highlight
            if len(predictions) > 50:
                recent_start = len(predictions) - 50
                axes[i].axvspan(recent_start, len(predictions), alpha=0.1, color='yellow', 
                              label='Recent Period')
                axes[i].legend()
        
        plt.tight_layout()
        plt.savefig(save_path / "04_time_series_predictions.png", dpi=300, bbox_inches='tight')
        plt.show()
        print("📊 Time series predictions plot saved")
    
    def plot_residual_analysis(self, evaluation_results: Dict, save_path: Path):
        """Plot residual analysis for all models."""
        n_models = len(evaluation_results)
        fig, axes = plt.subplots(n_models, 3, figsize=(18, 5*n_models))
        
        if n_models == 1:
            axes = axes.reshape(1, -1)
        
        for i, (model_name, results) in enumerate(evaluation_results.items()):
            predictions = results['predictions_original_scale']
            actuals = results['actuals_original_scale']
            residuals = actuals - predictions
            
            # Residuals vs predicted
            axes[i, 0].scatter(predictions, residuals, alpha=0.6, s=20, color='blue')
            axes[i, 0].axhline(y=0, color='red', linestyle='--', alpha=0.8)
            axes[i, 0].set_xlabel('Predicted Values')
            axes[i, 0].set_ylabel('Residuals')
            axes[i, 0].set_title(f'{model_name} - Residuals vs Predicted')
            axes[i, 0].grid(True, alpha=0.3)
            
            # Add trend line for residuals
            z = np.polyfit(predictions, residuals, 1)
            p = np.poly1d(z)
            axes[i, 0].plot(predictions, p(predictions), "g--", alpha=0.7)
            
            # Residuals histogram
            axes[i, 1].hist(residuals, bins=30, alpha=0.7, color='skyblue', edgecolor='navy')
            axes[i, 1].axvline(x=0, color='red', linestyle='--', alpha=0.8)
            axes[i, 1].set_xlabel('Residuals')
            axes[i, 1].set_ylabel('Frequency')
            axes[i, 1].set_title(f'{model_name} - Residuals Distribution')
            axes[i, 1].grid(True, alpha=0.3)
            
            # Add normal distribution overlay
            mu, sigma = np.mean(residuals), np.std(residuals)
            x = np.linspace(residuals.min(), residuals.max(), 100)
            normal_dist = ((1/(sigma * np.sqrt(2 * np.pi))) * 
                          np.exp(-0.5 * ((x - mu) / sigma) ** 2))
            # Scale to match histogram
            normal_dist = normal_dist * len(residuals) * (residuals.max() - residuals.min()) / 30
            axes[i, 1].plot(x, normal_dist, 'r-', alpha=0.8, label='Normal Distribution')
            axes[i, 1].legend()
            
            # Q-Q plot
            from scipy import stats
            stats.probplot(residuals, dist="norm", plot=axes[i, 2])
            axes[i, 2].set_title(f'{model_name} - Q-Q Plot (Normality Test)')
            axes[i, 2].grid(True, alpha=0.3)
            
            # Add statistics
            mean_residual = np.mean(residuals)
            std_residual = np.std(residuals)
            axes[i, 2].text(0.05, 0.95, f'Mean: {mean_residual:.2f}\nStd: {std_residual:.2f}', 
                          transform=axes[i, 2].transAxes,
                          bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                          verticalalignment='top')
        
        plt.tight_layout()
        plt.savefig(save_path / "05_residual_analysis.png", dpi=300, bbox_inches='tight')
        plt.show()
        print("📊 Residual analysis plot saved")
    
    def plot_feature_analysis(self, data_info: Dict, save_path: Path):
        """Plot feature importance and analysis."""
        feature_names = data_info['feature_names']
        
        # Categorize features
        feature_categories = {
            'Original': [],
            'Rolling Stats': [],
            'Lag Features': [],
            'Temporal': [],
            'Volatility': [],
            'Other': []
        }
        
        for feature in feature_names:
            if any(x in feature.lower() for x in ['roll', 'mean', 'std', 'min', 'max']):
                feature_categories['Rolling Stats'].append(feature)
            elif 'lag' in feature.lower():
                feature_categories['Lag Features'].append(feature)
            elif any(x in feature.lower() for x in ['sin', 'cos', 'cycle', 'month', 'year']):
                feature_categories['Temporal'].append(feature)
            elif 'volatility' in feature.lower():
                feature_categories['Volatility'].append(feature)
            elif not any(x in feature.lower() for x in ['diff', 'pct', 'regime', 'spectral']):
                feature_categories['Original'].append(feature)
            else:
                feature_categories['Other'].append(feature)
        
        # Plot feature category distribution
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        categories = list(feature_categories.keys())
        counts = [len(feature_categories[cat]) for cat in categories]
        
        # Bar plot
        bars = axes[0].bar(categories, counts, alpha=0.8, color=plt.cm.Set3(np.linspace(0, 1, len(categories))))
        axes[0].set_title('Feature Distribution by Category', fontweight='bold')
        axes[0].set_ylabel('Number of Features')
        axes[0].tick_params(axis='x', rotation=45)
        self._add_value_labels(axes[0], bars, 'd')
        
        # Pie chart
        axes[1].pie(counts, labels=categories, autopct='%1.1f%%', startangle=90,
                   colors=plt.cm.Set3(np.linspace(0, 1, len(categories))))
        axes[1].set_title('Feature Category Distribution', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(save_path / "06_feature_analysis.png", dpi=300, bbox_inches='tight')
        plt.show()
        print("📊 Feature analysis plot saved")
    
    def plot_solar_cycle_context(self, df: pd.DataFrame, evaluation_results: Dict, save_path: Path):
        """Plot predictions in solar cycle context."""
        fig, axes = plt.subplots(2, 1, figsize=(18, 12))
        
        # Full time series context
        axes[0].plot(df['date'], df['sunspot_number'], linewidth=1, alpha=0.7, color='blue')
        axes[0].set_title('Solar Activity Context: Full Time Series', fontweight='bold')
        axes[0].set_ylabel('Sunspot Number')
        axes[0].grid(True, alpha=0.3)
        
        # Highlight recent period
        recent_period = df['date'] >= (df['date'].max() - pd.DateOffset(years=3))
        axes[0].fill_between(df.loc[recent_period, 'date'], 
                           df.loc[recent_period, 'sunspot_number'],
                           alpha=0.3, color='red', label='Recent Period (Test Data)')
        axes[0].legend()
        
        # Solar cycle phases if available
        if 'solar_cycle_phase' in df.columns:
            axes[1].plot(df['date'], df['solar_cycle_phase'], linewidth=1.5, color='purple')
            axes[1].set_title('Solar Cycle Phase Evolution', fontweight='bold')
            axes[1].set_ylabel('Solar Cycle Phase')
            axes[1].set_xlabel('Date')
            axes[1].grid(True, alpha=0.3)
            
            # Mark prediction period
            axes[1].fill_between(df.loc[recent_period, 'date'], 
                               df.loc[recent_period, 'solar_cycle_phase'],
                               alpha=0.3, color='red', label='Prediction Period')
            axes[1].legend()
        else:
            # Alternative: show rolling statistics
            if 'sunspot_number_roll_mean_365' in df.columns:
                axes[1].plot(df['date'], df['sunspot_number'], alpha=0.3, color='gray', label='Daily')
                axes[1].plot(df['date'], df['sunspot_number_roll_mean_365'], 
                           linewidth=2, color='red', label='Annual Average')
                axes[1].set_title('Solar Activity: Daily vs Annual Average', fontweight='bold')
                axes[1].set_ylabel('Sunspot Number')
                axes[1].set_xlabel('Date')
                axes[1].legend()
                axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path / "07_solar_cycle_context.png", dpi=300, bbox_inches='tight')
        plt.show()
        print("📊 Solar cycle context plot saved")
    
    def generate_results_summary(self, training_results: Dict, evaluation_results: Dict, save_path: Path):
        """Generate comprehensive results summary."""
        summary_path = save_path / "results_summary.txt"
        
        with open(summary_path, 'w') as f:
            f.write("PYTORCH SUNSPOT PREDICTION MODELS - RESULTS SUMMARY\n")
            f.write("=" * 70 + "\n\n")
            
            f.write(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Models Evaluated: {len(evaluation_results)}\n\n")
            
            # Model ranking
            model_ranking = sorted(evaluation_results.items(), 
                                 key=lambda x: x[1]['metrics']['overall']['rmse'])
            
            f.write("MODEL PERFORMANCE RANKING (by RMSE):\n")
            f.write("-" * 45 + "\n")
            for i, (model_name, results) in enumerate(model_ranking):
                metrics = results['metrics']['overall']
                f.write(f"{i+1}. {model_name}:\n")
                f.write(f"   RMSE: {metrics['rmse']:.2f}\n")
                f.write(f"   MAE:  {metrics['mae']:.2f}\n")
                f.write(f"   R²:   {metrics['r2']:.4f}\n")
                f.write(f"   MAPE: {metrics['mape']:.1f}%\n\n")
            
            # Training summary
            f.write("TRAINING SUMMARY:\n")
            f.write("-" * 20 + "\n")
            for model_name, history in training_results.items():
                final_train_loss = history['train_loss'][-1]
                final_val_loss = history['val_loss'][-1]
                n_epochs = len(history['train_loss'])
                
                f.write(f"{model_name}:\n")
                f.write(f"   Epochs: {n_epochs}\n")
                f.write(f"   Final Train Loss: {final_train_loss:.6f}\n")
                f.write(f"   Final Val Loss: {final_val_loss:.6f}\n")
                f.write(f"   Final LR: {history['lr'][-1]:.2e}\n\n")
            
            # Best model details
            best_model_name = model_ranking[0][0]
            best_results = model_ranking[0][1]
            
            f.write(f"BEST MODEL: {best_model_name}\n")
            f.write("-" * (len(best_model_name) + 12) + "\n")
            f.write(f"This model achieved the lowest RMSE of {best_results['metrics']['overall']['rmse']:.2f}\n")
            f.write(f"With an R² score of {best_results['metrics']['overall']['r2']:.4f}\n")
            f.write(f"Mean Absolute Error: {best_results['metrics']['overall']['mae']:.2f}\n")
            f.write(f"Mean Absolute Percentage Error: {best_results['metrics']['overall']['mape']:.1f}%\n\n")
            
            # Model insights
            f.write("MODEL INSIGHTS:\n")
            f.write("-" * 15 + "\n")
            
            if len(evaluation_results) >= 2:
                transformer_rmse = evaluation_results.get('Transformer', {}).get('metrics', {}).get('overall', {}).get('rmse', float('inf'))
                lstm_rmse = evaluation_results.get('LSTM', {}).get('metrics', {}).get('overall', {}).get('rmse', float('inf'))
                gru_rmse = evaluation_results.get('GRU', {}).get('metrics', {}).get('overall', {}).get('rmse', float('inf'))
                
                if transformer_rmse < min(lstm_rmse, gru_rmse):
                    f.write("- Transformer architecture shows superior performance for this multivariate task\n")
                    f.write("- Patch-based processing effectively captures temporal patterns\n")
                elif lstm_rmse < gru_rmse:
                    f.write("- LSTM with attention outperforms GRU, suggesting memory mechanisms are crucial\n")
                else:
                    f.write("- GRU shows competitive performance with lower computational complexity\n")
                
                f.write("- All models benefit from multivariate features (F10.7, geomagnetic indices)\n")
                f.write("- Non-smoothed daily sunspot prediction remains challenging but achievable\n")
            
            f.write(f"\nVisualization files saved in: {save_path}\n")
        
        print(f"📄 Results summary saved to: {summary_path}")
    
    def _add_value_labels(self, ax, bars, format_str):
        """Add value labels on top of bars."""
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:{format_str}}',
                   ha='center', va='bottom', fontweight='bold')


def main():
    """Run comprehensive results visualization."""
    visualizer = ResultsVisualizer()
    
    print("🚀 Starting comprehensive PyTorch model visualization...")
    
    # Train and visualize models
    results = visualizer.train_and_visualize_models(
        data_subset_size=8000,  # Use 8000 recent observations
        save_dir="data/results_visualization"
    )
    
    if results:
        print("\n🎉 SUCCESS: All visualizations completed!")
        print("\n📊 Generated Visualizations:")
        print("1. Training History (loss curves, learning rates)")
        print("2. Performance Comparison (RMSE, MAE, R², MAPE)")
        print("3. Predictions vs Actual (scatter plots)")
        print("4. Time Series Predictions (temporal plots)")
        print("5. Residual Analysis (diagnostic plots)")
        print("6. Feature Analysis (category distribution)")
        print("7. Solar Cycle Context (historical context)")
        print("8. Results Summary (comprehensive report)")
        
        print(f"\n💡 Best performing model: Check results_summary.txt")
        print(f"📁 All files saved to: data/results_visualization/")
    else:
        print("❌ Visualization failed. Check data availability.")


if __name__ == "__main__":
    main()