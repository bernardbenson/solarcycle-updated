"""
Comprehensive model evaluation framework with uncertainty quantification 
and Solar Cycle 25 validation for sunspot prediction models.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

from pytorch_models import ModelTrainer


class ModelEvaluator:
    """
    Comprehensive evaluation framework for solar prediction models.
    """
    
    def __init__(self, trainer: ModelTrainer, data_info: Dict):
        self.trainer = trainer
        self.data_info = data_info
        self.evaluation_results = {}
        
    def evaluate_all_models(self, batch_size: int = 64) -> Dict:
        """Evaluate all trained models comprehensively."""
        print("🔍 COMPREHENSIVE MODEL EVALUATION")
        print("=" * 60)
        
        results = {}
        
        for model_name in self.trainer.models.keys():
            print(f"\nEvaluating {model_name}...")
            
            # Standard evaluation
            eval_result = self.trainer.evaluate_model(model_name, self.data_info, batch_size)
            
            # Enhanced evaluation
            enhanced_result = self.enhanced_evaluation(model_name, batch_size)
            
            # Uncertainty quantification
            uncertainty_result = self.uncertainty_quantification(model_name, batch_size)
            
            # Combine results
            results[model_name] = {
                'standard_metrics': eval_result['metrics'],
                'predictions': eval_result['predictions_original_scale'],
                'actuals': eval_result['actuals_original_scale'],
                'enhanced_metrics': enhanced_result,
                'uncertainty': uncertainty_result
            }
        
        self.evaluation_results = results
        return results
    
    def enhanced_evaluation(self, model_name: str, batch_size: int = 64) -> Dict:
        """Enhanced evaluation with additional metrics and analysis."""
        model = self.trainer.models[model_name]
        model.eval()
        
        from torch.utils.data import DataLoader
        test_loader = DataLoader(
            self.data_info['test_dataset'], batch_size=batch_size, shuffle=False
        )
        
        predictions = []
        actuals = []
        
        with torch.no_grad():
            for data, target in test_loader:
                data = data.to(self.trainer.device)
                
                if hasattr(model, 'forward') and 'AttentionLSTM' in model_name or 'AttentionGRU' in model_name:
                    output, _ = model(data)
                else:
                    output = model(data)
                
                pred = output.cpu().numpy()
                actual = target.numpy()
                
                predictions.append(pred)
                actuals.append(actual)
        
        predictions = np.concatenate(predictions, axis=0)
        actuals = np.concatenate(actuals, axis=0)
        
        # Convert to original scale
        pred_orig = self.trainer.scalers['target'].inverse_transform(
            predictions.reshape(-1, 1)
        ).ravel()
        actual_orig = self.trainer.scalers['target'].inverse_transform(
            actuals.reshape(-1, 1)
        ).ravel()
        
        # Enhanced metrics
        enhanced_metrics = {}
        
        # Directional accuracy
        pred_direction = np.diff(pred_orig) > 0
        actual_direction = np.diff(actual_orig) > 0
        directional_accuracy = np.mean(pred_direction == actual_direction) * 100
        
        # Peak detection accuracy
        from scipy.signal import find_peaks
        pred_peaks, _ = find_peaks(pred_orig, height=np.mean(pred_orig))
        actual_peaks, _ = find_peaks(actual_orig, height=np.mean(actual_orig))
        
        # Solar activity regime accuracy
        low_threshold = np.percentile(actual_orig, 33)
        high_threshold = np.percentile(actual_orig, 67)
        
        pred_regime = np.where(pred_orig < low_threshold, 0, 
                              np.where(pred_orig > high_threshold, 2, 1))
        actual_regime = np.where(actual_orig < low_threshold, 0,
                                np.where(actual_orig > high_threshold, 2, 1))
        
        regime_accuracy = np.mean(pred_regime == actual_regime) * 100
        
        # Statistical tests
        # Shapiro-Wilk test for residual normality
        residuals = actual_orig - pred_orig
        shapiro_stat, shapiro_p = stats.shapiro(residuals[:5000])  # Limit for computational efficiency
        
        # Durbin-Watson test for autocorrelation
        def durbin_watson(residuals):
            diff_residuals = np.diff(residuals)
            return np.sum(diff_residuals**2) / np.sum(residuals**2)
        
        dw_statistic = durbin_watson(residuals)
        
        enhanced_metrics = {
            'directional_accuracy': directional_accuracy,
            'n_predicted_peaks': len(pred_peaks),
            'n_actual_peaks': len(actual_peaks),
            'peak_detection_ratio': len(pred_peaks) / max(len(actual_peaks), 1),
            'regime_accuracy': regime_accuracy,
            'residual_normality_p': shapiro_p,
            'durbin_watson_stat': dw_statistic,
            'forecast_bias': np.mean(residuals),
            'forecast_variance': np.var(residuals),
            'symmetric_mape': np.mean(2 * np.abs(residuals) / (np.abs(pred_orig) + np.abs(actual_orig))) * 100
        }
        
        return enhanced_metrics
    
    def uncertainty_quantification(self, model_name: str, batch_size: int = 64, 
                                 n_samples: int = 50) -> Dict:
        """
        Uncertainty quantification using Monte Carlo dropout.
        """
        model = self.trainer.models[model_name]
        
        from torch.utils.data import DataLoader
        test_loader = DataLoader(
            self.data_info['test_dataset'], batch_size=batch_size, shuffle=False
        )
        
        # Enable dropout for uncertainty estimation
        def enable_dropout(m):
            if type(m) == torch.nn.Dropout:
                m.train()
                
        model.apply(enable_dropout)
        
        all_predictions = []
        actuals = []
        
        # Multiple forward passes for uncertainty estimation
        for _ in range(n_samples):
            batch_predictions = []
            batch_actuals = []
            
            with torch.no_grad():
                for data, target in test_loader:
                    data = data.to(self.trainer.device)
                    
                    if hasattr(model, 'forward') and ('AttentionLSTM' in model_name or 'AttentionGRU' in model_name):
                        output, _ = model(data)
                    else:
                        output = model(data)
                    
                    pred = output.cpu().numpy()
                    actual = target.numpy()
                    
                    batch_predictions.append(pred)
                    if len(batch_actuals) == 0:  # Only collect actuals once
                        batch_actuals.append(actual)
            
            predictions = np.concatenate(batch_predictions, axis=0)
            all_predictions.append(predictions)
            
            if len(actuals) == 0:
                actuals = np.concatenate(batch_actuals, axis=0)
        
        # Stack predictions: (n_samples, n_test, prediction_horizon)
        all_predictions = np.stack(all_predictions, axis=0)
        
        # Convert to original scale
        pred_mean = np.mean(all_predictions, axis=0)
        pred_std = np.std(all_predictions, axis=0)
        
        pred_mean_orig = self.trainer.scalers['target'].inverse_transform(
            pred_mean.reshape(-1, 1)
        ).ravel()
        pred_std_orig = pred_std.reshape(-1, 1).ravel() * self.trainer.scalers['target'].scale_[0]
        
        actual_orig = self.trainer.scalers['target'].inverse_transform(
            actuals.reshape(-1, 1)
        ).ravel()
        
        # Uncertainty metrics
        # Prediction intervals
        confidence_levels = [0.68, 0.95]  # 1σ and 2σ
        prediction_intervals = {}
        
        for conf_level in confidence_levels:
            alpha = 1 - conf_level
            z_score = stats.norm.ppf(1 - alpha/2)
            
            lower_bound = pred_mean_orig - z_score * pred_std_orig
            upper_bound = pred_mean_orig + z_score * pred_std_orig
            
            # Coverage probability
            coverage = np.mean((actual_orig >= lower_bound) & (actual_orig <= upper_bound))
            
            # Average interval width
            avg_width = np.mean(upper_bound - lower_bound)
            
            prediction_intervals[f'{conf_level:.0%}'] = {
                'lower_bound': lower_bound,
                'upper_bound': upper_bound,
                'coverage': coverage,
                'average_width': avg_width
            }
        
        uncertainty_metrics = {
            'prediction_intervals': prediction_intervals,
            'epistemic_uncertainty': np.mean(pred_std_orig),
            'prediction_std': pred_std_orig,
            'prediction_mean': pred_mean_orig
        }
        
        # Set model back to eval mode
        model.eval()
        
        return uncertainty_metrics
    
    def solar_cycle_25_validation(self, cutoff_date: str = '2020-01-01') -> Dict:
        """
        Validate model performance specifically on Solar Cycle 25 period.
        """
        print("\n🌞 SOLAR CYCLE 25 VALIDATION")
        print("=" * 40)
        
        # Load original data to get dates
        data_path = Path("data/engineered_multivariate_data.csv")
        df = pd.read_csv(data_path)
        df['date'] = pd.to_datetime(df['date'])
        
        # Find Solar Cycle 25 period in test data
        cycle_25_mask = df['date'] >= cutoff_date
        cycle_25_data = df[cycle_25_mask].reset_index(drop=True)
        
        print(f"Solar Cycle 25 period: {cutoff_date} to {df['date'].max()}")
        print(f"Data points: {len(cycle_25_data)}")
        
        # Get test data dates (approximate)
        test_start_idx = int(len(df) * 0.85)  # Assuming 85% train+val, 15% test
        test_dates = df.iloc[test_start_idx:]['date'].reset_index(drop=True)
        
        validation_results = {}
        
        for model_name, results in self.evaluation_results.items():
            predictions = results['predictions']
            actuals = results['actuals']
            
            # Find overlap with Solar Cycle 25
            if len(test_dates) >= len(predictions):
                test_dates_subset = test_dates.iloc[:len(predictions)]
                cycle_25_test_mask = test_dates_subset >= cutoff_date
                
                if cycle_25_test_mask.sum() > 0:
                    cycle_25_pred = predictions[cycle_25_test_mask]
                    cycle_25_actual = actuals[cycle_25_test_mask]
                    
                    # Calculate metrics for Solar Cycle 25 period
                    cycle_25_metrics = {
                        'mae': mean_absolute_error(cycle_25_actual, cycle_25_pred),
                        'rmse': np.sqrt(mean_squared_error(cycle_25_actual, cycle_25_pred)),
                        'r2': r2_score(cycle_25_actual, cycle_25_pred),
                        'mape': np.mean(np.abs((cycle_25_actual - cycle_25_pred) / 
                                             (cycle_25_actual + 1e-8))) * 100,
                        'n_samples': len(cycle_25_pred)
                    }
                    
                    validation_results[model_name] = cycle_25_metrics
                    
                    print(f"{model_name} - Cycle 25 RMSE: {cycle_25_metrics['rmse']:.2f}")
        
        return validation_results
    
    def create_comparison_plots(self, save_dir: str = "data/evaluation_results"):
        """Create comprehensive comparison plots for all models."""
        save_path = Path(save_dir)
        save_path.mkdir(exist_ok=True, parents=True)
        
        if not self.evaluation_results:
            print("No evaluation results found. Run evaluate_all_models first.")
            return
        
        # 1. Model Performance Comparison
        self._plot_model_comparison(save_path)
        
        # 2. Prediction vs Actual plots
        self._plot_predictions_vs_actual(save_path)
        
        # 3. Uncertainty quantification plots
        self._plot_uncertainty_quantification(save_path)
        
        # 4. Residual analysis
        self._plot_residual_analysis(save_path)
        
        print(f"Evaluation plots saved to: {save_path}")
    
    def _plot_model_comparison(self, save_path: Path):
        """Plot model performance comparison."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        models = list(self.evaluation_results.keys())
        
        # Extract metrics
        rmse_values = [self.evaluation_results[m]['standard_metrics']['overall']['rmse'] for m in models]
        r2_values = [self.evaluation_results[m]['standard_metrics']['overall']['r2'] for m in models]
        mae_values = [self.evaluation_results[m]['standard_metrics']['overall']['mae'] for m in models]
        mape_values = [self.evaluation_results[m]['standard_metrics']['overall']['mape'] for m in models]
        
        # RMSE comparison
        axes[0, 0].bar(models, rmse_values, alpha=0.7, color='skyblue')
        axes[0, 0].set_title('Root Mean Square Error (RMSE)')
        axes[0, 0].set_ylabel('RMSE')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # R² comparison
        axes[0, 1].bar(models, r2_values, alpha=0.7, color='lightgreen')
        axes[0, 1].set_title('Coefficient of Determination (R²)')
        axes[0, 1].set_ylabel('R²')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # MAE comparison
        axes[1, 0].bar(models, mae_values, alpha=0.7, color='salmon')
        axes[1, 0].set_title('Mean Absolute Error (MAE)')
        axes[1, 0].set_ylabel('MAE')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # MAPE comparison
        axes[1, 1].bar(models, mape_values, alpha=0.7, color='gold')
        axes[1, 1].set_title('Mean Absolute Percentage Error (MAPE)')
        axes[1, 1].set_ylabel('MAPE (%)')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(save_path / "model_comparison.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_predictions_vs_actual(self, save_path: Path):
        """Plot predictions vs actual values for all models."""
        n_models = len(self.evaluation_results)
        fig, axes = plt.subplots(1, n_models, figsize=(5*n_models, 5))
        
        if n_models == 1:
            axes = [axes]
        
        for i, (model_name, results) in enumerate(self.evaluation_results.items()):
            predictions = results['predictions']
            actuals = results['actuals']
            
            # Scatter plot
            axes[i].scatter(actuals, predictions, alpha=0.6, s=20)
            
            # Perfect prediction line
            min_val = min(actuals.min(), predictions.min())
            max_val = max(actuals.max(), predictions.max())
            axes[i].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
            
            # Add R² score
            r2 = results['standard_metrics']['overall']['r2']
            axes[i].text(0.05, 0.95, f'R² = {r2:.3f}', transform=axes[i].transAxes,
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            axes[i].set_xlabel('Actual Sunspot Number')
            axes[i].set_ylabel('Predicted Sunspot Number')
            axes[i].set_title(f'{model_name}')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path / "predictions_vs_actual.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_uncertainty_quantification(self, save_path: Path):
        """Plot uncertainty quantification results."""
        fig, axes = plt.subplots(len(self.evaluation_results), 1, 
                                figsize=(15, 5*len(self.evaluation_results)))
        
        if len(self.evaluation_results) == 1:
            axes = [axes]
        
        for i, (model_name, results) in enumerate(self.evaluation_results.items()):
            if 'uncertainty' not in results:
                continue
                
            uncertainty = results['uncertainty']
            actuals = results['actuals']
            
            pred_mean = uncertainty['prediction_mean']
            intervals_95 = uncertainty['prediction_intervals']['95%']
            
            # Time series plot with uncertainty bands
            x = np.arange(len(actuals))
            
            axes[i].plot(x, actuals, 'b-', label='Actual', linewidth=1)
            axes[i].plot(x, pred_mean, 'r-', label='Predicted', linewidth=1)
            axes[i].fill_between(x, intervals_95['lower_bound'], intervals_95['upper_bound'],
                               alpha=0.3, color='red', label='95% Prediction Interval')
            
            axes[i].set_title(f'{model_name} - Uncertainty Quantification')
            axes[i].set_xlabel('Time Steps')
            axes[i].set_ylabel('Sunspot Number')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
            
            # Add coverage information
            coverage = intervals_95['coverage']
            axes[i].text(0.02, 0.95, f'95% Coverage: {coverage:.1%}', 
                        transform=axes[i].transAxes,
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(save_path / "uncertainty_quantification.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_residual_analysis(self, save_path: Path):
        """Plot residual analysis for all models."""
        fig, axes = plt.subplots(2, len(self.evaluation_results), 
                                figsize=(5*len(self.evaluation_results), 10))
        
        if len(self.evaluation_results) == 1:
            axes = axes.reshape(-1, 1)
        
        for i, (model_name, results) in enumerate(self.evaluation_results.items()):
            predictions = results['predictions']
            actuals = results['actuals']
            residuals = actuals - predictions
            
            # Residuals vs predicted
            axes[0, i].scatter(predictions, residuals, alpha=0.6, s=20)
            axes[0, i].axhline(y=0, color='r', linestyle='--')
            axes[0, i].set_xlabel('Predicted Values')
            axes[0, i].set_ylabel('Residuals')
            axes[0, i].set_title(f'{model_name} - Residuals vs Predicted')
            axes[0, i].grid(True, alpha=0.3)
            
            # Q-Q plot
            stats.probplot(residuals, dist="norm", plot=axes[1, i])
            axes[1, i].set_title(f'{model_name} - Q-Q Plot')
            axes[1, i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path / "residual_analysis.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_evaluation_report(self, save_path: str = "data/evaluation_results/evaluation_report.txt"):
        """Generate comprehensive evaluation report."""
        Path(save_path).parent.mkdir(exist_ok=True, parents=True)
        
        with open(save_path, 'w') as f:
            f.write("COMPREHENSIVE MODEL EVALUATION REPORT\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"Evaluation Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Models Evaluated: {len(self.evaluation_results)}\n")
            f.write(f"Test Samples: {len(list(self.evaluation_results.values())[0]['predictions'])}\n\n")
            
            # Model ranking
            rmse_ranking = sorted(self.evaluation_results.items(), 
                                key=lambda x: x[1]['standard_metrics']['overall']['rmse'])
            
            f.write("MODEL RANKING (by RMSE):\n")
            f.write("-" * 30 + "\n")
            for i, (model_name, results) in enumerate(rmse_ranking):
                rmse = results['standard_metrics']['overall']['rmse']
                r2 = results['standard_metrics']['overall']['r2']
                f.write(f"{i+1}. {model_name}: RMSE={rmse:.2f}, R²={r2:.4f}\n")
            
            f.write("\n\nDETAILED METRICS:\n")
            f.write("=" * 40 + "\n")
            
            for model_name, results in self.evaluation_results.items():
                f.write(f"\n{model_name.upper()}:\n")
                f.write("-" * len(model_name) + "\n")
                
                # Standard metrics
                std_metrics = results['standard_metrics']['overall']
                f.write(f"RMSE: {std_metrics['rmse']:.2f}\n")
                f.write(f"MAE: {std_metrics['mae']:.2f}\n")
                f.write(f"R²: {std_metrics['r2']:.4f}\n")
                f.write(f"MAPE: {std_metrics['mape']:.2f}%\n")
                
                # Enhanced metrics
                if 'enhanced_metrics' in results:
                    enh_metrics = results['enhanced_metrics']
                    f.write(f"Directional Accuracy: {enh_metrics['directional_accuracy']:.1f}%\n")
                    f.write(f"Regime Accuracy: {enh_metrics['regime_accuracy']:.1f}%\n")
                    f.write(f"Forecast Bias: {enh_metrics['forecast_bias']:.2f}\n")
                
                # Uncertainty metrics
                if 'uncertainty' in results:
                    unc_metrics = results['uncertainty']
                    coverage_95 = unc_metrics['prediction_intervals']['95%']['coverage']
                    f.write(f"95% Prediction Interval Coverage: {coverage_95:.1%}\n")
                    f.write(f"Average Epistemic Uncertainty: {unc_metrics['epistemic_uncertainty']:.2f}\n")
        
        print(f"Evaluation report saved to: {save_path}")


def main():
    """Example usage of model evaluation framework."""
    print("Model evaluation framework loaded successfully!")
    print("Use ModelEvaluator class with trained models for comprehensive evaluation.")


if __name__ == "__main__":
    main()