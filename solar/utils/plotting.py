"""
Enhanced plotting utilities for solar cycle prediction with uncertainty visualization.
Supports TensorFlow-style plots with uncertainty bands and advanced visualizations.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple, Any
import pandas as pd
import seaborn as sns
from datetime import datetime, timedelta
import warnings

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class SolarCyclePlotter:
    """Enhanced plotter for solar cycle predictions with uncertainty."""
    
    def __init__(self, style: str = 'publication', figsize: Tuple[int, int] = (12, 8)):
        """
        Args:
            style: Plot style ('publication', 'presentation', 'notebook')
            figsize: Default figure size
        """
        self.style = style
        self.figsize = figsize
        self.colors = {
            'actual': '#2E86AB',
            'prediction': '#A23B72', 
            'uncertainty': '#F18F01',
            'trend': '#C73E1D',
            'seasonal': '#6A994E',
            'peak': '#FF6B35'
        }
        
        self._setup_style()
    
    def _setup_style(self):
        """Configure matplotlib style based on usage context."""
        if self.style == 'publication':
            plt.rcParams.update({
                'font.size': 12,
                'axes.titlesize': 14,
                'axes.labelsize': 12,
                'xtick.labelsize': 10,
                'ytick.labelsize': 10,
                'legend.fontsize': 10,
                'figure.titlesize': 16,
                'lines.linewidth': 2,
                'axes.grid': True,
                'grid.alpha': 0.3
            })
        elif self.style == 'presentation':
            plt.rcParams.update({
                'font.size': 14,
                'axes.titlesize': 18,
                'axes.labelsize': 16,
                'xtick.labelsize': 12,
                'ytick.labelsize': 12,
                'legend.fontsize': 12,
                'figure.titlesize': 20,
                'lines.linewidth': 3,
                'axes.grid': True,
                'grid.alpha': 0.3
            })
    
    def plot_single_cycle_with_uncertainty(self, 
                                         actual: Optional[np.ndarray] = None,
                                         prediction: Optional[np.ndarray] = None,
                                         uncertainty: Optional[Dict[str, np.ndarray]] = None,
                                         title: str = "Solar Cycle Prediction",
                                         xlabel: str = "Months into Cycle",
                                         ylabel: str = "Sunspot Number",
                                         save_path: Optional[Path] = None,
                                         show_peak: bool = True,
                                         cycle_info: Optional[Dict] = None) -> plt.Figure:
        """
        Plot single cycle with uncertainty bands.
        
        Args:
            actual: Actual sunspot data
            prediction: Predicted values (point estimate)
            uncertainty: Dict with 'lower', 'upper', 'q10', 'q50', 'q90' arrays
            title: Plot title
            xlabel: X-axis label
            ylabel: Y-axis label
            save_path: Path to save figure
            show_peak: Whether to highlight peaks
            cycle_info: Additional cycle information for annotations
        
        Returns:
            Figure object
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Determine sequence lengths
        max_len = 0
        if actual is not None:
            max_len = max(max_len, len(actual))
        if prediction is not None:
            max_len = max(max_len, len(prediction))
        
        months = np.arange(max_len)
        
        # Plot actual data
        if actual is not None:
            actual_months = np.arange(len(actual))
            ax.plot(actual_months, actual, 
                   color=self.colors['actual'], linewidth=2.5, 
                   label='Actual', alpha=0.9)
            
            # Highlight actual peak
            if show_peak:
                peak_idx = np.argmax(actual)
                ax.scatter(peak_idx, actual[peak_idx], 
                          color=self.colors['actual'], s=100, 
                          zorder=5, edgecolor='white', linewidth=2)
        
        # Plot prediction
        if prediction is not None:
            pred_months = np.arange(len(prediction))
            ax.plot(pred_months, prediction, 
                   color=self.colors['prediction'], linewidth=2.5,
                   label='Prediction', alpha=0.9)
            
            # Highlight predicted peak
            if show_peak:
                peak_idx = np.argmax(prediction)
                ax.scatter(peak_idx, prediction[peak_idx], 
                          color=self.colors['prediction'], s=100, 
                          zorder=5, edgecolor='white', linewidth=2)
        
        # Plot uncertainty bands
        if uncertainty is not None and prediction is not None:
            pred_months = np.arange(len(prediction))
            
            # Main uncertainty band (e.g., 80% interval)
            if 'lower' in uncertainty and 'upper' in uncertainty:
                ax.fill_between(pred_months, uncertainty['lower'], uncertainty['upper'],
                               color=self.colors['uncertainty'], alpha=0.3, 
                               label='80% Prediction Interval')
            
            # Quantile bands
            elif 'q10' in uncertainty and 'q90' in uncertainty:
                ax.fill_between(pred_months, uncertainty['q10'], uncertainty['q90'],
                               color=self.colors['uncertainty'], alpha=0.3,
                               label='80% Prediction Interval')
                
                # Median line if available
                if 'q50' in uncertainty:
                    ax.plot(pred_months, uncertainty['q50'], 
                           color=self.colors['prediction'], linewidth=1.5,
                           linestyle='--', alpha=0.7, label='Median Prediction')
        
        # Annotations
        if cycle_info:
            # Add cycle information
            info_text = []
            if 'peak_month' in cycle_info:
                info_text.append(f"Peak Month: {cycle_info['peak_month']}")
            if 'peak_value' in cycle_info:
                info_text.append(f"Peak Value: {cycle_info['peak_value']:.1f}")
            if 'rmse' in cycle_info:
                info_text.append(f"RMSE: {cycle_info['rmse']:.1f}")
            if 'mae' in cycle_info:
                info_text.append(f"MAE: {cycle_info['mae']:.1f}")
            
            if info_text:
                ax.text(0.02, 0.98, '\n'.join(info_text),
                       transform=ax.transAxes, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                       fontsize=10)
        
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        
        return fig
    
    def plot_overlapping_cycles_enhanced(self,
                                       prediction_results: Dict[str, Dict],
                                       save_dir: Optional[Path] = None,
                                       show_uncertainty: bool = True,
                                       figsize: Optional[Tuple[int, int]] = None) -> plt.Figure:
        """
        Enhanced version of overlapping cycle plots with uncertainty.
        
        Args:
            prediction_results: Dictionary with cycle predictions and uncertainty
            save_dir: Directory to save individual plots
            show_uncertainty: Whether to show uncertainty bands
            figsize: Custom figure size
        
        Returns:
            Combined figure with all cycles
        """
        if figsize is None:
            figsize = (16, 12)
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        axes = axes.flatten()
        
        cycle_names = ['cycle_22', 'cycle_23', 'cycle_24', 'cycle_25']
        
        for i, cycle_name in enumerate(cycle_names):
            if cycle_name not in prediction_results:
                axes[i].set_visible(False)
                continue
            
            result = prediction_results[cycle_name]
            ax = axes[i]
            
            # Extract data
            prediction = result.get('prediction', result.get('mean'))
            actual = result.get('actual')
            uncertainty = result.get('uncertainty', {})
            
            if prediction is None:
                axes[i].set_visible(False)
                continue
            
            # Plot prediction
            pred_months = np.arange(len(prediction))
            ax.plot(pred_months, prediction, 
                   color=self.colors['prediction'], linewidth=2.5,
                   label='Prediction', alpha=0.9)
            
            # Plot actual if available
            if actual is not None:
                actual_months = np.arange(len(actual))
                ax.plot(actual_months, actual, 
                       color=self.colors['actual'], linewidth=2.5,
                       label='Actual', alpha=0.9)
            
            # Plot uncertainty
            if show_uncertainty and uncertainty:
                if 'q10' in uncertainty and 'q90' in uncertainty:
                    ax.fill_between(pred_months, uncertainty['q10'], uncertainty['q90'],
                                   color=self.colors['uncertainty'], alpha=0.3,
                                   label='80% PI')
                elif 'std' in uncertainty:
                    lower = prediction - 1.28 * uncertainty['std']  # ~80% interval
                    upper = prediction + 1.28 * uncertainty['std']
                    ax.fill_between(pred_months, lower, upper,
                                   color=self.colors['uncertainty'], alpha=0.3,
                                   label='80% PI')
            
            # Peak markers
            pred_peak_idx = np.argmax(prediction)
            ax.scatter(pred_peak_idx, prediction[pred_peak_idx], 
                      color=self.colors['peak'], s=80, zorder=5,
                      edgecolor='white', linewidth=1.5)
            
            if actual is not None:
                actual_peak_idx = np.argmax(actual)
                ax.scatter(actual_peak_idx, actual[actual_peak_idx], 
                          color=self.colors['actual'], s=80, zorder=5,
                          edgecolor='white', linewidth=1.5)
            
            # Title and metrics
            title = result.get('description', cycle_name.replace('_', ' ').title())
            if 'rmse' in result:
                title += f"\nRMSE: {result['rmse']:.1f}"
            if 'mae' in result:
                title += f", MAE: {result['mae']:.1f}"
            
            # For cycle 25, show max prediction
            if cycle_name == 'cycle_25' and actual is None:
                max_val = np.max(prediction)
                max_month = np.argmax(prediction) + 1
                title += f"\nMax: {max_val:.1f} at month {max_month}"
            
            ax.set_title(title, fontsize=11, fontweight='bold')
            ax.set_xlabel('Months into Cycle')
            ax.set_ylabel('Sunspot Number')
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
        
        plt.suptitle('Solar Cycle Predictions with Uncertainty', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_dir / "overlapping_cycles_enhanced.png", 
                       dpi=300, bbox_inches='tight', facecolor='white')
        
        return fig
    
    def plot_training_history_enhanced(self, 
                                     metrics_history: Dict[str, List],
                                     save_path: Optional[Path] = None) -> plt.Figure:
        """Enhanced training history plot with multiple metrics."""
        
        # Determine number of subplots needed
        metric_groups = {
            'Loss': ['train_loss', 'val_loss', 'train_total_loss', 'val_total_loss'],
            'Learning Rate': ['lr'],
            'Teacher Forcing': ['teacher_forcing_ratio'],
            'Coverage': [k for k in metrics_history.keys() if 'coverage' in k.lower()]
        }
        
        # Filter out empty groups
        active_groups = {k: v for k, v in metric_groups.items() 
                        if any(metric in metrics_history for metric in v)}
        
        n_plots = len(active_groups)
        if n_plots == 0:
            return None
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, (group_name, metrics) in enumerate(active_groups.items()):
            if i >= 4:  # Max 4 subplots
                break
                
            ax = axes[i]
            
            for metric in metrics:
                if metric in metrics_history and len(metrics_history[metric]) > 0:
                    epochs = range(len(metrics_history[metric]))
                    
                    if group_name == 'Learning Rate':
                        ax.semilogy(epochs, metrics_history[metric], 
                                   label=metric, linewidth=2)
                    else:
                        ax.plot(epochs, metrics_history[metric], 
                               label=metric, linewidth=2)
            
            ax.set_title(group_name, fontweight='bold')
            ax.set_xlabel('Epoch')
            ax.set_ylabel(group_name)
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(len(active_groups), 4):
            axes[i].set_visible(False)
        
        plt.suptitle('Training History', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        
        return fig
    
    def plot_mc_dropout_uncertainty(self, 
                                   mc_samples: np.ndarray,
                                   actual: Optional[np.ndarray] = None,
                                   title: str = "MC-Dropout Uncertainty",
                                   save_path: Optional[Path] = None) -> plt.Figure:
        """
        Plot MC-Dropout uncertainty with sample distribution.
        
        Args:
            mc_samples: MC samples (seq_len, n_samples)
            actual: Actual values for comparison
            title: Plot title
            save_path: Path to save figure
        
        Returns:
            Figure object
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        seq_len, n_samples = mc_samples.shape
        months = np.arange(seq_len)
        
        # Main prediction plot
        mean_pred = np.mean(mc_samples, axis=1)
        std_pred = np.std(mc_samples, axis=1)
        
        # Plot individual samples (subset for clarity)
        sample_indices = np.linspace(0, n_samples-1, min(20, n_samples), dtype=int)
        for i in sample_indices:
            ax1.plot(months, mc_samples[:, i], 
                    color=self.colors['uncertainty'], alpha=0.1, linewidth=0.5)
        
        # Plot mean and confidence intervals
        ax1.plot(months, mean_pred, color=self.colors['prediction'], 
                linewidth=2.5, label='Mean Prediction')
        
        ax1.fill_between(months, mean_pred - 2*std_pred, mean_pred + 2*std_pred,
                        color=self.colors['uncertainty'], alpha=0.3, label='95% CI')
        ax1.fill_between(months, mean_pred - std_pred, mean_pred + std_pred,
                        color=self.colors['uncertainty'], alpha=0.5, label='68% CI')
        
        # Plot actual if available
        if actual is not None:
            actual_months = np.arange(len(actual))
            ax1.plot(actual_months, actual, color=self.colors['actual'], 
                    linewidth=2.5, label='Actual')
        
        ax1.set_title(f"{title} - Prediction Uncertainty")
        ax1.set_xlabel('Months')
        ax1.set_ylabel('Sunspot Number')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Uncertainty evolution plot
        ax2.plot(months, std_pred, color=self.colors['trend'], 
                linewidth=2, label='Prediction Std')
        ax2.fill_between(months, 0, std_pred, 
                        color=self.colors['trend'], alpha=0.3)
        
        ax2.set_title('Prediction Uncertainty over Time')
        ax2.set_xlabel('Months')
        ax2.set_ylabel('Standard Deviation')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        
        return fig
    
    def plot_peak_distribution(self,
                             prediction_samples: np.ndarray,
                             actual_peak: Optional[Tuple[int, float]] = None,
                             title: str = "Peak Distribution",
                             save_path: Optional[Path] = None) -> plt.Figure:
        """
        Plot distribution of predicted peaks from MC samples.
        
        Args:
            prediction_samples: MC samples (seq_len, n_samples)
            actual_peak: Tuple of (peak_month, peak_value) if known
            title: Plot title
            save_path: Path to save figure
        
        Returns:
            Figure object
        """
        # Extract peaks from each sample
        peak_months = []
        peak_values = []
        
        for i in range(prediction_samples.shape[1]):
            sample = prediction_samples[:, i]
            peak_idx = np.argmax(sample)
            peak_val = sample[peak_idx]
            peak_months.append(peak_idx)
            peak_values.append(peak_val)
        
        peak_months = np.array(peak_months)
        peak_values = np.array(peak_values)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Peak month distribution
        ax1.hist(peak_months, bins=min(20, len(np.unique(peak_months))), 
                color=self.colors['uncertainty'], alpha=0.7, edgecolor='black')
        
        if actual_peak:
            ax1.axvline(actual_peak[0], color=self.colors['actual'], 
                       linewidth=3, label=f'Actual: {actual_peak[0]}')
        
        # Add statistics
        mean_month = np.mean(peak_months)
        std_month = np.std(peak_months)
        ax1.axvline(mean_month, color=self.colors['prediction'], 
                   linewidth=3, label=f'Mean: {mean_month:.1f}±{std_month:.1f}')
        
        ax1.set_title('Peak Month Distribution')
        ax1.set_xlabel('Month')
        ax1.set_ylabel('Frequency')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Peak value distribution
        ax2.hist(peak_values, bins=20, color=self.colors['uncertainty'], 
                alpha=0.7, edgecolor='black')
        
        if actual_peak:
            ax2.axvline(actual_peak[1], color=self.colors['actual'], 
                       linewidth=3, label=f'Actual: {actual_peak[1]:.1f}')
        
        mean_value = np.mean(peak_values)
        std_value = np.std(peak_values)
        ax2.axvline(mean_value, color=self.colors['prediction'], 
                   linewidth=3, label=f'Mean: {mean_value:.1f}±{std_value:.1f}')
        
        ax2.set_title('Peak Value Distribution')
        ax2.set_xlabel('Sunspot Number')
        ax2.set_ylabel('Frequency')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        
        return fig
    
    def plot_nbeats_decomposition(self,
                                decomposition: Dict[str, np.ndarray],
                                actual: Optional[np.ndarray] = None,
                                title: str = "N-BEATS Decomposition",
                                save_path: Optional[Path] = None) -> plt.Figure:
        """Plot N-BEATS interpretable decomposition."""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        components = ['trend', 'seasonal', 'generic', 'total']
        months = np.arange(len(decomposition['total']))
        
        for i, component in enumerate(components):
            if component not in decomposition:
                axes[i].set_visible(False)
                continue
            
            ax = axes[i]
            
            # Plot component
            ax.plot(months, decomposition[component], 
                   color=self.colors.get(component, self.colors['prediction']),
                   linewidth=2.5, label=component.title())
            
            # For total, also plot actual if available
            if component == 'total' and actual is not None:
                actual_months = np.arange(len(actual))
                ax.plot(actual_months, actual, 
                       color=self.colors['actual'], linewidth=2.5,
                       label='Actual', alpha=0.8)
            
            ax.set_title(f"{component.title()} Component", fontweight='bold')
            ax.set_xlabel('Months')
            ax.set_ylabel('Sunspot Number')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        
        return fig


def create_summary_report(results: Dict[str, Any], 
                         output_dir: Path,
                         experiment_name: str = "Solar Cycle Prediction") -> None:
    """Create a comprehensive summary report with all visualizations."""
    
    plotter = SolarCyclePlotter(style='publication')
    
    # Create summary document
    report_lines = [
        f"# {experiment_name} - Results Summary",
        f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Model Performance",
    ]
    
    if 'training_results' in results:
        training = results['training_results']
        report_lines.extend([
            f"- Best Validation Loss: {training.get('best_val_loss', 'N/A'):.4f}",
            f"- Best Epoch: {training.get('best_epoch', 'N/A')}",
            f"- Total Epochs: {training.get('total_epochs', 'N/A')}",
            f"- Early Stopped: {training.get('early_stopped', 'N/A')}",
            ""
        ])
    
    if 'prediction_results' in results:
        report_lines.append("## Cycle Predictions")
        for cycle_name, result in results['prediction_results'].items():
            if 'rmse' in result:
                report_lines.append(f"- {cycle_name}: RMSE={result['rmse']:.1f}, MAE={result.get('mae', 'N/A'):.1f}")
            else:
                max_val = np.max(result['prediction']) if 'prediction' in result else 'N/A'
                report_lines.append(f"- {cycle_name}: Max prediction={max_val:.1f}")
        report_lines.append("")
    
    # Save report
    with open(output_dir / "summary_report.md", 'w') as f:
        f.write('\n'.join(report_lines))
    
    print(f"Summary report saved to {output_dir / 'summary_report.md'}")


if __name__ == "__main__":
    # Test the plotting utilities
    print("Testing enhanced plotting utilities...")
    
    # Create test data
    np.random.seed(42)
    months = np.arange(132)
    
    # Create synthetic cycle with noise
    actual = 50 + 40 * np.sin(np.linspace(0, 2*np.pi, 132)) + np.random.normal(0, 5, 132)
    prediction = actual + np.random.normal(0, 8, 132)
    
    # Create uncertainty data
    uncertainty = {
        'q10': prediction - 15,
        'q50': prediction,
        'q90': prediction + 15,
        'std': np.full_like(prediction, 10)
    }
    
    # Initialize plotter
    plotter = SolarCyclePlotter()
    
    # Test single cycle plot
    print("Testing single cycle plot with uncertainty...")
    fig1 = plotter.plot_single_cycle_with_uncertainty(
        actual=actual,
        prediction=prediction,
        uncertainty=uncertainty,
        title="Test Solar Cycle",
        cycle_info={'peak_month': 66, 'peak_value': 120, 'rmse': 8.5}
    )
    
    # Test MC-Dropout plot
    print("Testing MC-Dropout uncertainty plot...")
    mc_samples = np.random.normal(prediction[:, np.newaxis], 10, (132, 30))
    fig2 = plotter.plot_mc_dropout_uncertainty(
        mc_samples=mc_samples,
        actual=actual,
        title="Test MC-Dropout"
    )
    
    # Test peak distribution
    print("Testing peak distribution plot...")
    fig3 = plotter.plot_peak_distribution(
        prediction_samples=mc_samples,
        actual_peak=(66, 120),
        title="Test Peak Distribution"
    )
    
    plt.show()
    
    print("\n✅ Enhanced plotting utilities implemented and tested!")