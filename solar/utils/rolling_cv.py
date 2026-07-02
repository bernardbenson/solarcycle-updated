"""
Rolling-origin cross-validation for time series forecasting.
Implements time-blocked CV splits for robust evaluation of solar cycle models.
"""

import numpy as np
import torch
from typing import List, Tuple, Dict, Optional, Iterator
from dataclasses import dataclass
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings


@dataclass
class CVFold:
    """Single cross-validation fold with train/test indices."""
    fold_id: int
    train_start: int
    train_end: int
    test_start: int
    test_end: int
    description: str


class RollingOriginCV:
    """
    Rolling-origin cross-validation for time series data.
    
    Creates expanding windows where each fold uses progressively more training data
    and tests on a fixed-size future window.
    """
    
    def __init__(self, n_folds: int = 5, test_size: int = 132, 
                 min_train_size: int = 1000, step_size: Optional[int] = None):
        """
        Args:
            n_folds: Number of CV folds
            test_size: Size of test window (e.g., 132 months = 11 years)
            min_train_size: Minimum training set size
            step_size: Step size between folds (if None, uses test_size)
        """
        self.n_folds = n_folds
        self.test_size = test_size
        self.min_train_size = min_train_size
        self.step_size = step_size or test_size
        
    def split(self, data_length: int) -> List[CVFold]:
        """
        Generate rolling-origin CV splits.
        
        Args:
            data_length: Total length of time series
        
        Returns:
            List of CVFold objects
        """
        folds = []
        
        # Calculate starting position for first fold
        # Ensure we have enough data for all folds
        total_required = self.min_train_size + self.n_folds * self.step_size + self.test_size
        
        if data_length < total_required:
            warnings.warn(
                f"Data length {data_length} may be insufficient for {self.n_folds} folds. "
                f"Consider reducing n_folds or test_size."
            )
        
        # Start with minimum training size
        for fold_id in range(self.n_folds):
            # Calculate fold boundaries
            test_start = self.min_train_size + fold_id * self.step_size
            test_end = test_start + self.test_size
            
            # Check if we have enough data for this fold
            if test_end > data_length:
                break
            
            train_start = 0
            train_end = test_start
            
            fold = CVFold(
                fold_id=fold_id,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                description=f"Fold {fold_id}: Train[{train_start}:{train_end}], Test[{test_start}:{test_end}]"
            )
            
            folds.append(fold)
        
        return folds
    
    def get_fold_data(self, data: np.ndarray, fold: CVFold) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract train and test data for a specific fold.
        
        Args:
            data: Time series data
            fold: CVFold object
        
        Returns:
            Tuple of (train_data, test_data)
        """
        train_data = data[fold.train_start:fold.train_end]
        test_data = data[fold.test_start:fold.test_end]
        
        return train_data, test_data


class BlockedTimeSeriesCV:
    """
    Blocked time series cross-validation with non-overlapping train/test periods.
    Includes gap between train and test to avoid data leakage.
    """
    
    def __init__(self, n_folds: int = 5, test_size: int = 132, 
                 gap_size: int = 0, train_size: Optional[int] = None):
        """
        Args:
            n_folds: Number of CV folds
            test_size: Size of test blocks
            gap_size: Gap between train and test periods
            train_size: Fixed training size (if None, use expanding window)
        """
        self.n_folds = n_folds
        self.test_size = test_size
        self.gap_size = gap_size
        self.train_size = train_size
        
    def split(self, data_length: int) -> List[CVFold]:
        """Generate blocked time series CV splits."""
        folds = []
        
        # Calculate block size including gap
        block_size = self.test_size + self.gap_size
        total_test_blocks = self.n_folds * block_size
        
        if self.train_size is None:
            # Expanding window: use remaining data for training
            min_train_size = data_length - total_test_blocks
            
            if min_train_size < 100:  # Minimum viable training size
                raise ValueError(f"Insufficient data for {self.n_folds} folds with test_size {self.test_size}")
            
            for fold_id in range(self.n_folds):
                test_start = data_length - (self.n_folds - fold_id) * block_size
                test_end = test_start + self.test_size
                
                train_start = 0
                train_end = test_start - self.gap_size
                
                if train_end <= train_start:
                    break
                
                fold = CVFold(
                    fold_id=fold_id,
                    train_start=train_start,
                    train_end=train_end,
                    test_start=test_start,
                    test_end=test_end,
                    description=f"Blocked Fold {fold_id}: Train[{train_start}:{train_end}], Gap[{train_end}:{test_start}], Test[{test_start}:{test_end}]"
                )
                
                folds.append(fold)
        
        else:
            # Fixed window size
            for fold_id in range(self.n_folds):
                test_start = data_length - (self.n_folds - fold_id) * block_size
                test_end = test_start + self.test_size
                
                train_end = test_start - self.gap_size
                train_start = max(0, train_end - self.train_size)
                
                if train_end <= train_start or test_end > data_length:
                    break
                
                fold = CVFold(
                    fold_id=fold_id,
                    train_start=train_start,
                    train_end=train_end,
                    test_start=test_start,
                    test_end=test_end,
                    description=f"Fixed Fold {fold_id}: Train[{train_start}:{train_end}], Gap[{train_end}:{test_start}], Test[{test_start}:{test_end}]"
                )
                
                folds.append(fold)
        
        return folds


class TimeSeriesMetrics:
    """Comprehensive metrics for time series forecasting evaluation."""
    
    @staticmethod
    def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                       prefix: str = "") -> Dict[str, float]:
        """
        Compute comprehensive forecasting metrics.
        
        Args:
            y_true: Ground truth values
            y_pred: Predicted values
            prefix: Prefix for metric names
        
        Returns:
            Dictionary of computed metrics
        """
        # Ensure same length
        min_len = min(len(y_true), len(y_pred))
        y_true = y_true[:min_len]
        y_pred = y_pred[:min_len]
        
        metrics = {}
        prefix = f"{prefix}_" if prefix else ""
        
        # Basic metrics
        metrics[f"{prefix}mae"] = mean_absolute_error(y_true, y_pred)
        metrics[f"{prefix}mse"] = mean_squared_error(y_true, y_pred)
        metrics[f"{prefix}rmse"] = np.sqrt(metrics[f"{prefix}mse"])
        
        # Relative metrics
        mae_baseline = np.mean(np.abs(y_true - np.mean(y_true)))
        metrics[f"{prefix}mae_ratio"] = metrics[f"{prefix}mae"] / mae_baseline if mae_baseline > 0 else np.inf
        
        # R-squared
        try:
            metrics[f"{prefix}r2"] = r2_score(y_true, y_pred)
        except:
            metrics[f"{prefix}r2"] = -np.inf
        
        # Mean Absolute Percentage Error (MAPE)
        mask = y_true != 0
        if np.any(mask):
            mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
            metrics[f"{prefix}mape"] = mape
        else:
            metrics[f"{prefix}mape"] = np.inf
        
        # Symmetric MAPE (more robust)
        smape = np.mean(2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred))) * 100
        metrics[f"{prefix}smape"] = smape
        
        return metrics
    
    @staticmethod
    def compute_dtw_distance(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Compute Dynamic Time Warping distance for shape similarity.
        Simplified DTW implementation.
        """
        try:
            n, m = len(y_true), len(y_pred)
            
            # Create cost matrix
            dtw_matrix = np.full((n + 1, m + 1), np.inf)
            dtw_matrix[0, 0] = 0
            
            for i in range(1, n + 1):
                for j in range(1, m + 1):
                    cost = abs(y_true[i-1] - y_pred[j-1])
                    dtw_matrix[i, j] = cost + min(
                        dtw_matrix[i-1, j],      # insertion
                        dtw_matrix[i, j-1],      # deletion
                        dtw_matrix[i-1, j-1]     # match
                    )
            
            return dtw_matrix[n, m] / max(n, m)  # Normalized DTW
        
        except:
            # Fallback to simple euclidean distance if DTW fails
            min_len = min(len(y_true), len(y_pred))
            return np.sqrt(np.mean((y_true[:min_len] - y_pred[:min_len])**2))


def evaluate_cv_results(cv_results: List[Dict], metric_names: List[str] = None) -> Dict[str, Dict[str, float]]:
    """
    Aggregate cross-validation results across folds.
    
    Args:
        cv_results: List of dictionaries containing metrics for each fold
        metric_names: Specific metrics to aggregate (if None, use all)
    
    Returns:
        Dictionary with aggregated statistics
    """
    if not cv_results:
        return {}
    
    if metric_names is None:
        metric_names = list(cv_results[0].keys())
    
    aggregated = {}
    
    for metric in metric_names:
        if metric in cv_results[0]:
            values = [fold_result[metric] for fold_result in cv_results if metric in fold_result]
            
            if values and all(isinstance(v, (int, float)) and not np.isnan(v) for v in values):
                aggregated[metric] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'median': np.median(values),
                    'values': values
                }
    
    return aggregated


if __name__ == "__main__":
    # Test rolling-origin CV
    np.random.seed(42)
    
    # Generate synthetic time series
    data_length = 2000
    time_series = np.cumsum(np.random.randn(data_length)) + 100
    
    print("Testing Rolling-Origin Cross-Validation...")
    
    # Test rolling CV
    rolling_cv = RollingOriginCV(n_folds=5, test_size=132, min_train_size=1000)
    rolling_folds = rolling_cv.split(data_length)
    
    print(f"\nRolling CV: Generated {len(rolling_folds)} folds")
    for fold in rolling_folds:
        print(f"  {fold.description}")
        train_data, test_data = rolling_cv.get_fold_data(time_series, fold)
        print(f"    Train size: {len(train_data)}, Test size: {len(test_data)}")
    
    # Test blocked CV
    print("\nTesting Blocked Time Series CV...")
    blocked_cv = BlockedTimeSeriesCV(n_folds=3, test_size=132, gap_size=12)
    blocked_folds = blocked_cv.split(data_length)
    
    print(f"\nBlocked CV: Generated {len(blocked_folds)} folds")
    for fold in blocked_folds:
        print(f"  {fold.description}")
    
    # Test metrics
    print("\nTesting Time Series Metrics...")
    y_true = np.sin(np.linspace(0, 4*np.pi, 100)) * 50 + 100
    y_pred = y_true + np.random.normal(0, 5, 100)  # Add noise
    
    metrics = TimeSeriesMetrics.compute_metrics(y_true, y_pred, "test")
    print("Computed metrics:")
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}")
    
    # Test DTW
    dtw_dist = TimeSeriesMetrics.compute_dtw_distance(y_true, y_pred)
    print(f"  DTW distance: {dtw_dist:.4f}")
    
    print("\n✅ Rolling-origin CV system implemented and tested!")