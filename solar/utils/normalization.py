"""
Robust normalization utilities for solar cycle data.
Provides variance-stabilizing transforms and robust scaling.
"""

import numpy as np
import json
from typing import Dict, Optional, Tuple, Any
from pathlib import Path
from sklearn.preprocessing import StandardScaler, RobustScaler as SklearnRobustScaler


class RobustScaler:
    """Enhanced robust scaler with configurable transforms and persistence."""
    
    def __init__(self, method: str = "robust", transform: str = "identity", 
                 quantile_range: Tuple[float, float] = (25.0, 75.0)):
        """
        Args:
            method: "robust", "standard", or "none"
            transform: "identity", "log1p", or "sqrt" 
            quantile_range: IQR range for robust scaling
        """
        self.method = method
        self.transform_type = transform  # Renamed to avoid conflict with transform() method
        self.quantile_range = quantile_range
        
        # Initialize scalers (accept either a plain string or a str-Enum).
        method_str = str(getattr(method, 'value', method)).lower()

        if method_str == "robust":
            self.scaler = SklearnRobustScaler(quantile_range=quantile_range)
        elif method_str == "standard":
            self.scaler = StandardScaler()
        else:
            self.scaler = None
            
        # Transform parameters
        self.transform_params = {}
        self.fitted = False
    
    def _apply_transform(self, data: np.ndarray) -> np.ndarray:
        """Apply variance-stabilizing transform."""
        transform_str = str(getattr(self.transform_type, 'value', self.transform_type)).lower()

        if transform_str == "log1p":
            # Ensure positive values for log transform
            if np.any(data < 0):
                # Shift to make all values positive
                shift = abs(np.min(data)) + 1
                self.transform_params['log1p_shift'] = shift
                data = data + shift
            return np.log1p(data)
        elif transform_str == "sqrt":
            # Ensure non-negative values for sqrt
            if np.any(data < 0):
                shift = abs(np.min(data))
                self.transform_params['sqrt_shift'] = shift
                data = data + shift
            return np.sqrt(data)
        else:  # identity
            return data
    
    def _inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """Inverse variance-stabilizing transform."""
        transform_str = str(getattr(self.transform_type, 'value', self.transform_type)).lower()

        if transform_str == "log1p":
            data = np.expm1(data)
            if 'log1p_shift' in self.transform_params:
                data = data - self.transform_params['log1p_shift']
        elif transform_str == "sqrt":
            data = np.square(data)
            if 'sqrt_shift' in self.transform_params:
                data = data - self.transform_params['sqrt_shift']
        return data
    
    def fit(self, data: np.ndarray) -> 'RobustScaler':
        """Fit the scaler to data."""
        data = data.reshape(-1, 1) if data.ndim == 1 else data
        
        # Apply variance-stabilizing transform
        transformed_data = self._apply_transform(data)
        
        # Fit scaler if not "none"
        if self.scaler is not None:
            self.scaler.fit(transformed_data)
        
        self.fitted = True
        return self
    
    def transform(self, data: np.ndarray) -> np.ndarray:
        """Transform data using fitted parameters."""
        if not self.fitted:
            raise ValueError("Scaler not fitted. Call fit() first.")
        
        original_shape = data.shape
        data = data.reshape(-1, 1) if data.ndim == 1 else data
        
        # Apply variance-stabilizing transform
        transformed_data = self._apply_transform(data)
        
        # Apply scaling if not "none"
        if self.scaler is not None:
            scaled_data = self.scaler.transform(transformed_data)
        else:
            scaled_data = transformed_data
        
        return scaled_data.reshape(original_shape)
    
    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """Inverse transform data back to original scale."""
        if not self.fitted:
            raise ValueError("Scaler not fitted. Call fit() first.")
        
        original_shape = data.shape
        data = data.reshape(-1, 1) if data.ndim == 1 else data
        
        # Inverse scaling if not "none"
        if self.scaler is not None:
            unscaled_data = self.scaler.inverse_transform(data)
        else:
            unscaled_data = data
        
        # Inverse variance-stabilizing transform
        original_data = self._inverse_transform(unscaled_data)
        
        return original_data.reshape(original_shape)
    
    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        """Fit and transform data in one step."""
        self.fit(data)
        return self.transform(data)
    
    def save_params(self, filepath: Path) -> None:
        """Save scaler parameters to JSON file."""
        if not self.fitted:
            raise ValueError("Cannot save unfitted scaler.")
        
        params = {
            'method': self.method,
            'transform': self.transform_type,
            'quantile_range': self.quantile_range,
            'transform_params': self.transform_params,
            'fitted': self.fitted
        }
        
        # Add scaler-specific parameters
        if self.scaler is not None:
            if hasattr(self.scaler, 'center_'):
                params['center_'] = self.scaler.center_.tolist()
            if hasattr(self.scaler, 'scale_'):
                params['scale_'] = self.scaler.scale_.tolist()
            if hasattr(self.scaler, 'mean_'):
                params['mean_'] = self.scaler.mean_.tolist()
            if hasattr(self.scaler, 'var_'):
                params['var_'] = self.scaler.var_.tolist()
        
        with open(filepath, 'w') as f:
            json.dump(params, f, indent=2)
    
    @classmethod
    def load_params(cls, filepath: Path) -> 'RobustScaler':
        """Load scaler parameters from JSON file."""
        with open(filepath, 'r') as f:
            params = json.load(f)
        
        # Create scaler instance
        scaler = cls(
            method=params['method'],
            transform=params['transform'],
            quantile_range=tuple(params['quantile_range'])
        )
        
        # Restore transform parameters
        scaler.transform_params = params['transform_params']
        scaler.fitted = params['fitted']
        
        # Restore scaler parameters
        if scaler.scaler is not None and params['fitted']:
            if 'center_' in params:
                scaler.scaler.center_ = np.array(params['center_'])
            if 'scale_' in params:
                scaler.scaler.scale_ = np.array(params['scale_'])
            if 'mean_' in params:
                scaler.scaler.mean_ = np.array(params['mean_'])
            if 'var_' in params:
                scaler.scaler.var_ = np.array(params['var_'])
        
        return scaler


def create_solar_features(monthly_data: np.ndarray, window: int = 13) -> Dict[str, np.ndarray]:
    """
    Create additional features for solar cycle prediction.
    
    Args:
        monthly_data: Monthly sunspot numbers
        window: Window size for moving average (default: 13 months)
    
    Returns:
        Dictionary of feature arrays
    """
    n_months = len(monthly_data)
    features = {}
    
    # Month indices
    month_indices = np.arange(n_months)
    
    # Sine/cosine encoding for cycle position (approximate 11-year cycle)
    cycle_period = 132  # 11 years * 12 months
    features['cycle_sin'] = np.sin(2 * np.pi * month_indices / cycle_period)
    features['cycle_cos'] = np.cos(2 * np.pi * month_indices / cycle_period)
    
    # Calendar month sine/cosine (seasonal effects)
    calendar_period = 12
    features['calendar_sin'] = np.sin(2 * np.pi * (month_indices % 12) / calendar_period)
    features['calendar_cos'] = np.cos(2 * np.pi * (month_indices % 12) / calendar_period)
    
    # Running mean (smoothed trend)
    from scipy.ndimage import uniform_filter1d
    features['running_mean'] = uniform_filter1d(
        monthly_data.astype(float), size=window, mode='nearest'
    )
    
    # Rising/declining phase indicator
    # Use derivative of smoothed data
    smoothed = uniform_filter1d(monthly_data.astype(float), size=5, mode='nearest')
    derivative = np.gradient(smoothed)
    features['rising_phase'] = (derivative > 0).astype(float)
    
    # Lag features (previous values)
    for lag in [1, 3, 6, 12]:
        lag_feature = np.zeros_like(monthly_data, dtype=float)
        lag_feature[lag:] = monthly_data[:-lag]
        features[f'lag_{lag}'] = lag_feature
    
    return features


def prepare_enhanced_monthly_data(df, target_col: str = 'sunspot_number', 
                                start_year: int = 1749, 
                                scaler_config: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, np.ndarray], RobustScaler]:
    """
    Enhanced data preparation with features and configurable scaling.
    
    Args:
        df: DataFrame with solar data
        target_col: Target column name
        start_year: Starting year for data
        scaler_config: Scaler configuration dict
    
    Returns:
        Tuple of (scaled_target, features_dict, fitted_scaler)
    """
    # Default scaler config
    if scaler_config is None:
        scaler_config = {
            'method': 'robust',
            'transform': 'sqrt',  # Variance-stabilizing for sunspot data
            'quantile_range': (25.0, 75.0)
        }
    
    # Prepare monthly data
    import pandas as pd
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')
    df = df[df.index.year >= start_year]
    
    monthly_data = df[target_col].resample('ME').mean().dropna()
    monthly_values = monthly_data.values
    
    # Create features
    features = create_solar_features(monthly_values)
    
    # Fit and transform target
    scaler = RobustScaler(**scaler_config)
    scaled_target = scaler.fit_transform(monthly_values)
    
    return scaled_target, features, scaler


if __name__ == "__main__":
    # Test the robust scaler
    import matplotlib.pyplot as plt
    
    # Generate test data with outliers
    np.random.seed(42)
    data = np.random.normal(100, 20, 1000)
    data[50:60] = 300  # Add outliers
    
    # Test different scalers
    scalers = {
        'Standard': RobustScaler(method='standard', transform='identity'),
        'Robust': RobustScaler(method='robust', transform='identity'),
        'Robust + Log': RobustScaler(method='robust', transform='log1p'),
        'Robust + Sqrt': RobustScaler(method='robust', transform='sqrt')
    }
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    
    for i, (name, scaler) in enumerate(scalers.items()):
        scaled_data = scaler.fit_transform(data)
        reconstructed = scaler.inverse_transform(scaled_data)
        
        axes[i].hist(scaled_data, bins=50, alpha=0.7, label='Scaled')
        axes[i].set_title(f'{name}\nMean: {np.mean(scaled_data):.2f}, Std: {np.std(scaled_data):.2f}')
        axes[i].legend()
    
    plt.tight_layout()
    plt.savefig('scaler_comparison.png', dpi=150)
    plt.show()
    
    print("✅ Normalization utilities implemented and tested!")