"""
Feature engineering module for multivariate sunspot prediction.
Creates advanced features from raw solar and geomagnetic data.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy import signal
from statsmodels.tsa.seasonal import seasonal_decompose
from typing import Tuple, List, Optional
import warnings
warnings.filterwarnings('ignore')


class SolarFeatureEngineer:
    """Advanced feature engineering for solar time series data."""
    
    def __init__(self):
        self.scalers = {}
        self.feature_names = []
        
    def create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create temporal features from date column."""
        df = df.copy()
        
        # Basic temporal features
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['day_of_year'] = df['date'].dt.dayofyear
        df['quarter'] = df['date'].dt.quarter
        
        # Cyclical encoding for seasonal patterns
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365.25)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365.25)
        
        # Solar cycle features (11-year cycle)
        solar_cycle_years = (df['year'] - 1755) % 11  # Solar cycles started counting from 1755
        df['solar_cycle_phase'] = solar_cycle_years / 11
        df['solar_cycle_sin'] = np.sin(2 * np.pi * df['solar_cycle_phase'])
        df['solar_cycle_cos'] = np.cos(2 * np.pi * df['solar_cycle_phase'])
        
        return df
    
    def create_lag_features(self, df: pd.DataFrame, target_col: str = 'sunspot_number', 
                           lags: List[int] = [1, 7, 14, 30, 90, 180, 365]) -> pd.DataFrame:
        """Create lagged features for time series prediction."""
        df = df.copy()
        
        for lag in lags:
            df[f'{target_col}_lag_{lag}'] = df[target_col].shift(lag)
            
        # Create lagged features for other important variables
        other_cols = ['f107_F10.7_ADJ', 'ap_avg', 'kp_sum']
        for col in other_cols:
            if col in df.columns:
                for lag in [1, 7, 30]:  # Shorter lags for auxiliary features
                    df[f'{col}_lag_{lag}'] = df[col].shift(lag)
        
        return df
    
    def create_rolling_statistics(self, df: pd.DataFrame, target_col: str = 'sunspot_number',
                                 windows: List[int] = [7, 14, 30, 90, 180, 365]) -> pd.DataFrame:
        """Create rolling window statistics."""
        df = df.copy()
        
        for window in windows:
            # Rolling statistics for sunspot numbers
            df[f'{target_col}_roll_mean_{window}'] = df[target_col].rolling(window=window, min_periods=1).mean()
            df[f'{target_col}_roll_std_{window}'] = df[target_col].rolling(window=window, min_periods=1).std()
            df[f'{target_col}_roll_min_{window}'] = df[target_col].rolling(window=window, min_periods=1).min()
            df[f'{target_col}_roll_max_{window}'] = df[target_col].rolling(window=window, min_periods=1).max()
            df[f'{target_col}_roll_median_{window}'] = df[target_col].rolling(window=window, min_periods=1).median()
            
            # Rolling statistics for F10.7
            if 'f107_F10.7_ADJ' in df.columns:
                df[f'f107_roll_mean_{window}'] = df['f107_F10.7_ADJ'].rolling(window=window, min_periods=1).mean()
                df[f'f107_roll_std_{window}'] = df['f107_F10.7_ADJ'].rolling(window=window, min_periods=1).std()
        
        return df
    
    def create_difference_features(self, df: pd.DataFrame, target_col: str = 'sunspot_number') -> pd.DataFrame:
        """Create differenced features to capture trends and changes."""
        df = df.copy()
        
        # First and second differences
        df[f'{target_col}_diff_1'] = df[target_col].diff()
        df[f'{target_col}_diff_2'] = df[target_col].diff().diff()
        
        # Percentage changes
        df[f'{target_col}_pct_change_1'] = df[target_col].pct_change()
        df[f'{target_col}_pct_change_7'] = df[target_col].pct_change(periods=7)
        df[f'{target_col}_pct_change_30'] = df[target_col].pct_change(periods=30)
        
        # Rate of change over different periods
        for period in [7, 14, 30, 90]:
            df[f'{target_col}_roc_{period}'] = (df[target_col] - df[target_col].shift(period)) / period
        
        return df
    
    def create_seasonal_decomposition_features(self, df: pd.DataFrame, 
                                             target_col: str = 'sunspot_number',
                                             period: int = 365) -> pd.DataFrame:
        """Create seasonal decomposition features."""
        df = df.copy()
        
        if len(df) < 2 * period:
            print(f"Warning: Not enough data for seasonal decomposition (need at least {2*period} points)")
            return df
        
        try:
            # Handle missing values by interpolation
            series = df[target_col].interpolate()
            
            # Seasonal decomposition
            decomposition = seasonal_decompose(series, model='additive', period=period, extrapolate_trend='freq')
            
            df[f'{target_col}_trend'] = decomposition.trend
            df[f'{target_col}_seasonal'] = decomposition.seasonal
            df[f'{target_col}_residual'] = decomposition.resid
            
            # Additional seasonal features
            df[f'{target_col}_detrended'] = series - decomposition.trend
            df[f'{target_col}_deseasonalized'] = series - decomposition.seasonal
            
        except Exception as e:
            print(f"Warning: Seasonal decomposition failed: {e}")
            
        return df
    
    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features between different variables."""
        df = df.copy()
        
        # Sunspot-F10.7 interactions
        if 'f107_F10.7_ADJ' in df.columns:
            df['sunspot_f107_ratio'] = df['sunspot_number'] / (df['f107_F10.7_ADJ'] + 1e-8)
            df['sunspot_f107_product'] = df['sunspot_number'] * df['f107_F10.7_ADJ']
            df['sunspot_f107_diff'] = df['sunspot_number'] - df['f107_F10.7_ADJ']
        
        # Geomagnetic activity interactions
        if 'ap_avg' in df.columns:
            df['sunspot_ap_ratio'] = df['sunspot_number'] / (df['ap_avg'] + 1e-8)
            df['sunspot_ap_product'] = df['sunspot_number'] * df['ap_avg']
        
        # Solar cycle phase interactions
        if 'solar_cycle_phase' in df.columns:
            df['sunspot_cycle_interaction'] = df['sunspot_number'] * df['solar_cycle_phase']
            df['f107_cycle_interaction'] = df.get('f107_F10.7_ADJ', 0) * df['solar_cycle_phase']
        
        return df
    
    def create_spectral_features(self, df: pd.DataFrame, target_col: str = 'sunspot_number') -> pd.DataFrame:
        """Create spectral analysis features using FFT."""
        df = df.copy()
        
        if len(df) < 100:  # Need sufficient data for spectral analysis
            return df
            
        try:
            # Remove NaN values for FFT
            series = df[target_col].dropna()
            if len(series) < 50:
                return df
            
            # FFT analysis
            fft = np.fft.fft(series.values)
            freqs = np.fft.fftfreq(len(series))
            
            # Power spectral density
            psd = np.abs(fft) ** 2
            
            # Find dominant frequencies
            dominant_freq_idx = np.argsort(psd)[-5:]  # Top 5 frequencies
            
            # Create features from dominant frequencies
            for i, idx in enumerate(dominant_freq_idx):
                df[f'{target_col}_dominant_freq_{i}'] = freqs[idx]
                df[f'{target_col}_dominant_power_{i}'] = psd[idx]
            
            # Spectral centroid and bandwidth
            spectral_centroid = np.sum(freqs[:len(freqs)//2] * psd[:len(psd)//2]) / np.sum(psd[:len(psd)//2])
            spectral_bandwidth = np.sqrt(np.sum((freqs[:len(freqs)//2] - spectral_centroid)**2 * psd[:len(psd)//2]) / np.sum(psd[:len(psd)//2]))
            
            df[f'{target_col}_spectral_centroid'] = spectral_centroid
            df[f'{target_col}_spectral_bandwidth'] = spectral_bandwidth
            
        except Exception as e:
            print(f"Warning: Spectral analysis failed: {e}")
        
        return df
    
    def create_volatility_features(self, df: pd.DataFrame, target_col: str = 'sunspot_number') -> pd.DataFrame:
        """Create volatility and variability features."""
        df = df.copy()
        
        # GARCH-like volatility (rolling standard deviation of returns)
        returns = df[target_col].pct_change()
        for window in [7, 14, 30, 90]:
            df[f'{target_col}_volatility_{window}'] = returns.rolling(window=window).std()
            df[f'{target_col}_volatility_ewm_{window}'] = returns.ewm(span=window).std()
        
        # Range-based volatility (Garman-Klass estimator adapted)
        for window in [7, 14, 30]:
            high = df[target_col].rolling(window=window).max()
            low = df[target_col].rolling(window=window).min()
            df[f'{target_col}_range_volatility_{window}'] = (high - low) / df[target_col].rolling(window=window).mean()
        
        return df
    
    def create_regime_features(self, df: pd.DataFrame, target_col: str = 'sunspot_number') -> pd.DataFrame:
        """Create features to identify different solar activity regimes."""
        df = df.copy()
        
        # Define thresholds for different activity levels
        low_threshold = df[target_col].quantile(0.33)
        high_threshold = df[target_col].quantile(0.67)
        
        # Activity regime indicators
        df[f'{target_col}_regime_low'] = (df[target_col] <= low_threshold).astype(int)
        df[f'{target_col}_regime_medium'] = ((df[target_col] > low_threshold) & (df[target_col] <= high_threshold)).astype(int)
        df[f'{target_col}_regime_high'] = (df[target_col] > high_threshold).astype(int)
        
        # Time since regime change
        regime = df[target_col].apply(lambda x: 0 if x <= low_threshold else (1 if x <= high_threshold else 2))
        regime_changes = (regime != regime.shift()).cumsum()
        df[f'{target_col}_regime_duration'] = df.groupby(regime_changes).cumcount() + 1
        
        return df
    
    def engineer_all_features(self, df: pd.DataFrame, target_col: str = 'sunspot_number') -> pd.DataFrame:
        """Apply all feature engineering steps."""
        print("Creating temporal features...")
        df = self.create_temporal_features(df)
        
        print("Creating lag features...")
        df = self.create_lag_features(df, target_col)
        
        print("Creating rolling statistics...")
        df = self.create_rolling_statistics(df, target_col)
        
        print("Creating difference features...")
        df = self.create_difference_features(df, target_col)
        
        print("Creating seasonal decomposition features...")
        df = self.create_seasonal_decomposition_features(df, target_col)
        
        print("Creating interaction features...")
        df = self.create_interaction_features(df)
        
        print("Creating spectral features...")
        df = self.create_spectral_features(df, target_col)
        
        print("Creating volatility features...")
        df = self.create_volatility_features(df, target_col)
        
        print("Creating regime features...")
        df = self.create_regime_features(df, target_col)
        
        # Store feature names (excluding date and target)
        self.feature_names = [col for col in df.columns if col not in ['date', target_col]]
        
        print(f"Feature engineering complete. Created {len(self.feature_names)} features.")
        return df
    
    def prepare_sequences(self, df: pd.DataFrame, target_col: str = 'sunspot_number',
                         sequence_length: int = 60, prediction_horizon: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare sequences for PyTorch time series models."""
        # Remove non-numeric columns and handle missing values
        numeric_df = df.select_dtypes(include=[np.number])
        numeric_df = numeric_df.fillna(numeric_df.mean())
        
        # Separate features and target
        feature_cols = [col for col in numeric_df.columns if col != target_col]
        X_data = numeric_df[feature_cols].values
        y_data = numeric_df[target_col].values
        
        # Create sequences
        X_sequences = []
        y_sequences = []
        
        for i in range(sequence_length, len(X_data) - prediction_horizon + 1):
            X_sequences.append(X_data[i-sequence_length:i])
            y_sequences.append(y_data[i:i+prediction_horizon])
        
        return np.array(X_sequences), np.array(y_sequences)
    
    def scale_features(self, df: pd.DataFrame, target_col: str = 'sunspot_number',
                      scaler_type: str = 'standard') -> pd.DataFrame:
        """Scale features for model training."""
        df = df.copy()
        
        # Select numeric columns only
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if 'date' in numeric_cols:
            numeric_cols.remove('date')
        
        # Initialize scaler
        if scaler_type == 'standard':
            scaler = StandardScaler()
        elif scaler_type == 'minmax':
            scaler = MinMaxScaler()
        else:
            raise ValueError("scaler_type must be 'standard' or 'minmax'")
        
        # Fit and transform features (excluding target)
        feature_cols = [col for col in numeric_cols if col != target_col]
        if feature_cols:
            df[feature_cols] = scaler.fit_transform(df[feature_cols])
            self.scalers['features'] = scaler
        
        # Scale target separately
        target_scaler = StandardScaler() if scaler_type == 'standard' else MinMaxScaler()
        df[[target_col]] = target_scaler.fit_transform(df[[target_col]])
        self.scalers['target'] = target_scaler
        
        return df


def main():
    """Example usage of feature engineering."""
    # Load sample data
    df = pd.read_csv('data/raw_multivariate_data.csv')
    df['date'] = pd.to_datetime(df['date'])
    
    # Initialize feature engineer
    engineer = SolarFeatureEngineer()
    
    # Apply feature engineering
    df_engineered = engineer.engineer_all_features(df, target_col='sunspot_number')
    
    print(f"\nOriginal features: {df.shape[1]}")
    print(f"Engineered features: {df_engineered.shape[1]}")
    print(f"Added {df_engineered.shape[1] - df.shape[1]} new features")
    
    # Save engineered dataset
    df_engineered.to_csv('data/engineered_multivariate_data.csv', index=False)
    print("Engineered dataset saved to 'data/engineered_multivariate_data.csv'")


if __name__ == "__main__":
    main()