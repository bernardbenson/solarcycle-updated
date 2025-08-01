"""
PyTorch implementation of advanced time series models for sunspot prediction.
Includes Transformer (PatchTST-style), LSTM/GRU with attention, and ensemble methods.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict, List
import math
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')


class SolarTimeSeriesDataset(Dataset):
    """PyTorch Dataset for multivariate solar time series data."""
    
    def __init__(self, data: np.ndarray, targets: np.ndarray, 
                 sequence_length: int = 365, prediction_horizon: int = 30):
        """
        Args:
            data: Input features array (n_samples, n_features)
            targets: Target values array (n_samples,)
            sequence_length: Length of input sequences (days)
            prediction_horizon: Number of days to predict ahead
        """
        self.data = torch.FloatTensor(data)
        self.targets = torch.FloatTensor(targets)
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        
        # Create sequences
        self.sequences = []
        self.target_sequences = []
        
        for i in range(sequence_length, len(data) - prediction_horizon + 1):
            # Input sequence
            seq = self.data[i-sequence_length:i]
            self.sequences.append(seq)
            
            # Target sequence (can be single value or multiple values)
            if prediction_horizon == 1:
                target = self.targets[i]
            else:
                target = self.targets[i:i+prediction_horizon]
            self.target_sequences.append(target)
        
        self.sequences = torch.stack(self.sequences)
        self.target_sequences = torch.stack(self.target_sequences)
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.target_sequences[idx]


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer models."""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:x.size(0), :]


class PatchEmbedding(nn.Module):
    """Patch embedding layer for PatchTST-style models."""
    
    def __init__(self, patch_size: int, n_features: int, d_model: int):
        super().__init__()
        self.patch_size = patch_size
        self.n_features = n_features
        self.d_model = d_model
        
        # Linear projection for patches
        self.projection = nn.Linear(patch_size * n_features, d_model)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, n_features)
        batch_size, seq_len, n_features = x.shape
        
        # Create patches
        n_patches = seq_len // self.patch_size
        if seq_len % self.patch_size != 0:
            # Pad the sequence if needed
            pad_size = self.patch_size - (seq_len % self.patch_size)
            x = F.pad(x, (0, 0, 0, pad_size), mode='replicate')
            n_patches += 1
        
        # Reshape to patches: (batch_size, n_patches, patch_size * n_features)
        x = x[:, :n_patches * self.patch_size, :].reshape(
            batch_size, n_patches, self.patch_size * n_features
        )
        
        # Project patches to d_model dimension
        x = self.projection(x)  # (batch_size, n_patches, d_model)
        
        return x


class SolarTransformerModel(nn.Module):
    """
    PatchTST-inspired transformer model for multivariate sunspot prediction.
    Uses patch-based embeddings and multi-head attention.
    """
    
    def __init__(self, n_features: int, d_model: int = 128, n_heads: int = 8,
                 n_layers: int = 6, patch_size: int = 16, prediction_horizon: int = 30,
                 dropout: float = 0.1):
        super().__init__()
        
        self.n_features = n_features
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.patch_size = patch_size
        self.prediction_horizon = prediction_horizon
        
        # Patch embedding
        self.patch_embedding = PatchEmbedding(patch_size, n_features, d_model)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, n_layers)
        
        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, prediction_horizon)
        )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, n_features)
        
        # Patch embedding
        x = self.patch_embedding(x)  # (batch_size, n_patches, d_model)
        
        # Add positional encoding
        x = x.transpose(0, 1)  # (n_patches, batch_size, d_model)
        x = self.pos_encoding(x)
        x = x.transpose(0, 1)  # (batch_size, n_patches, d_model)
        
        # Layer normalization
        x = self.layer_norm(x)
        
        # Transformer encoding
        x = self.transformer(x)  # (batch_size, n_patches, d_model)
        
        # Global average pooling over patches
        x = x.mean(dim=1)  # (batch_size, d_model)
        
        # Output projection
        output = self.output_projection(x)  # (batch_size, prediction_horizon)
        
        return output


class AttentionLSTMModel(nn.Module):
    """
    LSTM model with attention mechanism for multivariate sunspot prediction.
    """
    
    def __init__(self, n_features: int, hidden_size: int = 128, n_layers: int = 3,
                 prediction_horizon: int = 30, dropout: float = 0.2, 
                 bidirectional: bool = True):
        super().__init__()
        
        self.n_features = n_features
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.prediction_horizon = prediction_horizon
        self.bidirectional = bidirectional
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=n_layers,
            dropout=dropout if n_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )
        
        # Attention mechanism
        lstm_output_size = hidden_size * (2 if bidirectional else 1)
        self.attention = nn.MultiheadAttention(
            embed_dim=lstm_output_size,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        # Output layers
        self.output_layers = nn.Sequential(
            nn.Linear(lstm_output_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, prediction_horizon)
        )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(lstm_output_size)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, n_features)
        
        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(x)
        # lstm_out shape: (batch_size, seq_len, hidden_size * directions)
        
        # Layer normalization
        lstm_out = self.layer_norm(lstm_out)
        
        # Self-attention
        attn_out, attn_weights = self.attention(lstm_out, lstm_out, lstm_out)
        # attn_out shape: (batch_size, seq_len, hidden_size * directions)
        
        # Global average pooling over time dimension
        pooled = attn_out.mean(dim=1)  # (batch_size, hidden_size * directions)
        
        # Output projection
        output = self.output_layers(pooled)  # (batch_size, prediction_horizon)
        
        return output, attn_weights


class AttentionGRUModel(nn.Module):
    """
    GRU model with attention mechanism for multivariate sunspot prediction.
    """
    
    def __init__(self, n_features: int, hidden_size: int = 128, n_layers: int = 3,
                 prediction_horizon: int = 30, dropout: float = 0.2,
                 bidirectional: bool = True):
        super().__init__()
        
        self.n_features = n_features
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.prediction_horizon = prediction_horizon
        self.bidirectional = bidirectional
        
        # GRU layers
        self.gru = nn.GRU(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=n_layers,
            dropout=dropout if n_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )
        
        # Attention mechanism
        gru_output_size = hidden_size * (2 if bidirectional else 1)
        self.attention = nn.MultiheadAttention(
            embed_dim=gru_output_size,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        # Output layers
        self.output_layers = nn.Sequential(
            nn.Linear(gru_output_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, prediction_horizon)
        )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(gru_output_size)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, n_features)
        
        # GRU forward pass
        gru_out, hidden = self.gru(x)
        # gru_out shape: (batch_size, seq_len, hidden_size * directions)
        
        # Layer normalization
        gru_out = self.layer_norm(gru_out)
        
        # Self-attention
        attn_out, attn_weights = self.attention(gru_out, gru_out, gru_out)
        # attn_out shape: (batch_size, seq_len, hidden_size * directions)
        
        # Global average pooling over time dimension
        pooled = attn_out.mean(dim=1)  # (batch_size, hidden_size * directions)
        
        # Output projection
        output = self.output_layers(pooled)  # (batch_size, prediction_horizon)
        
        return output, attn_weights


class EnsembleModel(nn.Module):
    """
    Ensemble model combining Transformer, LSTM, and GRU predictions.
    """
    
    def __init__(self, models: List[nn.Module], weights: Optional[List[float]] = None):
        super().__init__()
        
        self.models = nn.ModuleList(models)
        
        if weights is None:
            self.weights = nn.Parameter(torch.ones(len(models)) / len(models))
        else:
            self.weights = nn.Parameter(torch.tensor(weights, dtype=torch.float32))
        
    def forward(self, x):
        outputs = []
        
        for model in self.models:
            if isinstance(model, SolarTransformerModel):
                out = model(x)
            else:  # LSTM or GRU models
                out, _ = model(x)
            outputs.append(out)
        
        # Weighted ensemble
        stacked_outputs = torch.stack(outputs, dim=0)  # (n_models, batch_size, prediction_horizon)
        weights = F.softmax(self.weights, dim=0).unsqueeze(1).unsqueeze(2)
        
        ensemble_output = (stacked_outputs * weights).sum(dim=0)
        
        return ensemble_output


class ModelTrainer:
    """
    Comprehensive trainer for PyTorch time series models with uncertainty quantification.
    """
    
    def __init__(self, device: str = 'auto'):
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"Using device: {self.device}")
        
        self.scalers = {}
        self.models = {}
        self.training_history = {}
        
    def prepare_data(self, df: pd.DataFrame, target_col: str = 'sunspot_number',
                    sequence_length: int = 365, prediction_horizon: int = 30,
                    train_ratio: float = 0.8, val_ratio: float = 0.1) -> Dict:
        """
        Prepare data for PyTorch training with proper scaling and splitting.
        """
        print(f"Preparing data for PyTorch training...")
        print(f"Original dataset shape: {df.shape}")
        
        # Sort by date and select numeric features
        df = df.sort_values('date').reset_index(drop=True)
        
        # Remove date and non-predictive columns
        exclude_cols = ['date', target_col]  # Exclude target from features
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        # Separate features and target
        X = df[feature_cols].select_dtypes(include=[np.number])  # Only numeric features
        y = df[target_col]
        
        print(f"Feature columns: {len(X.columns)}")
        print(f"Target column: {target_col}")
        
        # Handle missing values and infinite values
        print("Cleaning data (handling NaN and infinity values)...")
        
        # Replace infinite values with NaN, then fill with mean
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(X.mean())
        
        # Clean target variable
        y = y.replace([np.inf, -np.inf], np.nan)
        y = y.fillna(y.mean())
        
        # Final check for any remaining invalid values
        invalid_features = X.columns[X.isin([np.inf, -np.inf, np.nan]).any()].tolist()
        if invalid_features:
            print(f"Warning: Found invalid values in features: {invalid_features}")
            X = X.fillna(0)  # Final safety net
        
        if y.isin([np.inf, -np.inf, np.nan]).any():
            print("Warning: Found invalid values in target variable")
            y = y.fillna(y.mean())
        
        print(f"Data cleaning complete. Shape: {X.shape}")
        
        # Scale features and target
        feature_scaler = StandardScaler()
        target_scaler = StandardScaler()
        
        X_scaled = feature_scaler.fit_transform(X)
        y_scaled = target_scaler.fit_transform(y.values.reshape(-1, 1)).ravel()
        
        self.scalers['features'] = feature_scaler
        self.scalers['target'] = target_scaler
        
        # Time series split (no shuffling to preserve temporal order)
        n_samples = len(X_scaled)
        train_end = int(n_samples * train_ratio)
        val_end = int(n_samples * (train_ratio + val_ratio))
        
        # Create datasets
        train_dataset = SolarTimeSeriesDataset(
            X_scaled[:train_end], y_scaled[:train_end],
            sequence_length, prediction_horizon
        )
        
        val_dataset = SolarTimeSeriesDataset(
            X_scaled[train_end:val_end], y_scaled[train_end:val_end],
            sequence_length, prediction_horizon
        )
        
        test_dataset = SolarTimeSeriesDataset(
            X_scaled[val_end:], y_scaled[val_end:],
            sequence_length, prediction_horizon
        )
        
        data_info = {
            'train_dataset': train_dataset,
            'val_dataset': val_dataset,
            'test_dataset': test_dataset,
            'n_features': X_scaled.shape[1],
            'sequence_length': sequence_length,
            'prediction_horizon': prediction_horizon,
            'feature_names': list(X.columns),
            'target_name': target_col
        }
        
        print(f"Training samples: {len(train_dataset)}")
        print(f"Validation samples: {len(val_dataset)}")
        print(f"Test samples: {len(test_dataset)}")
        
        return data_info
    
    def train_model(self, model: nn.Module, data_info: Dict, model_name: str,
                   batch_size: int = 32, epochs: int = 100, lr: float = 1e-3,
                   patience: int = 15, min_delta: float = 1e-4) -> Dict:
        """
        Train a PyTorch model with early stopping and validation monitoring.
        """
        print(f"\n{'='*60}")
        print(f"Training {model_name}")
        print(f"{'='*60}")
        
        model = model.to(self.device)
        
        # Data loaders
        train_loader = DataLoader(
            data_info['train_dataset'], batch_size=batch_size, shuffle=True
        )
        val_loader = DataLoader(
            data_info['val_dataset'], batch_size=batch_size, shuffle=False
        )
        
        # Optimizer and scheduler
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=patience//2
        )
        
        # Loss function
        criterion = nn.MSELoss()
        
        # Training history
        history = {
            'train_loss': [],
            'val_loss': [],
            'lr': []
        }
        
        # Early stopping
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        
        print(f"Training for up to {epochs} epochs...")
        
        for epoch in range(epochs):
            # Training phase
            model.train()
            train_losses = []
            
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                
                # Forward pass
                if isinstance(model, (AttentionLSTMModel, AttentionGRUModel)):
                    output, _ = model(data)
                else:
                    output = model(data)
                
                # Handle different target shapes
                if target.dim() == 1:
                    target = target.unsqueeze(1)
                if output.shape != target.shape:
                    if target.shape[1] == 1:
                        target = target.expand_as(output)
                
                loss = criterion(output, target)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                train_losses.append(loss.item())
            
            # Validation phase
            model.eval()
            val_losses = []
            
            with torch.no_grad():
                for data, target in val_loader:
                    data, target = data.to(self.device), target.to(self.device)
                    
                    if isinstance(model, (AttentionLSTMModel, AttentionGRUModel)):
                        output, _ = model(data)
                    else:
                        output = model(data)
                    
                    if target.dim() == 1:
                        target = target.unsqueeze(1)
                    if output.shape != target.shape:
                        if target.shape[1] == 1:
                            target = target.expand_as(output)
                    
                    loss = criterion(output, target)
                    val_losses.append(loss.item())
            
            # Calculate average losses
            avg_train_loss = np.mean(train_losses)
            avg_val_loss = np.mean(val_losses)
            current_lr = optimizer.param_groups[0]['lr']
            
            history['train_loss'].append(avg_train_loss)
            history['val_loss'].append(avg_val_loss)
            history['lr'].append(current_lr)
            
            # Print progress
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"Epoch {epoch+1:3d}/{epochs}: "
                      f"Train Loss: {avg_train_loss:.6f}, "
                      f"Val Loss: {avg_val_loss:.6f}, "
                      f"LR: {current_lr:.2e}")
            
            # Learning rate scheduling
            scheduler.step(avg_val_loss)
            
            # Early stopping check
            if avg_val_loss < best_val_loss - min_delta:
                best_val_loss = avg_val_loss
                patience_counter = 0
                best_model_state = model.state_dict().copy()
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                print(f"Best validation loss: {best_val_loss:.6f}")
                break
        
        # Load best model
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        
        # Store model and history
        self.models[model_name] = model
        self.training_history[model_name] = history
        
        print(f"Training completed for {model_name}")
        print(f"Final validation loss: {best_val_loss:.6f}")
        
        return history
    
    def evaluate_model(self, model_name: str, data_info: Dict, batch_size: int = 64) -> Dict:
        """
        Evaluate a trained model on test data with comprehensive metrics.
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found. Available models: {list(self.models.keys())}")
        
        model = self.models[model_name]
        model.eval()
        
        test_loader = DataLoader(
            data_info['test_dataset'], batch_size=batch_size, shuffle=False
        )
        
        predictions = []
        actuals = []
        
        print(f"Evaluating {model_name} on test data...")
        
        with torch.no_grad():
            for data, target in test_loader:
                data = data.to(self.device)
                
                if isinstance(model, (AttentionLSTMModel, AttentionGRUModel)):
                    output, _ = model(data)
                else:
                    output = model(data)
                
                # Convert to numpy for metric calculation
                pred = output.cpu().numpy()
                actual = target.numpy()
                
                predictions.append(pred)
                actuals.append(actual)
        
        # Concatenate all predictions and actuals
        predictions = np.concatenate(predictions, axis=0)
        actuals = np.concatenate(actuals, axis=0)
        
        # Handle different shapes
        if actuals.ndim == 1:
            actuals = actuals.reshape(-1, 1)
        if predictions.shape != actuals.shape:
            if actuals.shape[1] == 1:
                actuals = np.repeat(actuals, predictions.shape[1], axis=1)
        
        # Calculate metrics for each prediction horizon
        metrics = {}
        prediction_horizon = predictions.shape[1]
        
        for h in range(prediction_horizon):
            pred_h = predictions[:, h]
            actual_h = actuals[:, h] if actuals.shape[1] > 1 else actuals[:, 0]
            
            # Inverse transform to original scale
            pred_h_orig = self.scalers['target'].inverse_transform(pred_h.reshape(-1, 1)).ravel()
            actual_h_orig = self.scalers['target'].inverse_transform(actual_h.reshape(-1, 1)).ravel()
            
            metrics[f'horizon_{h+1}'] = {
                'mae': mean_absolute_error(actual_h_orig, pred_h_orig),
                'rmse': np.sqrt(mean_squared_error(actual_h_orig, pred_h_orig)),
                'r2': r2_score(actual_h_orig, pred_h_orig),
                'mape': np.mean(np.abs((actual_h_orig - pred_h_orig) / (actual_h_orig + 1e-8))) * 100
            }
        
        # Overall metrics (average across horizons)
        pred_orig = self.scalers['target'].inverse_transform(predictions.reshape(-1, 1)).ravel()
        actual_orig = self.scalers['target'].inverse_transform(actuals.reshape(-1, 1)).ravel()
        
        metrics['overall'] = {
            'mae': mean_absolute_error(actual_orig, pred_orig),
            'rmse': np.sqrt(mean_squared_error(actual_orig, pred_orig)),
            'r2': r2_score(actual_orig, pred_orig),
            'mape': np.mean(np.abs((actual_orig - pred_orig) / (actual_orig + 1e-8))) * 100
        }
        
        print(f"Evaluation complete for {model_name}")
        print(f"Overall RMSE: {metrics['overall']['rmse']:.2f}")
        print(f"Overall R²: {metrics['overall']['r2']:.4f}")
        
        return {
            'metrics': metrics,
            'predictions': predictions,
            'actuals': actuals,
            'predictions_original_scale': pred_orig,
            'actuals_original_scale': actual_orig
        }


def main():
    """Example usage of PyTorch models."""
    # This will be integrated with the main training pipeline
    print("PyTorch models module loaded successfully!")
    print("Use ModelTrainer class to train transformer and LSTM/GRU models.")


if __name__ == "__main__":
    main()