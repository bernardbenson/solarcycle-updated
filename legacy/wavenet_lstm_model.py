"""
WaveNet + LSTM implementation for Solar Cycle 25 forecasting.
Based on the paper "Forecasting Solar Cycle 25 using Deep Neural Networks" (arXiv:2005.12406).
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Dict
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


class SolarDataset(Dataset):
    """Simple dataset for sunspot time series data."""
    
    def __init__(self, data: np.ndarray, sequence_length: int = 120, 
                 prediction_horizon: int = 132):
        """
        Args:
            data: Time series data (n_samples,)
            sequence_length: Length of input sequences (months)
            prediction_horizon: Number of months to predict ahead
        """
        self.data = torch.FloatTensor(data)
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        
        # Create sequences
        self.sequences = []
        self.targets = []
        
        # Ensure we have enough data
        min_required = sequence_length + prediction_horizon
        if len(data) < min_required:
            raise ValueError(f"Data too short. Need at least {min_required} samples, got {len(data)}")
        
        for i in range(sequence_length, len(data) - prediction_horizon + 1):
            seq = self.data[i-sequence_length:i]
            target = self.data[i:i+prediction_horizon]
            
            self.sequences.append(seq.unsqueeze(-1))  # Add feature dimension
            self.targets.append(target)
        
        if len(self.sequences) == 0:
            raise ValueError(f"No valid sequences created. Check data size and parameters.")
            
        self.sequences = torch.stack(self.sequences)
        self.targets = torch.stack(self.targets)
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]


class CausalConv1d(nn.Module):
    """Causal convolution ensuring no future information leakage."""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, 
                 dilation: int = 1):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size, 
            padding=self.padding, dilation=dilation
        )
        
    def forward(self, x):
        x = self.conv(x)
        return x[:, :, :-self.padding] if self.padding > 0 else x


class WaveNetBlock(nn.Module):
    """WaveNet residual block with gated activation."""
    
    def __init__(self, residual_channels: int, gate_channels: int, 
                 skip_channels: int, kernel_size: int = 2, 
                 dilation: int = 1, dropout: float = 0.1):
        super().__init__()
        
        # Dilated causal convolutions
        self.filter_conv = CausalConv1d(
            residual_channels, gate_channels, kernel_size, dilation
        )
        self.gate_conv = CausalConv1d(
            residual_channels, gate_channels, kernel_size, dilation
        )
        
        # 1x1 convolutions
        self.residual_conv = nn.Conv1d(gate_channels, residual_channels, 1)
        self.skip_conv = nn.Conv1d(gate_channels, skip_channels, 1)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # Gated activation unit
        filter_out = torch.tanh(self.filter_conv(x))
        gate_out = torch.sigmoid(self.gate_conv(x))
        
        # Gated output
        gated = filter_out * gate_out
        gated = self.dropout(gated)
        
        # Residual connection
        residual = self.residual_conv(gated)
        if residual.shape == x.shape:
            residual = residual + x
        
        # Skip connection
        skip = self.skip_conv(gated)
        
        return residual, skip


class SimpleWaveNet(nn.Module):
    """Simplified WaveNet-inspired feature extractor for efficiency."""
    
    def __init__(self, n_features: int = 1, channels: int = 32, 
                 n_layers: int = 6, kernel_size: int = 3, dropout: float = 0.1):
        super().__init__()
        
        self.n_features = n_features
        self.channels = channels
        
        # Input projection
        self.input_conv = nn.Conv1d(n_features, channels, 1)
        
        # Simplified dilated convolutions
        self.conv_layers = nn.ModuleList()
        for i in range(n_layers):
            dilation = 2 ** (i % 4)  # Limit max dilation
            self.conv_layers.append(
                nn.Sequential(
                    nn.Conv1d(channels, channels, kernel_size, 
                             padding=(kernel_size-1)*dilation//2, dilation=dilation),
                    nn.BatchNorm1d(channels),
                    nn.ReLU(),
                    nn.Dropout(dropout)
                )
            )
        
        # Output projection
        self.output_conv = nn.Conv1d(channels, channels, 1)
        
    def forward(self, x):
        # x: (batch_size, seq_len, n_features)
        x = x.transpose(1, 2)  # (batch_size, n_features, seq_len)
        
        # Input projection
        x = self.input_conv(x)
        
        # Apply dilated convolutions with residual connections
        for conv_layer in self.conv_layers:
            residual = x
            x = conv_layer(x)
            if x.shape == residual.shape:
                x = x + residual  # Residual connection
        
        # Output projection
        x = self.output_conv(x)
        
        # Convert back to (batch_size, seq_len, features)
        return x.transpose(1, 2)


class WaveNetLSTM(nn.Module):
    """
    WaveNet + LSTM model for Solar Cycle 25 forecasting.
    Based on paper methodology.
    """
    
    def __init__(self, n_features: int = 1, wavenet_channels: int = 64,
                 lstm_hidden_size: int = 128, lstm_layers: int = 2,
                 prediction_horizon: int = 132, dropout: float = 0.1):
        super().__init__()
        
        self.prediction_horizon = prediction_horizon
        
        # WaveNet feature extractor
        self.wavenet = SimpleWaveNet(
            n_features=n_features,
            channels=wavenet_channels,
            n_layers=6,  # Increased for paper methodology
            dropout=dropout
        )
        
        # LSTM for long-term sequence modeling
        self.lstm = nn.LSTM(
            input_size=wavenet_channels,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_layers,
            dropout=dropout if lstm_layers > 1 else 0,
            batch_first=True,
            bidirectional=False
        )
        
        # Output layers for paper methodology
        self.output_layers = nn.Sequential(
            nn.Linear(lstm_hidden_size, lstm_hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden_size // 2, prediction_horizon)
        )
        
    def forward(self, x):
        # Extract WaveNet features
        wavenet_features = self.wavenet(x)
        
        # LSTM processing
        lstm_out, _ = self.lstm(wavenet_features)
        
        # Use last hidden state for prediction
        final_hidden = lstm_out[:, -1, :]
        
        # Generate predictions
        output = self.output_layers(final_hidden)
        
        return output


class SolarCycleTrainer:
    """Trainer for WaveNet+LSTM solar cycle prediction."""
    
    def __init__(self, device: str = 'auto'):
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"Using device: {self.device}")
        
        self.scaler = StandardScaler()
        self.model = None
        self.training_history = {}
        
    def prepare_monthly_data(self, df: pd.DataFrame, target_col: str = 'sunspot_number',
                           start_year: int = 1749) -> np.ndarray:
        """
        Prepare monthly averaged data as in the paper.
        """
        print("Preparing monthly averaged data...")
        
        # Convert to monthly averages
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')
        
        # Filter from start year
        df = df[df.index.year >= start_year]
        
        # Resample to monthly averages
        monthly_data = df[target_col].resample('M').mean()
        
        # Remove NaN values
        monthly_data = monthly_data.dropna()
        
        print(f"Monthly data shape: {monthly_data.shape}")
        print(f"Date range: {monthly_data.index[0]} to {monthly_data.index[-1]}")
        print(f"Years of data: {len(monthly_data) / 12:.1f}")
        
        return monthly_data.values
    
    def create_datasets(self, data: np.ndarray, sequence_length: int = 120,
                       prediction_horizon: int = 132, train_ratio: float = 0.8) -> Dict:
        """Create train/test datasets with proper temporal splitting."""
        
        # Scale the data
        data_scaled = self.scaler.fit_transform(data.reshape(-1, 1)).ravel()
        
        # Temporal split (no shuffling)
        split_idx = int(len(data_scaled) * train_ratio)
        
        train_data = data_scaled[:split_idx]
        test_data = data_scaled[split_idx:]
        
        # Create datasets
        train_dataset = SolarDataset(train_data, sequence_length, prediction_horizon)
        test_dataset = SolarDataset(test_data, sequence_length, prediction_horizon)
        
        print(f"Training samples: {len(train_dataset)}")
        print(f"Test samples: {len(test_dataset)}")
        
        return {
            'train_dataset': train_dataset,
            'test_dataset': test_dataset,
            'data_scaled': data_scaled,
            'split_idx': split_idx
        }
    
    def train(self, data: np.ndarray, epochs: int = 200, batch_size: int = 16,
              lr: float = 1e-3, patience: int = 20) -> Dict:
        """Train the WaveNet+LSTM model."""
        
        # Prepare data
        data_info = self.create_datasets(data)
        
        # Initialize model for paper methodology
        self.model = WaveNetLSTM(
            n_features=1,
            wavenet_channels=64,   # Paper configuration
            lstm_hidden_size=128,  # Paper configuration
            lstm_layers=2,         # Paper configuration
            prediction_horizon=132, # 11 years = 132 months (paper)
            dropout=0.1
        ).to(self.device)
        
        # Data loaders
        train_loader = DataLoader(
            data_info['train_dataset'], batch_size=batch_size, shuffle=True
        )
        test_loader = DataLoader(
            data_info['test_dataset'], batch_size=batch_size, shuffle=False
        )
        
        # Optimizer and scheduler
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=patience//2
        )
        
        criterion = nn.MSELoss()
        
        # Training history
        history = {'train_loss': [], 'test_loss': [], 'lr': []}
        
        # Early stopping
        best_test_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        
        print(f"\nTraining WaveNet+LSTM for {epochs} epochs...")
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_losses = []
            
            for data_batch, target_batch in train_loader:
                data_batch = data_batch.to(self.device)
                target_batch = target_batch.to(self.device)
                
                optimizer.zero_grad()
                
                output = self.model(data_batch)
                loss = criterion(output, target_batch)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                
                train_losses.append(loss.item())
            
            # Testing
            self.model.eval()
            test_losses = []
            
            with torch.no_grad():
                for data_batch, target_batch in test_loader:
                    data_batch = data_batch.to(self.device)
                    target_batch = target_batch.to(self.device)
                    
                    output = self.model(data_batch)
                    loss = criterion(output, target_batch)
                    test_losses.append(loss.item())
            
            # Calculate averages
            avg_train_loss = np.mean(train_losses)
            avg_test_loss = np.mean(test_losses)
            current_lr = optimizer.param_groups[0]['lr']
            
            history['train_loss'].append(avg_train_loss)
            history['test_loss'].append(avg_test_loss)
            history['lr'].append(current_lr)
            
            # Print progress
            if (epoch + 1) % 20 == 0 or epoch == 0:
                print(f"Epoch {epoch+1:3d}/{epochs}: "
                      f"Train Loss: {avg_train_loss:.6f}, "
                      f"Test Loss: {avg_test_loss:.6f}, "
                      f"LR: {current_lr:.2e}")
            
            # Learning rate scheduling
            scheduler.step(avg_test_loss)
            
            # Early stopping
            if avg_test_loss < best_test_loss:
                best_test_loss = avg_test_loss
                patience_counter = 0
                best_model_state = self.model.state_dict().copy()
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break
        
        # Load best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
        
        self.training_history = history
        
        print(f"Training completed. Best test loss: {best_test_loss:.6f}")
        
        return {
            'history': history,
            'best_test_loss': best_test_loss,
            'data_info': data_info
        }
    
    def predict_solar_cycle_25(self, data: np.ndarray, last_n_months: int = 120) -> Dict:
        """Generate Solar Cycle 25 predictions."""
        
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        self.model.eval()
        
        # Use last N months as input
        recent_data = data[-last_n_months:]
        recent_scaled = self.scaler.transform(recent_data.reshape(-1, 1)).ravel()
        
        # Prepare input
        input_seq = torch.FloatTensor(recent_scaled).unsqueeze(0).unsqueeze(-1).to(self.device)
        
        # Generate prediction
        with torch.no_grad():
            prediction_scaled = self.model(input_seq).cpu().numpy().ravel()
        
        # Inverse transform
        prediction = self.scaler.inverse_transform(
            prediction_scaled.reshape(-1, 1)
        ).ravel()
        
        # Calculate statistics
        max_sunspot_idx = np.argmax(prediction)
        max_sunspot_value = prediction[max_sunspot_idx]
        
        return {
            'prediction': prediction,
            'max_sunspot_number': max_sunspot_value,
            'max_sunspot_month': max_sunspot_idx + 1,
            'prediction_years': len(prediction) / 12
        }
    
    def evaluate(self, data_info: Dict) -> Dict:
        """Evaluate model performance."""
        
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        self.model.eval()
        
        test_loader = DataLoader(
            data_info['test_dataset'], batch_size=32, shuffle=False
        )
        
        predictions = []
        actuals = []
        
        with torch.no_grad():
            for data_batch, target_batch in test_loader:
                data_batch = data_batch.to(self.device)
                
                output = self.model(data_batch)
                
                pred = output.cpu().numpy()
                actual = target_batch.numpy()
                
                predictions.append(pred)
                actuals.append(actual)
        
        predictions = np.concatenate(predictions, axis=0)
        actuals = np.concatenate(actuals, axis=0)
        
        # Inverse transform to original scale
        pred_orig = self.scaler.inverse_transform(
            predictions.reshape(-1, 1)
        ).reshape(predictions.shape)
        actual_orig = self.scaler.inverse_transform(
            actuals.reshape(-1, 1)
        ).reshape(actuals.shape)
        
        # Calculate metrics
        mae = mean_absolute_error(actual_orig.ravel(), pred_orig.ravel())
        rmse = np.sqrt(mean_squared_error(actual_orig.ravel(), pred_orig.ravel()))
        r2 = r2_score(actual_orig.ravel(), pred_orig.ravel())
        
        return {
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'predictions': pred_orig,
            'actuals': actual_orig
        }
    
    def plot_training_history(self, save_path: str = None):
        """Plot training history."""
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Loss plot
        axes[0].plot(self.training_history['train_loss'], label='Train Loss')
        axes[0].plot(self.training_history['test_loss'], label='Test Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training History')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Learning rate plot
        axes[1].plot(self.training_history['lr'])
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Learning Rate')
        axes[1].set_title('Learning Rate Schedule')
        axes[1].grid(True, alpha=0.3)
        axes[1].set_yscale('log')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def backtest_historical_cycles(self, data: np.ndarray, cycle_info: Dict) -> Dict:
        """
        Backtest the model on historical solar cycles to validate performance.
        
        Args:
            data: Full historical sunspot data
            cycle_info: Dictionary with solar cycle start/end dates
        """
        
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        print("Running historical backtest...")
        
        backtest_results = {}
        
        # Test on the last few solar cycles
        test_cycles = [20, 21, 22, 23, 24]  # Solar cycles to backtest
        
        for cycle_num in test_cycles:
            if str(cycle_num) not in cycle_info:
                continue
                
            cycle_start = cycle_info[str(cycle_num)]['start_month']
            cycle_end = cycle_info[str(cycle_num)]['end_month']
            
            # Use 10 years before cycle start as training context
            context_start = max(0, cycle_start - 120)
            context_data = data[context_start:cycle_start]
            
            if len(context_data) < 120:
                continue  # Skip if not enough context
            
            # Make prediction for this cycle
            context_scaled = self.scaler.transform(context_data.reshape(-1, 1)).ravel()
            input_seq = torch.FloatTensor(context_scaled[-120:]).unsqueeze(0).unsqueeze(-1).to(self.device)
            
            with torch.no_grad():
                prediction_scaled = self.model(input_seq).cpu().numpy().ravel()
            
            prediction = self.scaler.inverse_transform(
                prediction_scaled.reshape(-1, 1)
            ).ravel()
            
            # Get actual cycle data
            actual_cycle_length = min(132, cycle_end - cycle_start)
            actual_data = data[cycle_start:cycle_start + actual_cycle_length]
            
            # Trim prediction to match actual length
            prediction = prediction[:len(actual_data)]
            
            # Calculate metrics
            mae = mean_absolute_error(actual_data, prediction)
            rmse = np.sqrt(mean_squared_error(actual_data, prediction))
            
            # Peak prediction accuracy
            actual_max = np.max(actual_data)
            predicted_max = np.max(prediction)
            peak_error = abs(actual_max - predicted_max)
            
            backtest_results[cycle_num] = {
                'actual_data': actual_data,
                'predicted_data': prediction,
                'actual_max': actual_max,
                'predicted_max': predicted_max,
                'peak_error': peak_error,
                'mae': mae,
                'rmse': rmse,
                'cycle_start_month': cycle_start,
                'cycle_length_months': len(actual_data)
            }
            
            print(f"Solar Cycle {cycle_num}: RMSE={rmse:.1f}, Peak Error={peak_error:.1f}")
        
        return backtest_results
    
    def plot_backtest_results(self, backtest_results: Dict, save_path: str = None):
        """Plot backtest results for historical solar cycles."""
        
        n_cycles = len(backtest_results)
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for i, (cycle_num, results) in enumerate(backtest_results.items()):
            if i >= 6:  # Limit to 6 subplots
                break
                
            actual = results['actual_data']
            predicted = results['predicted_data']
            months = np.arange(len(actual))
            
            axes[i].plot(months, actual, 'b-', linewidth=2, label='Actual', alpha=0.8)
            axes[i].plot(months, predicted, 'r-', linewidth=2, label='Predicted', alpha=0.8)
            
            # Mark peaks
            axes[i].axhline(y=results['actual_max'], color='blue', linestyle='--', alpha=0.5)
            axes[i].axhline(y=results['predicted_max'], color='red', linestyle='--', alpha=0.5)
            
            axes[i].set_title(f'Solar Cycle {cycle_num}\nRMSE: {results["rmse"]:.1f}, Peak Error: {results["peak_error"]:.1f}')
            axes[i].set_xlabel('Months into Cycle')
            axes[i].set_ylabel('Sunspot Number')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(n_cycles, 6):
            axes[i].set_visible(False)
        
        plt.suptitle('Historical Solar Cycle Backtesting - WaveNet+LSTM Model', fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_predictions(self, results: Dict, save_path: str = None):
        """Plot Solar Cycle 25 predictions."""
        
        prediction = results['prediction']
        months = np.arange(1, len(prediction) + 1)
        years = months / 12
        
        plt.figure(figsize=(12, 6))
        plt.plot(years, prediction, 'r-', linewidth=2, label='Solar Cycle 25 Prediction')
        plt.axhline(y=results['max_sunspot_number'], color='orange', linestyle='--', 
                   alpha=0.7, label=f'Max: {results["max_sunspot_number"]:.1f}')
        
        plt.xlabel('Years into Solar Cycle 25')
        plt.ylabel('Sunspot Number')
        plt.title('Solar Cycle 25 Forecast - WaveNet+LSTM Model')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Add text annotation for maximum
        plt.text(results['max_sunspot_month']/12, results['max_sunspot_number'] + 5,
                f'Peak: {results["max_sunspot_number"]:.1f}\nMonth {results["max_sunspot_month"]}',
                ha='center', va='bottom', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()


def main():
    """Example usage of WaveNet+LSTM model."""
    print("WaveNet+LSTM Solar Cycle Prediction Model")
    print("Based on paper: Forecasting Solar Cycle 25 using Deep Neural Networks")
    print("=" * 60)
    
    # This would be integrated with your data loading pipeline
    # For now, just demonstrate the model structure
    trainer = SolarCycleTrainer()
    
    # Example: create dummy data to test the model
    dummy_data = np.random.randn(300) * 50 + 100  # ~25 years of monthly data (smaller for testing)
    
    print("Testing with dummy data...")
    
    # Train model (fewer epochs for testing)
    results = trainer.train(dummy_data, epochs=20, batch_size=8)
    
    # Make predictions
    predictions = trainer.predict_solar_cycle_25(dummy_data)
    
    print(f"\nSolar Cycle 25 Forecast:")
    print(f"Maximum sunspot number: {predictions['max_sunspot_number']:.1f}")
    print(f"Peak month: {predictions['max_sunspot_month']}")
    print(f"Forecast duration: {predictions['prediction_years']:.1f} years")
    
    # Evaluate model
    evaluation = trainer.evaluate(results['data_info'])
    print(f"\nModel Performance:")
    print(f"RMSE: {evaluation['rmse']:.2f}")
    print(f"MAE: {evaluation['mae']:.2f}")
    print(f"R²: {evaluation['r2']:.4f}")


if __name__ == "__main__":
    main()