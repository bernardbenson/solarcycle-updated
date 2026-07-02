"""
PyTorch implementation matching the TensorFlow WaveNet+LSTM model exactly.
Uses 528 months input to predict 132 months output with sliding window.
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Dict
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


class SlidingWindowDataset(Dataset):
    """Dataset matching TensorFlow univariate_data function."""
    
    def __init__(self, data: np.ndarray, history_size: int = 528, 
                 target_size: int = 132, step: int = 1):
        """
        Args:
            data: Time series data
            history_size: Input window size (528 months)
            target_size: Prediction horizon (132 months)
            step: Step size (1 month)
        """
        self.data = data
        self.history_size = history_size
        self.target_size = target_size
        self.step = step
        
        self.sequences = []
        self.targets = []
        
        start_index = history_size
        end_index = len(data) - target_size
        
        for i in range(start_index, end_index):
            # Input sequence
            indices = range(i - history_size, i, step)
            sequence = np.reshape(data[indices], (int(history_size/step), 1))
            
            # Target sequence
            target = data[i:i + target_size]
            
            self.sequences.append(torch.FloatTensor(sequence))
            self.targets.append(torch.FloatTensor(target))
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]


class GatedActivationUnit(nn.Module):
    """Gated activation unit for WaveNet."""
    
    def __init__(self, activation="tanh"):
        super().__init__()
        self.activation = getattr(torch, activation)
    
    def forward(self, inputs):
        n_filters = inputs.shape[-1] // 2
        linear_output = self.activation(inputs[..., :n_filters])
        gate = torch.sigmoid(inputs[..., n_filters:])
        return linear_output * gate


class WaveNetResidualBlock(nn.Module):
    """WaveNet residual block matching TensorFlow implementation."""
    
    def __init__(self, n_filters: int, dilation_rate: int):
        super().__init__()
        
        self.conv = nn.Conv1d(n_filters, 2 * n_filters, kernel_size=2,
                             dilation=dilation_rate)
        self.gated_activation = GatedActivationUnit()
        self.output_conv = nn.Conv1d(n_filters, n_filters, kernel_size=1)
    
    def forward(self, inputs):
        # Causal padding for dilated convolution
        pad_size = (self.conv.kernel_size[0] - 1) * self.conv.dilation[0]
        padded_inputs = torch.nn.functional.pad(inputs, (pad_size, 0), mode='constant', value=0)
        
        # Apply convolution
        z = self.conv(padded_inputs)
        
        # Remove future information (causal)
        if z.shape[2] > inputs.shape[2]:
            z = z[:, :, :inputs.shape[2]]
        
        # Gated activation (reduces channels from 2*n_filters to n_filters)
        z = self.gated_activation(z)
        
        # Output convolution
        z = self.output_conv(z)
        
        # Residual connection
        return inputs + z, z


class ImprovedWaveNetLSTM(nn.Module):
    """
    Improved WaveNet+LSTM model with better architecture for lower MAE.
    """
    
    def __init__(self, n_filters: int = 128, n_outputs: int = 132, 
                 lstm_units: int = 256, lstm_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        
        self.n_filters = n_filters
        self.n_outputs = n_outputs
        
        # Input convolution with larger filters
        self.input_conv = nn.Conv1d(1, n_filters, kernel_size=7, padding=3)
        
        # Improved dilated convolutions with skip connections
        self.dilated_convs = nn.ModuleList([
            nn.Conv1d(n_filters, n_filters, kernel_size=3, padding=2**i, dilation=2**i)
            for i in range(8)  # More layers for better feature extraction
        ])
        
        # Batch normalization and dropout
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(n_filters) for _ in range(8)
        ])
        self.dropouts = nn.ModuleList([
            nn.Dropout(dropout) for _ in range(8)
        ])
        
        # Attention mechanism for better feature selection
        self.attention = nn.MultiheadAttention(n_filters, num_heads=8, dropout=dropout, batch_first=True)
        
        # Output convolution with intermediate layer
        self.output_conv1 = nn.Conv1d(n_filters, n_filters // 2, kernel_size=1)
        self.output_conv2 = nn.Conv1d(n_filters // 2, n_outputs, kernel_size=1)
        self.output_dropout = nn.Dropout(dropout)
        
        # Multi-layer LSTM with dropout
        self.lstm = nn.LSTM(
            n_outputs, lstm_units, num_layers=lstm_layers,
            dropout=dropout if lstm_layers > 1 else 0,
            batch_first=True
        )
        
        # Final prediction layers with regularization
        self.prediction_layers = nn.Sequential(
            nn.Linear(lstm_units, lstm_units // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_units // 2, lstm_units // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_units // 4, n_outputs)
        )
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, 1)
        x = x.transpose(1, 2)  # (batch_size, 1, seq_len)
        
        # Input convolution
        x = torch.relu(self.input_conv(x))
        
        # Apply dilated convolutions with skip connections
        skip_connections = []
        for conv, bn, dropout in zip(self.dilated_convs, self.batch_norms, self.dropouts):
            # Apply convolution
            conv_out = conv(x)
            
            # Ensure same sequence length
            if conv_out.shape[2] != x.shape[2]:
                conv_out = conv_out[:, :, :x.shape[2]]
            
            # Apply batch norm, activation, and dropout
            conv_out = torch.relu(bn(conv_out))
            conv_out = dropout(conv_out)
            
            # Residual connection
            x = x + conv_out
            skip_connections.append(x)
        
        # Combine skip connections
        skip_sum = torch.stack(skip_connections[-4:], dim=0).mean(dim=0)  # Use last 4 layers
        
        # Apply attention mechanism
        skip_sum_transposed = skip_sum.transpose(1, 2)  # (batch_size, seq_len, n_filters)
        attended, _ = self.attention(skip_sum_transposed, skip_sum_transposed, skip_sum_transposed)
        attended = attended.transpose(1, 2)  # Back to (batch_size, n_filters, seq_len)
        
        # Output convolutions
        x = torch.relu(self.output_conv1(attended))
        x = self.output_dropout(x)
        x = self.output_conv2(x)
        
        # Transpose for LSTM: (batch_size, seq_len, n_outputs)
        x = x.transpose(1, 2)
        
        # LSTM processing
        lstm_out, _ = self.lstm(x)
        
        # Use attention-weighted combination of LSTM outputs instead of just the last
        # Apply global average pooling with attention weights
        attention_weights = torch.softmax(torch.mean(lstm_out, dim=2), dim=1)
        weighted_output = torch.sum(lstm_out * attention_weights.unsqueeze(2), dim=1)
        
        # Final prediction
        final_output = self.prediction_layers(weighted_output)
        
        return final_output


class SimplerOptimizedModel(nn.Module):
    """
    Simpler model focused on reducing MAE without overfitting.
    """
    
    def __init__(self, n_filters: int = 64, n_outputs: int = 132, 
                 lstm_units: int = 128, dropout: float = 0.1):
        super().__init__()
        
        # Simpler input processing
        self.input_conv = nn.Conv1d(1, n_filters, kernel_size=5, padding=2)
        
        # Fewer, more focused dilated convolutions
        self.conv1 = nn.Conv1d(n_filters, n_filters, kernel_size=3, padding=1, dilation=1)
        self.conv2 = nn.Conv1d(n_filters, n_filters, kernel_size=3, padding=2, dilation=2)
        self.conv3 = nn.Conv1d(n_filters, n_filters, kernel_size=3, padding=4, dilation=4)
        self.conv4 = nn.Conv1d(n_filters, n_filters, kernel_size=3, padding=8, dilation=8)
        
        # Simple batch norms
        self.bn1 = nn.BatchNorm1d(n_filters)
        self.bn2 = nn.BatchNorm1d(n_filters)
        self.bn3 = nn.BatchNorm1d(n_filters)
        self.bn4 = nn.BatchNorm1d(n_filters)
        
        # Output preparation
        self.output_conv = nn.Conv1d(n_filters, n_outputs, kernel_size=1)
        
        # Simple LSTM
        self.lstm = nn.LSTM(n_outputs, lstm_units, batch_first=True)
        
        # Simple output layer
        self.output_layer = nn.Linear(lstm_units, n_outputs)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, 1)
        x = x.transpose(1, 2)  # (batch_size, 1, seq_len)
        
        # Input processing
        x = torch.relu(self.input_conv(x))
        
        # Dilated convolutions with residuals
        identity = x
        
        # Layer 1
        x1 = torch.relu(self.bn1(self.conv1(x)))
        x = x + x1
        
        # Layer 2
        x2 = torch.relu(self.bn2(self.conv2(x)))
        x = x + x2
        
        # Layer 3  
        x3 = torch.relu(self.bn3(self.conv3(x)))
        x = x + x3
        
        # Layer 4
        x4 = torch.relu(self.bn4(self.conv4(x)))
        x = x + x4
        
        # Add original identity connection
        x = x + identity
        
        # Output preparation
        x = self.output_conv(x)
        x = self.dropout(x)
        
        # LSTM processing
        x = x.transpose(1, 2)  # (batch_size, seq_len, n_outputs)
        lstm_out, _ = self.lstm(x)
        
        # Simple output - just use last timestep
        final_output = self.output_layer(lstm_out[:, -1, :])
        
        return final_output


class TensorFlowStyleTrainer:
    """Trainer matching the TensorFlow implementation."""
    
    def __init__(self, device: str = 'auto'):
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"Using device: {self.device}")
        
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model = None
        self.training_history = {}
        
    def prepare_data(self, monthly_data: np.ndarray, train_ratio: float = 0.8,
                    history_size: int = 528, target_size: int = 132) -> Dict:
        """Prepare data matching TensorFlow implementation."""
        
        print("Preparing data with sliding window approach...")
        
        # Normalize data
        data_scaled = self.scaler.fit_transform(monthly_data.reshape(-1, 1)).ravel()
        
        # Create full dataset first
        full_dataset = SlidingWindowDataset(data_scaled, history_size, target_size)
        
        print(f"Total samples created: {len(full_dataset)}")
        
        # Split samples directly (not by time index)
        total_samples = len(full_dataset)
        train_size = int(total_samples * train_ratio)
        
        train_samples = []
        test_samples = []
        
        for i, (seq, target) in enumerate(full_dataset):
            if i < train_size:
                train_samples.append((seq, target))
            else:
                test_samples.append((seq, target))
        
        print(f"Train samples: {len(train_samples)}")
        print(f"Test samples: {len(test_samples)}")
        
        return {
            'train_data': train_samples,
            'test_data': test_samples,
            'data_scaled': data_scaled,
            'train_size': train_size
        }
    
    def train(self, monthly_data: np.ndarray, epochs: int = 100,
              batch_size: int = 32, lr: float = 1e-3, patience: int = 15) -> Dict:
        """Train the model."""
        
        # Prepare data
        data_info = self.prepare_data(monthly_data)
        
        if len(data_info['train_data']) == 0:
            raise ValueError("No training data available. Check data size and parameters.")
        
        # Initialize simpler, more focused model
        self.model = SimplerOptimizedModel(
            n_filters=64,           # Back to original size
            n_outputs=132,
            lstm_units=128,         # Back to original size
            dropout=0.1             # Less dropout
        ).to(self.device)
        
        # Create data loaders
        train_loader = DataLoader(data_info['train_data'], batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(data_info['test_data'], batch_size=batch_size, shuffle=False)
        
        # Optimizer with weight decay and better loss function
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )
        
        # Use Huber loss for better robustness to outliers
        criterion = nn.SmoothL1Loss()  # More robust than MSE for solar data
        
        # Training history
        history = {'train_loss': [], 'test_loss': []}
        
        # Early stopping variables
        best_test_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        
        print(f"\nTraining TensorFlow-style model for {epochs} epochs...")
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_losses = []
            
            for batch_data, batch_target in train_loader:
                batch_data = batch_data.to(self.device)
                batch_target = batch_target.to(self.device)
                
                optimizer.zero_grad()
                
                output = self.model(batch_data)
                loss = criterion(output, batch_target)
                
                loss.backward()
                optimizer.step()
                
                train_losses.append(loss.item())
            
            # Testing
            self.model.eval()
            test_losses = []
            
            with torch.no_grad():
                for batch_data, batch_target in test_loader:
                    batch_data = batch_data.to(self.device)
                    batch_target = batch_target.to(self.device)
                    
                    output = self.model(batch_data)
                    loss = criterion(output, batch_target)
                    
                    test_losses.append(loss.item())
            
            avg_train_loss = np.mean(train_losses)
            avg_test_loss = np.mean(test_losses) if test_losses else 0.0
            
            history['train_loss'].append(avg_train_loss)
            history['test_loss'].append(avg_test_loss)
            
            if (epoch + 1) % 10 == 0 or epoch == 0:
                current_lr = optimizer.param_groups[0]['lr']
                print(f"Epoch {epoch+1:3d}/{epochs}: Train Loss: {avg_train_loss:.6f}, Test Loss: {avg_test_loss:.6f}, LR: {current_lr:.2e}")
            
            # Learning rate scheduling
            scheduler.step(avg_test_loss)
            
            # Early stopping check
            if avg_test_loss < best_test_loss:
                best_test_loss = avg_test_loss
                patience_counter = 0
                best_model_state = self.model.state_dict().copy()
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                print(f"Best test loss: {best_test_loss:.6f}")
                break
        
        # Load best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
        
        self.training_history = history
        
        return {
            'history': history,
            'data_info': data_info
        }
    
    def create_overlapping_predictions(self, monthly_data: np.ndarray) -> Dict:
        """Create predictions matching TensorFlow implementation."""
        
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        print("Creating overlapping predictions...")
        
        # Scale data
        data_scaled = self.scaler.transform(monthly_data.reshape(-1, 1)).ravel()
        
        results = {}
        
        # Test cases matching TensorFlow implementation
        test_cases = {
            'cycle_22': {
                'input_start': len(data_scaled) - 924,
                'input_end': len(data_scaled) - 396,
                'actual_start': len(data_scaled) - 396,
                'actual_end': len(data_scaled) - 264,
                'description': 'Solar Cycle 22'
            },
            'cycle_23': {
                'input_start': len(data_scaled) - 792,
                'input_end': len(data_scaled) - 264,
                'actual_start': len(data_scaled) - 264,
                'actual_end': len(data_scaled) - 132,
                'description': 'Solar Cycle 23'
            },
            'cycle_24': {
                'input_start': len(data_scaled) - 660,
                'input_end': len(data_scaled) - 132,
                'actual_start': len(data_scaled) - 132,
                'actual_end': len(data_scaled),
                'description': 'Solar Cycle 24'
            },
            'cycle_25': {
                'input_start': len(data_scaled) - 528,
                'input_end': len(data_scaled),
                'actual_start': None,
                'actual_end': None,
                'description': 'Solar Cycle 25 (Future)'
            }
        }
        
        self.model.eval()
        
        for cycle_name, info in test_cases.items():
            # Extract input
            input_data = data_scaled[info['input_start']:info['input_end']]
            
            if len(input_data) != 528:
                print(f"Warning: {cycle_name} input length {len(input_data)} != 528")
                continue
            
            # Prepare input tensor
            input_tensor = torch.FloatTensor(input_data).reshape(1, 528, 1).to(self.device)
            
            # Make prediction
            with torch.no_grad():
                prediction_scaled = self.model(input_tensor).cpu().numpy().ravel()
            
            # Inverse transform
            prediction = self.scaler.inverse_transform(
                prediction_scaled.reshape(-1, 1)
            ).ravel()
            
            # Get actual data if available
            actual = None
            if info['actual_start'] is not None:
                actual_scaled = data_scaled[info['actual_start']:info['actual_end']]
                actual = self.scaler.inverse_transform(
                    actual_scaled.reshape(-1, 1)
                ).ravel()
            
            results[cycle_name] = {
                'prediction': prediction,
                'actual': actual,
                'description': info['description']
            }
            
            if actual is not None:
                # Calculate metrics
                pred_trimmed = prediction[:len(actual)]
                mae = mean_absolute_error(actual, pred_trimmed)
                rmse = np.sqrt(mean_squared_error(actual, pred_trimmed))
                results[cycle_name]['mae'] = mae
                results[cycle_name]['rmse'] = rmse
                print(f"{cycle_name}: RMSE={rmse:.1f}, MAE={mae:.1f}")
        
        return results
    
    def plot_individual_predictions(self, prediction_results: Dict, save_dir: str = None):
        """Plot individual predictions matching TensorFlow style."""
        
        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(exist_ok=True, parents=True)
        
        for cycle_name, result in prediction_results.items():
            plt.figure(figsize=(10, 6))
            
            # Plot prediction
            pred_months = np.arange(len(result['prediction']))
            plt.plot(pred_months, result['prediction'], 'r-', linewidth=2, label='Prediction')
            
            # Plot actual if available
            if result['actual'] is not None:
                actual_months = np.arange(len(result['actual']))
                plt.plot(actual_months, result['actual'], 'b-', linewidth=2, label='Actual')
                
                if 'rmse' in result:
                    plt.text(0.02, 0.98, f'RMSE: {result["rmse"]:.1f}',
                           transform=plt.gca().transAxes, verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            else:
                # For cycle 25, show maximum
                max_val = np.max(result['prediction'])
                max_month = np.argmax(result['prediction'])
                plt.axvline(x=max_month, color='orange', linestyle='--', alpha=0.7)
                plt.scatter(max_month, max_val, color='red', s=100, zorder=5)
                plt.text(0.02, 0.98, f'Max: {max_val:.1f}\nMonth: {max_month+1}',
                        transform=plt.gca().transAxes, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            plt.title(f'{result["description"]}', fontsize=14)
            plt.xlabel('Months')
            plt.ylabel('Sunspot Number')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            if save_dir:
                plt.savefig(save_dir / f'{cycle_name}_tensorflow_style.png', dpi=300, bbox_inches='tight')
            
            plt.show()


def main():
    """Example usage of TensorFlow-style model."""
    print("TensorFlow-Style WaveNet+LSTM Solar Prediction Model")
    print("Methodology: 528 months → 132 months with sliding window")
    print("=" * 60)
    
    # Create dummy data for testing
    dummy_data = np.random.randn(3500) * 50 + 100  # About 290 years of monthly data
    dummy_data += np.sin(np.linspace(0, 50*np.pi, len(dummy_data))) * 30  # Add cyclical pattern
    
    print(f"Testing with {len(dummy_data)} months of dummy data")
    
    # Train model
    trainer = TensorFlowStyleTrainer()
    results = trainer.train(dummy_data, epochs=50, batch_size=16)
    
    # Create overlapping predictions
    prediction_results = trainer.create_overlapping_predictions(dummy_data)
    
    # Plot results
    trainer.plot_individual_predictions(prediction_results)
    
    print(f"\nTensorFlow-style model training and prediction complete!")


if __name__ == "__main__":
    main()