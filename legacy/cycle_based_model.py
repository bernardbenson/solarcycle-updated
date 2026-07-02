"""
Cycle-based WaveNet+LSTM implementation matching the paper methodology.
Uses 4 solar cycles as input to predict 1 solar cycle as output.
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


class SolarCycleDataset(Dataset):
    """Dataset for cycle-based solar prediction (4 cycles in → 1 cycle out)."""
    
    def __init__(self, cycle_data: List[np.ndarray]):
        """
        Args:
            cycle_data: List of solar cycles, each as numpy array of monthly data
        """
        self.sequences = []
        self.targets = []
        
        # Create training pairs: 4 consecutive cycles → next cycle
        for i in range(len(cycle_data) - 4):
            # Input: 4 consecutive cycles
            input_cycles = cycle_data[i:i+4]
            
            # Concatenate 4 cycles and pad to fixed length
            input_sequence = self._prepare_input_sequence(input_cycles)
            
            # Target: next cycle (5th cycle)
            target_cycle = cycle_data[i+4]
            target_padded = self._pad_cycle(target_cycle, 132)  # 11 years max
            
            self.sequences.append(torch.FloatTensor(input_sequence).unsqueeze(-1))
            self.targets.append(torch.FloatTensor(target_padded))
        
        if len(self.sequences) == 0:
            raise ValueError("Not enough cycles to create training pairs. Need at least 5 cycles.")
    
    def _prepare_input_sequence(self, cycles: List[np.ndarray], max_length: int = 132) -> np.ndarray:
        """Prepare input sequence from 4 cycles."""
        # Pad each cycle to max_length and concatenate
        padded_cycles = [self._pad_cycle(cycle, max_length) for cycle in cycles]
        return np.concatenate(padded_cycles)  # Shape: (4 * max_length,)
    
    def _pad_cycle(self, cycle: np.ndarray, target_length: int) -> np.ndarray:
        """Pad or truncate cycle to target length."""
        if len(cycle) >= target_length:
            return cycle[:target_length]
        else:
            # Pad with mean value
            padding = np.full(target_length - len(cycle), np.mean(cycle))
            return np.concatenate([cycle, padding])
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]


class CycleBased_WaveNetLSTM(nn.Module):
    """
    WaveNet+LSTM model matching TensorFlow implementation.
    Input: 528 months (44 years) sliding window
    Output: 132 months (11 years) prediction
    """
    
    def __init__(self, input_length: int = 528, output_length: int = 132,
                 wavenet_channels: int = 64, lstm_hidden_size: int = 128,
                 lstm_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        
        self.input_length = input_length   # 528 months input
        self.output_length = output_length  # 132 months output
        
        # WaveNet feature extractor
        self.wavenet = self._build_wavenet(wavenet_channels, dropout)
        
        # LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=wavenet_channels,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_layers,
            dropout=dropout if lstm_layers > 1 else 0,
            batch_first=True,
            bidirectional=False
        )
        
        # Output layers
        self.output_layers = nn.Sequential(
            nn.Linear(lstm_hidden_size, lstm_hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden_size, cycle_length)  # Predict one cycle
        )
        
    def _build_wavenet(self, channels: int, dropout: float):
        """Build simplified WaveNet feature extractor."""
        return nn.Sequential(
            # Input projection
            nn.Conv1d(1, channels, kernel_size=1),
            
            # Dilated convolutions
            self._dilated_conv_block(channels, channels, dilation=1, dropout=dropout),
            self._dilated_conv_block(channels, channels, dilation=2, dropout=dropout),
            self._dilated_conv_block(channels, channels, dilation=4, dropout=dropout),
            self._dilated_conv_block(channels, channels, dilation=8, dropout=dropout),
            self._dilated_conv_block(channels, channels, dilation=16, dropout=dropout),
            
            # Output projection
            nn.Conv1d(channels, channels, kernel_size=1),
            nn.ReLU()
        )
    
    def _dilated_conv_block(self, in_channels: int, out_channels: int, 
                           dilation: int, dropout: float):
        """Create a dilated convolution block with residual connection."""
        return nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, 
                     padding=dilation, dilation=dilation),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        # x shape: (batch_size, sequence_length, 1)
        x = x.transpose(1, 2)  # (batch_size, 1, sequence_length)
        
        # WaveNet feature extraction
        wavenet_features = self.wavenet(x)  # (batch_size, channels, sequence_length)
        wavenet_features = wavenet_features.transpose(1, 2)  # (batch_size, sequence_length, channels)
        
        # LSTM processing
        lstm_out, _ = self.lstm(wavenet_features)
        
        # Use last hidden state
        final_hidden = lstm_out[:, -1, :]  # (batch_size, lstm_hidden_size)
        
        # Generate cycle prediction
        cycle_prediction = self.output_layers(final_hidden)  # (batch_size, cycle_length)
        
        return cycle_prediction


class CycleBasedTrainer:
    """Trainer for cycle-based solar prediction."""
    
    def __init__(self, device: str = 'auto'):
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"Using device: {self.device}")
        
        self.scaler = StandardScaler()
        self.model = None
        self.training_history = {}
        
    def segment_into_cycles(self, monthly_data: np.ndarray, 
                           cycle_starts: List[int]) -> List[np.ndarray]:
        """
        Segment monthly data into individual solar cycles.
        
        Args:
            monthly_data: Full time series of monthly sunspot numbers
            cycle_starts: List of month indices where each cycle starts
        
        Returns:
            List of individual solar cycles
        """
        cycles = []
        
        for i in range(len(cycle_starts) - 1):
            start_idx = cycle_starts[i]
            end_idx = cycle_starts[i + 1]
            cycle_data = monthly_data[start_idx:end_idx]
            
            if len(cycle_data) > 12:  # Only include cycles with more than 1 year of data
                cycles.append(cycle_data)
        
        print(f"Segmented {len(cycles)} solar cycles")
        print(f"Cycle lengths: {[len(c) for c in cycles]} months")
        
        return cycles
    
    def prepare_cycle_data(self, monthly_data: np.ndarray) -> Tuple[List[np.ndarray], Dict]:
        """
        Prepare data by segmenting into solar cycles.
        """
        # Approximate solar cycle starts (months from 1749)
        # These are rough estimates based on historical solar minima
        cycle_starts = [
            0,      # Cycle 1 start (1755)
            132,    # Cycle 2 start (1766)
            264,    # Cycle 3 start (1775)
            384,    # Cycle 4 start (1784)
            516,    # Cycle 5 start (1798)
            648,    # Cycle 6 start (1810)
            768,    # Cycle 7 start (1823)
            900,    # Cycle 8 start (1833)
            1020,   # Cycle 9 start (1843)
            1152,   # Cycle 10 start (1855)
            1284,   # Cycle 11 start (1867)
            1404,   # Cycle 12 start (1878)
            1524,   # Cycle 13 start (1889)
            1656,   # Cycle 14 start (1901)
            1776,   # Cycle 15 start (1913)
            1908,   # Cycle 16 start (1923)
            2040,   # Cycle 17 start (1933)
            2172,   # Cycle 18 start (1944)
            2304,   # Cycle 19 start (1954)
            2436,   # Cycle 20 start (1964)
            2568,   # Cycle 21 start (1976)
            2700,   # Cycle 22 start (1986)
            2832,   # Cycle 23 start (1996)
            2976,   # Cycle 24 start (2008)
            3120,   # Cycle 25 start (2019)
            len(monthly_data)  # End marker
        ]
        
        # Filter cycle starts that are within our data range
        valid_starts = [s for s in cycle_starts if s < len(monthly_data)]
        
        # Segment data into cycles
        cycles = self.segment_into_cycles(monthly_data, valid_starts)
        
        # Scale individual cycles
        scaled_cycles = []
        for cycle in cycles:
            cycle_scaled = self.scaler.fit_transform(cycle.reshape(-1, 1)).ravel()
            scaled_cycles.append(cycle_scaled)
        
        return scaled_cycles, {'cycle_starts': valid_starts, 'n_cycles': len(cycles)}
    
    def train(self, monthly_data: np.ndarray, epochs: int = 100, 
              batch_size: int = 8, lr: float = 1e-3, patience: int = 15) -> Dict:
        """Train the cycle-based model."""
        
        print("Preparing cycle-based training data...")
        
        # Segment data into cycles
        cycles, cycle_info = self.prepare_cycle_data(monthly_data)
        
        if len(cycles) < 5:
            raise ValueError(f"Need at least 5 cycles for training. Found {len(cycles)}")
        
        # Create dataset (4 cycles → 1 cycle)
        dataset = SolarCycleDataset(cycles)
        
        # Split into train/test (temporal split)
        train_size = max(1, len(dataset) - 2)  # Leave last 2 samples for testing
        test_size = len(dataset) - train_size
        
        train_dataset = torch.utils.data.Subset(dataset, range(train_size))
        test_dataset = torch.utils.data.Subset(dataset, range(train_size, len(dataset)))
        
        print(f"Training samples: {len(train_dataset)}")
        print(f"Test samples: {len(test_dataset)}")
        
        # Initialize model
        self.model = CycleBased_WaveNetLSTM(
            cycle_length=132,
            n_cycles_input=4,
            wavenet_channels=64,
            lstm_hidden_size=128,
            lstm_layers=2,
            dropout=0.1
        ).to(self.device)
        
        # Data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        # Optimizer and criterion
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
        
        print(f"\nTraining cycle-based WaveNet+LSTM for {epochs} epochs...")
        
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
            avg_test_loss = np.mean(test_losses) if test_losses else avg_train_loss
            current_lr = optimizer.param_groups[0]['lr']
            
            history['train_loss'].append(avg_train_loss)
            history['test_loss'].append(avg_test_loss)
            history['lr'].append(current_lr)
            
            # Print progress
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"Epoch {epoch+1:3d}/{epochs}: "
                      f"Train Loss: {avg_train_loss:.6f}, "
                      f"Test Loss: {avg_test_loss:.6f}, "
                      f"LR: {current_lr:.2e}")
            
            # Learning rate scheduling and early stopping
            scheduler.step(avg_test_loss)
            
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
            'cycle_info': cycle_info,
            'dataset_info': {
                'train_samples': len(train_dataset),
                'test_samples': len(test_dataset),
                'total_cycles': len(cycles)
            }
        }
    
    def create_overlapping_predictions(self, monthly_data: np.ndarray) -> Dict:
        """
        Create overlapping prediction plots like in the TensorFlow implementation.
        Tests predictions on cycles 22, 23, 24 and predicts cycle 25.
        """
        
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        print("Creating overlapping predictions for historical cycles...")
        
        # Use 528 months (44 years) as input window like in TensorFlow version
        input_window = 528
        prediction_horizon = 132  # 11 years
        
        results = {}
        
        # Test different input positions to predict known cycles
        test_cases = {
            'cycle_22': {
                'input_start': len(monthly_data) - 924,  # -924:-396
                'input_end': len(monthly_data) - 396,
                'actual_start': len(monthly_data) - 396,  # -396:-264
                'actual_end': len(monthly_data) - 264,
                'description': 'Solar Cycle 22'
            },
            'cycle_23': {
                'input_start': len(monthly_data) - 792,  # -792:-264
                'input_end': len(monthly_data) - 264,
                'actual_start': len(monthly_data) - 264,  # -264:-132
                'actual_end': len(monthly_data) - 132,
                'description': 'Solar Cycle 23'
            },
            'cycle_24': {
                'input_start': len(monthly_data) - 660,  # -660:-132
                'input_end': len(monthly_data) - 132,
                'actual_start': len(monthly_data) - 132,  # -132:end
                'actual_end': len(monthly_data),
                'description': 'Solar Cycle 24'
            },
            'cycle_25': {
                'input_start': len(monthly_data) - 528,  # -528:end (last 528 months)
                'input_end': len(monthly_data),
                'actual_start': None,  # No actual data for future
                'actual_end': None,
                'description': 'Solar Cycle 25 (Future Prediction)'
            }
        }
        
        for cycle_name, info in test_cases.items():
            # Extract input data
            input_data = monthly_data[info['input_start']:info['input_end']]
            
            if len(input_data) != input_window:
                print(f"Warning: {cycle_name} input length {len(input_data)} != {input_window}")
                continue
            
            # Scale input data
            input_scaled = self.scaler.transform(input_data.reshape(-1, 1)).ravel()
            
            # Prepare for model prediction
            input_tensor = torch.FloatTensor(input_scaled).unsqueeze(0).unsqueeze(-1).to(self.device)
            
            # Make prediction
            self.model.eval()
            with torch.no_grad():
                prediction_scaled = self.model(input_tensor).cpu().numpy().ravel()
            
            # Inverse transform prediction
            prediction = self.scaler.inverse_transform(
                prediction_scaled.reshape(-1, 1)
            ).ravel()
            
            # Get actual data if available
            actual = None
            if info['actual_start'] is not None and info['actual_end'] is not None:
                actual = monthly_data[info['actual_start']:info['actual_end']]
            
            results[cycle_name] = {
                'input_data': input_data,
                'prediction': prediction,
                'actual': actual,
                'description': info['description'],
                'input_months': len(input_data),
                'prediction_months': len(prediction)
            }
            
            if actual is not None:
                # Calculate metrics
                # Trim prediction to actual length for comparison
                pred_trimmed = prediction[:len(actual)]
                mae = mean_absolute_error(actual, pred_trimmed)
                rmse = np.sqrt(mean_squared_error(actual, pred_trimmed))
                
                results[cycle_name]['mae'] = mae
                results[cycle_name]['rmse'] = rmse
                
                print(f"{cycle_name}: RMSE={rmse:.1f}, MAE={mae:.1f}")
        
        return results
    
    def plot_overlapping_predictions(self, prediction_results: Dict, save_path: str = None):
        """
        Create overlapping prediction plots like in the TensorFlow implementation.
        Shows prediction vs actual for each historical cycle.
        """
        
        # Create subplot for each cycle
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        cycle_names = ['cycle_22', 'cycle_23', 'cycle_24', 'cycle_25']
        colors_pred = ['red', 'red', 'red', 'blue']
        colors_actual = ['blue', 'blue', 'blue', None]
        
        for i, cycle_name in enumerate(cycle_names):
            if cycle_name not in prediction_results:
                axes[i].set_visible(False)
                continue
            
            result = prediction_results[cycle_name]
            
            # Plot prediction
            pred_months = np.arange(len(result['prediction']))
            axes[i].plot(pred_months, result['prediction'], 
                        color=colors_pred[i], linewidth=2, label='Prediction')
            
            # Plot actual if available
            if result['actual'] is not None:
                actual_months = np.arange(len(result['actual']))
                axes[i].plot(actual_months, result['actual'], 
                           color=colors_actual[i], linewidth=2, label='Actual')
                
                # Add metrics to title
                if 'rmse' in result:
                    title = f"{result['description']}\nRMSE: {result['rmse']:.1f}, MAE: {result['mae']:.1f}"
                else:
                    title = result['description']
            else:
                title = result['description']
                # For cycle 25, add max prediction
                max_val = np.max(result['prediction'])
                max_month = np.argmax(result['prediction']) + 1
                title += f"\nMax: {max_val:.1f} at month {max_month}"
            
            axes[i].set_title(title, fontsize=12)
            axes[i].set_xlabel('Months into Cycle')
            axes[i].set_ylabel('Sunspot Number')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
        
        plt.suptitle('Overlapping Predictions vs Actual - Historical Validation', fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_tensorflow_style_individual(self, prediction_results: Dict, save_dir: str = None):
        """
        Create individual plots matching the TensorFlow implementation style.
        One plot per cycle showing prediction vs actual overlap.
        """
        
        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(exist_ok=True, parents=True)
        
        cycle_names = ['cycle_22', 'cycle_23', 'cycle_24', 'cycle_25']
        
        for cycle_name in cycle_names:
            if cycle_name not in prediction_results:
                continue
            
            result = prediction_results[cycle_name]
            
            plt.figure(figsize=(10, 6))
            
            # Plot prediction
            pred_months = np.arange(len(result['prediction']))
            plt.plot(pred_months, result['prediction'], 
                    'r-', linewidth=2, label='Prediction')
            
            # Plot actual if available
            if result['actual'] is not None:
                actual_months = np.arange(len(result['actual']))
                plt.plot(actual_months, result['actual'], 
                        'b-', linewidth=2, label='Actual')
                
                # Add metrics
                if 'rmse' in result:
                    plt.text(0.02, 0.98, f'RMSE: {result["rmse"]:.1f}\nMAE: {result["mae"]:.1f}',
                           transform=plt.gca().transAxes, verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            else:
                # For cycle 25, show max prediction
                max_val = np.max(result['prediction'])
                max_month = np.argmax(result['prediction'])
                plt.axvline(x=max_month, color='orange', linestyle='--', alpha=0.7)
                plt.scatter(max_month, max_val, color='red', s=100, zorder=5)
                plt.text(0.02, 0.98, f'Max: {max_val:.1f}\nMonth: {max_month+1}',
                        transform=plt.gca().transAxes, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            plt.title(f'{result["description"]} - Prediction vs Actual', fontsize=14)
            plt.xlabel('Months into Cycle')
            plt.ylabel('Sunspot Number')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            if save_dir:
                plt.savefig(save_dir / f'{cycle_name}_prediction.png', dpi=300, bbox_inches='tight')
            
            plt.show()

    def predict_solar_cycle_25(self, monthly_data: np.ndarray) -> Dict:
        """Predict Solar Cycle 25 using the last 4 cycles."""
        
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Get the last 4 cycles for prediction
        cycles, _ = self.prepare_cycle_data(monthly_data)
        
        if len(cycles) < 4:
            raise ValueError(f"Need at least 4 cycles for prediction. Found {len(cycles)}")
        
        # Use last 4 cycles as input
        last_4_cycles = cycles[-4:]
        
        # Prepare input sequence
        dataset = SolarCycleDataset(last_4_cycles + [np.zeros(132)])  # Dummy target
        input_seq, _ = dataset[0]  # Get the prepared input
        
        # Make prediction
        self.model.eval()
        with torch.no_grad():
            input_batch = input_seq.unsqueeze(0).to(self.device)
            prediction_scaled = self.model(input_batch).cpu().numpy().ravel()
        
        # Inverse transform
        prediction = self.scaler.inverse_transform(
            prediction_scaled.reshape(-1, 1)
        ).ravel()
        
        # Calculate statistics
        max_idx = np.argmax(prediction)
        max_value = prediction[max_idx]
        
        return {
            'prediction': prediction,
            'max_sunspot_number': max_value,
            'max_sunspot_month': max_idx + 1,
            'prediction_years': len(prediction) / 12,
            'input_cycles': len(last_4_cycles)
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
    
    def plot_cycle_prediction(self, results: Dict, save_path: str = None):
        """Plot Solar Cycle 25 prediction."""
        
        prediction = results['prediction']
        months = np.arange(1, len(prediction) + 1)
        years = months / 12
        
        plt.figure(figsize=(12, 6))
        plt.plot(years, prediction, 'r-', linewidth=2, label='Solar Cycle 25 Prediction')
        plt.axhline(y=results['max_sunspot_number'], color='orange', linestyle='--', 
                   alpha=0.7, label=f'Max: {results["max_sunspot_number"]:.1f}')
        
        plt.xlabel('Years into Solar Cycle 25')
        plt.ylabel('Sunspot Number')
        plt.title('Solar Cycle 25 Forecast - Cycle-Based WaveNet+LSTM\n(4 Cycles → 1 Cycle Prediction)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Add text annotation
        plt.text(results['max_sunspot_month']/12, results['max_sunspot_number'] + 5,
                f'Peak: {results["max_sunspot_number"]:.1f}\nMonth {results["max_sunspot_month"]}',
                ha='center', va='bottom', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()


def main():
    """Example usage of cycle-based model."""
    print("Cycle-Based WaveNet+LSTM Solar Prediction Model")
    print("Methodology: 4 Solar Cycles → 1 Solar Cycle Prediction")
    print("=" * 60)
    
    # Create dummy cycle data for testing
    dummy_cycles = []
    for i in range(8):  # Create 8 dummy cycles
        cycle_length = np.random.randint(100, 140)
        cycle = np.random.randn(cycle_length) * 30 + 80 + np.sin(np.linspace(0, 2*np.pi, cycle_length)) * 50
        dummy_cycles.append(cycle)
    
    dummy_data = np.concatenate(dummy_cycles)
    
    print(f"Testing with {len(dummy_cycles)} dummy cycles ({len(dummy_data)} months total)")
    
    # Train model
    trainer = CycleBasedTrainer()
    results = trainer.train(dummy_data, epochs=30, batch_size=4)
    
    # Make predictions
    predictions = trainer.predict_solar_cycle_25(dummy_data)
    
    print(f"\nSolar Cycle 25 Forecast:")
    print(f"Maximum sunspot number: {predictions['max_sunspot_number']:.1f}")
    print(f"Peak month: {predictions['max_sunspot_month']}")
    print(f"Input cycles used: {predictions['input_cycles']}")


if __name__ == "__main__":
    main()