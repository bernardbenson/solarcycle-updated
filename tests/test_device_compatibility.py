#!/usr/bin/env python3
"""
Test device compatibility for PyTorch on different platforms.
Tests CPU, CUDA, and Apple Silicon MPS support.
"""

import torch
import numpy as np
import sys
from pathlib import Path

def test_basic_pytorch():
    """Test basic PyTorch functionality."""
    print("=== Basic PyTorch Test ===")
    print(f"PyTorch version: {torch.__version__}")
    
    # Basic tensor operations
    x = torch.randn(3, 3)
    y = torch.randn(3, 3)
    z = torch.mm(x, y)
    
    print(f"✅ Basic tensor operations work: {z.shape}")
    return True

def test_cuda():
    """Test CUDA support."""
    print("\n=== CUDA Test ===")
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available")
        return False
    
    print(f"✅ CUDA available")
    print(f"CUDA device count: {torch.cuda.device_count()}")
    
    for i in range(torch.cuda.device_count()):
        print(f"Device {i}: {torch.cuda.get_device_name(i)}")
    
    try:
        # Test CUDA operations
        device = torch.device('cuda:0')
        x = torch.randn(1000, 1000, device=device)
        y = torch.randn(1000, 1000, device=device)
        z = torch.mm(x, y)
        
        print(f"✅ CUDA operations work: {z.shape}")
        return True
        
    except Exception as e:
        print(f"❌ CUDA operations failed: {e}")
        return False

def test_mps():
    """Test Apple Silicon MPS support."""
    print("\n=== Apple Silicon MPS Test ===")
    
    if not hasattr(torch.backends, 'mps'):
        print("❌ MPS backend not available (PyTorch version too old)")
        return False
    
    if not torch.backends.mps.is_available():
        print("❌ MPS not available (not on Apple Silicon or MPS not built)")
        return False
    
    print("✅ MPS available")
    
    try:
        # Test MPS operations
        device = torch.device('mps')
        x = torch.randn(1000, 1000, device=device)
        y = torch.randn(1000, 1000, device=device)
        z = torch.mm(x, y)
        
        print(f"✅ MPS operations work: {z.shape}")
        return True
        
    except Exception as e:
        print(f"❌ MPS operations failed: {e}")
        return False

def test_autocast():
    """Test automatic mixed precision."""
    print("\n=== AMP (Automatic Mixed Precision) Test ===")
    
    if torch.cuda.is_available():
        print("Testing AMP with CUDA...")
        try:
            from torch.cuda.amp import autocast, GradScaler
            
            device = torch.device('cuda')
            model = torch.nn.Linear(100, 10).to(device)
            optimizer = torch.optim.Adam(model.parameters())
            scaler = GradScaler()
            
            x = torch.randn(32, 100, device=device)
            target = torch.randn(32, 10, device=device)
            
            with autocast():
                output = model(x)
                loss = torch.nn.functional.mse_loss(output, target)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            print("✅ AMP with CUDA works")
            return True
            
        except Exception as e:
            print(f"❌ AMP with CUDA failed: {e}")
            return False
    
    else:
        print("❌ AMP requires CUDA (not available)")
        return False

def test_device_selection():
    """Test automatic device selection logic."""
    print("\n=== Device Selection Test ===")
    
    # Test our device selection logic
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"✅ Auto-selected: {device}")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
        print(f"✅ Auto-selected: {device}")
    else:
        device = torch.device('cpu')
        print(f"✅ Auto-selected: {device}")
    
    # Test tensor operations on selected device
    try:
        x = torch.randn(100, 100).to(device)
        y = torch.randn(100, 100).to(device)
        z = torch.mm(x, y)
        
        print(f"✅ Operations work on {device}: {z.shape}")
        return device
        
    except Exception as e:
        print(f"❌ Operations failed on {device}: {e}")
        return torch.device('cpu')

def test_solar_model_compatibility():
    """Test if our solar models can run on the available device."""
    print("\n=== Solar Model Compatibility Test ===")
    
    # Add the project root to the path
    sys.path.append(str(Path(__file__).parent))
    
    try:
        from solar.models.wavenet_attn_seq2seq import WaveNetAttnSeq2Seq
        from solar.models.tcn_only import TCNOnly
        from solar.models.nbeatsx import NBEATSx
        
        # Auto-select device
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
        
        print(f"Testing models on device: {device}")
        
        # Test WaveNet Attention Seq2Seq
        print("Testing WaveNet Attention Seq2Seq...")
        config = {
            'input_dim': 1,
            'd_model': 64,  # Smaller for testing
            'output_size': 132,
            'wavenet': {'stacks': 2, 'layers_per_stack': 2, 'channels': 64, 'dropout': 0.1},
            'encoder_bilstm_hidden': 64,
            'decoder_lstm_hidden': 64,
            'attention': 'scaled_dot',
            'head': 'mse',
            'dropout': 0.1
        }
        
        model = WaveNetAttnSeq2Seq(config).to(device)
        x = torch.randn(2, 100, 1).to(device)  # Small batch for testing
        outputs = model(x, teacher_forcing_ratio=0.0)
        print(f"✅ WaveNet Seq2Seq works: {outputs['predictions'].shape}")
        
        # Test TCN
        print("Testing TCN...")
        tcn_config = {
            'input_dim': 1,
            'd_model': 64,
            'output_size': 132,
            'tcn': {'num_blocks': 2, 'layers_per_block': 2, 'kernel_size': 3, 'dropout': 0.1},
            'head': 'mse'
        }
        
        tcn_model = TCNOnly(tcn_config).to(device)
        x = torch.randn(2, 100, 1).to(device)
        tcn_outputs = tcn_model(x)
        print(f"✅ TCN works: {tcn_outputs['predictions'].shape}")
        
        # Test N-BEATS
        print("Testing N-BEATS...")
        nbeats_config = {
            'input_window': 100,
            'output_size': 132,
            'd_model': 64,
            'nbeats': {'trend_blocks': 1, 'seasonal_blocks': 1, 'generic_blocks': 1, 'basis_size': 4, 'dropout': 0.1},
            'head': 'mse'
        }
        
        nbeats_model = NBEATSx(nbeats_config).to(device)
        x = torch.randn(2, 100, 1).to(device)
        nbeats_outputs = nbeats_model(x)
        print(f"✅ N-BEATS works: {nbeats_outputs['predictions'].shape}")
        
        print("✅ All solar models are compatible!")
        return True
        
    except Exception as e:
        print(f"❌ Solar model test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all compatibility tests."""
    print("🚀 PyTorch Device Compatibility Test")
    print("=" * 50)
    
    results = {}
    
    # Basic tests
    results['basic'] = test_basic_pytorch()
    results['cuda'] = test_cuda()
    results['mps'] = test_mps()
    results['amp'] = test_autocast()
    
    # Device selection
    selected_device = test_device_selection()
    
    # Solar model compatibility
    results['solar_models'] = test_solar_model_compatibility()
    
    # Summary
    print("\n" + "=" * 50)
    print("🎯 COMPATIBILITY SUMMARY")
    print("=" * 50)
    
    print(f"Selected device: {selected_device}")
    print(f"Basic PyTorch: {'✅' if results['basic'] else '❌'}")
    print(f"CUDA support: {'✅' if results['cuda'] else '❌'}")
    print(f"MPS support: {'✅' if results['mps'] else '❌'}")
    print(f"AMP support: {'✅' if results['amp'] else '❌'}")
    print(f"Solar models: {'✅' if results['solar_models'] else '❌'}")
    
    # Recommendations
    print("\n🔧 RECOMMENDATIONS:")
    
    if results['cuda']:
        print("- Use CUDA for best performance")
        print("- AMP is available for faster training")
    elif results['mps']:
        print("- Use MPS for Apple Silicon acceleration")
        print("- AMP not supported, but MPS provides good performance")
    else:
        print("- CPU-only mode available")
        print("- Consider smaller batch sizes and models")
    
    if not results['solar_models']:
        print("- Check solar model dependencies")
        print("- Try reducing model sizes if memory is limited")
    
    return 0 if all(results.values()) else 1

if __name__ == "__main__":
    exit(main())