#!/usr/bin/env python3
"""
Quick start example for multivariate sunspot prediction.
Run this to test the complete pipeline on a subset of data.
"""

import subprocess
import sys
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors."""
    print(f"\n{'='*60}")
    print(f"🚀 {description}")
    print(f"{'='*60}")
    print(f"Running: {command}")
    
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print("✅ Success!")
        if result.stdout:
            print("Output:", result.stdout[:500], "..." if len(result.stdout) > 500 else "")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {e}")
        if e.stderr:
            print("Error details:", e.stderr[:500])
        return False

def main():
    """Run the complete example pipeline."""
    print("🌞 MULTIVARIATE SUNSPOT PREDICTION - QUICK START EXAMPLE")
    print("=" * 70)
    print("This example will run the complete pipeline on recent data for demonstration.")
    print("Expected runtime: 5-10 minutes")
    
    # Check if uv is available
    try:
        subprocess.run("uv --version", shell=True, check=True, capture_output=True)
        print("✅ uv is available")
    except subprocess.CalledProcessError:
        print("❌ uv not found. Please install uv first:")
        print("curl -LsSf https://astral.sh/uv/install.sh | sh")
        sys.exit(1)
    
    # Define pipeline steps
    steps = [
        {
            "command": "uv run python main.py --phase collect --start-year 2015",
            "description": "Collect recent solar data (2015-2025)"
        },
        {
            "command": "uv run python main.py --phase preprocess",
            "description": "Apply feature engineering (create 131 features)"
        },
        {
            "command": "uv run python quick_pytorch_test.py",
            "description": "Test PyTorch models (quick training on subset)"
        },
        {
            "command": "uv run python quick_viz_test.py",
            "description": "Generate quick data visualizations"
        }
    ]
    
    # Optional full pipeline step
    full_pipeline = {
        "command": "uv run python main.py --phase visualize",
        "description": "Create comprehensive visualizations (may take longer)"
    }
    
    print(f"\nWill run {len(steps)} main steps:")
    for i, step in enumerate(steps, 1):
        print(f"{i}. {step['description']}")
    
    # Ask user if they want to continue
    try:
        response = input(f"\nContinue with example? (y/n): ").lower().strip()
        if response not in ['y', 'yes']:
            print("Cancelled by user.")
            return
    except KeyboardInterrupt:
        print("\nCancelled by user.")
        return
    
    # Run pipeline steps
    success_count = 0
    
    for i, step in enumerate(steps, 1):
        print(f"\n📍 Step {i}/{len(steps)}")
        if run_command(step["command"], step["description"]):
            success_count += 1
        else:
            print(f"⚠️  Step {i} failed, but continuing...")
    
    # Ask about optional visualization
    if success_count >= 2:  # If at least data collection and preprocessing worked
        print(f"\n🎨 Optional: Run comprehensive visualizations?")
        print("This will create publication-quality plots but may take longer.")
        
        try:
            response = input("Run visualizations? (y/n): ").lower().strip()
            if response in ['y', 'yes']:
                if run_command(full_pipeline["command"], full_pipeline["description"]):
                    success_count += 1
        except KeyboardInterrupt:
            print("\nSkipping visualizations.")
    
    # Summary
    print(f"\n{'='*70}")
    print(f"🏁 EXAMPLE COMPLETE")
    print(f"{'='*70}")
    print(f"✅ Successful steps: {success_count}")
    
    if success_count >= 2:
        print(f"\n🎉 SUCCESS! The pipeline is working.")
        print(f"\n📁 Check these directories for results:")
        print(f"   • data/raw_multivariate_data.csv - Raw collected data")
        print(f"   • data/engineered_multivariate_data.csv - Processed features")
        print(f"   • data/visualizations/ - Data visualization plots")
        
        if Path("data/training_results").exists():
            print(f"   • data/training_results/ - Model training results")
        
        print(f"\n🚀 Next steps:")
        print(f"   • Run full training: uv run python main.py --phase train")
        print(f"   • Run evaluation: uv run python main.py --phase evaluate")
        print(f"   • Train on full historical data: uv run python main.py --phase collect")
        print(f"   • Create results plots: uv run python visualize_results.py")
        
    else:
        print(f"\n⚠️  Some steps failed. Check the error messages above.")
        print(f"Common solutions:")
        print(f"   • Check internet connection (needed for data collection)")
        print(f"   • Install uv: curl -LsSf https://astral.sh/uv/install.sh | sh")
        print(f"   • Check available disk space")
    
    print(f"\n📖 For full documentation, see README.md")
    print(f"🐛 For issues, create a GitHub issue")

if __name__ == "__main__":
    main()