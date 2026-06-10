"""
main.py - House Price Prediction Pipeline
Run this script to: generate data → explore → train models → predict future prices
"""

import subprocess
import sys
import os


def print_header(text):
    """Print a formatted header."""
    print("\n" + "="*70)
    print(f"  {text}")
    print("="*70)


def run_script(script_path, description):
    """Run a Python script and handle errors."""
    print_header(description)
    print(f"Running: {script_path}\n")
    
    try:
        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            text=True,
            cwd=os.path.dirname(os.path.abspath(__file__))
        )
        print(result.stdout)
        
        if result.stderr:
            # Only print warnings, not errors (ignore matplotlib font warnings etc)
            for line in result.stderr.split('\n'):
                if 'Error' in line or 'Traceback' in line:
                    print(f"⚠️  {line}")
            
        if result.returncode != 0:
            print(f"❌ Script failed with exit code {result.returncode}")
            print(f"Error details: {result.stderr}")
            return False
        
        return True
    except Exception as e:
        print(f"❌ Failed to run script: {e}")
        return False


def main():
    """Run the complete house price prediction pipeline."""
    print("\n" + "★"*70)
    print("  🏠  HOUSE PRICE PREDICTION PIPELINE  🏠")
    print("★"*70)
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Step 1: Generate dataset
    success = run_script(
        os.path.join(base_dir, 'generate_data.py'),
        "STEP 1: Generating House Price Dataset"
    )
    if not success:
        print("❌ Pipeline stopped at Step 1")
        return
    
    # Step 2: Explore data (visualizations only, non-critical)
    run_script(
        os.path.join(base_dir, 'explore_data.py'),
        "STEP 2: Exploring Data & Generating Visualizations"
    )
    
    # Step 3: Train models
    success = run_script(
        os.path.join(base_dir, 'train_model.py'),
        "STEP 3: Training Machine Learning Models"
    )
    if not success:
        print("❌ Pipeline stopped at Step 3")
        return
    
    # Step 4: Predict future prices
    run_script(
        os.path.join(base_dir, 'predict_future.py'),
        "STEP 4: Predicting Future House Prices"
    )
    
    # Summary
    print("★"*70)
    print("  ✅  PIPELINE COMPLETE!  ✅")
    print("★"*70)
    print(f"\n📁 Output files in: {os.path.join(base_dir, 'output')}/")
    print("   ├── price_distribution.png")
    print("   ├── correlation_heatmap.png")
    print("   ├── feature_relationships.png")
    print("   ├── price_over_time.png")
    print("   ├── multivariate_insights.png")
    print("   ├── model_performance.png")
    print("   ├── feature_importance.png")
    print("   ├── future_predictions.png")
    print("   ├── seasonality.png")
    print("   ├── model_comparison.csv")
    print("   └── future_predictions.csv")
    print(f"\n📁 Models saved in: {os.path.join(base_dir, 'models')}/")
    print(f"\n📁 Data saved in:   {os.path.join(base_dir, 'data')}/")
    print(f"   └── house_data.csv")
    print("\n💡 Tip: You can also run each step individually:")
    print("   python generate_data.py")
    print("   python explore_data.py")
    print("   python train_model.py")
    print("   python predict_future.py")


if __name__ == "__main__":
    main()
