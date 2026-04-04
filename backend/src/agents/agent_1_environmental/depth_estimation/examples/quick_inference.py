"""
Quick Inference Example
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference import quick_analyze
import ee

def main():
    print("="*70)
    print("  QUICK INFERENCE - FLOOD ANALYSIS")
    print("="*70)
    print("\nAnalyzing latest Sentinel-1 for Sylhet\n")
    
    model_path = '../../../models/flood_depth_model.h5'
    
    if not os.path.exists(model_path):
        print(f"✗ Model not found: {model_path}")
        print("Run quick_train.py first!")
        return
    
    # Initialize EE
    try:
        ee.Initialize()
    except:
        ee.Authenticate()
        ee.Initialize()
    
    # Analyze
    results = quick_analyze(region='sylhet', model_path=model_path)
    
    print("\n✓ Analysis complete!")
    print("Results saved to flood_analysis_*.json")

if __name__ == "__main__":
    main()