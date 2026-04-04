"""
Quick Training Example - Fixed with Correct Project ID
"""

import sys
import os
from typing import Dict, Any, Union

# Set up paths
current_file = os.path.abspath(__file__)
examples_dir = os.path.dirname(current_file)
depth_estimation_dir = os.path.dirname(examples_dir)
agent_dir = os.path.dirname(depth_estimation_dir)

# Add depth_estimation to Python path
if depth_estimation_dir not in sys.path:
    sys.path.insert(0, depth_estimation_dir)

print("="*70)
print("  FLOOD DEPTH ESTIMATION - TRAINING")
print("="*70)
print()
print("Training configuration:")
print("  Region: Sylhet, Bangladesh")
print("  Date range: 2022-05-01 to 2022-09-30")
print("  Training samples: 80")
print("  Validation samples: 20")
print("  Epochs: 20")
print("  Estimated time: 12-15 minutes")
print()

# Check dependencies
print("Checking dependencies...")
try:
    import numpy as np
    print("  ✓ numpy")
except ImportError:
    print("  ✗ numpy - Run: pip install numpy")
    sys.exit(1)

try:
    import tensorflow as tf
    print("  ✓ tensorflow")
except ImportError:
    print("  ✗ tensorflow - Run: pip install tensorflow>=2.13.0")
    sys.exit(1)

try:
    import ee
    print("  ✓ earthengine-api")
except ImportError:
    print("  ✗ earthengine-api - Run: pip install earthengine-api")
    sys.exit(1)

try:
    import yaml
    print("  ✓ pyyaml")
except ImportError:
    print("  ✗ pyyaml - Run: pip install pyyaml")
    sys.exit(1)

print()

# Check Earth Engine - WITH CORRECT PROJECT ID
print("Checking Earth Engine authentication...")
try:
    ee.Initialize(project='caramel-pulsar-475810-e7')
    print("  ✓ Earth Engine authenticated (project: caramel-pulsar-475810-e7)")
except Exception as e:
    print(f"  ✗ Earth Engine authentication failed: {e}")
    print()
    print("  Please run: earthengine authenticate --force")
    print("  Then try again")
    print()
    sys.exit(1)

print()
print("="*70)

response = input("Continue with training? (y/n): ")
if response.lower() != 'y':
    print("Training cancelled.")
    sys.exit(0)

print()
print("Starting training pipeline...")
print()

# Import training modules
try:
    from training.train_depth import DepthTrainer
    from training.dataset_generator import DepthDatasetGenerator
    from core.depth_model import LightweightDepthCNN
    from utils.config_loader import load_config
except ImportError as e:
    print(f"Error importing modules: {e}")
    print()
    print("Make sure you're running from agent_1_environmental directory")
    print("Current directory:", os.getcwd())
    sys.exit(1)

# Load config
aoi_coords: list = [91.8, 24.7, 92.2, 25.0]  # Default Sylhet
training_config: Dict[str, Any] = {
    'start_date': '2022-05-01',
    'end_date': '2022-09-30',
    'n_train_samples': 80,
    'n_val_samples': 20,
    'patch_size': 32,
    'epochs': 20,
    'batch_size': 8,
    'learning_rate': 0.001
}

try:
    config_path = os.path.join(depth_estimation_dir, 'config.yaml')
    loaded_config: Dict[str, Any] = load_config(config_path)
    
    # Use Sylhet region from config if available
    regions = loaded_config.get('regions')
    if isinstance(regions, dict):
        sylhet_coords = regions.get('sylhet')
        if isinstance(sylhet_coords, list):
            aoi_coords = sylhet_coords
    
    # Get training config
    training_cfg = loaded_config.get('training')
    if isinstance(training_cfg, dict):
        training_config = training_cfg
    
except Exception as e:
    print(f"Warning: Could not load config.yaml: {e}")
    print("Using default configuration...")

# Set up full config
full_config: Dict[str, Any] = {
    'aoi_coords': aoi_coords,
    **training_config
}

print("Configuration loaded:")
for key, value in full_config.items():
    print(f"  {key}: {value}")
print()

# Create models directory
models_dir = os.path.join(agent_dir, 'models')
os.makedirs(models_dir, exist_ok=True)
model_save_path = os.path.join(models_dir, 'flood_depth_model.h5')

print(f"Model will be saved to: {model_save_path}")
print()

# Train
try:
    print("="*70)
    print("  TRAINING PIPELINE")
    print("="*70)
    print()
    
    trainer = DepthTrainer(full_config)
    history = trainer.train(save_path=model_save_path)
    
    print()
    print("="*70)
    print("  TRAINING COMPLETE!")
    print("="*70)
    print()
    print(f"Model saved to: {model_save_path}")
    print()
    print("Next step: Run quick_inference.py to test the model")
    print()
    
except Exception as e:
    print()
    print("="*70)
    print("  ERROR DURING TRAINING")
    print("="*70)
    print()
    print(f"Error: {e}")
    print()
    import traceback
    traceback.print_exc()
    sys.exit(1)