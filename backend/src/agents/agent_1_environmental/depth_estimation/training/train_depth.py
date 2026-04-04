"""
Main Training Script for Depth Estimation
Run this to train the depth model
"""

import numpy as np
import json
import time
from datetime import datetime
from typing import Dict, Any, Optional

# Support both relative and absolute imports
try:
    from training.dataset_generator import DepthDatasetGenerator
    from core.depth_model import LightweightDepthCNN
except ImportError:
    from .dataset_generator import DepthDatasetGenerator
    from ..core.depth_model import LightweightDepthCNN


class DepthTrainer:
    """Complete training pipeline"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize trainer
        
        Args:
            config: dict with training parameters, or None for defaults
        """
        self.config = config or self._default_config()
        self.model: Optional[LightweightDepthCNN] = None
        self.history: Any = None
    
    def _default_config(self) -> Dict[str, Any]:
        """Default training configuration"""
        return {
            'aoi_coords': [91.8, 24.7, 92.2, 25.0],  # Sylhet
            'start_date': '2022-05-01',
            'end_date': '2022-09-30',
            'n_train_samples': 80,
            'n_val_samples': 20,
            'patch_size': 32,
            'epochs': 20,
            'batch_size': 8,
            'learning_rate': 0.001
        }
    
    def train(self, save_path: str = 'models/flood_depth_model.h5') -> Any:
        """
        Complete training pipeline
        
        Args:
            save_path: Where to save trained model
            
        Returns:
            history: Training history
        """
        print("="*70)
        print("  FLOOD DEPTH ESTIMATION - TRAINING PIPELINE")
        print("="*70)
        
        # Step 1: Generate training data
        print("\n[1/4] Generating training data...")
        train_gen = DepthDatasetGenerator(
            self.config['aoi_coords'],
            self.config['start_date'],
            self.config['end_date']
        )
        
        X_train, y_train = train_gen.generate_dataset(
            n_samples=self.config['n_train_samples'],
            patch_size=self.config['patch_size']
        )
        
        # Step 2: Generate validation data
        print("\n[2/4] Generating validation data...")
        val_gen = DepthDatasetGenerator(
            self.config['aoi_coords'],
            self.config['start_date'],
            self.config['end_date']
        )
        
        X_val, y_val = val_gen.generate_dataset(
            n_samples=self.config['n_val_samples'],
            patch_size=self.config['patch_size'],
            seed=999  # Different seed for validation
        )
        
        # Step 3: Build and train model
        print("\n[3/4] Training model...")
        input_shape = (self.config['patch_size'], self.config['patch_size'], 2)
        self.model = LightweightDepthCNN(input_shape=input_shape)
        self.model.compile(learning_rate=self.config['learning_rate'])
        
        print(f"\nModel parameters: {self.model.get_params():,}")
        
        start_time = time.time()
        
        self.history = self.model.train(
            X_train, y_train,
            X_val, y_val,
            epochs=self.config['epochs'],
            batch_size=self.config['batch_size']
        )
        
        training_time = time.time() - start_time
        
        # Step 4: Save model
        print("\n[4/4] Saving model...")
        self.model.save(save_path)
        
        # Save training info WITH FULL HISTORY FOR LOSS CURVES
        info = {
            'config': self.config,
            'training_time_minutes': training_time / 60,
            'final_metrics': {
                'val_loss': float(self.history.history['val_loss'][-1]),
                'val_mae': float(self.history.history['val_mae'][-1]),
                'val_rmse': float(self.history.history['val_rmse'][-1])
            },
            # FULL HISTORY FOR LOSS CURVES
            'history': {
                'loss': [float(x) for x in self.history.history['loss']],
                'val_loss': [float(x) for x in self.history.history['val_loss']],
                'mae': [float(x) for x in self.history.history['mae']],
                'val_mae': [float(x) for x in self.history.history['val_mae']],
                'rmse': [float(x) for x in self.history.history['rmse']],
                'val_rmse': [float(x) for x in self.history.history['val_rmse']]
            },
            'trained_at': datetime.now().isoformat()
        }
        
        info_path = save_path.replace('.h5', '_info.json')
        with open(info_path, 'w') as f:
            json.dump(info, f, indent=2)
        
        print(f"\n✓ Training info saved to {info_path}")
        
        # Print summary
        print("\n" + "="*70)
        print("  TRAINING COMPLETE!")
        print("="*70)
        print(f"\nTraining time: {training_time/60:.1f} minutes")
        print(f"\nFinal metrics:")
        print(f"  Validation MAE:  {info['final_metrics']['val_mae']:.3f} meters")
        print(f"  Validation RMSE: {info['final_metrics']['val_rmse']:.3f} meters")
        print(f"\nModel saved to: {save_path}")
        print("="*70)
        
        return self.history


def quick_train(region: str = 'sylhet', save_path: str = '../../../models/flood_depth_model.h5') -> Any:
    """
    Quick training function with preset regions
    
    Args:
        region: 'sylhet', 'dhaka', or 'chittagong'
        save_path: Where to save model
    """
    regions = {
        'sylhet': [91.8, 24.7, 92.2, 25.0],
        'dhaka': [90.3, 23.7, 90.5, 23.9],
        'chittagong': [91.7, 22.2, 92.0, 22.5]
    }
    
    config = {
        'aoi_coords': regions.get(region, regions['sylhet']),
        'start_date': '2022-05-01',
        'end_date': '2022-09-30',
        'n_train_samples': 80,
        'n_val_samples': 20,
        'patch_size': 32,
        'epochs': 20,
        'batch_size': 8,
        'learning_rate': 0.001
    }
    
    trainer = DepthTrainer(config)
    return trainer.train(save_path)


if __name__ == "__main__":
    # Run training
    quick_train(region='sylhet')