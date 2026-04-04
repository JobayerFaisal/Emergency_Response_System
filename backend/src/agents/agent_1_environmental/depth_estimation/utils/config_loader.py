"""Configuration loader utility"""

import yaml
import os
from typing import Dict, Any, Union


def load_config(config_path: str = 'config.yaml') -> Dict[str, Any]:
    """
    Load configuration from YAML file
    
    Args:
        config_path: Path to config file
        
    Returns:
        dict with configuration
    """
    if not os.path.exists(config_path):
        return get_default_config()
    
    with open(config_path, 'r') as f:
        config: Union[Dict[str, Any], list, str, int, float, None] = yaml.safe_load(f)
    
    # Ensure we return a dict (yaml.safe_load can return various types)
    if not isinstance(config, dict):
        return get_default_config()
    
    return config


def get_default_config() -> Dict[str, Any]:
    """Get default configuration"""
    return {
        'regions': {
            'sylhet': [91.8, 24.7, 92.2, 25.0],
            'dhaka': [90.3, 23.7, 90.5, 23.9],
            'chittagong': [91.7, 22.2, 92.0, 22.5]
        },
        'training': {
            'start_date': '2022-05-01',
            'end_date': '2022-09-30',
            'n_train_samples': 80,
            'n_val_samples': 20,
            'patch_size': 128,
            'epochs': 20,
            'batch_size': 8,
            'learning_rate': 0.001
        },
        'model': {
            'max_depth': 5.0,
            'input_shape': [128, 128, 2]
        },
        'thresholds': {
            'flood_min_depth': 0.1,
            'severity_low': 1.0,
            'severity_medium': 2.0
        }
    }


def save_config(config: Dict[str, Any], config_path: str = 'config.yaml') -> None:
    """
    Save configuration to YAML file
    
    Args:
        config: Configuration dict
        config_path: Path to save
    """
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"✓ Configuration saved to {config_path}")