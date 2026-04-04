"""
Standalone Flood Depth Estimation Module
Works independently from existing flood detection system
"""

__version__ = "1.0.0"
__author__ = "Mahmudul Hasan"

from .core.depth_predictor import DepthPredictor
from .core.depth_model import LightweightDepthCNN
from .training.train_depth import DepthTrainer

__all__ = ['DepthPredictor', 'LightweightDepthCNN', 'DepthTrainer']