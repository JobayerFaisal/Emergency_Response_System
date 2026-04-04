"""
Depth Predictor - Main interface for depth estimation
"""

import numpy as np
from typing import Dict, Any, Tuple, Optional
from .depth_model import LightweightDepthCNN


class DepthPredictor:
    """Easy-to-use depth prediction interface"""
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize predictor
        
        Args:
            model_path: Path to trained model (.h5 file)
        """
        self.model = LightweightDepthCNN()
        
        if model_path:
            self.model.load(model_path)
    
    def predict_depth(self, sar_array: np.ndarray) -> np.ndarray:
        """
        Predict flood depth from SAR image
        
        Args:
            sar_array: numpy array [H, W, 2] with VV and VH bands
            
        Returns:
            depth_map: numpy array [H, W, 1] with depth in meters
        """
        if self.model.model is None:
            raise ValueError("No model loaded. Provide model_path or train first.")
        
        return self.model.predict(sar_array)
    
    def analyze(self, sar_array: np.ndarray) -> Dict[str, Any]:
        """
        Complete depth analysis with statistics
        
        Args:
            sar_array: SAR image [H, W, 2]
            
        Returns:
            dict with depth_map, statistics, and severity
        """
        depth_map = self.predict_depth(sar_array)
        
        # Calculate statistics
        flooded = depth_map > 0.1
        
        if np.sum(flooded) > 0:
            flooded_depths = depth_map[flooded]
            mean_depth = float(np.mean(flooded_depths))
            max_depth = float(np.max(flooded_depths))
            std_depth = float(np.std(flooded_depths))
        else:
            mean_depth = max_depth = std_depth = 0.0
        
        # Calculate flood area
        total_pixels = depth_map.shape[0] * depth_map.shape[1]
        flooded_pixels = int(np.sum(flooded))
        flood_percent = 100.0 * flooded_pixels / total_pixels
        
        # Severity classification
        severity = self._classify_severity(depth_map)
        
        return {
            'depth_map': depth_map,
            'statistics': {
                'flood_area_percent': float(flood_percent),
                'flooded_pixels': flooded_pixels,
                'mean_depth_m': mean_depth,
                'max_depth_m': max_depth,
                'std_depth_m': std_depth
            },
            'severity': severity
        }
    
    def _classify_severity(self, depth_map: np.ndarray) -> np.ndarray:
        """
        Classify flood severity
        
        Returns:
            severity_map: 0=none, 1=low, 2=medium, 3=high
        """
        severity = np.zeros_like(depth_map, dtype=np.int32)
        severity[depth_map > 0.1] = 1   # Low (>10cm)
        severity[depth_map > 1.0] = 2   # Medium (>1m)
        severity[depth_map > 2.0] = 3   # High (>2m)
        
        return severity
    
    def get_warning_level(self, statistics: Dict[str, Any]) -> Tuple[int, str]:
        """
        Get warning level based on statistics
        
        Args:
            statistics: dict from analyze()
            
        Returns:
            level: 0-3 (none, low, medium, high)
            message: Warning message
        """
        max_depth = statistics['max_depth_m']
        flood_pct = statistics['flood_area_percent']
        
        level = 0
        messages = []
        
        # Depth-based
        if max_depth > 2.0:
            level = max(level, 3)
            messages.append(f"CRITICAL: Max depth {max_depth:.1f}m")
        elif max_depth > 1.0:
            level = max(level, 2)
            messages.append(f"HIGH: Max depth {max_depth:.1f}m")
        elif max_depth > 0.5:
            level = max(level, 1)
            messages.append(f"MODERATE: Max depth {max_depth:.1f}m")
        
        # Area-based
        if flood_pct > 50:
            level = max(level, 3)
            messages.append(f"CRITICAL: {flood_pct:.0f}% area flooded")
        elif flood_pct > 20:
            level = max(level, 2)
            messages.append(f"HIGH: {flood_pct:.0f}% area flooded")
        
        if level == 0:
            message = "No significant flood detected"
        else:
            severity_names = ['NONE', 'LOW', 'MEDIUM', 'HIGH']
            message = f"WARNING [{severity_names[level]}]: " + ", ".join(messages)
        
        return level, message