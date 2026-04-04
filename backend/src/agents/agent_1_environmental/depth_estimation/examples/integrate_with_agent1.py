"""
Integration Guide with Agent 1
Shows how to use depth estimation in your existing code
"""

import sys
import os
import numpy as np
from typing import Dict, Any, Optional
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.depth_predictor import DepthPredictor


class Agent1WithDepth:
    """
    Example Agent 1 with depth estimation
    NO changes to existing flood detection!
    """
    
    def __init__(self, depth_model_path: str = '../../../models/flood_depth_model.h5'):
        """
        Add depth estimation (optional)
        """
        # Your existing code stays unchanged
        # self.flood_detector = ... 
        
        # NEW: Add depth (optional)
        self.depth_predictor: Optional[DepthPredictor] = None
        if os.path.exists(depth_model_path):
            self.depth_predictor = DepthPredictor(depth_model_path)
            print("✓ Depth estimation enabled")
        else:
            print("ℹ Depth disabled (model not found)")
    
    def analyze_flood(self, sar_data: np.ndarray) -> Dict[str, Any]:
        """
        Analyze flood with optional depth
        """
        # Your existing flood detection
        results: Dict[str, Any] = {
            'flood_detected': True
        }
        
        # NEW: Add depth if available
        if self.depth_predictor:
            depth_results = self.depth_predictor.analyze(sar_data)
            results['depth'] = depth_results
            level, msg = self.depth_predictor.get_warning_level(depth_results['statistics'])
            results['warning'] = {
                'level': level, 
                'message': msg
            }
        
        return results
    
    def send_to_agent2(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Send to Agent 2"""
        data: Dict[str, Any] = {
            'flood_detected': results['flood_detected']
        }
        
        # Add depth data if available
        if 'depth' in results:
            stats = results['depth']['statistics']
            data['flood_area_percent'] = stats['flood_area_percent']
            data['max_depth_m'] = stats['max_depth_m']
            data['severity'] = results['depth']['severity']
            
            print(f"\n→ Sending to Agent 2:")
            print(f"  Area: {stats['flood_area_percent']:.1f}%")
            print(f"  Max depth: {stats['max_depth_m']:.2f}m")
        
        return data


def example_usage():
    """Example usage"""
    print("="*70)
    print("  INTEGRATION EXAMPLE")
    print("="*70)
    
    # Initialize
    agent = Agent1WithDepth()
    
    # Create sample data
    print("\n1. Creating sample SAR data...")
    sar_data = np.random.rand(128, 128, 2).astype(np.float32)
    
    # Analyze
    print("\n2. Analyzing flood...")
    results = agent.analyze_flood(sar_data)
    
    # Send to Agent 2
    print("\n3. Sending to Agent 2...")
    agent.send_to_agent2(results)
    
    print("\n" + "="*70)
    print("  INTEGRATION COMPLETE")
    print("="*70)
    print("\nKey points:")
    print("  ✓ No changes to existing code")
    print("  ✓ Depth is optional")
    print("  ✓ Compatible with Agent 2")


def minimal_example():
    """Minimal 3-line integration"""
    print("\n" + "="*70)
    print("  MINIMAL INTEGRATION (3 lines)")
    print("="*70 + "\n")
    
    print("In your Agent 1, add:\n")
    print("```python")
    print("from depth_estimation import DepthPredictor")
    print("")
    print("predictor = DepthPredictor('models/flood_depth_model.h5')")
    print("results = predictor.analyze(sar_array)")
    print("depth_map = results['depth_map']")
    print("```\n")
    print("That's it!")


if __name__ == "__main__":
    example_usage()
    minimal_example()
    
    print("\n" + "="*70)
    print("  NEXT STEPS")
    print("="*70)
    print("\n1. Train: python quick_train.py")
    print("2. Test: python quick_inference.py")
    print("3. Integrate: Add 3 lines to your Agent 1")
    print("="*70)