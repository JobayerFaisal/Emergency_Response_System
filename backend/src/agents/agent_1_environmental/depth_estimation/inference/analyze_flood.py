"""
Flood Analysis Script
Run depth estimation on SAR images
"""

import ee
import numpy as np
from datetime import datetime
from ..core.depth_predictor import DepthPredictor

try:
    ee.Initialize()
except:
    pass


class FloodAnalyzer:
    """Run depth analysis on flood events"""
    
    def __init__(self, model_path):
        """
        Args:
            model_path: Path to trained depth model
        """
        self.predictor = DepthPredictor(model_path)
        print(f"✓ Loaded depth model from {model_path}")
    
    def fetch_sar(self, aoi_coords, date=None):
        """
        Fetch Sentinel-1 SAR image
        
        Args:
            aoi_coords: [lon_min, lat_min, lon_max, lat_max]
            date: Date string 'YYYY-MM-DD' or None for latest
            
        Returns:
            sar_array: numpy array [H, W, 2]
            acq_date: Acquisition date
        """
        aoi = ee.Geometry.Rectangle(aoi_coords)
        
        if date is None:
            # Get latest image from last 3 days
            today = ee.Date(datetime.now())
            start = today.advance(-3, 'day')
            end = today
        else:
            start = ee.Date(date)
            end = start.advance(1, 'day')
        
        s1 = ee.ImageCollection('COPERNICUS/S1_GRD') \
            .filterBounds(aoi) \
            .filterDate(start, end) \
            .filter(ee.Filter.eq('instrumentMode', 'IW')) \
            .select(['VV', 'VH']) \
            .sort('system:time_start', False)
        
        image = s1.first()
        
        # Get acquisition time
        time_ms = image.get('system:time_start').getInfo()
        acq_date = datetime.fromtimestamp(time_ms / 1000)
        
        # Sample image
        pixels = image.sample(
            region=aoi,
            scale=10,
            numPixels=128*128,
            geometries=False
        )
        
        features = pixels.getInfo()['features']
        
        # Convert to array
        vv = [f['properties']['VV'] for f in features]
        vh = [f['properties']['VH'] for f in features]
        
        size = int(np.sqrt(len(vv)))
        vv_img = np.array(vv[:size*size]).reshape(size, size)
        vh_img = np.array(vh[:size*size]).reshape(size, size)
        
        sar_array = np.stack([vv_img, vh_img], axis=-1).astype(np.float32)
        
        return sar_array, acq_date
    
    def analyze_region(self, aoi_coords, date=None, save_results=True):
        """
        Complete flood analysis for a region
        
        Args:
            aoi_coords: [lon_min, lat_min, lon_max, lat_max]
            date: Date string or None for latest
            save_results: Whether to save results
            
        Returns:
            results: dict with depth map and statistics
        """
        print("\n" + "="*70)
        print("  FLOOD DEPTH ANALYSIS")
        print("="*70)
        
        # Fetch SAR data
        print(f"\n[1/3] Fetching Sentinel-1 data...")
        sar_array, acq_date = self.fetch_sar(aoi_coords, date)
        print(f"  ✓ Image acquired: {acq_date.strftime('%Y-%m-%d %H:%M UTC')}")
        print(f"  ✓ Size: {sar_array.shape}")
        
        # Run analysis
        print(f"\n[2/3] Running depth analysis...")
        results = self.predictor.analyze(sar_array)
        
        # Add metadata
        results['metadata'] = {
            'acquisition_time': acq_date.isoformat(),
            'analysis_time': datetime.now().isoformat(),
            'aoi_coords': aoi_coords
        }
        
        # Get warning
        level, message = self.predictor.get_warning_level(results['statistics'])
        results['warning'] = {'level': level, 'message': message}
        
        # Print results
        print(f"\n[3/3] Analysis complete!")
        print("\n" + "-"*70)
        print("  RESULTS")
        print("-"*70)
        stats = results['statistics']
        print(f"  Acquisition: {acq_date.strftime('%Y-%m-%d %H:%M UTC')}")
        print(f"  Flood area: {stats['flood_area_percent']:.1f}%")
        print(f"  Mean depth: {stats['mean_depth_m']:.2f} m")
        print(f"  Max depth: {stats['max_depth_m']:.2f} m")
        print(f"  Warning: {message}")
        print("-"*70)
        
        # Save results
        if save_results:
            import json
            timestamp = acq_date.strftime('%Y%m%d_%H%M')
            
            # Save JSON (without numpy arrays)
            json_results = {
                'metadata': results['metadata'],
                'statistics': results['statistics'],
                'warning': results['warning']
            }
            
            filename = f'flood_analysis_{timestamp}.json'
            with open(filename, 'w') as f:
                json.dump(json_results, f, indent=2)
            
            print(f"\n✓ Results saved to {filename}")
        
        return results


def quick_analyze(region='sylhet', model_path='../../../models/flood_depth_model.h5'):
    """
    Quick analysis function
    
    Args:
        region: 'sylhet', 'dhaka', or 'chittagong'
        model_path: Path to trained model
    """
    regions = {
        'sylhet': [91.8, 24.7, 92.2, 25.0],
        'dhaka': [90.3, 23.7, 90.5, 23.9],
        'chittagong': [91.7, 22.2, 92.0, 22.5]
    }
    
    analyzer = FloodAnalyzer(model_path)
    return analyzer.analyze_region(regions[region])


if __name__ == "__main__":
    # Run analysis
    quick_analyze(region='sylhet')