"""
Synthetic Depth Label Generator
Uses DEM to create training labels without ground truth
FIXED VERSION - Handles null/empty flood regions properly
"""

import ee
import numpy as np

try:
    ee.Initialize()
except:
    pass


class SyntheticDepthGenerator:
    """Generate synthetic depth labels from DEM"""
    
    def __init__(self, dem_asset: str = 'USGS/SRTMGL1_003'):
        self.dem = ee.Image(dem_asset).select('elevation')
    
    def generate(self, flood_mask: ee.Image, aoi: ee.Geometry, max_depth: float = 5.0) -> ee.Image:
        """
        Generate synthetic depth from topography
        
        Args:
            flood_mask: ee.Image with flood detection (1=flooded, 0=dry)
            aoi: ee.Geometry for area of interest
            max_depth: Maximum depth in meters
            
        Returns:
            ee.Image with depth values in meters
        """
        # Ensure flood_mask is binary (0 or 1)
        flood_binary = flood_mask.gt(0).selfMask()
        
        # Get elevation only in flooded areas
        flooded_elevation = self.dem.updateMask(flood_binary)
        
        # Calculate flood water level (90th percentile of flooded area elevation)
        stats = flooded_elevation.reduceRegion(
            reducer=ee.Reducer.percentile([90]),
            geometry=aoi,
            scale=30,
            maxPixels=1e9,
            bestEffort=True
        )
        
        # Handle potential null value with server-side conditional
        # If no flooded pixels, use a default water level based on mean DEM elevation
        water_level_raw = stats.get('elevation')
        
        # Get fallback value (mean elevation + 1m as default flood level)
        fallback_stats = self.dem.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=aoi,
            scale=100,
            maxPixels=1e8,
            bestEffort=True
        )
        fallback_level = ee.Number(fallback_stats.get('elevation')).add(1)
        
        # Use server-side conditional to handle null
        water_level = ee.Number(
            ee.Algorithms.If(
                water_level_raw,
                water_level_raw,
                fallback_level
            )
        )
        
        # Create constant image from water level
        water_level_image = ee.Image.constant(water_level)
        
        # Depth = water_level - ground_elevation
        depth = water_level_image.subtract(self.dem)
        
        # Only keep depth in flooded areas
        depth = depth.updateMask(flood_binary)
        
        # Clip to realistic range [0, max_depth]
        depth = depth.clamp(0, max_depth)
        
        # Add small noise for realism (±10%)
        # Use seed for reproducibility
        noise = ee.Image.random(42).multiply(0.2).subtract(0.1)
        depth = depth.multiply(ee.Image.constant(1).add(noise))
        
        # Final clamp and rename
        return depth.clamp(0, max_depth).rename('depth').toFloat()
    
    def generate_from_sar(self, sar_vv: ee.Image, aoi: ee.Geometry, 
                          threshold_percentile: int = 20, max_depth: float = 5.0) -> ee.Image:
        """
        Generate depth directly from SAR VV band
        Uses SAR backscatter intensity to estimate relative depth
        
        Args:
            sar_vv: ee.Image with VV polarization band
            aoi: ee.Geometry for area of interest  
            threshold_percentile: Percentile for flood detection threshold
            max_depth: Maximum depth in meters
            
        Returns:
            ee.Image with estimated depth values
        """
        # Detect flood using threshold
        stats = sar_vv.reduceRegion(
            reducer=ee.Reducer.percentile([threshold_percentile]),
            geometry=aoi,
            scale=10,
            maxPixels=int(1e9),
            bestEffort=True
        )
        
        threshold = ee.Number(
            ee.Algorithms.If(
                stats.get('VV'),
                stats.get('VV'),
                ee.Number(-15)  # Default threshold in dB
            )
        )
        
        # Flood mask: low backscatter = water
        flood_mask = sar_vv.lt(threshold)
        
        # Generate depth using DEM
        return self.generate(flood_mask, aoi, max_depth)
    
    def generate_simple(self, flood_mask: ee.Image, aoi: ee.Geometry, 
                        max_depth: float = 5.0) -> ee.Image:
        """
        Simplified depth generation using distance from flood edge
        More robust when DEM-based method fails
        
        Args:
            flood_mask: ee.Image with flood detection
            aoi: ee.Geometry for area of interest
            max_depth: Maximum depth in meters
            
        Returns:
            ee.Image with estimated depth values
        """
        # Ensure binary mask
        flood_binary = flood_mask.gt(0).selfMask()
        
        # Calculate distance to flood edge (proxy for depth)
        # Water is deeper toward the center of flooded areas
        distance = flood_binary.fastDistanceTransform().sqrt()
        
        # Normalize to max_depth range
        dist_stats = distance.reduceRegion(
            reducer=ee.Reducer.percentile([95]),
            geometry=aoi,
            scale=30,
            maxPixels=int(1e8),
            bestEffort=True
        )
        
        max_dist = ee.Number(
            ee.Algorithms.If(
                dist_stats.get('distance'),
                dist_stats.get('distance'),
                ee.Number(100)
            )
        )
        
        # Scale distance to depth (0 at edge, max_depth at center)
        depth = distance.divide(max_dist).multiply(max_depth)
        
        # Apply flood mask and clamp
        depth = depth.updateMask(flood_binary).clamp(0, max_depth)
        
        # Add small noise
        noise = ee.Image.random(42).multiply(0.2).subtract(0.1)
        depth = depth.multiply(ee.Image.constant(1).add(noise))
        
        return depth.clamp(0, max_depth).rename('depth').toFloat()