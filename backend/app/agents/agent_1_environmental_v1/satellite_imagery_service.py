"""
Satellite Imagery Service for Flood Detection
Uses Google Earth Engine (GEE) and Sentinel-1 SAR data
"""

import ee
import geemap
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class FloodDetectionResult:
    """Data class for flood detection results"""
    flood_area_km2: float
    flood_pixels: int
    detection_confidence: float
    affected_regions: List[Dict]
    geojson: Dict
    image_urls: Dict[str, str]
    metadata: Dict


class SatelliteImageryService:
    """
    Service for processing satellite imagery to detect floods
    Uses Sentinel-1 SAR data via Google Earth Engine
    """
    
    def __init__(self, service_account_key_path: Optional[str] = None):
        """
        Initialize the satellite imagery service
        
        Args:
            service_account_key_path: Path to GEE service account JSON key
                                     If None, uses default credentials
        """
        self.initialize_earth_engine(service_account_key_path)
        
        # Default parameters (can be tuned)
        self.params = {
            'SMOOTHING_RADIUS': 50,  # meters, for speckle filtering
            'DIFF_THRESHOLD': -3,     # dB, threshold for flood detection
            'POLARIZATION': 'VH',     # VH is better for flood detection
            'INSTRUMENT_MODE': 'IW',  # Interferometric Wide Swath
            'ORBIT_PASS': 'DESCENDING',  # Can be ASCENDING or DESCENDING
        }
        
    def initialize_earth_engine(self, key_path: Optional[str] = None):
        """Initialize Google Earth Engine with authentication"""
        try:
            if key_path:
                # Service account authentication (for production)
                credentials = ee.ServiceAccountCredentials(
                    email='your-service-account@project.iam.gserviceaccount.com',
                    key_file=key_path
                )
                ee.Initialize(credentials)
                logger.info("GEE initialized with service account")
            else:
                # Interactive authentication (for development)
                try:
                    ee.Initialize()
                    logger.info("GEE initialized with cached credentials")
                except Exception:
                    logger.info("Authenticating with GEE...")
                    ee.Authenticate()
                    ee.Initialize()
                    logger.info("GEE authentication successful")
                    
        except Exception as e:
            logger.error(f"Failed to initialize Earth Engine: {e}")
            raise
    
    def detect_flood(
        self,
        location: Tuple[float, float],  # (latitude, longitude)
        radius_km: float = 50,
        before_start: str = None,  # ISO format: "2024-11-01"
        before_end: str = None,
        after_start: str = None,
        after_end: str = None,
    ) -> FloodDetectionResult:
        """
        Detect floods in a given location using change detection
        
        Args:
            location: (lat, lon) tuple for center point
            radius_km: Radius around location to analyze
            before_start/end: Date range for "before flood" baseline
            after_start/end: Date range for "after flood" comparison
            
        Returns:
            FloodDetectionResult object with detection results
        """
        logger.info(f"Starting flood detection for location {location}")
        
        # Set default dates if not provided
        if not after_end:
            after_end = datetime.now().strftime('%Y-%m-%d')
        if not after_start:
            after_start = (datetime.now() - timedelta(days=3)).strftime('%Y-%m-%d')
        if not before_end:
            before_end = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
        if not before_start:
            before_start = (datetime.now() - timedelta(days=60)).strftime('%Y-%m-%d')
        
        # Create region of interest (ROI)
        lat, lon = location
        point = ee.Geometry.Point([lon, lat])
        roi = point.buffer(radius_km * 1000)  # Convert km to meters
        
        # Get Sentinel-1 image collections
        before_collection = self._get_sentinel1_collection(
            roi, before_start, before_end
        )
        after_collection = self._get_sentinel1_collection(
            roi, after_start, after_end
        )
        
        # Create mosaics (combine multiple images)
        before_image = before_collection.mosaic().clip(roi)
        after_image = after_collection.mosaic().clip(roi)
        
        # Apply speckle filtering
        before_filtered = self._apply_speckle_filter(before_image)
        after_filtered = self._apply_speckle_filter(after_image)
        
        # Perform change detection
        flood_mask = self._detect_change(before_filtered, after_filtered)
        
        # Remove permanent water bodies
        flood_mask = self._remove_permanent_water(flood_mask, roi)
        
        # Apply slope mask (floods don't occur on steep slopes)
        flood_mask = self._apply_slope_mask(flood_mask, roi)
        
        # Calculate statistics
        stats = self._calculate_statistics(flood_mask, roi)
        
        # Generate GeoJSON
        geojson = self._generate_geojson(flood_mask, roi)
        
        # Generate visualization URLs
        image_urls = self._generate_image_urls(
            before_filtered, after_filtered, flood_mask, roi
        )
        
        # Identify affected regions
        affected_regions = self._identify_affected_regions(flood_mask, roi, location)
        
        result = FloodDetectionResult(
            flood_area_km2=stats['area_km2'],
            flood_pixels=stats['pixel_count'],
            detection_confidence=stats['confidence'],
            affected_regions=affected_regions,
            geojson=geojson,
            image_urls=image_urls,
            metadata={
                'location': location,
                'before_dates': f"{before_start} to {before_end}",
                'after_dates': f"{after_start} to {after_end}",
                'parameters': self.params,
                'timestamp': datetime.now().isoformat()
            }
        )
        
        logger.info(f"Flood detection complete. Area: {stats['area_km2']:.2f} km²")
        return result
    
    def _get_sentinel1_collection(
        self, 
        roi: ee.Geometry, 
        start_date: str, 
        end_date: str
    ) -> ee.ImageCollection:
        """Get filtered Sentinel-1 image collection"""
        
        collection = ee.ImageCollection('COPERNICUS/S1_GRD') \
            .filterBounds(roi) \
            .filterDate(start_date, end_date) \
            .filter(ee.Filter.listContains(
                'transmitterReceiverPolarisation',
                self.params['POLARIZATION']
            )) \
            .filter(ee.Filter.eq(
                'instrumentMode',
                self.params['INSTRUMENT_MODE']
            )) \
            .filter(ee.Filter.eq(
                'orbitProperties_pass',
                self.params['ORBIT_PASS']
            )) \
            .select(self.params['POLARIZATION'])
        
        # Log collection size
        size = collection.size().getInfo()
        logger.info(f"Found {size} images from {start_date} to {end_date}")
        
        return collection
    
    def _apply_speckle_filter(self, image: ee.Image) -> ee.Image:
        """
        Apply speckle filtering to reduce noise in SAR images
        Uses focal median filter (simple but effective)
        """
        smoothing = self.params['SMOOTHING_RADIUS']
        return image.focal_median(smoothing, 'circle', 'meters')
    
    def _detect_change(
        self, 
        before: ee.Image, 
        after: ee.Image
    ) -> ee.Image:
        """
        Detect changes between before and after images
        
        Change detection formula:
        - Calculate difference: after - before (in dB)
        - Negative values indicate decreased backscatter (potential flood)
        - Apply threshold to identify significant changes
        """
        # Calculate difference
        difference = after.subtract(before)
        
        # Apply threshold to identify flooded areas
        # Flooded areas have large negative differences
        threshold = self.params['DIFF_THRESHOLD']
        flood_mask = difference.lt(threshold)
        
        return flood_mask
    
    def _remove_permanent_water(
        self, 
        flood_mask: ee.Image, 
        roi: ee.Geometry
    ) -> ee.Image:
        """
        Remove permanent water bodies from flood mask
        Uses JRC Global Surface Water dataset
        """
        # Get permanent water layer
        permanent_water = ee.Image('JRC/GSW1_3/GlobalSurfaceWater') \
            .select('occurrence') \
            .clip(roi)
        
        # Areas with >80% water occurrence are considered permanent water
        permanent_water_mask = permanent_water.gt(80)
        
        # Remove permanent water from flood mask
        flood_mask = flood_mask.And(permanent_water_mask.Not())
        
        return flood_mask
    
    def _apply_slope_mask(
        self, 
        flood_mask: ee.Image, 
        roi: ee.Geometry
    ) -> ee.Image:
        """
        Apply slope mask - floods don't occur on steep slopes
        Uses SRTM Digital Elevation Model
        """
        # Get elevation data
        dem = ee.Image('USGS/SRTMGL1_003').clip(roi)
        
        # Calculate slope in degrees
        slope = ee.Terrain.slope(dem)
        
        # Mask out areas with slope > 5 degrees
        slope_mask = slope.lt(5)
        
        # Apply slope mask to flood mask
        flood_mask = flood_mask.And(slope_mask)
        
        return flood_mask
    
    def _calculate_statistics(
        self, 
        flood_mask: ee.Image, 
        roi: ee.Geometry
    ) -> Dict:
        """Calculate flood statistics"""
        
        # Count flooded pixels
        pixel_count = flood_mask.reduceRegion(
            reducer=ee.Reducer.sum(),
            geometry=roi,
            scale=10,  # Sentinel-1 resolution is 10m
            maxPixels=1e9
        ).getInfo()
        
        pixel_count = pixel_count.get(self.params['POLARIZATION'], 0)
        
        # Calculate area in km²
        # Each pixel is 10m x 10m = 100 m² = 0.0001 km²
        area_km2 = pixel_count * 0.0001
        
        # Calculate confidence (simple heuristic)
        # Higher confidence for larger flood extents
        if area_km2 > 50:
            confidence = 0.9
        elif area_km2 > 10:
            confidence = 0.8
        elif area_km2 > 1:
            confidence = 0.7
        else:
            confidence = 0.6
        
        return {
            'pixel_count': int(pixel_count),
            'area_km2': area_km2,
            'confidence': confidence
        }
    
    def _generate_geojson(
        self, 
        flood_mask: ee.Image, 
        roi: ee.Geometry
    ) -> Dict:
        """
        Convert flood mask to GeoJSON format
        This can be used for mapping and analysis
        """
        # Convert raster to vector
        vectors = flood_mask.reduceToVectors(
            geometry=roi,
            scale=100,  # Use coarser scale for vector conversion
            geometryType='polygon',
            maxPixels=1e8
        )
        
        # Convert to GeoJSON
        geojson = vectors.getInfo()
        
        return geojson
    
    def _generate_image_urls(
        self,
        before: ee.Image,
        after: ee.Image,
        flood_mask: ee.Image,
        roi: ee.Geometry
    ) -> Dict[str, str]:
        """
        Generate map tile URLs for visualization
        These URLs can be used in web maps
        """
        # Visualization parameters
        vis_params_sar = {'min': -30, 'max': 0}
        vis_params_flood = {'palette': ['blue']}
        
        # Get map IDs
        before_map = before.clip(roi).getMapId(vis_params_sar)
        after_map = after.clip(roi).getMapId(vis_params_sar)
        flood_map = flood_mask.clip(roi).updateMask(flood_mask).getMapId(vis_params_flood)
        
        return {
            'before_flood': before_map['tile_fetcher'].url_format,
            'after_flood': after_map['tile_fetcher'].url_format,
            'flood_extent': flood_map['tile_fetcher'].url_format
        }
    
    def _identify_affected_regions(
        self,
        flood_mask: ee.Image,
        roi: ee.Geometry,
        center_location: Tuple[float, float]
    ) -> List[Dict]:
        """
        Identify specific affected regions/neighborhoods
        This is a placeholder - implement based on your region data
        """
        # For now, return the overall affected area
        # In production, you'd intersect with administrative boundaries
        
        affected = [{
            'name': 'Dhaka Metropolitan Area',
            'center': center_location,
            'flood_detected': True,
            'severity': 'moderate'
        }]
        
        return affected


class FloodPredictionService:
    """
    Service for predicting flood progression
    This is for Phase 3 - uses simple trend analysis initially
    """
    
    def predict_flood_progression(
        self,
        historical_data: List[FloodDetectionResult],
        hours_ahead: int = 6
    ) -> Dict:
        """
        Predict flood progression based on historical trend
        
        For your prototype, use simple linear trend
        In Phase 3, implement ML-based prediction
        """
        if len(historical_data) < 2:
            return {
                'predicted_area_km2': historical_data[-1].flood_area_km2,
                'confidence': 'low',
                'method': 'no trend data'
            }
        
        # Calculate area trend
        areas = [r.flood_area_km2 for r in historical_data]
        timestamps = range(len(areas))
        
        # Simple linear regression
        trend = np.polyfit(timestamps, areas, 1)[0]
        
        # Predict future area
        predicted_area = areas[-1] + (trend * (hours_ahead / 3))
        predicted_area = max(0, predicted_area)  # Can't be negative
        
        return {
            'predicted_area_km2': predicted_area,
            'trend': 'increasing' if trend > 0 else 'decreasing',
            'confidence': 'medium',
            'method': 'linear_trend'
        }


def example_usage():
    """Example of how to use the satellite imagery service"""
    
    # Initialize service
    service = SatelliteImageryService()
    
    # Dhaka coordinates
    dhaka_location = (23.8103, 90.4125)
    
    # Detect floods
    result = service.detect_flood(
        location=dhaka_location,
        radius_km=30,
        before_start='2024-10-01',
        before_end='2024-10-15',
        after_start='2024-11-20',
        after_end='2024-11-30'
    )
    
    # Print results
    print(f"Flood Area: {result.flood_area_km2:.2f} km²")
    print(f"Confidence: {result.detection_confidence:.2%}")
    print(f"Affected Regions: {len(result.affected_regions)}")
    
    # Save GeoJSON for mapping
    with open('flood_extent.geojson', 'w') as f:
        json.dump(result.geojson, f)
    
    print("Results saved to flood_extent.geojson")


if __name__ == "__main__":
    example_usage()
