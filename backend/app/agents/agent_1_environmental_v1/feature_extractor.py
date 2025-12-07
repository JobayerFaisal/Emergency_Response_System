"""
Feature Extraction Module for ML-Based Flood Detection
Extracts rich features from Sentinel-1 SAR data for ML models

Features Extracted:
1. Backscatter Statistics (VH, VV, ratios)
2. Texture Features (GLCM)
3. Temporal Features (change detection)
4. Spatial Features (terrain, distance)
5. Contextual Features (land cover, urban)
"""

import ee
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class FeatureSet:
    """Container for extracted features"""
    features: Dict[str, np.ndarray]
    feature_names: List[str]
    spatial_reference: ee.Geometry
    timestamp: str
    metadata: Dict


class SARFeatureExtractor:
    """
    Extract ML-ready features from Sentinel-1 SAR imagery
    
    This class is the BRIDGE between raw SAR data and ML models.
    It transforms satellite images into structured feature vectors
    that machine learning models can understand.
    """
    
    def __init__(self):
        """Initialize feature extractor"""
        
        self.feature_config = {
            # Which features to extract
            'backscatter_stats': True,
            'texture_features': True,
            'temporal_features': True,
            'spatial_features': True,
            'contextual_features': True,
        }
        
        # GLCM texture parameters
        self.texture_params = {
            'radius': 3,  # Neighborhood size
            'metrics': ['contrast', 'correlation', 'entropy', 'homogeneity']
        }
        
    def extract_all_features(
        self,
        before_image: ee.Image,
        after_image: ee.Image,
        roi: ee.Geometry,
        scale: int = 10
    ) -> FeatureSet:
        """
        Extract complete feature set for ML models
        
        This is the MAIN function that orchestrates all feature extraction.
        
        Args:
            before_image: Baseline SAR image (pre-flood)
            after_image: Current SAR image (post-flood)
            roi: Region of interest
            scale: Resolution in meters (10m for Sentinel-1)
            
        Returns:
            FeatureSet with all extracted features
        """
        logger.info("Extracting features for ML models...")
        
        features = {}
        feature_names = []
        
        # 1. BACKSCATTER FEATURES (Core SAR data)
        if self.feature_config['backscatter_stats']:
            backscatter_feats = self._extract_backscatter_features(
                before_image, after_image, roi, scale
            )
            features.update(backscatter_feats)
            feature_names.extend(backscatter_feats.keys())
        
        # 2. TEXTURE FEATURES (Spatial patterns)
        if self.feature_config['texture_features']:
            texture_feats = self._extract_texture_features(
                after_image, roi, scale
            )
            features.update(texture_feats)
            feature_names.extend(texture_feats.keys())
        
        # 3. TEMPORAL FEATURES (Change detection)
        if self.feature_config['temporal_features']:
            temporal_feats = self._extract_temporal_features(
                before_image, after_image, roi, scale
            )
            features.update(temporal_feats)
            feature_names.extend(temporal_feats.keys())
        
        # 4. SPATIAL FEATURES (Terrain, distance)
        if self.feature_config['spatial_features']:
            spatial_feats = self._extract_spatial_features(roi, scale)
            features.update(spatial_feats)
            feature_names.extend(spatial_feats.keys())
        
        # 5. CONTEXTUAL FEATURES (Land cover, urban)
        if self.feature_config['contextual_features']:
            contextual_feats = self._extract_contextual_features(roi, scale)
            features.update(contextual_feats)
            feature_names.extend(contextual_feats.keys())
        
        logger.info(f"Extracted {len(feature_names)} features total")
        
        return FeatureSet(
            features=features,
            feature_names=feature_names,
            spatial_reference=roi,
            timestamp=ee.Date(after_image.get('system:time_start')).format().getInfo(),
            metadata={
                'scale': scale,
                'feature_count': len(feature_names)
            }
        )
    
    def _extract_backscatter_features(
        self,
        before: ee.Image,
        after: ee.Image,
        roi: ee.Geometry,
        scale: int
    ) -> Dict[str, np.ndarray]:
        """
        Extract backscatter intensity features
        
        These are the MOST IMPORTANT features for flood detection!
        Water has LOW backscatter, land has HIGH backscatter.
        
        Features extracted:
        - VH_before, VH_after: Cross-polarization (best for flood)
        - VV_before, VV_after: Co-polarization (water detection)
        - VH_diff, VV_diff: Change in backscatter
        - VH_VV_ratio: Polarization ratio (water discrimination)
        """
        logger.info("Extracting backscatter features...")
        
        features = {}
        
        # Get VH polarization (best for flood detection)
        vh_before = before.select('VH')
        vh_after = after.select('VH')
        
        # Calculate difference (key for change detection!)
        vh_diff = vh_after.subtract(vh_before)
        
        # Calculate ratio (normalized change)
        vh_ratio = vh_after.divide(vh_before)
        
        # If VV available, extract it too
        try:
            vv_before = before.select('VV')
            vv_after = after.select('VV')
            vv_diff = vv_after.subtract(vv_before)
            
            # Cross-polarization ratio (VH/VV)
            # Very useful for distinguishing water from other dark targets
            vh_vv_ratio = vh_after.divide(vv_after)
            
            features['VV_before'] = self._image_to_array(vv_before, roi, scale)
            features['VV_after'] = self._image_to_array(vv_after, roi, scale)
            features['VV_diff'] = self._image_to_array(vv_diff, roi, scale)
            features['VH_VV_ratio'] = self._image_to_array(vh_vv_ratio, roi, scale)
            
        except Exception as e:
            logger.warning(f"VV polarization not available: {e}")
        
        # Store as numpy arrays for ML
        features['VH_before'] = self._image_to_array(vh_before, roi, scale)
        features['VH_after'] = self._image_to_array(vh_after, roi, scale)
        features['VH_diff'] = self._image_to_array(vh_diff, roi, scale)
        features['VH_ratio'] = self._image_to_array(vh_ratio, roi, scale)
        
        return features
    
    def _extract_texture_features(
        self,
        image: ee.Image,
        roi: ee.Geometry,
        scale: int
    ) -> Dict[str, np.ndarray]:
        """
        Extract texture features using GLCM (Gray Level Co-occurrence Matrix)
        
        WHY? Flooded areas have different TEXTURE than dry land:
        - Water: Smooth, homogeneous
        - Urban: Rough, high contrast
        - Vegetation: Medium texture
        
        GLCM captures these patterns!
        
        Features:
        - Contrast: Local variations
        - Correlation: Linear dependencies  
        - Entropy: Randomness
        - Homogeneity: Smoothness
        """
        logger.info("Extracting texture features...")
        
        features = {}
        
        # Select band for texture analysis (VH is best)
        band = image.select('VH')
        
        # Compute GLCM
        glcm = band.glcmTexture(
            size=self.texture_params['radius']
        )
        
        # Extract each texture metric
        for metric in self.texture_params['metrics']:
            # GLCM output format: {band}_{metric}
            texture_band = f"VH_{metric}"
            
            if metric in ['asm', 'contrast', 'corr', 'var', 'idm', 'savg', 
                         'svar', 'sent', 'ent', 'dvar', 'dent', 'imcorr1', 'imcorr2']:
                try:
                    texture_image = glcm.select(texture_band)
                    features[f'texture_{metric}'] = self._image_to_array(
                        texture_image, roi, scale
                    )
                except:
                    logger.debug(f"Texture metric {metric} not available")
        
        return features
    
    def _extract_temporal_features(
        self,
        before: ee.Image,
        after: ee.Image,
        roi: ee.Geometry,
        scale: int
    ) -> Dict[str, np.ndarray]:
        """
        Extract temporal change features
        
        These features capture HOW MUCH things changed over time.
        Floods cause RAPID, LARGE changes.
        
        Features:
        - Absolute difference
        - Relative change (%)
        - Log ratio
        - Change magnitude
        """
        logger.info("Extracting temporal features...")
        
        features = {}
        
        vh_before = before.select('VH')
        vh_after = after.select('VH')
        
        # Absolute difference (already computed in backscatter)
        diff = vh_after.subtract(vh_before)
        
        # Relative change (percentage)
        # (after - before) / before * 100
        relative_change = diff.divide(vh_before).multiply(100)
        
        # Log ratio (commonly used in SAR change detection)
        # log(after / before)
        log_ratio = vh_after.divide(vh_before).log()
        
        # Change magnitude (absolute value of diff)
        magnitude = diff.abs()
        
        features['temporal_diff'] = self._image_to_array(diff, roi, scale)
        features['temporal_rel_change'] = self._image_to_array(relative_change, roi, scale)
        features['temporal_log_ratio'] = self._image_to_array(log_ratio, roi, scale)
        features['temporal_magnitude'] = self._image_to_array(magnitude, roi, scale)
        
        return features
    
    def _extract_spatial_features(
        self,
        roi: ee.Geometry,
        scale: int
    ) -> Dict[str, np.ndarray]:
        """
        Extract spatial/terrain features
        
        WHY? Floods follow terrain! Water flows downhill.
        
        Features critical for DEPTH estimation:
        - Elevation: Lower areas flood more
        - Slope: Flat areas accumulate water
        - Distance to water: Closer = more likely to flood
        - Flow accumulation: Where water collects
        - Curvature: Concave areas hold water
        """
        logger.info("Extracting spatial features...")
        
        features = {}
        
        # 1. ELEVATION (from SRTM DEM)
        dem = ee.Image('USGS/SRTMGL1_003').clip(roi)
        elevation = dem.select('elevation')
        
        # 2. SLOPE (derived from elevation)
        slope = ee.Terrain.slope(dem)
        
        # 3. ASPECT (which direction slope faces)
        aspect = ee.Terrain.aspect(dem)
        
        # 4. CURVATURE (approximate using elevation gradient)
        # Concave (negative) = valleys, water accumulation
        # Convex (positive) = ridges, water flows away
        gradient = elevation.gradient()
        curvature = gradient.select('x').add(gradient.select('y'))
        
        # 5. DISTANCE TO PERMANENT WATER
        # Get permanent water bodies from JRC dataset
        water = ee.Image('JRC/GSW1_3/GlobalSurfaceWater') \
            .select('occurrence') \
            .clip(roi)
        
        # Create mask of permanent water (>80% occurrence)
        permanent_water = water.gt(80)
        
        # Calculate distance to nearest water (in meters)
        distance_to_water = permanent_water.fastDistanceTransform() \
            .sqrt() \
            .multiply(scale)
        
        # 6. FLOW ACCUMULATION (where water collects)
        # This requires more complex hydrology calculations
        # For now, use a simple proxy: inverted elevation
        flow_accumulation = elevation.multiply(-1)
        
        # Store features
        features['elevation'] = self._image_to_array(elevation, roi, scale)
        features['slope'] = self._image_to_array(slope, roi, scale)
        features['aspect'] = self._image_to_array(aspect, roi, scale)
        features['curvature'] = self._image_to_array(curvature, roi, scale)
        features['distance_to_water'] = self._image_to_array(distance_to_water, roi, scale)
        features['flow_accumulation'] = self._image_to_array(flow_accumulation, roi, scale)
        
        return features
    
    def _extract_contextual_features(
        self,
        roi: ee.Geometry,
        scale: int
    ) -> Dict[str, np.ndarray]:
        """
        Extract contextual features (land cover, urban areas)
        
        WHY? Flood detection varies by context:
        - Urban areas: Hard to detect (buildings scatter radar)
        - Vegetation: Medium difficulty
        - Open water/agriculture: Easy to detect
        
        Features:
        - Land cover type
        - Urban density
        - Vegetation index (if optical data available)
        """
        logger.info("Extracting contextual features...")
        
        features = {}
        
        try:
            # OPTION 1: Use MODIS Land Cover
            land_cover = ee.ImageCollection('MODIS/006/MCD12Q1') \
                .first() \
                .select('LC_Type1') \
                .clip(roi)
            
            features['land_cover'] = self._image_to_array(land_cover, roi, scale)
            
        except Exception as e:
            logger.warning(f"Land cover data not available: {e}")
        
        try:
            # OPTION 2: Use Global Human Settlement Layer (urban areas)
            urban = ee.Image('JRC/GHSL/P2016/BUILT_LDSMT_GLOBE_V1') \
                .select('built') \
                .clip(roi)
            
            features['urban_density'] = self._image_to_array(urban, roi, scale)
            
        except Exception as e:
            logger.warning(f"Urban data not available: {e}")
        
        return features
    
    def _image_to_array(
        self,
        image: ee.Image,
        roi: ee.Geometry,
        scale: int
    ) -> np.ndarray:
        """
        Convert Earth Engine image to numpy array
        
        This is the critical step that transforms cloud-based imagery
        into local arrays that ML models can process.
        """
        # Sample the image within ROI
        # Returns a dictionary with pixel values
        sampled = image.sample(
            region=roi,
            scale=scale,
            geometries=False,
            dropNulls=True
        )
        
        # Convert to numpy array
        # This downloads the data from GEE servers
        values = sampled.aggregate_array(image.bandNames().get(0)).getInfo()
        
        return np.array(values)
    
    def prepare_ml_dataset(
        self,
        feature_set: FeatureSet,
        labels: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Prepare features for ML model training/inference
        
        Combines all features into a single matrix where:
        - Rows = samples (pixels)
        - Columns = features
        
        Args:
            feature_set: Extracted features
            labels: Optional ground truth labels (for training)
            
        Returns:
            X: Feature matrix [n_samples, n_features]
            y: Label vector [n_samples] (if provided)
        """
        logger.info("Preparing ML dataset...")
        
        # Get feature arrays
        feature_arrays = []
        
        for name in feature_set.feature_names:
            if name in feature_set.features:
                arr = feature_set.features[name]
                # Reshape to column vector if needed
                if arr.ndim == 1:
                    arr = arr.reshape(-1, 1)
                feature_arrays.append(arr)
        
        # Stack all features horizontally
        # Result: [n_samples, n_features]
        X = np.hstack(feature_arrays)
        
        logger.info(f"Dataset shape: {X.shape}")
        logger.info(f"Features: {len(feature_set.feature_names)}")
        
        # Handle missing values
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Prepare labels if provided
        y = None
        if labels is not None:
            y = np.array(labels)
            logger.info(f"Labels shape: {y.shape}")
        
        return X, y
    
    def get_feature_importance_names(self) -> List[str]:
        """
        Get ordered list of feature names
        Useful for interpreting model feature importance
        """
        return [
            # Backscatter features (most important!)
            'VH_before', 'VH_after', 'VH_diff', 'VH_ratio',
            'VV_before', 'VV_after', 'VV_diff', 'VH_VV_ratio',
            
            # Texture features
            'texture_contrast', 'texture_correlation', 
            'texture_entropy', 'texture_homogeneity',
            
            # Temporal features
            'temporal_diff', 'temporal_rel_change',
            'temporal_log_ratio', 'temporal_magnitude',
            
            # Spatial features (critical for depth!)
            'elevation', 'slope', 'aspect', 'curvature',
            'distance_to_water', 'flow_accumulation',
            
            # Contextual features
            'land_cover', 'urban_density'
        ]


def example_usage():
    """
    Example: How to use the feature extractor
    """
    import ee
    ee.Initialize()
    
    # Initialize extractor
    extractor = SARFeatureExtractor()
    
    # Define location (Dhaka)
    dhaka = ee.Geometry.Point([90.4125, 23.8103])
    roi = dhaka.buffer(10000)  # 10km radius
    
    # Get Sentinel-1 images
    collection = ee.ImageCollection('COPERNICUS/S1_GRD') \
        .filterBounds(roi) \
        .filter(ee.Filter.eq('instrumentMode', 'IW')) \
        .select('VH')
    
    # Get before and after images
    before = collection.filterDate('2024-10-01', '2024-10-15').mosaic()
    after = collection.filterDate('2024-11-20', '2024-11-30').mosaic()
    
    # Extract features
    features = extractor.extract_all_features(
        before_image=before,
        after_image=after,
        roi=roi,
        scale=10
    )
    
    # Prepare for ML
    X, _ = extractor.prepare_ml_dataset(features)
    
    print(f"Feature matrix shape: {X.shape}")
    print(f"Ready for ML model!")
    
    return X


if __name__ == "__main__":
    # Test the feature extractor
    X = example_usage()
