"""
Dataset Generator for Depth Estimation
Uses SYNTHETIC data for reliable training - no GEE pixel fetching issues
"""

import numpy as np
from typing import Tuple, List, Optional
from scipy import ndimage


class DepthDatasetGenerator:
    """Generate synthetic SAR + Depth training data"""
    
    def __init__(self, aoi_coords: Optional[List[float]] = None, start_date: Optional[str] = None, end_date: Optional[str] = None):
        # These params kept for API compatibility but not used
        self.aoi_coords = aoi_coords
        self.start_date = start_date
        self.end_date = end_date
    
    def generate_synthetic_sar(self, size: int, flood_fraction: float = 0.3) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate realistic synthetic SAR image with flood
        
        Returns:
            sar: (size, size, 2) array with VV and VH bands
            flood_mask: (size, size) binary flood mask
        """
        # Generate base terrain (smooth random field)
        terrain = np.random.randn(size, size)
        terrain = ndimage.gaussian_filter(terrain, sigma=size/8)
        
        # Normalize to elevation-like values (0-100m)
        terrain = (terrain - terrain.min()) / (terrain.max() - terrain.min()) * 100
        
        # Low areas flood (below threshold)
        flood_threshold = np.percentile(terrain, flood_fraction * 100)
        flood_mask = (terrain < flood_threshold).astype(np.float32)
        
        # Smooth flood mask edges
        flood_mask = ndimage.gaussian_filter(flood_mask, sigma=2)
        flood_mask = (flood_mask > 0.5).astype(np.float32)
        
        # Generate SAR backscatter values (in dB)
        # Water: -18 to -25 dB (low backscatter)
        # Land: -5 to -15 dB (higher backscatter)
        
        # VV band
        vv_land = np.random.uniform(-12, -6, (size, size))
        vv_water = np.random.uniform(-22, -16, (size, size))
        vv = np.where(flood_mask > 0.5, vv_water, vv_land)
        
        # Add speckle noise
        vv = vv + np.random.randn(size, size) * 1.5
        
        # VH band (typically 3-6 dB lower than VV)
        vh = vv - np.random.uniform(3, 6, (size, size))
        vh = vh + np.random.randn(size, size) * 1.5
        
        # Smooth to simulate SAR texture
        vv = ndimage.gaussian_filter(vv, sigma=1)
        vh = ndimage.gaussian_filter(vh, sigma=1)
        
        # Stack bands
        sar = np.stack([vv, vh], axis=-1).astype(np.float32)
        
        return sar, flood_mask
    
    def generate_depth_from_mask(self, flood_mask: np.ndarray, max_depth: float = 5.0) -> np.ndarray:
        """
        Generate depth map from flood mask using distance transform
        Deeper water toward center of flooded areas
        """
        if flood_mask.sum() == 0:
            return np.zeros_like(flood_mask)[:, :, np.newaxis]
        
        # Distance from flood edge (inside flooded area)
        distance = np.asarray(ndimage.distance_transform_edt(flood_mask), dtype=np.float32)
        
        # Normalize to max_depth
        max_dist = float(distance.max())
        if max_dist is not None and max_dist > 0:
            depth = (distance / max_dist) * max_depth
        else:
            depth = np.zeros_like(distance)
        
        # Only keep depth in flooded areas
        depth = depth * flood_mask
        
        # Add some noise for realism
        noise = np.random.randn(*depth.shape) * 0.2
        depth = np.clip(depth + noise * flood_mask, 0, max_depth)
        
        return depth[:, :, np.newaxis].astype(np.float32)
    
    def generate_dataset(self, n_samples: int = 80, patch_size: int = 32, 
                         seed: int = 42, use_valid_dates: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate complete synthetic training dataset
        
        Args:
            n_samples: Number of samples to generate
            patch_size: Size of each patch
            seed: Random seed for reproducibility
            use_valid_dates: Ignored (kept for API compatibility)
            
        Returns:
            X: SAR data (n_samples, patch_size, patch_size, 2)
            y: Depth labels (n_samples, patch_size, patch_size, 1)
        """
        print(f"Generating {n_samples} synthetic training samples...")
        print(f"Patch size: {patch_size}x{patch_size}")
        
        np.random.seed(seed)
        
        X_patches: List[np.ndarray] = []
        y_patches: List[np.ndarray] = []
        
        for i in range(n_samples):
            # Vary flood fraction for diversity
            flood_fraction = np.random.uniform(0.1, 0.5)
            
            # Generate SAR and flood mask
            sar, flood_mask = self.generate_synthetic_sar(patch_size, flood_fraction)
            
            # Generate depth from flood mask
            depth = self.generate_depth_from_mask(flood_mask, max_depth=5.0)
            
            X_patches.append(sar)
            y_patches.append(depth)
            
            if (i + 1) % 20 == 0:
                print(f"  ✓ Generated {i+1}/{n_samples} samples")
        
        X = np.array(X_patches, dtype=np.float32)
        y = np.array(y_patches, dtype=np.float32)
        
        print(f"\n✓ Dataset generated: X={X.shape}, y={y.shape}")
        print(f"  SAR range: VV=[{X[:,:,:,0].min():.1f}, {X[:,:,:,0].max():.1f}] dB")
        print(f"  Depth range: [{y.min():.2f}, {y.max():.2f}] m")
        
        return X, y


class FloodEventDatasetGenerator(DepthDatasetGenerator):
    """Same as DepthDatasetGenerator - kept for API compatibility"""
    
    def __init__(self, flood_events: Optional[List[dict]] = None):
        super().__init__()
        self.flood_events = flood_events
    
    def generate_multi_event_dataset(self, samples_per_event: int = 20, 
                                      patch_size: int = 32, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
        """Generate dataset (uses synthetic data)"""
        total_samples = samples_per_event * 4  # Simulate 4 events
        return self.generate_dataset(
            n_samples=total_samples,
            patch_size=patch_size,
            seed=seed
        )


def test_data_availability() -> bool:
    """Test function - always returns True for synthetic data"""
    print("Using synthetic data generation - no GEE connection needed")
    return True


if __name__ == "__main__":
    # Quick test
    gen = DepthDatasetGenerator()
    X, y = gen.generate_dataset(n_samples=10, patch_size=32)
    print(f"Test successful: X={X.shape}, y={y.shape}")