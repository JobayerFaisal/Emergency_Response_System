"""
Flood Depth Estimation - Inference Pipeline
============================================
Runs predictions on SAR imagery and generates report-ready outputs

Author: Mahmudul Hasan
Project: Autonomous Multi-Agent System for Real-Time Urban Flood Response
Agent: Environmental Intelligence Agent (Agent 1)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy import ndimage
from tensorflow import keras
from datetime import datetime
from typing import Tuple, Dict, Optional, Any
import json
import os


class FloodDepthPredictor:
    """
    Flood Depth Estimation Inference Pipeline
    
    Takes SAR imagery (VV, VH bands) and predicts water depth in meters.
    Designed for integration with the Environmental Intelligence Agent.
    """
    
    def __init__(self, model_path: str):
        """
        Initialize the predictor with a trained model.
        
        Args:
            model_path: Path to the trained .h5 model file
        """
        self.model_path = model_path
        self.model = keras.models.load_model(model_path, compile=False)
        self.input_shape = self.model.input_shape[1:3]  # (height, width)
        
        print(f"✓ Model loaded: {model_path}")
        print(f"  Input shape: {self.input_shape[0]}×{self.input_shape[1]}×2")
        print(f"  Parameters: {self.model.count_params():,}")
    
    def preprocess(self, sar_vv: np.ndarray, sar_vh: np.ndarray) -> np.ndarray:
        """
        Preprocess SAR data for model input.
        
        Args:
            sar_vv: VV polarization band (dB values)
            sar_vh: VH polarization band (dB values)
            
        Returns:
            Normalized SAR array ready for prediction
        """
        # Stack bands
        sar = np.stack([sar_vv, sar_vh], axis=-1).astype(np.float32)
        
        # Add batch dimension if needed
        if len(sar.shape) == 3:
            sar = sar[np.newaxis, ...]
        
        # Normalize (zero mean, unit variance)
        sar_norm = (sar - sar.mean()) / (sar.std() + 1e-7)
        
        return sar_norm
    
    def predict(self, sar_vv: np.ndarray, sar_vh: np.ndarray) -> np.ndarray:
        """
        Predict flood depth from SAR imagery.
        
        Args:
            sar_vv: VV polarization band (dB values)
            sar_vh: VH polarization band (dB values)
            
        Returns:
            Depth map in meters (0-5m range)
        """
        # Preprocess
        sar_input = self.preprocess(sar_vv, sar_vh)
        
        # Predict
        depth = self.model.predict(sar_input, verbose=0)
        
        # Remove batch dimension
        if depth.shape[0] == 1:
            depth = depth[0, :, :, 0]
        
        return depth
    
    def predict_with_flood_mask(self, sar_vv: np.ndarray, sar_vh: np.ndarray,
                                 flood_threshold: float = -15.0) -> Dict[str, Any]:
        """
        Predict depth with automatic flood detection.
        
        Args:
            sar_vv: VV polarization band (dB values)
            sar_vh: VH polarization band (dB values)
            flood_threshold: VV threshold for flood detection (dB)
            
        Returns:
            Dictionary with depth, flood_mask, and masked_depth
        """
        # Detect flood (low backscatter = water)
        flood_mask = (sar_vv < flood_threshold).astype(np.float32)
        
        # Predict depth
        depth = self.predict(sar_vv, sar_vh)
        
        # Apply flood mask (depth only in flooded areas)
        masked_depth = depth * flood_mask
        
        return {
            'depth': depth,
            'flood_mask': flood_mask,
            'masked_depth': masked_depth,
            'flood_threshold': flood_threshold
        }
    
    def analyze_flood(self, depth: np.ndarray, flood_mask: np.ndarray,
                      pixel_size_m: float = 30.0) -> Dict[str, Any]:
        """
        Analyze flood statistics from predictions.
        
        Args:
            depth: Predicted depth map
            flood_mask: Binary flood mask
            pixel_size_m: Pixel size in meters (default 30m for Sentinel-1)
            
        Returns:
            Dictionary with flood statistics
        """
        # Flooded pixels
        flooded_pixels = flood_mask.sum()
        total_pixels = flood_mask.size
        flood_fraction = flooded_pixels / total_pixels
        
        # Area calculation
        pixel_area_km2 = (pixel_size_m ** 2) / 1e6
        flooded_area_km2 = flooded_pixels * pixel_area_km2
        
        # Depth statistics (only in flooded areas)
        flooded_depths = depth[flood_mask > 0.5]
        
        if len(flooded_depths) > 0:
            depth_stats = {
                'mean': float(flooded_depths.mean()),
                'std': float(flooded_depths.std()),
                'min': float(flooded_depths.min()),
                'max': float(flooded_depths.max()),
                'median': float(np.median(flooded_depths))
            }
        else:
            depth_stats = {'mean': 0, 'std': 0, 'min': 0, 'max': 0, 'median': 0}
        
        # Depth categories
        depth_categories = {
            'shallow_0_1m': float(((flooded_depths > 0) & (flooded_depths <= 1)).sum() / max(len(flooded_depths), 1)),
            'moderate_1_2m': float(((flooded_depths > 1) & (flooded_depths <= 2)).sum() / max(len(flooded_depths), 1)),
            'deep_2_3m': float(((flooded_depths > 2) & (flooded_depths <= 3)).sum() / max(len(flooded_depths), 1)),
            'very_deep_3m_plus': float((flooded_depths > 3).sum() / max(len(flooded_depths), 1))
        }
        
        return {
            'flooded_pixels': int(flooded_pixels),
            'total_pixels': int(total_pixels),
            'flood_fraction': float(flood_fraction),
            'flooded_area_km2': float(flooded_area_km2),
            'depth_statistics': depth_stats,
            'depth_categories': depth_categories
        }


class FloodVisualization:
    """Generate report-ready flood visualizations"""
    
    # Custom colormap for depth
    DEPTH_COLORS = ['#FFFFFF', '#E3F2FD', '#90CAF9', '#42A5F5', '#1E88E5', '#1565C0', '#0D47A1']
    
    def __init__(self):
        self.depth_cmap = LinearSegmentedColormap.from_list('flood_depth', self.DEPTH_COLORS)
        plt.rcParams['figure.dpi'] = 150
        plt.rcParams['savefig.dpi'] = 300
    
    def plot_prediction_result(self, sar_vv: np.ndarray, depth: np.ndarray,
                                flood_mask: np.ndarray, save_path: Optional[str] = None,
                                title: str = "Flood Depth Estimation") -> None:
        """
        Create comprehensive prediction visualization.
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. SAR VV Band
        im0 = axes[0, 0].imshow(sar_vv, cmap='gray', vmin=-25, vmax=-5)
        axes[0, 0].set_title('(a) SAR VV Band', fontweight='bold')
        axes[0, 0].axis('off')
        plt.colorbar(im0, ax=axes[0, 0], label='Backscatter (dB)', shrink=0.8)
        
        # 2. Flood Detection
        im1 = axes[0, 1].imshow(flood_mask, cmap='Blues', vmin=0, vmax=1)
        axes[0, 1].set_title('(b) Flood Detection', fontweight='bold')
        axes[0, 1].axis('off')
        cbar1 = plt.colorbar(im1, ax=axes[0, 1], shrink=0.8)
        cbar1.set_ticks([0, 1])
        cbar1.set_ticklabels(['Dry', 'Flooded'])
        
        # 3. Predicted Depth
        im2 = axes[1, 0].imshow(depth, cmap=self.depth_cmap, vmin=0, vmax=5)
        axes[1, 0].set_title('(c) Predicted Flood Depth', fontweight='bold')
        axes[1, 0].axis('off')
        plt.colorbar(im2, ax=axes[1, 0], label='Depth (meters)', shrink=0.8)
        
        # 4. Masked Depth (only flooded areas)
        masked_depth = np.where(flood_mask > 0.5, depth, np.nan)
        im3 = axes[1, 1].imshow(masked_depth, cmap=self.depth_cmap, vmin=0, vmax=5)
        axes[1, 1].set_title('(d) Flood Depth (Masked)', fontweight='bold')
        axes[1, 1].axis('off')
        plt.colorbar(im3, ax=axes[1, 1], label='Depth (meters)', shrink=0.8)
        
        plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', facecolor='white')
            print(f"✓ Saved: {save_path}")
        
        plt.close()
    
    def plot_flood_analysis(self, analysis: Dict, save_path: Optional[str] = None,
                            location: str = "Study Area") -> None:
        """
        Create flood analysis summary figure.
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # 1. Depth categories pie chart
        categories = analysis['depth_categories']
        labels = ['Shallow\n(0-1m)', 'Moderate\n(1-2m)', 'Deep\n(2-3m)', 'Very Deep\n(>3m)']
        sizes = [categories['shallow_0_1m'], categories['moderate_1_2m'],
                 categories['deep_2_3m'], categories['very_deep_3m_plus']]
        colors = ['#90CAF9', '#42A5F5', '#1E88E5', '#0D47A1']
        
        # Filter out zero values
        non_zero = [(l, s, c) for l, s, c in zip(labels, sizes, colors) if s > 0]
        if non_zero:
            labels, sizes, colors = zip(*non_zero)
            axes[0].pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
                       startangle=90, explode=[0.02]*len(sizes))
        axes[0].set_title('Flood Depth Distribution', fontweight='bold')
        
        # 2. Statistics summary
        axes[1].axis('off')
        stats = analysis['depth_statistics']
        
        summary_text = f"""
╔════════════════════════════════════════════╗
║        FLOOD ANALYSIS REPORT               ║
║        {location:^30}       ║
╠════════════════════════════════════════════╣
║                                            ║
║  FLOOD EXTENT                              ║
║  ────────────                              ║
║  • Flooded Area: {analysis['flooded_area_km2']:.2f} km²               ║
║  • Flood Coverage: {analysis['flood_fraction']*100:.1f}%                  ║
║  • Flooded Pixels: {analysis['flooded_pixels']:,}              ║
║                                            ║
║  DEPTH STATISTICS                          ║
║  ────────────────                          ║
║  • Mean Depth: {stats['mean']:.2f} m                    ║
║  • Max Depth: {stats['max']:.2f} m                     ║
║  • Median Depth: {stats['median']:.2f} m                 ║
║  • Std Deviation: {stats['std']:.2f} m                 ║
║                                            ║
║  RISK ASSESSMENT                           ║
║  ───────────────                           ║
║  • Shallow (<1m): {analysis['depth_categories']['shallow_0_1m']*100:.1f}%                  ║
║  • Moderate (1-2m): {analysis['depth_categories']['moderate_1_2m']*100:.1f}%                ║
║  • Deep (2-3m): {analysis['depth_categories']['deep_2_3m']*100:.1f}%                    ║
║  • Very Deep (>3m): {analysis['depth_categories']['very_deep_3m_plus']*100:.1f}%                ║
║                                            ║
╚════════════════════════════════════════════╝
        """
        
        axes[1].text(0.5, 0.5, summary_text, transform=axes[1].transAxes,
                    fontsize=10, fontfamily='monospace',
                    verticalalignment='center', horizontalalignment='center',
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', facecolor='white')
            print(f"✓ Saved: {save_path}")
        
        plt.close()


def generate_synthetic_sar(size: int = 32, flood_fraction: float = 0.3,
                           seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic SAR data for testing/demo.
    
    Returns:
        Tuple of (sar_vv, sar_vh) arrays
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Generate terrain
    terrain = np.random.randn(size, size)
    terrain = ndimage.gaussian_filter(terrain, sigma=size/8)
    terrain = (terrain - terrain.min()) / (terrain.max() - terrain.min()) * 100
    
    # Create flood mask
    flood_threshold = np.percentile(terrain, flood_fraction * 100)
    flood_mask = (terrain < flood_threshold).astype(np.float32)
    flood_mask = ndimage.gaussian_filter(flood_mask, sigma=2)
    flood_mask = (flood_mask > 0.5).astype(np.float32)
    
    # Generate SAR values
    vv_land = np.random.uniform(-12, -6, (size, size))
    vv_water = np.random.uniform(-22, -16, (size, size))
    vv = np.where(flood_mask > 0.5, vv_water, vv_land)
    vv = ndimage.gaussian_filter(vv + np.random.randn(size, size) * 1.5, sigma=1)
    
    vh = vv - np.random.uniform(3, 6, (size, size))
    vh = ndimage.gaussian_filter(vh + np.random.randn(size, size) * 1.5, sigma=1)
    
    return vv.astype(np.float32), vh.astype(np.float32)


def run_demo(model_dir: str, output_dir: Optional[str] = None) -> None:
    """
    Run complete inference demo with visualizations.
    """
    if output_dir is None:
        output_dir = os.path.join(model_dir, 'inference_results')
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print("  FLOOD DEPTH ESTIMATION - INFERENCE DEMO")
    print("=" * 60)
    
    # Load predictor
    model_path = os.path.join(model_dir, 'flood_depth_model.h5')
    predictor = FloodDepthPredictor(model_path)
    viz = FloodVisualization()
    
    # Generate test scenarios
    scenarios = [
        {"name": "Light Flooding", "flood_fraction": 0.15, "seed": 100},
        {"name": "Moderate Flooding", "flood_fraction": 0.30, "seed": 200},
        {"name": "Severe Flooding", "flood_fraction": 0.45, "seed": 300},
    ]
    
    all_results = []
    
    for i, scenario in enumerate(scenarios):
        print(f"\n[{i+1}/{len(scenarios)}] Processing: {scenario['name']}...")
        
        # Generate SAR data
        sar_vv, sar_vh = generate_synthetic_sar(
            size=32,
            flood_fraction=scenario['flood_fraction'],
            seed=scenario['seed']
        )
        
        # Run prediction
        result = predictor.predict_with_flood_mask(sar_vv, sar_vh)
        
        # Analyze flood
        analysis = predictor.analyze_flood(result['depth'], result['flood_mask'])
        analysis['scenario'] = scenario['name']
        all_results.append(analysis)
        
        # Generate visualizations
        viz.plot_prediction_result(
            sar_vv, result['depth'], result['flood_mask'],
            save_path=os.path.join(output_dir, f'prediction_{i+1}_{scenario["name"].lower().replace(" ", "_")}.png'),
            title=f"Flood Depth Estimation - {scenario['name']}"
        )
        
        viz.plot_flood_analysis(
            analysis,
            save_path=os.path.join(output_dir, f'analysis_{i+1}_{scenario["name"].lower().replace(" ", "_")}.png'),
            location=f"Scenario: {scenario['name']}"
        )
        
        print(f"  ✓ Flood coverage: {analysis['flood_fraction']*100:.1f}%")
        print(f"  ✓ Mean depth: {analysis['depth_statistics']['mean']:.2f}m")
    
    # Save results to JSON
    results_path = os.path.join(output_dir, 'inference_results.json')
    with open(results_path, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'model': model_path,
            'scenarios': all_results
        }, f, indent=2)
    
    print(f"\n✓ Results saved to: {results_path}")
    
    # Create comparison figure
    create_comparison_figure(all_results, output_dir)
    
    print("\n" + "=" * 60)
    print(f"  INFERENCE COMPLETE - Results in: {output_dir}")
    print("=" * 60)


def create_comparison_figure(results: list, output_dir: str) -> None:
    """Create scenario comparison figure for report"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    scenarios = [r['scenario'] for r in results]
    colors = ['#90CAF9', '#42A5F5', '#0D47A1']
    
    # 1. Flood Coverage Comparison
    coverages = [r['flood_fraction'] * 100 for r in results]
    bars1 = axes[0].bar(scenarios, coverages, color=colors, edgecolor='black')
    axes[0].set_ylabel('Flood Coverage (%)')
    axes[0].set_title('(a) Flood Extent Comparison', fontweight='bold')
    for bar, val in zip(bars1, coverages):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{val:.1f}%', ha='center', fontweight='bold')
    axes[0].set_ylim(0, max(coverages) * 1.2)
    
    # 2. Mean Depth Comparison
    depths = [r['depth_statistics']['mean'] for r in results]
    bars2 = axes[1].bar(scenarios, depths, color=colors, edgecolor='black')
    axes[1].set_ylabel('Mean Depth (m)')
    axes[1].set_title('(b) Mean Flood Depth', fontweight='bold')
    for bar, val in zip(bars2, depths):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f'{val:.2f}m', ha='center', fontweight='bold')
    axes[1].set_ylim(0, max(depths) * 1.3)
    
    # 3. Depth Category Distribution (stacked bar)
    shallow = [r['depth_categories']['shallow_0_1m'] * 100 for r in results]
    moderate = [r['depth_categories']['moderate_1_2m'] * 100 for r in results]
    deep = [r['depth_categories']['deep_2_3m'] * 100 for r in results]
    very_deep = [r['depth_categories']['very_deep_3m_plus'] * 100 for r in results]
    
    x = np.arange(len(scenarios))
    width = 0.6
    
    axes[2].bar(x, shallow, width, label='0-1m', color='#E3F2FD')
    axes[2].bar(x, moderate, width, bottom=shallow, label='1-2m', color='#90CAF9')
    axes[2].bar(x, deep, width, bottom=np.array(shallow)+np.array(moderate), label='2-3m', color='#42A5F5')
    axes[2].bar(x, very_deep, width, bottom=np.array(shallow)+np.array(moderate)+np.array(deep), label='>3m', color='#0D47A1')
    
    axes[2].set_ylabel('Percentage (%)')
    axes[2].set_title('(c) Depth Category Distribution', fontweight='bold')
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(scenarios)
    axes[2].legend(loc='upper right', title='Depth')
    
    plt.suptitle('Flood Scenario Comparison Analysis', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, 'scenario_comparison.png')
    plt.savefig(save_path, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✓ Saved: {save_path}")


if __name__ == "__main__":
    model_dir = r"D:\project\Ai agent\disaster-response-system\src\agents\agent_1_environmental\models"
    run_demo(model_dir)