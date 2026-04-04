"""
Depth Map Visualization
Create publication-quality visualizations
"""

import numpy as np
import matplotlib.pyplot as plt


class DepthVisualizer:
    """Create depth map visualizations"""
    
    def __init__(self):
        self.default_dpi = 150
    
    def plot_depth_analysis(self, sar_array, depth_map, severity_map, 
                           statistics=None, save_path=None):
        """
        Create 4-panel visualization
        
        Args:
            sar_array: SAR image [H, W, 2]
            depth_map: Depth map [H, W, 1]
            severity_map: Severity [H, W, 1]
            statistics: dict with stats
            save_path: Path to save figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        
        # 1. SAR Image (VV band)
        ax = axes[0, 0]
        im1 = ax.imshow(sar_array[:, :, 0], cmap='gray')
        ax.set_title('Sentinel-1 SAR (VV Band)', fontsize=12, fontweight='bold')
        ax.axis('off')
        plt.colorbar(im1, ax=ax, fraction=0.046)
        
        # 2. Flood Mask
        ax = axes[0, 1]
        flood_mask = (depth_map[:, :, 0] > 0.1).astype(float)
        im2 = ax.imshow(flood_mask, cmap='Blues', vmin=0, vmax=1)
        ax.set_title('Flood Detection', fontsize=12, fontweight='bold')
        ax.axis('off')
        plt.colorbar(im2, ax=ax, fraction=0.046, label='Flooded')
        
        # 3. Depth Map
        ax = axes[1, 0]
        depth_display = np.ma.masked_where(
            depth_map[:, :, 0] == 0, 
            depth_map[:, :, 0]
        )
        im3 = ax.imshow(depth_display, cmap='YlOrRd', vmin=0, vmax=5)
        ax.set_title('Estimated Flood Depth', fontsize=12, fontweight='bold')
        ax.axis('off')
        plt.colorbar(im3, ax=ax, fraction=0.046, label='Depth (meters)')
        
        # 4. Severity Map
        ax = axes[1, 1]
        im4 = ax.imshow(severity_map[:, :, 0], cmap='RdYlGn_r', vmin=0, vmax=3)
        ax.set_title('Flood Severity', fontsize=12, fontweight='bold')
        ax.axis('off')
        cbar = plt.colorbar(im4, ax=ax, fraction=0.046, ticks=[0, 1, 2, 3])
        cbar.ax.set_yticklabels(['None', 'Low', 'Medium', 'High'])
        
        # Add statistics
        if statistics:
            stats_text = (
                f"Flood Statistics:\n"
                f"  Area: {statistics['flood_area_percent']:.1f}%\n"
                f"  Mean depth: {statistics['mean_depth_m']:.2f} m\n"
                f"  Max depth: {statistics['max_depth_m']:.2f} m"
            )
            fig.text(0.5, 0.02, stats_text, ha='center', fontsize=11,
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout(rect=[0, 0.05, 1, 1])
        
        if save_path:
            plt.savefig(save_path, dpi=self.default_dpi, bbox_inches='tight')
            print(f"✓ Visualization saved to {save_path}")
        
        plt.show()
        return fig
    
    def plot_depth_only(self, depth_map, title='Flood Depth Map', 
                        save_path=None):
        """
        Plot depth map only
        
        Args:
            depth_map: Depth array [H, W, 1] or [H, W]
            title: Plot title
            save_path: Path to save
        """
        if len(depth_map.shape) == 3:
            depth_map = depth_map[:, :, 0]
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Mask zero values
        depth_display = np.ma.masked_where(depth_map == 0, depth_map)
        
        im = ax.imshow(depth_display, cmap='YlOrRd', vmin=0, vmax=5)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.axis('off')
        
        cbar = plt.colorbar(im, ax=ax, fraction=0.046)
        cbar.set_label('Depth (meters)', fontsize=12)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.default_dpi, bbox_inches='tight')
            print(f"✓ Depth map saved to {save_path}")
        
        plt.show()
        return fig


def visualize_results(results, save_prefix='flood_analysis'):
    """
    Quick visualization function
    
    Args:
        results: dict from FloodAnalyzer.analyze_region()
        save_prefix: Prefix for saved files
    """
    viz = DepthVisualizer()
    
    if 'depth_map' in results:
        viz.plot_depth_only(
            results['depth_map'],
            save_path=f'{save_prefix}_depth.png'
        )
    
    if 'severity' in results:
        severity = results['severity']
        if len(severity.shape) == 3:
            severity = severity[:, :, 0]
        
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(severity, cmap='RdYlGn_r', vmin=0, vmax=3)
        ax.set_title('Flood Severity', fontsize=14, fontweight='bold')
        ax.axis('off')
        
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, ticks=[0, 1, 2, 3])
        cbar.ax.set_yticklabels(['No Flood', 'Low (<1m)', 'Medium (1-2m)', 'High (>2m)'])
        
        plt.tight_layout()
        plt.savefig(f'{save_prefix}_severity.png', dpi=150, bbox_inches='tight')
        print(f"✓ Severity map saved to {save_prefix}_severity.png")
        plt.show()