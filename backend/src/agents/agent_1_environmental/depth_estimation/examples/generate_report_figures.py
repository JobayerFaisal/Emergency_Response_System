"""
Generate Training Visualizations for Report
Creates publication-ready figures including loss curves
"""

import numpy as np
import matplotlib.pyplot as plt
import json
import os
from typing import Optional, Dict, Any
from tensorflow import keras
from scipy import ndimage

# Set style for publication-quality figures
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 11


def plot_loss_curves(history: Dict[str, list], save_path: str) -> None:
    """Plot training and validation loss curves"""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    
    epochs = range(1, len(history['loss']) + 1)
    
    # Loss
    axes[0].plot(epochs, history['loss'], 'b-o', linewidth=2, markersize=4, label='Training')
    axes[0].plot(epochs, history['val_loss'], 'r-s', linewidth=2, markersize=4, label='Validation')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss (MSE)')
    axes[0].set_title('(a) Training & Validation Loss', fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # MAE
    axes[1].plot(epochs, history['mae'], 'b-o', linewidth=2, markersize=4, label='Training')
    axes[1].plot(epochs, history['val_mae'], 'r-s', linewidth=2, markersize=4, label='Validation')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('MAE (meters)')
    axes[1].set_title('(b) Mean Absolute Error', fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # RMSE
    axes[2].plot(epochs, history['rmse'], 'b-o', linewidth=2, markersize=4, label='Training')
    axes[2].plot(epochs, history['val_rmse'], 'r-s', linewidth=2, markersize=4, label='Validation')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('RMSE (meters)')
    axes[2].set_title('(c) Root Mean Square Error', fontweight='bold')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.suptitle('Flood Depth CNN - Training Progress', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✓ Saved: {save_path}")


def plot_loss_curve_single(history: Dict[str, list], save_path: str) -> None:
    """Plot single detailed loss curve"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    epochs = range(1, len(history['loss']) + 1)
    
    ax.plot(epochs, history['loss'], 'b-o', linewidth=2, markersize=5, label='Training Loss')
    ax.plot(epochs, history['val_loss'], 'r-s', linewidth=2, markersize=5, label='Validation Loss')
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss (MSE)', fontsize=12)
    ax.set_title('Flood Depth Estimation CNN - Training Loss Curve', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Add annotations for min validation loss
    min_val_idx = int(np.argmin(history['val_loss']))
    min_val = float(history['val_loss'][min_val_idx])
    ax.annotate(f'Best: {min_val:.4f}\n(Epoch {min_val_idx+1})',
                xy=(float(min_val_idx+1), min_val),
                xytext=(min_val_idx+3, min_val+0.1),
                fontsize=10,
                arrowprops=dict(arrowstyle='->', color='red'),
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Final values annotation
    final_train = history['loss'][-1]
    final_val = history['val_loss'][-1]
    ax.annotate(f'Final Train: {final_train:.4f}\nFinal Val: {final_val:.4f}',
                xy=(0.98, 0.98), xycoords='axes fraction',
                ha='right', va='top', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✓ Saved: {save_path}")


def plot_model_architecture(save_path: str) -> None:
    """Create a visual representation of model architecture"""
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('off')
    
    arch_text = """
    ┌─────────────────────────────────────────────────────────────────┐
    │                LIGHTWEIGHT DEPTH CNN ARCHITECTURE                │
    │                    (U-Net Style, ~58K Parameters)                │
    ├─────────────────────────────────────────────────────────────────┤
    │                                                                  │
    │   INPUT: SAR Image (32 × 32 × 2)  [VV, VH bands]                │
    │      │                                                           │
    │      ▼                                                           │
    │   ┌─────────────────────────────────────────┐                   │
    │   │  ENCODER                                 │                   │
    │   │  ├── Conv2D(16, 3×3) + BatchNorm + ReLU │ ──── Skip 1       │
    │   │  ├── MaxPool2D(2×2)                      │                   │
    │   │  ├── Conv2D(32, 3×3) + BatchNorm + ReLU │ ──── Skip 2       │
    │   │  └── MaxPool2D(2×2)                      │                   │
    │   └─────────────────────────────────────────┘                   │
    │      │                                                           │
    │      ▼                                                           │
    │   ┌─────────────────────────────────────────┐                   │
    │   │  BOTTLENECK                              │                   │
    │   │  └── Conv2D(64, 3×3) + BatchNorm + ReLU │                   │
    │   └─────────────────────────────────────────┘                   │
    │      │                                                           │
    │      ▼                                                           │
    │   ┌─────────────────────────────────────────┐                   │
    │   │  DECODER                                 │                   │
    │   │  ├── UpSample2D(2×2) + Concat(Skip 2)   │                   │
    │   │  ├── Conv2D(32, 3×3) + BatchNorm + ReLU │                   │
    │   │  ├── UpSample2D(2×2) + Concat(Skip 1)   │                   │
    │   │  └── Conv2D(16, 3×3) + BatchNorm + ReLU │                   │
    │   └─────────────────────────────────────────┘                   │
    │      │                                                           │
    │      ▼                                                           │
    │   ┌─────────────────────────────────────────┐                   │
    │   │  OUTPUT                                  │                   │
    │   │  └── Conv2D(1, 1×1) + ReLU              │                   │
    │   └─────────────────────────────────────────┘                   │
    │      │                                                           │
    │      ▼                                                           │
    │   OUTPUT: Depth Map (32 × 32 × 1)  [meters]                     │
    │                                                                  │
    ├─────────────────────────────────────────────────────────────────┤
    │  Total Parameters: 58,641  |  Optimized for CPU Training        │
    │  Loss: MSE  |  Metrics: MAE, RMSE  |  Optimizer: Adam           │
    └─────────────────────────────────────────────────────────────────┘
    """
    
    ax.text(0.5, 0.5, arch_text, transform=ax.transAxes,
            fontsize=10, fontfamily='monospace',
            verticalalignment='center', horizontalalignment='center',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
    
    plt.title('Flood Depth Estimation CNN Architecture', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✓ Saved: {save_path}")


def plot_final_metrics(info: Dict[str, Any], save_path: str) -> None:
    """Create bar chart of final metrics"""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    metrics = info['final_metrics']
    
    names = ['Validation Loss\n(MSE)', 'Validation MAE\n(meters)', 'Validation RMSE\n(meters)']
    values = [metrics['val_loss'], metrics['val_mae'], metrics['val_rmse']]
    colors = ['#3498db', '#2ecc71', '#9b59b6']
    
    bars = ax.bar(names, values, color=colors, edgecolor='black', linewidth=1.2)
    
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.3f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax.set_ylabel('Value', fontsize=12)
    ax.set_title('Flood Depth Estimation Model - Final Performance Metrics', 
                 fontsize=14, fontweight='bold')
    ax.set_ylim(0, max(values) * 1.2)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✓ Saved: {save_path}")


def plot_training_config(info: Dict[str, Any], save_path: str) -> None:
    """Create summary figure with training configuration"""
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.axis('off')
    
    config = info['config']
    metrics = info['final_metrics']
    
    # Check if history exists for epoch count
    if 'history' in info:
        epochs_trained = len(info['history']['loss'])
    else:
        epochs_trained = config.get('epochs', 20)
    
    summary_text = f"""
    ╔══════════════════════════════════════════════════════════════╗
    ║          FLOOD DEPTH ESTIMATION - TRAINING REPORT            ║
    ╠══════════════════════════════════════════════════════════════╣
    ║                                                              ║
    ║  MODEL ARCHITECTURE                                          ║
    ║  ─────────────────                                           ║
    ║  • Type: Lightweight U-Net CNN                               ║
    ║  • Parameters: 58,641                                        ║
    ║  • Input: {config['patch_size']} × {config['patch_size']} × 2 (SAR VV, VH bands)               ║
    ║  • Output: {config['patch_size']} × {config['patch_size']} × 1 (Depth in meters)              ║
    ║                                                              ║
    ╠══════════════════════════════════════════════════════════════╣
    ║                                                              ║
    ║  TRAINING CONFIGURATION                                      ║
    ║  ──────────────────────                                      ║
    ║  • Training Samples: {config['n_train_samples']}                                     ║
    ║  • Validation Samples: {config['n_val_samples']}                                   ║
    ║  • Patch Size: {config['patch_size']} × {config['patch_size']} pixels                            ║
    ║  • Epochs Trained: {epochs_trained}                                      ║
    ║  • Batch Size: {config['batch_size']}                                        ║
    ║  • Learning Rate: {config['learning_rate']}                                   ║
    ║  • Optimizer: Adam                                           ║
    ║  • Loss Function: Mean Squared Error (MSE)                   ║
    ║                                                              ║
    ╠══════════════════════════════════════════════════════════════╣
    ║                                                              ║
    ║  STUDY AREA                                                  ║
    ║  ──────────                                                  ║
    ║  • Region: Sylhet, Bangladesh                                ║
    ║  • Coordinates: [{config['aoi_coords'][0]}, {config['aoi_coords'][1]}] to [{config['aoi_coords'][2]}, {config['aoi_coords'][3]}]           ║
    ║  • Date Range: {config['start_date']} to {config['end_date']}          ║
    ║                                                              ║
    ╠══════════════════════════════════════════════════════════════╣
    ║                                                              ║
    ║  FINAL PERFORMANCE METRICS                                   ║
    ║  ─────────────────────────                                   ║
    ║  • Validation Loss (MSE): {metrics['val_loss']:.4f}                        ║
    ║  • Validation MAE: {metrics['val_mae']:.3f} meters                       ║
    ║  • Validation RMSE: {metrics['val_rmse']:.3f} meters                      ║
    ║                                                              ║
    ║  Training Time: {info['training_time_minutes']:.2f} minutes                          ║
    ║  Trained: {info['trained_at'][:10]}                                  ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    
    ax.text(0.5, 0.5, summary_text, transform=ax.transAxes,
            fontsize=10, fontfamily='monospace',
            verticalalignment='center', horizontalalignment='center',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    plt.title('Training Summary Report', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✓ Saved: {save_path}")


def plot_sample_predictions(model_path: str, save_path: str, n_samples: int = 4) -> None:
    """Generate sample predictions and compare with ground truth"""
    
    model = keras.models.load_model(model_path, compile=False)
    np.random.seed(123)
    
    fig, axes = plt.subplots(n_samples, 4, figsize=(14, 3.5 * n_samples))
    
    for i in range(n_samples):
        size = 32
        flood_fraction = np.random.uniform(0.2, 0.4)
        
        terrain = np.random.randn(size, size)
        terrain = ndimage.gaussian_filter(terrain, sigma=size/8)
        terrain = (terrain - terrain.min()) / (terrain.max() - terrain.min()) * 100
        
        flood_threshold = np.percentile(terrain, flood_fraction * 100)
        flood_mask = (terrain < flood_threshold).astype(np.float32)
        flood_mask = ndimage.gaussian_filter(flood_mask, sigma=2)
        flood_mask = (flood_mask > 0.5).astype(np.float32)
        
        vv_land = np.random.uniform(-12, -6, (size, size))
        vv_water = np.random.uniform(-22, -16, (size, size))
        vv = np.where(flood_mask > 0.5, vv_water, vv_land)
        vv = ndimage.gaussian_filter(vv + np.random.randn(size, size) * 1.5, sigma=1)
        vh = vv - np.random.uniform(3, 6, (size, size))
        vh = ndimage.gaussian_filter(vh + np.random.randn(size, size) * 1.5, sigma=1)
        
        sar = np.stack([vv, vh], axis=-1).astype(np.float32)
        
        distance = np.asarray(ndimage.distance_transform_edt(flood_mask), dtype=np.float32)
        max_dist = float(distance.max())
        if max_dist > 0:
            gt_depth = (distance / max_dist) * 5.0
        else:
            gt_depth = np.zeros_like(distance)
        gt_depth = gt_depth * flood_mask
        
        sar_input = sar[np.newaxis, ...]
        sar_norm = (sar_input - sar_input.mean()) / (sar_input.std() + 1e-7)
        pred_depth = model.predict(sar_norm, verbose=0)[0, :, :, 0]
        
        im0 = axes[i, 0].imshow(vv, cmap='gray', vmin=-25, vmax=-5)
        axes[i, 0].set_title('SAR VV Band (dB)' if i == 0 else '')
        axes[i, 0].axis('off')
        if i == n_samples - 1:
            plt.colorbar(im0, ax=axes[i, 0], fraction=0.046)
        
        im1 = axes[i, 1].imshow(gt_depth, cmap='Blues', vmin=0, vmax=5)
        axes[i, 1].set_title('Ground Truth Depth' if i == 0 else '')
        axes[i, 1].axis('off')
        if i == n_samples - 1:
            plt.colorbar(im1, ax=axes[i, 1], fraction=0.046, label='Depth (m)')
        
        im2 = axes[i, 2].imshow(pred_depth, cmap='Blues', vmin=0, vmax=5)
        axes[i, 2].set_title('Predicted Depth' if i == 0 else '')
        axes[i, 2].axis('off')
        if i == n_samples - 1:
            plt.colorbar(im2, ax=axes[i, 2], fraction=0.046, label='Depth (m)')
        
        error = np.abs(pred_depth - gt_depth)
        mae = error.mean()
        im3 = axes[i, 3].imshow(error, cmap='Reds', vmin=0, vmax=2)
        axes[i, 3].set_title(f'Absolute Error (MAE={mae:.2f}m)' if i == 0 else f'MAE={mae:.2f}m')
        axes[i, 3].axis('off')
        if i == n_samples - 1:
            plt.colorbar(im3, ax=axes[i, 3], fraction=0.046, label='Error (m)')
        
        axes[i, 0].set_ylabel(f'Sample {i+1}', fontsize=11, fontweight='bold')
    
    plt.suptitle('Flood Depth Estimation: Model Predictions vs Ground Truth', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✓ Saved: {save_path}")


def plot_depth_distribution(model_path: str, save_path: str) -> None:
    """Plot distribution of predicted depths"""
    model = keras.models.load_model(model_path, compile=False)
    
    np.random.seed(42)
    all_gt_depths = []
    all_pred_depths = []
    
    for _ in range(20):
        size = 32
        flood_fraction = np.random.uniform(0.2, 0.4)
        
        terrain = np.random.randn(size, size)
        terrain = ndimage.gaussian_filter(terrain, sigma=size/8)
        terrain = (terrain - terrain.min()) / (terrain.max() - terrain.min()) * 100
        
        flood_threshold = np.percentile(terrain, flood_fraction * 100)
        flood_mask = (terrain < flood_threshold).astype(np.float32)
        flood_mask = ndimage.gaussian_filter(flood_mask, sigma=2)
        flood_mask = (flood_mask > 0.5).astype(np.float32)
        
        vv_land = np.random.uniform(-12, -6, (size, size))
        vv_water = np.random.uniform(-22, -16, (size, size))
        vv = np.where(flood_mask > 0.5, vv_water, vv_land)
        vv = ndimage.gaussian_filter(vv + np.random.randn(size, size) * 1.5, sigma=1)
        vh = vv - np.random.uniform(3, 6, (size, size))
        vh = ndimage.gaussian_filter(vh + np.random.randn(size, size) * 1.5, sigma=1)
        
        sar = np.stack([vv, vh], axis=-1).astype(np.float32)
        
        distance = np.asarray(ndimage.distance_transform_edt(flood_mask), dtype=np.float32)
        max_dist = float(distance.max())
        if max_dist > 0:
            gt_depth = (distance / max_dist) * 5.0
        else:
            gt_depth = np.zeros_like(distance)
        gt_depth = gt_depth * flood_mask
        
        sar_input = sar[np.newaxis, ...]
        sar_norm = (sar_input - sar_input.mean()) / (sar_input.std() + 1e-7)
        pred_depth = model.predict(sar_norm, verbose=0)[0, :, :, 0]
        
        mask = flood_mask > 0.5
        all_gt_depths.extend(gt_depth[mask].flatten())
        all_pred_depths.extend(pred_depth[mask].flatten())
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    axes[0].hist(all_gt_depths, bins=30, alpha=0.7, label='Ground Truth', color='blue')
    axes[0].hist(all_pred_depths, bins=30, alpha=0.7, label='Predicted', color='red')
    axes[0].set_xlabel('Depth (meters)')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Distribution of Flood Depths')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].scatter(all_gt_depths, all_pred_depths, alpha=0.3, s=5)
    axes[1].plot([0, 5], [0, 5], 'r--', linewidth=2, label='Perfect Prediction')
    axes[1].set_xlabel('Ground Truth Depth (m)')
    axes[1].set_ylabel('Predicted Depth (m)')
    axes[1].set_title('Predicted vs Ground Truth')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlim(0, 5)
    axes[1].set_ylim(0, 5)
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✓ Saved: {save_path}")


def generate_all_figures(model_dir: str, output_dir: Optional[str] = None) -> None:
    """Generate all figures for report"""
    
    if output_dir is None:
        output_dir = os.path.join(model_dir, 'figures')
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print("  GENERATING REPORT FIGURES")
    print("=" * 60)
    
    info_path = os.path.join(model_dir, 'flood_depth_model_info.json')
    model_path = os.path.join(model_dir, 'flood_depth_model.h5')
    
    if not os.path.exists(info_path):
        print(f"ERROR: {info_path} not found!")
        return
    
    with open(info_path, 'r') as f:
        info = json.load(f)
    
    print("\n1. Generating model architecture diagram...")
    plot_model_architecture(os.path.join(output_dir, 'model_architecture.png'))
    
    print("2. Generating final metrics chart...")
    plot_final_metrics(info, os.path.join(output_dir, 'final_metrics.png'))
    
    print("3. Generating training summary...")
    plot_training_config(info, os.path.join(output_dir, 'training_summary.png'))
    
    # NEW: Generate loss curves if history is available
    if 'history' in info:
        print("4. Generating loss curves...")
        plot_loss_curves(info['history'], os.path.join(output_dir, 'loss_curves.png'))
        
        print("5. Generating detailed loss curve...")
        plot_loss_curve_single(info['history'], os.path.join(output_dir, 'loss_curve_detailed.png'))
    else:
        print("4. Skipping loss curves (no history in JSON - retrain to generate)")
    
    if os.path.exists(model_path):
        print("6. Generating sample predictions...")
        plot_sample_predictions(model_path, os.path.join(output_dir, 'sample_predictions.png'))
        
        print("7. Generating depth distribution analysis...")
        plot_depth_distribution(model_path, os.path.join(output_dir, 'depth_distribution.png'))
    else:
        print(f"WARNING: Model file not found: {model_path}")
    
    print("\n" + "=" * 60)
    print(f"  ALL FIGURES SAVED TO: {output_dir}")
    print("=" * 60)
    print("\nGenerated files:")
    for f in sorted(os.listdir(output_dir)):
        if f.endswith('.png'):
            fpath = os.path.join(output_dir, f)
            size_kb = os.path.getsize(fpath) / 1024
            print(f"  - {f} ({size_kb:.1f} KB)")


if __name__ == "__main__":
    model_dir = r"D:\project\Ai agent\disaster-response-system\src\agents\agent_1_environmental\models"
    generate_all_figures(model_dir)