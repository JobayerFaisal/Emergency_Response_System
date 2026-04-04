"""
Flood Detection with CHANGE DETECTION - FIXED VERSION
"""

import ee
import numpy as np
import os
import json
from datetime import datetime
import time

ee.Initialize(project='caramel-pulsar-475810-e7')

import tensorflow as tf
from tensorflow import keras

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

print("=" * 60)
print("🌊 FLOOD DETECTION - CHANGE DETECTION APPROACH")
print("=" * 60)

# ============================================================
# CUSTOM METRICS & LOSS - MUST BE DEFINED BEFORE LOADING MODEL
# ============================================================

def dice_coef(y_true, y_pred):
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (2. * intersection + 1) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + 1)

def dice_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)


# ============================================================
# FLOOD-PRONE TEST AREAS
# ============================================================

FLOOD_PRONE_AREAS = {
    'sylhet_surma_river': {
        'bounds': [91.85, 24.88, 91.95, 24.98],
        'name': 'Sylhet - Surma River Basin',
        'description': 'Major river, floods annually'
    },
    'sunamganj_haor': {
        'bounds': [91.20, 25.00, 91.35, 25.12],
        'name': 'Sunamganj Haor Region',
        'description': 'Wetland area, extremely flood-prone'
    },
    'sirajganj_jamuna': {
        'bounds': [89.65, 24.40, 89.80, 24.55],
        'name': 'Sirajganj - Jamuna River',
        'description': 'Major river floodplain'
    },
    'kurigram_brahmaputra': {
        'bounds': [89.60, 25.70, 89.75, 25.85],
        'name': 'Kurigram - Brahmaputra Basin',
        'description': 'Northern flood zone'
    },
    'chandpur_confluence': {
        'bounds': [90.60, 23.20, 90.75, 23.35],
        'name': 'Chandpur - River Confluence',
        'description': 'Padma-Meghna confluence'
    },
}

MODEL_PATH = 'models/flood_fast_best.keras'
OUTPUT_DIR = 'results/flood_detection'


# ============================================================
# SAR DOWNLOAD FUNCTIONS
# ============================================================

def download_sar_image(bounds, start_date, end_date):
    """Download Sentinel-1 SAR image"""
    
    geometry = ee.Geometry.Rectangle(bounds)
    
    collection = (ee.ImageCollection('COPERNICUS/S1_GRD')
        .filterBounds(geometry)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
        .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
        .filter(ee.Filter.eq('instrumentMode', 'IW'))
        .select(['VV', 'VH']))
    
    count = collection.size().getInfo()
    if count == 0:
        return None, None
    
    image = collection.median()
    
    first_img = collection.sort('system:time_start', False).first()
    timestamp = first_img.get('system:time_start').getInfo()
    acq_date = datetime.fromtimestamp(timestamp / 1000).strftime('%Y-%m-%d')
    
    try:
        sample = image.sampleRectangle(region=geometry, defaultValue=-999).getInfo()
        vv = np.array(sample['properties']['VV'])
        vh = np.array(sample['properties']['VH'])
        vv[vv == -999] = np.nan
        vh[vh == -999] = np.nan
        return np.stack([vv, vh], axis=-1), acq_date
    except Exception as e:
        print(f"      Error: {str(e)[:50]}")
        return None, None


def preprocess_image(sar_image, target_size=64):
    """Preprocess for model input"""
    
    if sar_image is None:
        return None
    
    from scipy.ndimage import zoom
    
    sar_image = np.nan_to_num(sar_image, nan=-15.0)
    
    h, w = sar_image.shape[:2]
    resized = zoom(sar_image, (target_size/h, target_size/w, 1), order=1)
    
    normalized = (resized + 35) / 35
    normalized = np.clip(np.array(normalized), 0, 1)
    
    return normalized.astype(np.float32)


# ============================================================
# CHANGE DETECTION
# ============================================================

def detect_flood_change(model, reference_img, current_img):
    """Detect floods using change detection"""
    
    ref_pred = model.predict(np.expand_dims(reference_img, 0), verbose=0)[0, :, :, 0]
    cur_pred = model.predict(np.expand_dims(current_img, 0), verbose=0)[0, :, :, 0]
    
    ref_water = (ref_pred > 0.5).astype(np.float32)
    cur_water = (cur_pred > 0.5).astype(np.float32)
    
    flood_mask = np.maximum(0, cur_water - ref_water)
    vv_change = current_img[:, :, 0] - reference_img[:, :, 0]
    
    return {
        'ref_water': ref_pred,
        'cur_water': cur_pred,
        'flood_mask': flood_mask,
        'vv_change': vv_change
    }


def analyze_flood(detection):
    """Analyze detection results"""
    
    flood = detection['flood_mask']
    total = flood.size
    flooded = (flood > 0.5).sum()
    pct = (flooded / total) * 100
    
    perm_water = (detection['ref_water'] > 0.5).sum() / total * 100
    cur_water = (detection['cur_water'] > 0.5).sum() / total * 100
    
    if pct > 25:
        status, risk = "🔴 SEVERE FLOODING", "CRITICAL"
    elif pct > 15:
        status, risk = "🟠 SIGNIFICANT FLOODING", "HIGH"
    elif pct > 8:
        status, risk = "🟡 MODERATE FLOODING", "MEDIUM"
    elif pct > 3:
        status, risk = "🟡 MINOR FLOODING", "LOW"
    else:
        status, risk = "🟢 NO SIGNIFICANT FLOODING", "MINIMAL"
    
    return {
        'flood_pct': pct,
        'perm_water_pct': perm_water,
        'cur_water_pct': cur_water,
        'status': status,
        'risk': risk
    }


# ============================================================
# VISUALIZATION
# ============================================================

def visualize_results(area_name, ref_img, cur_img, detection, analysis, 
                      ref_date, cur_date, output_path):
    """Create visualization"""
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f'Flood Detection: {area_name}\nRef: {ref_date} → Current: {cur_date}', 
                 fontsize=14, fontweight='bold')
    
    ref_vv = ref_img[:, :, 0] * 35 - 35
    axes[0, 0].imshow(ref_vv, cmap='gray', vmin=-25, vmax=-5)
    axes[0, 0].set_title(f'Reference (Dry)\n{ref_date}')
    axes[0, 0].axis('off')
    
    cur_vv = cur_img[:, :, 0] * 35 - 35
    axes[0, 1].imshow(cur_vv, cmap='gray', vmin=-25, vmax=-5)
    axes[0, 1].set_title(f'Current (Monsoon)\n{cur_date}')
    axes[0, 1].axis('off')
    
    vv_change = detection['vv_change'] * 35
    im3 = axes[0, 2].imshow(vv_change, cmap='RdBu', vmin=-10, vmax=10)
    axes[0, 2].set_title('SAR Change\n(Blue=Water Increase)')
    axes[0, 2].axis('off')
    plt.colorbar(im3, ax=axes[0, 2], fraction=0.046)
    
    axes[1, 0].imshow(detection['ref_water'], cmap='Blues', vmin=0, vmax=1)
    axes[1, 0].set_title(f'Permanent Water\n({analysis["perm_water_pct"]:.1f}%)')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(detection['flood_mask'], cmap='Reds', vmin=0, vmax=1)
    axes[1, 1].set_title(f'NEW Flooding\n({analysis["flood_pct"]:.1f}%)')
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(cur_vv, cmap='gray', vmin=-25, vmax=-5)
    flood_overlay = np.ma.masked_where(detection['flood_mask'] < 0.5, detection['flood_mask'])
    axes[1, 2].imshow(flood_overlay, cmap='Reds', alpha=0.7)
    axes[1, 2].contour(detection['ref_water'], levels=[0.5], colors='blue', linewidths=1)
    axes[1, 2].set_title(f'{analysis["status"]}\nRisk: {analysis["risk"]}')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"      📊 Saved: {output_path}")


# ============================================================
# MAIN
# ============================================================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load model with ALL custom objects
    print("\n📦 Loading model...")
    try:
        model = keras.models.load_model(
            MODEL_PATH, 
            custom_objects={
                'dice_coef': dice_coef,
                'dice_loss': dice_loss
            }
        )
        print(f"   ✅ Model loaded successfully!")
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return
    
    ref_start, ref_end = '2024-01-01', '2024-03-31'
    cur_start, cur_end = '2024-06-01', '2024-09-30'
    
    print(f"\n📅 Reference (Dry): {ref_start} to {ref_end}")
    print(f"📅 Current (Monsoon): {cur_start} to {cur_end}")
    
    results = []
    
    print("\n" + "=" * 60)
    print("🛰️ TESTING FLOOD-PRONE AREAS")
    print("=" * 60)
    
    for area_id, area_info in FLOOD_PRONE_AREAS.items():
        print(f"\n📍 {area_info['name']}")
        
        print("   📥 Downloading reference...")
        ref_raw, ref_date = download_sar_image(area_info['bounds'], ref_start, ref_end)
        if ref_raw is None:
            print("   ⚠️ No reference data")
            continue
        
        time.sleep(0.3)
        
        print("   📥 Downloading current...")
        cur_raw, cur_date = download_sar_image(area_info['bounds'], cur_start, cur_end)
        if cur_raw is None:
            print("   ⚠️ No current data")
            continue
        
        ref_proc = preprocess_image(ref_raw, 64)
        cur_proc = preprocess_image(cur_raw, 64)
        
        print("   🔍 Running detection...")
        detection = detect_flood_change(model, ref_proc, cur_proc)
        
        analysis = analyze_flood(detection)
        
        print(f"   {analysis['status']}")
        print(f"   Flood: {analysis['flood_pct']:.1f}%")
        
        output_path = os.path.join(OUTPUT_DIR, f'{area_id}.png')
        visualize_results(
            area_info['name'], ref_proc, cur_proc,
            detection, analysis, ref_date, cur_date, output_path
        )
        
        results.append({
            'area': area_info['name'],
            'flood_pct': analysis['flood_pct'],
            'risk': analysis['risk'],
            'status': analysis['status']
        })
        
        time.sleep(0.3)
    
    print("\n" + "=" * 60)
    print("📊 SUMMARY")
    print("=" * 60)
    print(f"\n{'Area':<35} {'Flood %':<10} {'Risk'}")
    print("-" * 60)
    for r in results:
        print(f"{r['area']:<35} {r['flood_pct']:<10.1f} {r['risk']}")
    
    with open(os.path.join(OUTPUT_DIR, 'summary.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n📁 Results saved to: {OUTPUT_DIR}/")


if __name__ == "__main__":
    from scipy.ndimage import zoom
    main()
    print("\n🎉 Done!")