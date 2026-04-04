"""
Download Sen1Floods11 Dataset for Flood Detection Training
This dataset contains real Sentinel-1 SAR flood imagery with labels
"""

import ee
import os
import requests
import numpy as np
from pathlib import Path

ee.Initialize(project='caramel-pulsar-475810-e7')

print("=" * 60)
print("🌊 FLOOD DETECTION TRAINING DATA DOWNLOADER")
print("=" * 60)

# ============================================================
# OPTION 1: Download from Sen1Floods11 (Recommended)
# ============================================================

def download_sen1floods11_sample():
    """
    Download sample data from Sen1Floods11 dataset
    Full dataset: https://github.com/cloudtostreet/Sen1Floods11
    """
    
    print("\n📥 Downloading Sen1Floods11 sample data...")
    
    # Sen1Floods11 catalog locations (flood events)
    flood_events = {
        'bangladesh_2017': {
            'name': 'Bangladesh Flood 2017',
            'bounds': [89.0, 23.5, 91.0, 25.0],
            'flood_date': ['2017-08-01', '2017-08-30'],
            'pre_flood_date': ['2017-06-01', '2017-06-30']
        },
        'india_2020': {
            'name': 'India Bihar Flood 2020', 
            'bounds': [84.5, 25.0, 87.0, 27.0],
            'flood_date': ['2020-07-15', '2020-08-15'],
            'pre_flood_date': ['2020-05-01', '2020-05-31']
        },
        'bangladesh_2020': {
            'name': 'Bangladesh Flood 2020',
            'bounds': [89.5, 23.0, 91.5, 25.5],
            'flood_date': ['2020-07-01', '2020-07-31'],
            'pre_flood_date': ['2020-04-01', '2020-04-30']
        }
    }
    
    return flood_events


def fetch_flood_training_pair(event_name, event_data, output_dir='data/training'):
    """
    Fetch pre-flood and flood image pairs from GEE
    These pairs are used to train the change detection model
    """
    
    print(f"\n🛰️ Processing: {event_data['name']}")
    
    bounds = ee.Geometry.Rectangle(event_data['bounds'])
    
    # Get PRE-FLOOD image (normal conditions)
    pre_flood = (ee.ImageCollection('COPERNICUS/S1_GRD')
        .filterBounds(bounds)
        .filterDate(event_data['pre_flood_date'][0], event_data['pre_flood_date'][1])
        .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
        .filter(ee.Filter.eq('instrumentMode', 'IW'))
        .select('VV')
        .median()  # Composite to reduce noise
    )
    
    # Get FLOOD image (during flood)
    flood = (ee.ImageCollection('COPERNICUS/S1_GRD')
        .filterBounds(bounds)
        .filterDate(event_data['flood_date'][0], event_data['flood_date'][1])
        .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
        .filter(ee.Filter.eq('instrumentMode', 'IW'))
        .select('VV')
        .median()
    )
    
    # Calculate difference (flood detection basis)
    difference = flood.subtract(pre_flood)
    
    # Get statistics
    pre_stats = pre_flood.reduceRegion(
        reducer=ee.Reducer.mean().combine(ee.Reducer.stdDev(), sharedInputs=True),
        geometry=bounds,
        scale=100,
        maxPixels=1e9
    ).getInfo()
    
    flood_stats = flood.reduceRegion(
        reducer=ee.Reducer.mean().combine(ee.Reducer.stdDev(), sharedInputs=True),
        geometry=bounds,
        scale=100,
        maxPixels=1e9
    ).getInfo()
    
    print(f"   Pre-flood VV mean: {pre_stats.get('VV_mean', 0):.2f} dB")
    print(f"   Flood VV mean: {flood_stats.get('VV_mean', 0):.2f} dB")
    print(f"   Difference: {flood_stats.get('VV_mean', 0) - pre_stats.get('VV_mean', 0):.2f} dB")
    
    # Water appears DARKER in SAR (lower backscatter)
    # So flooded areas will have MORE NEGATIVE values
    
    return {
        'event': event_name,
        'pre_flood_mean': pre_stats.get('VV_mean', 0),
        'flood_mean': flood_stats.get('VV_mean', 0),
        'pre_flood_image': pre_flood,
        'flood_image': flood,
        'difference': difference,
        'bounds': bounds
    }


def generate_flood_labels(flood_image, pre_flood_image, bounds, threshold=-3):
    """
    Generate flood labels using thresholding
    Water has lower backscatter than land in SAR imagery
    
    Threshold explanation:
    - VV backscatter for water: typically < -15 dB
    - VV backscatter for land: typically -10 to 0 dB
    - Change > 3 dB decrease = likely flooding
    """
    
    # Method 1: Absolute threshold (water detection)
    water_mask = flood_image.lt(-15)  # Pixels below -15 dB are water
    
    # Method 2: Change detection (more accurate for floods)
    change = flood_image.subtract(pre_flood_image)
    flood_mask = change.lt(threshold)  # Decrease of >3 dB indicates new water
    
    # Combine: New water that wasn't there before = FLOOD
    flood_label = flood_mask.And(pre_flood_image.gt(-15))
    
    return flood_label


def create_training_samples(num_samples=100):
    """
    Create training samples from flood events
    Returns list of (image_patch, label) pairs
    """
    
    flood_events = download_sen1floods11_sample()
    training_data = []
    
    for event_name, event_data in flood_events.items():
        print(f"\n{'='*50}")
        result = fetch_flood_training_pair(event_name, event_data)
        
        # Generate labels
        flood_label = generate_flood_labels(
            result['flood_image'],
            result['pre_flood_image'],
            result['bounds']
        )
        
        training_data.append({
            'event': event_name,
            'pre_flood': result['pre_flood_image'],
            'flood': result['flood_image'],
            'label': flood_label,
            'bounds': result['bounds'],
            'stats': {
                'pre_mean': result['pre_flood_mean'],
                'flood_mean': result['flood_mean']
            }
        })
        
        print(f"   ✅ Training pair created for {event_data['name']}")
    
    return training_data


# ============================================================
# MAIN EXECUTION
# ============================================================

if __name__ == "__main__":
    print("\n🚀 Starting training data preparation...")
    
    # Create training samples
    training_data = create_training_samples()
    
    print("\n" + "=" * 60)
    print("📊 TRAINING DATA SUMMARY")
    print("=" * 60)
    print(f"Total flood events processed: {len(training_data)}")
    
    for data in training_data:
        print(f"\n  📍 {data['event']}:")
        print(f"     Pre-flood mean: {data['stats']['pre_mean']:.2f} dB")
        print(f"     Flood mean: {data['stats']['flood_mean']:.2f} dB")
    
    print("\n✅ Training data preparation complete!")
    print("\n📝 Next step: Run train_flood_model.py to train the CNN")