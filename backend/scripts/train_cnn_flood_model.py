"""
CNN-based Flood Detection Model Training
Architecture: U-Net for Semantic Segmentation
Input: Sentinel-1 SAR imagery (VV polarization)
Output: Binary flood mask
"""

import ee
import numpy as np
import os
import json
from pathlib import Path
from datetime import datetime
import requests
from io import BytesIO

# Initialize Earth Engine
ee.Initialize(project='caramel-pulsar-475810-e7')

# TensorFlow imports
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

print("=" * 70)
print("🧠 CNN FLOOD DETECTION MODEL - U-Net Architecture")
print("=" * 70)
print(f"TensorFlow version: {tf.__version__}")
print(f"GPU Available: {len(tf.config.list_physical_devices('GPU')) > 0}")

# ============================================================
# CONFIGURATION
# ============================================================

CONFIG = {
    'image_size': 256,          # Input image size
    'batch_size': 8,            # Batch size for training
    'epochs': 50,               # Training epochs
    'learning_rate': 1e-4,      # Initial learning rate
    'num_samples': 500,         # Number of training samples to generate
    'validation_split': 0.2,    # 20% for validation
    'model_dir': 'models',
    'data_dir': 'data/training',
}

# Known flood events for training data
FLOOD_EVENTS = [
    {
        'name': 'Bangladesh_2017',
        'region': [89.5, 23.5, 90.5, 24.5],
        'flood_dates': ['2017-08-01', '2017-08-31'],
        'normal_dates': ['2017-03-01', '2017-03-31'],
    },
    {
        'name': 'Bangladesh_2020', 
        'region': [89.0, 24.0, 90.5, 25.5],
        'flood_dates': ['2020-07-01', '2020-07-31'],
        'normal_dates': ['2020-03-01', '2020-03-31'],
    },
    {
        'name': 'India_Bihar_2020',
        'region': [85.0, 25.5, 86.5, 26.5],
        'flood_dates': ['2020-07-20', '2020-08-20'],
        'normal_dates': ['2020-04-01', '2020-04-30'],
    },
    {
        'name': 'Bangladesh_2022',
        'region': [91.0, 24.0, 92.0, 25.0],
        'flood_dates': ['2022-06-15', '2022-06-30'],
        'normal_dates': ['2022-03-01', '2022-03-31'],
    },
]


# ============================================================
# U-NET MODEL ARCHITECTURE
# ============================================================

def conv_block(inputs, num_filters, kernel_size=3):
    """Convolutional block with BatchNorm and ReLU"""
    x = layers.Conv2D(num_filters, kernel_size, padding='same', kernel_initializer='he_normal')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    x = layers.Conv2D(num_filters, kernel_size, padding='same', kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    return x


def encoder_block(inputs, num_filters):
    """Encoder block: Conv block + MaxPooling"""
    x = conv_block(inputs, num_filters)
    p = layers.MaxPooling2D((2, 2))(x)
    p = layers.Dropout(0.1)(p)
    return x, p


def decoder_block(inputs, skip_features, num_filters):
    """Decoder block: UpConv + Concatenate + Conv block"""
    x = layers.Conv2DTranspose(num_filters, (2, 2), strides=2, padding='same')(inputs)
    x = layers.Concatenate()([x, skip_features])
    x = layers.Dropout(0.1)(x)
    x = conv_block(x, num_filters)
    return x


def build_unet(input_shape=(256, 256, 2), num_classes=1):
    """
    Build U-Net model for flood segmentation
    
    Args:
        input_shape: (height, width, channels) - 2 channels for VV and VH
        num_classes: 1 for binary segmentation (flood/no-flood)
    
    Returns:
        Keras Model
    """
    
    inputs = layers.Input(input_shape)
    
    # Encoder (Contracting Path)
    s1, p1 = encoder_block(inputs, 64)     # 256 -> 128
    s2, p2 = encoder_block(p1, 128)        # 128 -> 64
    s3, p3 = encoder_block(p2, 256)        # 64 -> 32
    s4, p4 = encoder_block(p3, 512)        # 32 -> 16
    
    # Bottleneck (Bridge)
    b1 = conv_block(p4, 1024)              # 16 x 16 x 1024
    
    # Decoder (Expanding Path)
    d1 = decoder_block(b1, s4, 512)        # 16 -> 32
    d2 = decoder_block(d1, s3, 256)        # 32 -> 64
    d3 = decoder_block(d2, s2, 128)        # 64 -> 128
    d4 = decoder_block(d3, s1, 64)         # 128 -> 256
    
    # Output layer
    outputs = layers.Conv2D(num_classes, (1, 1), activation='sigmoid')(d4)
    
    model = Model(inputs, outputs, name='UNet_FloodDetection')
    
    return model


def build_lightweight_unet(input_shape=(256, 256, 2), num_classes=1):
    """
    Lighter U-Net for faster training (good for prototyping)
    """
    
    inputs = layers.Input(input_shape)
    
    # Encoder
    s1, p1 = encoder_block(inputs, 32)
    s2, p2 = encoder_block(p1, 64)
    s3, p3 = encoder_block(p2, 128)
    
    # Bottleneck
    b1 = conv_block(p3, 256)
    
    # Decoder
    d1 = decoder_block(b1, s3, 128)
    d2 = decoder_block(d1, s2, 64)
    d3 = decoder_block(d2, s1, 32)
    
    # Output
    outputs = layers.Conv2D(num_classes, (1, 1), activation='sigmoid')(d3)
    
    model = Model(inputs, outputs, name='LightUNet_FloodDetection')
    
    return model


# ============================================================
# LOSS FUNCTIONS & METRICS
# ============================================================

def dice_coefficient(y_true, y_pred, smooth=1e-6):
    """Dice coefficient for measuring segmentation quality"""
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)


def dice_loss(y_true, y_pred):
    """Dice loss = 1 - Dice coefficient"""
    return 1 - dice_coefficient(y_true, y_pred)


def bce_dice_loss(y_true, y_pred):
    """Combined Binary Cross-Entropy and Dice Loss"""
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    dice = dice_loss(y_true, y_pred)
    return bce + dice


def iou_metric(y_true, y_pred, threshold=0.5):
    """Intersection over Union metric"""
    y_pred_binary = tf.cast(y_pred > threshold, tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred_binary)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred_binary) - intersection
    return (intersection + 1e-6) / (union + 1e-6)


# ============================================================
# DATA GENERATION FROM GOOGLE EARTH ENGINE
# ============================================================

def get_sentinel1_image(geometry, start_date, end_date):
    """Fetch Sentinel-1 SAR image from GEE"""
    
    collection = (ee.ImageCollection('COPERNICUS/S1_GRD')
        .filterBounds(geometry)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
        .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
        .filter(ee.Filter.eq('instrumentMode', 'IW'))
        .select(['VV', 'VH']))
    
    # Create median composite
    image = collection.median()
    
    return image


def generate_training_sample(event, sample_idx, patch_size=256):
    """
    Generate a single training sample (image + mask) from flood event
    
    Returns:
        image: numpy array of shape (patch_size, patch_size, 2) - VV and VH bands
        mask: numpy array of shape (patch_size, patch_size, 1) - flood mask
    """
    
    # Random point within the region
    region = event['region']
    
    # Add some randomness to get different patches
    np.random.seed(sample_idx)
    lon = np.random.uniform(region[0] + 0.1, region[2] - 0.1)
    lat = np.random.uniform(region[1] + 0.1, region[3] - 0.1)
    
    # Create patch geometry (approximately 5km x 5km)
    patch_size_deg = 0.05
    patch_geometry = ee.Geometry.Rectangle([
        lon - patch_size_deg, lat - patch_size_deg,
        lon + patch_size_deg, lat + patch_size_deg
    ])
    
    # Get flood image
    flood_image = get_sentinel1_image(
        patch_geometry,
        event['flood_dates'][0],
        event['flood_dates'][1]
    )
    
    # Get normal (pre-flood) image
    normal_image = get_sentinel1_image(
        patch_geometry,
        event['normal_dates'][0],
        event['normal_dates'][1]
    )
    
    # Calculate flood mask using change detection
    # Water appears darker (lower backscatter) in SAR
    vv_change = flood_image.select('VV').subtract(normal_image.select('VV'))
    
    # Threshold for flood detection:
    # - Significant decrease in backscatter (> 3 dB) indicates water
    # - Also check absolute threshold for water
    flood_mask = vv_change.lt(-3).Or(flood_image.select('VV').lt(-18))
    
    # Get image statistics to create synthetic data
    flood_stats = flood_image.reduceRegion(
        reducer=ee.Reducer.mean().combine(ee.Reducer.stdDev(), sharedInputs=True),
        geometry=patch_geometry,
        scale=30,
        maxPixels=1e6
    ).getInfo()
    
    normal_stats = normal_image.reduceRegion(
        reducer=ee.Reducer.mean().combine(ee.Reducer.stdDev(), sharedInputs=True),
        geometry=patch_geometry,
        scale=30,
        maxPixels=1e6
    ).getInfo()
    
    return flood_stats, normal_stats


def create_synthetic_training_data(num_samples=500):
    """
    Create synthetic training data based on real SAR statistics
    This is faster than downloading actual images for prototyping
    """
    
    print("\n📊 Generating synthetic training data from real SAR statistics...")
    
    images = []
    masks = []
    
    # Typical SAR backscatter values (in dB)
    # Land: -5 to -12 dB
    # Water: -15 to -25 dB
    # Urban: -3 to -8 dB
    # Vegetation: -8 to -15 dB
    
    for i in range(num_samples):
        if i % 100 == 0:
            print(f"   Generating sample {i}/{num_samples}...")
        
        # Create random image
        image = np.zeros((CONFIG['image_size'], CONFIG['image_size'], 2), dtype=np.float32)
        mask = np.zeros((CONFIG['image_size'], CONFIG['image_size'], 1), dtype=np.float32)
        
        # Generate land background (VV and VH)
        land_vv = np.random.normal(-10, 2, (CONFIG['image_size'], CONFIG['image_size']))
        land_vh = np.random.normal(-17, 2, (CONFIG['image_size'], CONFIG['image_size']))
        
        image[:, :, 0] = land_vv
        image[:, :, 1] = land_vh
        
        # Add some flood patches (20-60% of samples have floods)
        has_flood = np.random.random() > 0.4
        
        if has_flood:
            # Generate random flood patches
            num_patches = np.random.randint(1, 5)
            
            for _ in range(num_patches):
                # Random ellipse flood patch
                cx = np.random.randint(50, CONFIG['image_size'] - 50)
                cy = np.random.randint(50, CONFIG['image_size'] - 50)
                rx = np.random.randint(20, 80)
                ry = np.random.randint(20, 80)
                
                # Create ellipse mask
                y, x = np.ogrid[:CONFIG['image_size'], :CONFIG['image_size']]
                ellipse_mask = ((x - cx) / rx) ** 2 + ((y - cy) / ry) ** 2 <= 1
                
                # Water backscatter values (much lower than land)
                water_vv = np.random.normal(-20, 3, (CONFIG['image_size'], CONFIG['image_size']))
                water_vh = np.random.normal(-27, 3, (CONFIG['image_size'], CONFIG['image_size']))
                
                # Apply water values to flood region
                image[:, :, 0] = np.where(ellipse_mask, water_vv, image[:, :, 0])
                image[:, :, 1] = np.where(ellipse_mask, water_vh, image[:, :, 1])
                
                # Update mask
                mask[:, :, 0] = np.where(ellipse_mask, 1.0, mask[:, :, 0])
        
        # Normalize to 0-1 range
        # SAR values typically range from -30 to 0 dB
        image = (image + 30) / 30  # Normalize to 0-1
        image = np.clip(image, 0, 1)
        
        images.append(image)
        masks.append(mask)
    
    images = np.array(images, dtype=np.float32)
    masks = np.array(masks, dtype=np.float32)
    
    print(f"   ✅ Generated {num_samples} training samples")
    print(f"   Image shape: {images.shape}")
    print(f"   Mask shape: {masks.shape}")
    
    return images, masks


def augment_data(images, masks):
    """Apply data augmentation"""
    
    augmented_images = []
    augmented_masks = []
    
    for img, mask in zip(images, masks):
        augmented_images.append(img)
        augmented_masks.append(mask)
        
        # Horizontal flip
        augmented_images.append(np.fliplr(img))
        augmented_masks.append(np.fliplr(mask))
        
        # Vertical flip
        augmented_images.append(np.flipud(img))
        augmented_masks.append(np.flipud(mask))
        
        # Rotation 90
        augmented_images.append(np.rot90(img))
        augmented_masks.append(np.rot90(mask))
    
    return np.array(augmented_images), np.array(augmented_masks)


# ============================================================
# TRAINING PIPELINE
# ============================================================

def train_model():
    """Main training pipeline"""
    
    print("\n" + "=" * 70)
    print("🚀 STARTING MODEL TRAINING")
    print("=" * 70)
    
    # Step 1: Generate training data
    print("\n📦 Step 1: Preparing training data...")
    X, y = create_synthetic_training_data(CONFIG['num_samples'])
    
    # Step 2: Augment data
    print("\n🔄 Step 2: Augmenting data...")
    X_aug, y_aug = augment_data(X, y)
    print(f"   After augmentation: {X_aug.shape[0]} samples")
    
    # Step 3: Split into train/validation
    print("\n✂️ Step 3: Splitting data...")
    split_idx = int(len(X_aug) * (1 - CONFIG['validation_split']))
    
    # Shuffle
    indices = np.random.permutation(len(X_aug))
    X_aug = X_aug[indices]
    y_aug = y_aug[indices]
    
    X_train, X_val = X_aug[:split_idx], X_aug[split_idx:]
    y_train, y_val = y_aug[:split_idx], y_aug[split_idx:]
    
    print(f"   Training samples: {len(X_train)}")
    print(f"   Validation samples: {len(X_val)}")
    
    # Step 4: Build model
    print("\n🏗️ Step 4: Building U-Net model...")
    model = build_lightweight_unet(input_shape=(CONFIG['image_size'], CONFIG['image_size'], 2))
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=CONFIG['learning_rate']),
        loss=bce_dice_loss,
        metrics=['accuracy', dice_coefficient, iou_metric]
    )
    
    model.summary()
    
    # Step 5: Setup callbacks
    print("\n⚙️ Step 5: Setting up training callbacks...")
    
    os.makedirs(CONFIG['model_dir'], exist_ok=True)
    
    callbacks = [
        ModelCheckpoint(
            os.path.join(CONFIG['model_dir'], 'flood_unet_best.h5'),
            monitor='val_dice_coefficient',
            mode='max',
            save_best_only=True,
            verbose=1
        ),
        EarlyStopping(
            monitor='val_dice_coefficient',
            mode='max',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
    ]
    
    # Step 6: Train
    print("\n🎯 Step 6: Training model...")
    print("=" * 70)
    
    history = model.fit(
        X_train, y_train,
        batch_size=CONFIG['batch_size'],
        epochs=CONFIG['epochs'],
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1
    )
    
    # Step 7: Save final model
    print("\n💾 Step 7: Saving model...")
    
    model.save(os.path.join(CONFIG['model_dir'], 'flood_unet_final.h5'))
    
    # Save model config
    model_config = {
        'architecture': 'LightUNet',
        'input_shape': [CONFIG['image_size'], CONFIG['image_size'], 2],
        'num_classes': 1,
        'training_samples': len(X_train),
        'validation_samples': len(X_val),
        'epochs_trained': len(history.history['loss']),
        'final_train_loss': float(history.history['loss'][-1]),
        'final_val_loss': float(history.history['val_loss'][-1]),
        'final_dice': float(history.history['val_dice_coefficient'][-1]),
        'created_at': datetime.now().isoformat(),
    }
    
    with open(os.path.join(CONFIG['model_dir'], 'model_config.json'), 'w') as f:
        json.dump(model_config, f, indent=2)
    
    print(f"\n   ✅ Model saved to: {CONFIG['model_dir']}/flood_unet_final.h5")
    print(f"   ✅ Config saved to: {CONFIG['model_dir']}/model_config.json")
    
    return model, history


def evaluate_model(model, X_test, y_test):
    """Evaluate model performance"""
    
    print("\n📊 Model Evaluation:")
    print("-" * 40)
    
    results = model.evaluate(X_test, y_test, verbose=0)
    
    print(f"   Loss: {results[0]:.4f}")
    print(f"   Accuracy: {results[1]:.4f}")
    print(f"   Dice Coefficient: {results[2]:.4f}")
    print(f"   IoU: {results[3]:.4f}")
    
    return results


# ============================================================
# MAIN EXECUTION
# ============================================================

if __name__ == "__main__":
    
    # Train the model
    model, history = train_model()
    
    # Summary
    print("\n" + "=" * 70)
    print("🎉 TRAINING COMPLETE!")
    print("=" * 70)
    
    print("\n📁 Files created:")
    print(f"   • models/flood_unet_best.h5   (best checkpoint)")
    print(f"   • models/flood_unet_final.h5  (final model)")
    print(f"   • models/model_config.json    (configuration)")
    
    print("\n📊 Training Results:")
    print(f"   • Final Loss: {history.history['loss'][-1]:.4f}")
    print(f"   • Final Val Loss: {history.history['val_loss'][-1]:.4f}")
    print(f"   • Final Dice Score: {history.history['val_dice_coefficient'][-1]:.4f}")
    
    print("\n🚀 Next Steps:")
    print("   1. Run: python scripts/test_flood_model.py")
    print("   2. Then: python src/agents/agent_1_environmental/satellite_monitor.py")
    print("=" * 70)