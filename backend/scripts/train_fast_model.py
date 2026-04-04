"""
FAST Flood Detection Model - With Data Visualization
Saves sample images so you can see what the model trains on
"""

import ee
import numpy as np
import os
import json
from datetime import datetime

ee.Initialize(project='caramel-pulsar-475810-e7')

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

# For saving images
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

print("=" * 60)
print("⚡ FAST FLOOD DETECTION MODEL (with data saving)")
print("=" * 60)

# ============================================================
# CONFIGURATION
# ============================================================

CONFIG = {
    'image_size': 64,
    'batch_size': 32,
    'epochs': 30,
    'learning_rate': 1e-3,
    'num_samples': 300,
    'model_dir': 'models',
    'data_dir': 'data/training',
    'save_samples': True,  # Save sample images to disk
    'num_samples_to_save': 20,  # How many to save
}


# ============================================================
# LIGHTWEIGHT U-NET
# ============================================================

def build_fast_unet(input_shape=(64, 64, 2)):
    inputs = layers.Input(input_shape)
    
    # Encoder
    c1 = layers.Conv2D(16, 3, padding='same', activation='relu')(inputs)
    c1 = layers.Conv2D(16, 3, padding='same', activation='relu')(c1)
    p1 = layers.MaxPooling2D(2)(c1)
    
    c2 = layers.Conv2D(32, 3, padding='same', activation='relu')(p1)
    c2 = layers.Conv2D(32, 3, padding='same', activation='relu')(c2)
    p2 = layers.MaxPooling2D(2)(c2)
    
    c3 = layers.Conv2D(64, 3, padding='same', activation='relu')(p2)
    c3 = layers.Conv2D(64, 3, padding='same', activation='relu')(c3)
    p3 = layers.MaxPooling2D(2)(c3)
    
    # Bottleneck
    b = layers.Conv2D(128, 3, padding='same', activation='relu')(p3)
    b = layers.Dropout(0.3)(b)
    b = layers.Conv2D(128, 3, padding='same', activation='relu')(b)
    
    # Decoder
    u1 = layers.UpSampling2D(2)(b)
    u1 = layers.Concatenate()([u1, c3])
    d1 = layers.Conv2D(64, 3, padding='same', activation='relu')(u1)
    
    u2 = layers.UpSampling2D(2)(d1)
    u2 = layers.Concatenate()([u2, c2])
    d2 = layers.Conv2D(32, 3, padding='same', activation='relu')(u2)
    
    u3 = layers.UpSampling2D(2)(d2)
    u3 = layers.Concatenate()([u3, c1])
    d3 = layers.Conv2D(16, 3, padding='same', activation='relu')(u3)
    
    outputs = layers.Conv2D(1, 1, activation='sigmoid')(d3)
    
    return Model(inputs, outputs, name='FastUNet')


# ============================================================
# METRICS
# ============================================================

def dice_coef(y_true, y_pred):
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (2. * intersection + 1) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + 1)

def dice_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)


# ============================================================
# REALISTIC DATA GENERATION
# ============================================================

def create_realistic_data(num_samples=300):
    """Create challenging synthetic SAR data"""
    
    from scipy.ndimage import gaussian_filter
    
    print(f"\n📊 Generating {num_samples} realistic training samples...")
    
    size = CONFIG['image_size']
    images = []
    masks = []
    
    for i in range(num_samples):
        img = np.zeros((size, size, 2), dtype=np.float32)
        mask = np.zeros((size, size, 1), dtype=np.float32)
        
        # === BASE LAND TEXTURE ===
        land_type = np.random.choice(['urban', 'vegetation', 'bare', 'mixed'])
        
        if land_type == 'urban':
            base_vv, base_vh, noise_std = -6, -12, 2.5
        elif land_type == 'vegetation':
            base_vv, base_vh, noise_std = -10, -16, 3.0
        elif land_type == 'bare':
            base_vv, base_vh, noise_std = -12, -18, 2.0
        else:
            base_vv, base_vh, noise_std = -9, -15, 3.5
        
        # Generate with spatial correlation
        vv = np.random.normal(base_vv, noise_std, (size, size))
        vh = np.random.normal(base_vh, noise_std, (size, size))
        
        vv = gaussian_filter(vv, sigma=1.5)
        vh = gaussian_filter(vh, sigma=1.5)
        
        # Speckle noise
        speckle = np.random.gamma(4, 0.25, (size, size))
        vv = vv * speckle
        vh = vh * speckle
        
        img[:, :, 0] = vv
        img[:, :, 1] = vh
        
        # === ADD FLOOD (50% of samples) ===
        has_flood = np.random.random() > 0.5
        
        if has_flood:
            num_floods = np.random.randint(1, 4)
            
            for _ in range(num_floods):
                cx = np.random.randint(10, size - 10)
                cy = np.random.randint(10, size - 10)
                base_r = np.random.randint(5, 18)
                
                y_grid, x_grid = np.ogrid[:size, :size]
                dist = np.sqrt((x_grid - cx)**2 + (y_grid - cy)**2)
                
                # Irregular boundary
                angle = np.arctan2(y_grid - cy, x_grid - cx)
                radius_noise = np.sin(angle * np.random.randint(3, 7)) * np.random.uniform(2, 4)
                effective_radius = base_r + radius_noise
                
                flood_region = dist < effective_radius
                
                # Water values
                water_vv = np.random.uniform(-24, -18)
                water_vh = np.random.uniform(-30, -24)
                water_noise = np.random.normal(0, 1.5, (size, size))
                
                # Soft boundary
                soft_mask = np.clip((effective_radius - dist) / 2, 0, 1)
                
                img[:, :, 0] = img[:, :, 0] * (1 - soft_mask) + (water_vv + water_noise) * soft_mask
                img[:, :, 1] = img[:, :, 1] * (1 - soft_mask) + (water_vh + water_noise) * soft_mask
                
                mask[:, :, 0] = np.maximum(mask[:, :, 0], flood_region.astype(np.float32))
        
        # === PERMANENT WATER (NOT flood) ===
        if np.random.random() > 0.7:
            river_y = np.random.randint(10, size - 10)
            river_width = np.random.randint(2, 5)
            
            for x in range(size):
                y_offset = int(4 * np.sin(x / 8))
                y_start = max(0, river_y + y_offset - river_width)
                y_end = min(size, river_y + y_offset + river_width)
                
                img[y_start:y_end, x, 0] = np.random.uniform(-23, -19)
                img[y_start:y_end, x, 1] = np.random.uniform(-29, -25)
        
        # === NORMALIZE ===
        img = (img + 35) / 35
        img = np.clip(img, 0, 1)
        img += np.random.normal(0, 0.02, img.shape)
        img = np.clip(img, 0, 1)
        
        images.append(img.astype(np.float32))
        masks.append(mask.astype(np.float32))
        
        if (i + 1) % 100 == 0:
            print(f"   Generated {i + 1}/{num_samples}")
    
    X = np.array(images)
    y = np.array(masks)
    
    flood_ratio = y.sum() / y.size * 100
    print(f"\n   ✅ Data ready: {X.shape}")
    print(f"   Flood pixel ratio: {flood_ratio:.1f}%")
    
    return X, y


def save_sample_images(X, y, num_samples=20):
    """Save sample images to disk for visualization"""
    
    print(f"\n💾 Saving {num_samples} sample images to disk...")
    
    # Create directories
    img_dir = os.path.join(CONFIG['data_dir'], 'images')
    mask_dir = os.path.join(CONFIG['data_dir'], 'labels')
    viz_dir = os.path.join(CONFIG['data_dir'], 'visualizations')
    
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)
    os.makedirs(viz_dir, exist_ok=True)
    
    # Save samples
    for i in range(min(num_samples, len(X))):
        # Save as numpy
        np.save(os.path.join(img_dir, f'sample_{i:03d}.npy'), X[i])
        np.save(os.path.join(mask_dir, f'mask_{i:03d}.npy'), y[i])
        
        # Save visualization
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        
        # VV band
        axes[0].imshow(X[i, :, :, 0], cmap='gray')
        axes[0].set_title(f'VV Band (Sample {i})')
        axes[0].axis('off')
        
        # VH band
        axes[1].imshow(X[i, :, :, 1], cmap='gray')
        axes[1].set_title('VH Band')
        axes[1].axis('off')
        
        # Flood mask
        axes[2].imshow(y[i, :, :, 0], cmap='Blues')
        axes[2].set_title(f'Flood Mask (pixels: {y[i].sum():.0f})')
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, f'sample_{i:03d}.png'), dpi=100)
        plt.close()
    
    print(f"   ✅ Saved to:")
    print(f"      • {img_dir}/")
    print(f"      • {mask_dir}/")
    print(f"      • {viz_dir}/")


def augment_fast(X, y):
    """Quick augmentation"""
    X_aug = [X]
    y_aug = [y]
    
    X_aug.append(np.flip(X, axis=2))
    y_aug.append(np.flip(y, axis=2))
    
    X_aug.append(np.flip(X, axis=1))
    y_aug.append(np.flip(y, axis=1))
    
    return np.concatenate(X_aug), np.concatenate(y_aug)


# ============================================================
# TRAINING
# ============================================================

def train():
    print("\n🚀 Starting training...")
    
    # Check scipy
    try:
        from scipy.ndimage import gaussian_filter
    except ImportError:
        print("Installing scipy...")
        os.system('pip install scipy')
        from scipy.ndimage import gaussian_filter
    
    # Generate data
    X, y = create_realistic_data(CONFIG['num_samples'])
    
    # Save sample images
    if CONFIG['save_samples']:
        save_sample_images(X, y, CONFIG['num_samples_to_save'])
    
    # Augment
    print("\n🔄 Augmenting...")
    X, y = augment_fast(X, y)
    print(f"   After augmentation: {len(X)} samples")
    
    # Shuffle and split
    idx = np.random.permutation(len(X))
    X, y = X[idx], y[idx]
    
    split = int(0.8 * len(X))
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]
    
    print(f"   Train: {len(X_train)}, Val: {len(X_val)}")
    
    # Build model
    print("\n🏗️ Building Fast U-Net...")
    model = build_fast_unet()
    model.compile(
        optimizer=keras.optimizers.Adam(CONFIG['learning_rate']),
        loss=dice_loss,
        metrics=['accuracy', dice_coef]
    )
    
    print(f"   Parameters: {model.count_params():,}")
    
    # Callbacks
    os.makedirs(CONFIG['model_dir'], exist_ok=True)
    
    callbacks = [
        ModelCheckpoint(
            f"{CONFIG['model_dir']}/flood_fast_best.keras",
            monitor='val_dice_coef',
            mode='max',
            save_best_only=True,
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=8,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=4,
            verbose=1
        )
    ]
    
    # Train
    print("\n🎯 Training (should take ~5-10 minutes)...")
    print("=" * 60)
    
    history = model.fit(
        X_train, y_train,
        batch_size=CONFIG['batch_size'],
        epochs=CONFIG['epochs'],
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1
    )
    
    # Save model
    model.save(f"{CONFIG['model_dir']}/flood_fast_final.keras")
    
    # Save training history plot
    save_training_plot(history)
    
    # Save config
    config = {
        'input_shape': [CONFIG['image_size'], CONFIG['image_size'], 2],
        'parameters': model.count_params(),
        'epochs_trained': len(history.history['loss']),
        'final_val_accuracy': float(history.history['val_accuracy'][-1]),
        'final_val_dice': float(history.history['val_dice_coef'][-1]),
        'best_val_dice': float(max(history.history['val_dice_coef'])),
    }
    
    with open(f"{CONFIG['model_dir']}/fast_model_config.json", 'w') as f:
        json.dump(config, f, indent=2)
    
    return model, history


def save_training_plot(history):
    """Save training history visualization"""
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Loss
    axes[0].plot(history.history['loss'], label='Train')
    axes[0].plot(history.history['val_loss'], label='Val')
    axes[0].set_title('Loss')
    axes[0].legend()
    axes[0].set_xlabel('Epoch')
    
    # Accuracy
    axes[1].plot(history.history['accuracy'], label='Train')
    axes[1].plot(history.history['val_accuracy'], label='Val')
    axes[1].set_title('Accuracy')
    axes[1].legend()
    axes[1].set_xlabel('Epoch')
    
    # Dice
    axes[2].plot(history.history['dice_coef'], label='Train')
    axes[2].plot(history.history['val_dice_coef'], label='Val')
    axes[2].set_title('Dice Coefficient')
    axes[2].legend()
    axes[2].set_xlabel('Epoch')
    
    plt.tight_layout()
    plt.savefig(f"{CONFIG['model_dir']}/training_history.png", dpi=150)
    plt.close()
    
    print(f"\n   📊 Training plot saved to: {CONFIG['model_dir']}/training_history.png")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    
    model, history = train()
    
    print("\n" + "=" * 60)
    print("🎉 TRAINING COMPLETE!")
    print("=" * 60)
    
    final_acc = history.history['val_accuracy'][-1]
    final_dice = history.history['val_dice_coef'][-1]
    best_dice = max(history.history['val_dice_coef'])
    
    print(f"\n📊 Results:")
    print(f"   Val Accuracy: {final_acc:.2%}")
    print(f"   Val Dice: {final_dice:.4f}")
    print(f"   Best Dice: {best_dice:.4f}")
    
    print(f"\n📁 Files Created:")
    print(f"   Models:")
    print(f"      • models/flood_fast_best.keras")
    print(f"      • models/flood_fast_final.keras")
    print(f"      • models/fast_model_config.json")
    print(f"      • models/training_history.png")
    print(f"   Training Data:")
    print(f"      • data/training/images/ (numpy arrays)")
    print(f"      • data/training/labels/ (numpy arrays)")
    print(f"      • data/training/visualizations/ (PNG images)")
    
    print("\n" + "=" * 60)