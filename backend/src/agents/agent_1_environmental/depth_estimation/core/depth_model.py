"""
Lightweight Depth Estimation CNN (~95K parameters)
U-Net style architecture optimized for CPU
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
from typing import Dict, Any, Tuple, Optional



class LightweightDepthCNN:
    """Efficient CNN for depth estimation"""
    
    def __init__(self, input_shape: Tuple[int, int, int] = (32, 32, 2)):
        self.input_shape = input_shape
        self.model: Optional[keras.Model] = None
        
    def build(self) -> keras.Model:
        """Build U-Net style architecture"""
        inputs = keras.Input(shape=self.input_shape)
        
        # Encoder
        x = keras.layers.Conv2D(16, 3, padding='same', activation='relu')(inputs)
        x = keras.layers.BatchNormalization()(x)
        skip1 = x
        x = keras.layers.MaxPooling2D(2)(x)
        
        x = keras.layers.Conv2D(32, 3, padding='same', activation='relu')(x)
        x = keras.layers.BatchNormalization()(x)
        skip2 = x
        x = keras.layers.MaxPooling2D(2)(x)
        
        # Bottleneck
        x = keras.layers.Conv2D(64, 3, padding='same', activation='relu')(x)
        x = keras.layers.BatchNormalization()(x)
        
        # Decoder
        x = keras.layers.UpSampling2D(2)(x)
        x = keras.layers.Concatenate()([x, skip2])
        x = keras.layers.Conv2D(32, 3, padding='same', activation='relu')(x)
        x = keras.layers.BatchNormalization()(x)
        
        x = keras.layers.UpSampling2D(2)(x)
        x = keras.layers.Concatenate()([x, skip1])
        x = keras.layers.Conv2D(16, 3, padding='same', activation='relu')(x)
        
        # Output
        outputs = keras.layers.Conv2D(1, 1, activation='relu')(x)
        
        self.model = keras.Model(inputs, outputs, name='depth_cnn')
        return self.model
    
    def compile(self, learning_rate: float = 0.001) -> None:
        """Compile model"""
        if self.model is None:
            self.build()
            
        assert self.model is not None  # Type guard for Pylance
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate),
            loss='mse',
            metrics=['mae', keras.metrics.RootMeanSquaredError(name='rmse')]
        )
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              X_val: np.ndarray, y_val: np.ndarray, 
              epochs: int = 20, batch_size: int = 8) -> Any:
        """Train the model"""
        if self.model is None:
            self.compile()
        
        assert self.model is not None  # Type guard for Pylance
        
        # Normalize
        X_train_norm = (X_train - X_train.mean()) / (X_train.std() + 1e-7)
        X_val_norm = (X_val - X_val.mean()) / (X_val.std() + 1e-7)
        
        callbacks = [
            keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3)
        ]
        
        history = self.model.fit(
            X_train_norm, y_train,
            validation_data=(X_val_norm, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        return history
    
    def predict(self, sar_image: np.ndarray) -> np.ndarray:
        """Predict depth from SAR image"""
        assert self.model is not None, "Model not loaded or trained"
        
        # Handle single image or batch
        if len(sar_image.shape) == 3:
            sar_image = sar_image[np.newaxis, ...]
        
        # Normalize
        sar_norm = (sar_image - sar_image.mean()) / (sar_image.std() + 1e-7)
        
        # Predict
        depth = self.model.predict(sar_norm, verbose=0)
        
        return depth[0] if depth.shape[0] == 1 else depth
    
    def save(self, filepath: str) -> None:
        """Save model"""
        if self.model is not None:
            self.model.save(filepath)
            print(f"✓ Model saved to {filepath}")
    
    def load(self, filepath: str) -> keras.Model:
        """Load model"""
        self.model = keras.models.load_model(filepath)
        print(f"✓ Model loaded from {filepath}")
        return self.model
    
    def get_params(self) -> int:
        """Get parameter count"""
        if self.model is None:
            self.build()
        
        assert self.model is not None  # Type guard for Pylance
        return self.model.count_params()