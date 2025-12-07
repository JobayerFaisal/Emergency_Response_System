"""
Flood Depth Estimation using Machine Learning
THIS IS YOUR KEY CONTRIBUTION AS A DATA SCIENCE STUDENT!

Problem: Detect not just WHERE floods occur, but HOW DEEP they are
Solution: ML Regression model using SAR backscatter + terrain data

Why This is Novel and Impressive:
- Few papers have done this
- Combines SAR with hydrology
- Practical impact: depth → damage assessment
- Shows advanced ML skills
"""

import numpy as np
import pickle
from typing import Dict, Tuple, Optional
from dataclasses import dataclass
import logging

# ML libraries
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Optional: Neural Network for depth estimation
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DepthEstimationMetrics:
    """Metrics for depth estimation model"""
    mae: float  # Mean Absolute Error (meters)
    rmse: float  # Root Mean Squared Error (meters)
    r2: float  # R² score (goodness of fit)
    
    def __str__(self):
        return f"""
Depth Estimation Metrics:
────────────────────────────
MAE:  {self.mae:.3f} meters
RMSE: {self.rmse:.3f} meters
R²:   {self.r2:.4f}

Interpretation:
- MAE = Average prediction error
- RMSE = Typical prediction error (penalizes large errors)
- R² = How well model explains variance (1.0 = perfect)
        """


class FloodDepthEstimator:
    """
    ML Model for Estimating Flood Depth
    
    INPUT FEATURES:
    ═══════════════════════════════════════════════════════════
    SAR Features (from satellite):
    - VH/VV backscatter intensity (water = low)
    - Backscatter change (before → after)
    - Texture (smooth water vs rough)
    
    Terrain Features (from DEM):
    - Elevation (lower = deeper floods)
    - Slope (flat = water accumulation)
    - Distance to water body (closer = deeper)
    - Curvature (concave = water collects)
    - Flow accumulation (where water flows)
    
    Contextual Features:
    - Land cover type
    - Urban density
    - Rainfall amount (if available)
    ═══════════════════════════════════════════════════════════
    
    OUTPUT:
    - Estimated flood depth in meters (0-5m typical range)
    
    
    HOW IT WORKS - THE INTUITION:
    ═══════════════════════════════════════════════════════════
    Think of it this way:
    
    1. Water reflects less radar → darker SAR image
       MORE water (deeper) → DARKER image → LOWER backscatter
       
    2. Water flows downhill → elevation matters
       LOWER elevation → MORE water accumulation → DEEPER
       
    3. Flat areas accumulate water → slope matters
       FLATTER terrain → water pools → DEEPER
       
    4. Distance from river matters
       CLOSER to river → more likely to be deep
       
    The ML model learns these relationships from data!
    ═══════════════════════════════════════════════════════════
    """
    
    def __init__(
        self,
        model_type: str = 'random_forest',
        **model_params
    ):
        """
        Initialize depth estimator
        
        Args:
            model_type: 'random_forest', 'gradient_boosting', 'neural_network'
            **model_params: Additional parameters for the model
        """
        self.model_type = model_type
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_names = []
        self.training_metrics = None
        
        # Create model based on type
        if model_type == 'random_forest':
            self.model = RandomForestRegressor(
                n_estimators=model_params.get('n_estimators', 200),
                max_depth=model_params.get('max_depth', 30),
                min_samples_split=model_params.get('min_samples_split', 5),
                min_samples_leaf=model_params.get('min_samples_leaf', 2),
                random_state=42,
                n_jobs=-1,
                verbose=1
            )
            
        elif model_type == 'gradient_boosting':
            self.model = GradientBoostingRegressor(
                n_estimators=model_params.get('n_estimators', 200),
                max_depth=model_params.get('max_depth', 5),
                learning_rate=model_params.get('learning_rate', 0.1),
                random_state=42,
                verbose=1
            )
            
        elif model_type == 'neural_network':
            if not TF_AVAILABLE:
                raise ImportError("TensorFlow required for neural network")
            self.model = self._create_neural_network(model_params)
            
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def _create_neural_network(self, params: Dict) -> keras.Model:
        """
        Create neural network for depth estimation
        
        Architecture designed for regression:
        - Multiple hidden layers with decreasing size
        - Dropout for regularization
        - ReLU activation (works well for regression)
        - Linear output (for depth prediction)
        """
        n_features = params.get('n_features', 20)
        
        model = keras.Sequential([
            # Input layer
            layers.Dense(128, activation='relu', input_shape=(n_features,)),
            layers.Dropout(0.3),
            
            # Hidden layers
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.2),
            
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.1),
            
            # Output layer - single neuron for depth prediction
            layers.Dense(1, activation='linear')  # Linear for regression
        ])
        
        # Compile with MAE loss (robust to outliers)
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mae',  # Mean Absolute Error
            metrics=['mse', 'mae']
        )
        
        return model
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        epochs: int = 50
    ) -> DepthEstimationMetrics:
        """
        Train depth estimation model
        
        Args:
            X_train: Training features [n_samples, n_features]
            y_train: Training depths [n_samples] (in meters)
            X_val: Validation features
            y_val: Validation depths
            epochs: Number of epochs (for neural network)
            
        Returns:
            Training metrics
        """
        logger.info(f"Training Flood Depth Estimator ({self.model_type})...")
        logger.info(f"Training samples: {X_train.shape[0]}")
        logger.info(f"Features: {X_train.shape[1]}")
        logger.info(f"Depth range: {y_train.min():.2f} - {y_train.max():.2f} meters")
        logger.info(f"Mean depth: {y_train.mean():.2f} meters")
        
        # Normalize features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Train model
        if self.model_type == 'neural_network':
            # Neural network training with validation
            if X_val is not None and y_val is not None:
                X_val_scaled = self.scaler.transform(X_val)
                
                history = self.model.fit(
                    X_train_scaled, y_train,
                    validation_data=(X_val_scaled, y_val),
                    epochs=epochs,
                    batch_size=32,
                    verbose=1,
                    callbacks=[
                        keras.callbacks.EarlyStopping(
                            patience=10,
                            restore_best_weights=True
                        )
                    ]
                )
            else:
                self.model.fit(
                    X_train_scaled, y_train,
                    epochs=epochs,
                    batch_size=32,
                    verbose=1
                )
        else:
            # Sklearn models
            self.model.fit(X_train_scaled, y_train)
        
        self.is_trained = True
        
        # Evaluate
        if X_val is not None and y_val is not None:
            metrics = self.evaluate(X_val, y_val)
        else:
            metrics = self.evaluate(X_train, y_train)
        
        self.training_metrics = metrics
        return metrics
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict flood depth
        
        Args:
            X: Feature matrix [n_samples, n_features]
            
        Returns:
            Predicted depths [n_samples] in meters
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        X_scaled = self.scaler.transform(X)
        
        if self.model_type == 'neural_network':
            predictions = self.model.predict(X_scaled, verbose=0).flatten()
        else:
            predictions = self.model.predict(X_scaled)
        
        # Ensure non-negative depths
        predictions = np.maximum(predictions, 0)
        
        return predictions
    
    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> DepthEstimationMetrics:
        """
        Evaluate depth estimation performance
        
        Returns:
            Metrics including MAE, RMSE, R²
        """
        logger.info("Evaluating depth estimator...")
        
        # Get predictions
        y_pred = self.predict(X_test)
        
        # Calculate metrics
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        metrics = DepthEstimationMetrics(
            mae=mae,
            rmse=rmse,
            r2=r2
        )
        
        logger.info(str(metrics))
        return metrics
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance (only for tree-based models)
        
        Shows which features are most important for depth prediction.
        Critical for understanding the model!
        """
        if self.model_type not in ['random_forest', 'gradient_boosting']:
            logger.warning("Feature importance only available for tree-based models")
            return {}
        
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        
        importances = self.model.feature_importances_
        
        # Sort by importance
        indices = np.argsort(importances)[::-1]
        
        feature_importance = {}
        for i in indices:
            if i < len(self.feature_names):
                feature_importance[self.feature_names[i]] = float(importances[i])
        
        return feature_importance
    
    def classify_depth_severity(self, depths: np.ndarray) -> np.ndarray:
        """
        Classify depths into severity categories
        
        Categories:
        0: No flood (0m)
        1: Minor (0-0.5m) - ankle deep
        2: Moderate (0.5-1.5m) - knee to waist deep
        3: Major (1.5-3m) - chest deep, dangerous
        4: Severe (>3m) - life-threatening
        
        Useful for risk assessment and visualization!
        """
        categories = np.zeros_like(depths, dtype=int)
        
        categories[depths == 0] = 0
        categories[(depths > 0) & (depths <= 0.5)] = 1
        categories[(depths > 0.5) & (depths <= 1.5)] = 2
        categories[(depths > 1.5) & (depths <= 3.0)] = 3
        categories[depths > 3.0] = 4
        
        return categories
    
    def save(self, filepath: str):
        """Save model to disk"""
        logger.info(f"Saving depth estimator to {filepath}")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'model_type': self.model_type,
            'feature_names': self.feature_names,
            'training_metrics': self.training_metrics,
            'is_trained': self.is_trained
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info("Model saved successfully")
    
    def load(self, filepath: str):
        """Load model from disk"""
        logger.info(f"Loading depth estimator from {filepath}")
        
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.model_type = model_data['model_type']
        self.feature_names = model_data['feature_names']
        self.training_metrics = model_data['training_metrics']
        self.is_trained = model_data['is_trained']
        
        logger.info(f"Model loaded: {self.model_type}")


def create_synthetic_depth_data(n_samples: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create synthetic depth training data
    
    Simulates realistic flood depth patterns based on:
    - Backscatter intensity (darker = deeper)
    - Elevation (lower = deeper)
    - Slope (flatter = deeper)
    - Distance to water (closer = deeper)
    
    This generates PHYSICALLY PLAUSIBLE data for testing!
    """
    logger.info(f"Creating {n_samples} synthetic depth samples...")
    
    n_features = 15
    X = np.random.randn(n_samples, n_features)
    
    # Simulate depth based on physical relationships
    # (In reality, you'll train on real labeled data)
    
    # Key features for depth:
    backscatter = X[:, 0]  # VH backscatter
    elevation = X[:, 10]  # Elevation
    slope = X[:, 11]  # Slope
    distance_to_water = X[:, 12]  # Distance to water
    
    # Depth model (simplified physics):
    # - Lower backscatter → deeper (water is dark)
    # - Lower elevation → deeper (water accumulates)
    # - Lower slope → deeper (flat areas flood more)
    # - Closer to water → deeper
    
    depth = (
        -0.5 * backscatter  # Darker SAR = deeper
        - 0.8 * elevation  # Lower elevation = deeper
        - 0.3 * slope  # Flatter = deeper
        - 0.4 * distance_to_water  # Closer = deeper
        + np.random.normal(0, 0.3, n_samples)  # Add noise
    )
    
    # Ensure realistic depth range (0-5 meters)
    depth = np.clip(depth, 0, 5)
    
    logger.info(f"Depth range: {depth.min():.2f} - {depth.max():.2f} meters")
    logger.info(f"Mean depth: {depth.mean():.2f} meters")
    logger.info(f"Samples with depth > 0: {np.sum(depth > 0)} ({np.sum(depth > 0)/len(depth)*100:.1f}%)")
    
    return X, depth


def example_usage():
    """
    Example: Train and use depth estimator
    """
    logger.info("="*60)
    logger.info("FLOOD DEPTH ESTIMATION EXAMPLE")
    logger.info("="*60)
    
    # 1. Create synthetic data
    X, y = create_synthetic_depth_data(n_samples=2000)
    
    # 2. Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    logger.info(f"\nTrain set: {X_train.shape[0]} samples")
    logger.info(f"Test set: {X_test.shape[0]} samples")
    
    # 3. Train Random Forest depth estimator
    logger.info("\n" + "="*60)
    logger.info("Training Random Forest Depth Estimator...")
    logger.info("="*60)
    
    depth_model = FloodDepthEstimator(
        model_type='random_forest',
        n_estimators=200,
        max_depth=30
    )
    
    train_metrics = depth_model.train(X_train, y_train, X_test, y_test)
    
    # 4. Evaluate on test set
    logger.info("\n" + "="*60)
    logger.info("Testing depth estimator...")
    logger.info("="*60)
    
    test_metrics = depth_model.evaluate(X_test, y_test)
    
    # 5. Make predictions
    logger.info("\n" + "="*60)
    logger.info("Making depth predictions...")
    logger.info("="*60)
    
    sample = X_test[:10]
    actual_depths = y_test[:10]
    predicted_depths = depth_model.predict(sample)
    
    logger.info("\nSample predictions:")
    logger.info("Actual vs Predicted (meters):")
    for i in range(len(predicted_depths)):
        error = abs(predicted_depths[i] - actual_depths[i])
        logger.info(f"  {actual_depths[i]:.2f} → {predicted_depths[i]:.2f} (error: {error:.2f}m)")
    
    # 6. Classify severity
    categories = depth_model.classify_depth_severity(predicted_depths)
    severity_names = ['None', 'Minor', 'Moderate', 'Major', 'Severe']
    
    logger.info("\nSeverity classification:")
    for i in range(len(categories)):
        logger.info(f"  {predicted_depths[i]:.2f}m → {severity_names[categories[i]]}")
    
    # 7. Save model
    logger.info("\n" + "="*60)
    logger.info("Saving model...")
    logger.info("="*60)
    
    depth_model.save('/mnt/user-data/outputs/depth_estimator_rf.pkl')
    
    logger.info("\n✅ Training complete!")
    logger.info(f"Final Test MAE: {test_metrics.mae:.3f} meters")
    logger.info(f"Final Test RMSE: {test_metrics.rmse:.3f} meters")
    logger.info(f"Final Test R²: {test_metrics.r2:.4f}")
    
    return depth_model


if __name__ == "__main__":
    # Run example
    model = example_usage()
