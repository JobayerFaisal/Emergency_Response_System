"""
ML Models for Flood Detection
Implements multiple algorithms with unified interface

Models:
1. Random Forest Classifier (fast, interpretable)
2. XGBoost Classifier (high accuracy)
3. CNN Classifier (spatial patterns) - advanced
"""

import numpy as np
import pickle
import json
from pathlib import Path
from typing import Dict, Tuple, Optional, Union
from abc import ABC, abstractmethod
from dataclasses import dataclass
import logging

# ML libraries
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler
import joblib

# Optional: XGBoost (install with: pip install xgboost)
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logging.warning("XGBoost not available. Install with: pip install xgboost")

# Optional: TensorFlow/Keras for Neural Networks
try:
    import tensorflow as tf
    # from tensorflow import keras
    from tensorflow.keras import layers
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    logging.warning("TensorFlow not available. Install with: pip install tensorflow")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ModelMetrics:
    """Container for model evaluation metrics"""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    auc_roc: float
    confusion_matrix: np.ndarray
    
    def __str__(self):
        return f"""
Model Metrics:
─────────────────────────────
Accuracy:  {self.accuracy:.4f}
Precision: {self.precision:.4f}
Recall:    {self.recall:.4f}
F1 Score:  {self.f1_score:.4f}
AUC-ROC:   {self.auc_roc:.4f}

Confusion Matrix:
{self.confusion_matrix}
        """


class BaseFloodModel(ABC):
    """
    Abstract base class for all flood detection models
    
    This ensures all models have the same interface,
    making them interchangeable and easy to compare!
    """
    
    def __init__(self, model_name: str, model_version: str = "1.0"):
        """
        Initialize base model
        
        Args:
            model_name: Name of the model (e.g., "RandomForest", "XGBoost")
            model_version: Version string for model tracking
        """
        self.model_name = model_name
        self.model_version = model_version
        self.model = None
        self.scaler = StandardScaler()  # For feature normalization
        self.is_trained = False
        self.feature_names = []
        self.training_metrics = None
        
    @abstractmethod
    def train(
        self, 
        X_train: np.ndarray, 
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None
    ) -> ModelMetrics:
        """
        Train the model
        
        Must be implemented by each specific model
        """
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions
        
        Args:
            X: Feature matrix [n_samples, n_features]
            
        Returns:
            Predicted classes [n_samples]
        """
        pass
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities
        
        Returns:
            Probabilities [n_samples, n_classes]
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)
    
    def evaluate(
        self, 
        X_test: np.ndarray, 
        y_test: np.ndarray
    ) -> ModelMetrics:
        """
        Evaluate model on test data
        
        Returns comprehensive metrics
        """
        logger.info(f"Evaluating {self.model_name}...")
        
        # Get predictions
        y_pred = self.predict(X_test)
        y_proba = self.predict_proba(X_test)
        
        # Calculate metrics
        metrics = ModelMetrics(
            accuracy=accuracy_score(y_test, y_pred),
            precision=precision_score(y_test, y_pred, zero_division=0),
            recall=recall_score(y_test, y_pred, zero_division=0),
            f1_score=f1_score(y_test, y_pred, zero_division=0),
            auc_roc=roc_auc_score(y_test, y_proba[:, 1]) if y_proba.shape[1] == 2 else 0,
            confusion_matrix=confusion_matrix(y_test, y_pred)
        )
        
        logger.info(str(metrics))
        return metrics
    
    def save(self, filepath: str):
        """
        Save model to disk
        
        Saves:
        - Model weights/parameters
        - Scaler
        - Metadata (feature names, metrics, etc.)
        """
        logger.info(f"Saving model to {filepath}")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'model_name': self.model_name,
            'model_version': self.model_version,
            'feature_names': self.feature_names,
            'training_metrics': self.training_metrics,
            'is_trained': self.is_trained
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info("Model saved successfully")
    
    def load(self, filepath: str):
        """
        Load model from disk
        """
        logger.info(f"Loading model from {filepath}")
        
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.model_name = model_data['model_name']
        self.model_version = model_data['model_version']
        self.feature_names = model_data['feature_names']
        self.training_metrics = model_data['training_metrics']
        self.is_trained = model_data['is_trained']
        
        logger.info(f"Model loaded: {self.model_name} v{self.model_version}")


class RandomForestFloodDetector(BaseFloodModel):
    """
    Random Forest Classifier for Flood Detection
    
    WHY Random Forest?
    ✅ Handles non-linear relationships well
    ✅ Robust to outliers and noise
    ✅ Feature importance (interpretability!)
    ✅ Fast training and inference
    ✅ No complex hyperparameter tuning needed
    ✅ Works well with imbalanced data
    
    PERFECT for your capstone - easy to explain and visualize!
    """
    
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: Optional[int] = 20,
        min_samples_split: int = 10,
        class_weight: str = 'balanced'
    ):
        """
        Initialize Random Forest model
        
        Args:
            n_estimators: Number of trees (more = better, but slower)
            max_depth: Maximum tree depth (prevent overfitting)
            min_samples_split: Minimum samples to split a node
            class_weight: 'balanced' handles imbalanced data automatically
        """
        super().__init__("RandomForest", "1.0")
        
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            class_weight=class_weight,
            random_state=42,  # Reproducibility
            n_jobs=-1,  # Use all CPU cores
            verbose=1
        )
        
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None
    ) -> ModelMetrics:
        """
        Train Random Forest model
        
        Process:
        1. Normalize features (StandardScaler)
        2. Fit Random Forest
        3. Evaluate on validation set
        4. Return metrics
        """
        logger.info("Training Random Forest Flood Detector...")
        logger.info(f"Training samples: {X_train.shape[0]}")
        logger.info(f"Features: {X_train.shape[1]}")
        logger.info(f"Flood samples: {np.sum(y_train == 1)}")
        logger.info(f"Non-flood samples: {np.sum(y_train == 0)}")
        
        # 1. Normalize features
        logger.info("Normalizing features...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # 2. Train model
        logger.info("Training Random Forest...")
        self.model.fit(X_train_scaled, y_train)
        
        self.is_trained = True
        
        # 3. Evaluate
        if X_val is not None and y_val is not None:
            logger.info("Evaluating on validation set...")
            metrics = self.evaluate(X_val, y_val)
            self.training_metrics = metrics
        else:
            # Evaluate on training set (less reliable)
            logger.info("Evaluating on training set...")
            metrics = self.evaluate(X_train, y_train)
            self.training_metrics = metrics
        
        return metrics
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict flood (1) or no flood (0)
        
        Args:
            X: Feature matrix
            
        Returns:
            Binary predictions (0 or 1)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance scores
        
        This is SUPER useful for your defense:
        - Shows which features matter most
        - Validates your feature engineering
        - Makes model interpretable
        
        Returns:
            Dictionary: {feature_name: importance_score}
        """
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        
        importances = self.model.feature_importances_
        
        # Sort by importance
        indices = np.argsort(importances)[::-1]
        
        # Create dictionary
        feature_importance = {}
        for i in indices:
            if i < len(self.feature_names):
                feature_importance[self.feature_names[i]] = float(importances[i])
        
        return feature_importance
    
    def plot_feature_importance(self, top_n: int = 15):
        """
        Plot top N most important features
        
        Great for your presentation!
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("Matplotlib not available for plotting")
            return
        
        importance = self.get_feature_importance()
        
        # Get top N
        sorted_features = sorted(
            importance.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:top_n]
        
        features, scores = zip(*sorted_features)
        
        # Plot
        plt.figure(figsize=(10, 6))
        plt.barh(range(len(features)), scores)
        plt.yticks(range(len(features)), features)
        plt.xlabel('Importance Score')
        plt.title(f'Top {top_n} Most Important Features')
        plt.tight_layout()
        
        return plt


class XGBoostFloodDetector(BaseFloodModel):
    """
    XGBoost Classifier for Flood Detection
    
    WHY XGBoost?
    ✅ Often more accurate than Random Forest
    ✅ Handles imbalanced data well
    ✅ Built-in cross-validation
    ✅ Regularization to prevent overfitting
    
    Use this if you want MAXIMUM accuracy!
    """
    
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        scale_pos_weight: Optional[float] = None
    ):
        """
        Initialize XGBoost model
        
        Args:
            n_estimators: Number of boosting rounds
            max_depth: Maximum tree depth
            learning_rate: Step size shrinkage
            scale_pos_weight: Balance of positive vs negative weights
        """
        super().__init__("XGBoost", "1.0")
        
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost not installed. Install with: pip install xgboost")
        
        self.model = xgb.XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss'
        )
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None
    ) -> ModelMetrics:
        """
        Train XGBoost model with early stopping
        """
        logger.info("Training XGBoost Flood Detector...")
        
        # Normalize features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Train with validation set if provided
        if X_val is not None and y_val is not None:
            X_val_scaled = self.scaler.transform(X_val)
            
            self.model.fit(
                X_train_scaled, y_train,
                eval_set=[(X_val_scaled, y_val)],
                early_stopping_rounds=10,
                verbose=True
            )
        else:
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
        """Predict flood or no flood"""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)


def create_synthetic_training_data(n_samples: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create synthetic training data for testing
    
    This generates FAKE data that mimics real flood patterns.
    Useful for initial testing before you have real labeled data!
    
    Returns:
        X: Feature matrix [n_samples, n_features]
        y: Labels [n_samples] (0=no flood, 1=flood)
    """
    logger.info(f"Creating {n_samples} synthetic training samples...")
    
    n_features = 20
    X = np.random.randn(n_samples, n_features)
    
    # Create realistic patterns
    # Flood areas have:
    # - LOW backscatter (features 0-3)
    # - LOW elevation (feature 10)
    # - SMALL slope (feature 11)
    
    # Simulate flood conditions
    flood_mask = (
        (X[:, 0] < -0.5) &  # Low VH_after
        (X[:, 10] < -0.3) &  # Low elevation
        (X[:, 11] < 0.2)     # Small slope
    )
    
    y = flood_mask.astype(int)
    
    logger.info(f"Created {np.sum(y==1)} flood samples ({np.sum(y==1)/len(y)*100:.1f}%)")
    logger.info(f"Created {np.sum(y==0)} non-flood samples ({np.sum(y==0)/len(y)*100:.1f}%)")
    
    return X, y


def example_usage():
    """
    Example: How to train and use the flood detector
    """
    logger.info("="*60)
    logger.info("FLOOD DETECTOR TRAINING EXAMPLE")
    logger.info("="*60)
    
    # 1. Create synthetic training data
    X, y = create_synthetic_training_data(n_samples=2000)
    
    # 2. Split into train/validation/test
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    logger.info(f"Train set: {X_train.shape[0]} samples")
    logger.info(f"Val set: {X_val.shape[0]} samples")
    logger.info(f"Test set: {X_test.shape[0]} samples")
    
    # 3. Train Random Forest
    logger.info("\n" + "="*60)
    logger.info("Training Random Forest...")
    logger.info("="*60)
    
    rf_model = RandomForestFloodDetector(
        n_estimators=100,
        max_depth=20,
        class_weight='balanced'
    )
    
    rf_metrics = rf_model.train(X_train, y_train, X_val, y_val)
    
    # 4. Evaluate on test set
    logger.info("\n" + "="*60)
    logger.info("Testing on held-out test set...")
    logger.info("="*60)
    
    test_metrics = rf_model.evaluate(X_test, y_test)
    
    # 5. Make predictions
    logger.info("\n" + "="*60)
    logger.info("Making predictions...")
    logger.info("="*60)
    
    sample = X_test[:5]
    predictions = rf_model.predict(sample)
    probabilities = rf_model.predict_proba(sample)
    
    logger.info("Sample predictions:")
    for i in range(len(predictions)):
        logger.info(f"  Sample {i+1}: {predictions[i]} (prob: {probabilities[i][1]:.3f})")
    
    # 6. Save model
    logger.info("\n" + "="*60)
    logger.info("Saving model...")
    logger.info("="*60)
    
    rf_model.save('/mnt/user-data/outputs/flood_detector_rf.pkl')
    
    logger.info("\n✅ Training complete!")
    logger.info(f"Final Test Accuracy: {test_metrics.accuracy:.4f}")
    logger.info(f"Final Test F1 Score: {test_metrics.f1_score:.4f}")
    
    return rf_model


if __name__ == "__main__":
    # Run example
    model = example_usage()
