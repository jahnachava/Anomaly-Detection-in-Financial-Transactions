"""
Base class for supervised anomaly detection models.

This module provides a base class that defines the interface for all
supervised anomaly detection algorithms.
"""

import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple, List
from loguru import logger
import joblib
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix

class BaseSupervisedDetector(ABC):
    """
    Abstract base class for supervised anomaly detection models.
    """
    
    def __init__(self, config: Dict[str, Any], model_name: str):
        """
        Initialize the base supervised detector.
        
        Args:
            config: Configuration dictionary
            model_name: Name of the model
        """
        self.config = config
        self.model_name = model_name
        self.model = None
        self.is_fitted = False
        self.logger = logger
        self.model_params = {}
        self.feature_importance = None
        
    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'BaseSupervisedDetector':
        """
        Fit the anomaly detection model.
        
        Args:
            X: Feature matrix
            y: Target variable (0 for normal, 1 for anomaly)
            
        Returns:
            Self
        """
        pass
        
    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict anomalies in the data.
        
        Args:
            X: Feature matrix
            
        Returns:
            Array of predictions (1 for anomaly, 0 for normal)
        """
        pass
        
    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict probability of being an anomaly.
        
        Args:
            X: Feature matrix
            
        Returns:
            Array of probabilities [prob_normal, prob_anomaly]
        """
        pass
        
    def predict_with_threshold(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        """
        Predict anomalies using a custom probability threshold.
        
        Args:
            X: Feature matrix
            threshold: Probability threshold for anomaly detection
            
        Returns:
            Array of predictions (1 for anomaly, 0 for normal)
        """
        try:
            if not self.is_fitted:
                raise ValueError("Model must be fitted before making predictions")
                
            # Get probabilities
            probabilities = self.predict_proba(X)
            
            # Use threshold to make predictions
            predictions = (probabilities[:, 1] > threshold).astype(int)
            
            self.logger.info(f"Predicted {np.sum(predictions)} anomalies using threshold {threshold}")
            
            return predictions
            
        except Exception as e:
            self.logger.error(f"Error making predictions with threshold: {e}")
            raise
            
    def get_feature_importance(self) -> Optional[np.ndarray]:
        """
        Get feature importance if available.
        
        Returns:
            Array of feature importance scores or None if not available
        """
        try:
            if hasattr(self.model, 'feature_importances_'):
                return self.model.feature_importances_
            elif hasattr(self.model, 'coef_'):
                return np.abs(self.model.coef_.flatten())
            else:
                self.logger.warning("Feature importance not available for this model")
                return None
                
        except Exception as e:
            self.logger.error(f"Error getting feature importance: {e}")
            return None
            
    def get_top_features(self, n_features: int = 10) -> List[Tuple[str, float]]:
        """
        Get top N most important features.
        
        Args:
            n_features: Number of top features to return
            
        Returns:
            List of tuples (feature_name, importance_score)
        """
        try:
            importance = self.get_feature_importance()
            if importance is None:
                return []
                
            # Get feature names from the last fitted data
            if hasattr(self, 'feature_names_'):
                feature_names = self.feature_names_
            else:
                feature_names = [f"feature_{i}" for i in range(len(importance))]
                
            # Sort by importance
            feature_importance_pairs = list(zip(feature_names, importance))
            feature_importance_pairs.sort(key=lambda x: x[1], reverse=True)
            
            return feature_importance_pairs[:n_features]
            
        except Exception as e:
            self.logger.error(f"Error getting top features: {e}")
            return []
            
    def evaluate(self, X: pd.DataFrame, y: pd.Series, 
                 threshold: float = 0.5) -> Dict[str, Any]:
        """
        Evaluate the model performance.
        
        Args:
            X: Feature matrix
            y: True labels
            threshold: Probability threshold for predictions
            
        Returns:
            Dictionary containing evaluation metrics
        """
        try:
            if not self.is_fitted:
                raise ValueError("Model must be fitted before evaluation")
                
            # Make predictions
            y_pred = self.predict_with_threshold(X, threshold)
            y_proba = self.predict_proba(X)[:, 1]
            
            # Calculate metrics
            from sklearn.metrics import (
                accuracy_score, precision_score, recall_score, f1_score,
                roc_auc_score, average_precision_score, confusion_matrix
            )
            
            metrics = {
                'accuracy': accuracy_score(y, y_pred),
                'precision': precision_score(y, y_pred, zero_division=0),
                'recall': recall_score(y, y_pred, zero_division=0),
                'f1_score': f1_score(y, y_pred, zero_division=0),
                'roc_auc': roc_auc_score(y, y_proba),
                'average_precision': average_precision_score(y, y_proba),
                'confusion_matrix': confusion_matrix(y, y_pred).tolist(),
                'threshold': threshold
            }
            
            # Add classification report
            metrics['classification_report'] = classification_report(
                y, y_pred, output_dict=True, zero_division=0
            )
            
            self.logger.info(f"Model evaluation completed:")
            self.logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
            self.logger.info(f"  Precision: {metrics['precision']:.4f}")
            self.logger.info(f"  Recall: {metrics['recall']:.4f}")
            self.logger.info(f"  F1-Score: {metrics['f1_score']:.4f}")
            self.logger.info(f"  ROC-AUC: {metrics['roc_auc']:.4f}")
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error evaluating model: {e}")
            raise
            
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the model.
        
        Returns:
            Dictionary containing model information
        """
        info = {
            'model_name': self.model_name,
            'is_fitted': self.is_fitted,
            'model_params': self.model_params,
            'config': self.config
        }
        
        if self.is_fitted:
            # Add feature importance if available
            importance = self.get_feature_importance()
            if importance is not None:
                info['feature_importance_available'] = True
                info['n_features'] = len(importance)
            else:
                info['feature_importance_available'] = False
                
        return info
        
    def save_model(self, file_path: str) -> None:
        """
        Save the trained model to disk.
        
        Args:
            file_path: Path to save the model
        """
        try:
            if not self.is_fitted:
                raise ValueError("Model must be fitted before saving")
                
            model_data = {
                'model': self.model,
                'model_params': self.model_params,
                'is_fitted': self.is_fitted,
                'model_name': self.model_name,
                'feature_names_': getattr(self, 'feature_names_', None)
            }
            
            joblib.dump(model_data, file_path)
            self.logger.info(f"Model saved to {file_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving model: {e}")
            raise
            
    def load_model(self, file_path: str) -> None:
        """
        Load a trained model from disk.
        
        Args:
            file_path: Path to load the model from
        """
        try:
            if not Path(file_path).exists():
                raise FileNotFoundError(f"Model file not found: {file_path}")
                
            model_data = joblib.load(file_path)
            
            self.model = model_data['model']
            self.model_params = model_data['model_params']
            self.is_fitted = model_data['is_fitted']
            self.model_name = model_data['model_name']
            self.feature_names_ = model_data.get('feature_names_', None)
            
            self.logger.info(f"Model loaded from {file_path}")
            
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            raise
            
    def validate_input(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> None:
        """
        Validate input data.
        
        Args:
            X: Feature matrix to validate
            y: Target variable to validate (optional)
            
        Raises:
            ValueError: If input is invalid
        """
        if X is None or X.empty:
            raise ValueError("Input data cannot be None or empty")
            
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input data must be a pandas DataFrame")
            
        if X.shape[0] == 0:
            raise ValueError("Input data must have at least one sample")
            
        if X.shape[1] == 0:
            raise ValueError("Input data must have at least one feature")
            
        if y is not None:
            if len(y) != len(X):
                raise ValueError("Target variable length must match feature matrix length")
                
            if not isinstance(y, pd.Series):
                raise ValueError("Target variable must be a pandas Series")
                
            # Check for binary classification
            unique_values = y.unique()
            if not all(val in [0, 1] for val in unique_values):
                raise ValueError("Target variable must contain only 0 and 1 values")
                
    def set_params(self, **params) -> 'BaseSupervisedDetector':
        """
        Set parameters for the model.
        
        Args:
            **params: Parameters to set
            
        Returns:
            Self
        """
        try:
            if hasattr(self.model, 'set_params'):
                self.model.set_params(**params)
            else:
                self.model_params.update(params)
                
            self.logger.info(f"Parameters updated: {params}")
            return self
            
        except Exception as e:
            self.logger.error(f"Error setting parameters: {e}")
            raise
            
    def get_params(self) -> Dict[str, Any]:
        """
        Get current parameters of the model.
        
        Returns:
            Dictionary of current parameters
        """
        try:
            if hasattr(self.model, 'get_params'):
                return self.model.get_params()
            else:
                return self.model_params.copy()
                
        except Exception as e:
            self.logger.error(f"Error getting parameters: {e}")
            return {}
