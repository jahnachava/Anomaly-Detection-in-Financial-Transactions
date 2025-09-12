"""
Base class for unsupervised anomaly detection models.

This module provides a base class that defines the interface for all
unsupervised anomaly detection algorithms.
"""

import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple, List
from loguru import logger
import joblib
from pathlib import Path

class BaseUnsupervisedDetector(ABC):
    """
    Abstract base class for unsupervised anomaly detection models.
    """
    
    def __init__(self, config: Dict[str, Any], model_name: str):
        """
        Initialize the base unsupervised detector.
        
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
        
    @abstractmethod
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'BaseUnsupervisedDetector':
        """
        Fit the anomaly detection model.
        
        Args:
            X: Feature matrix
            y: Target variable (optional, for consistency with supervised models)
            
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
    def decision_function(self, X: pd.DataFrame) -> np.ndarray:
        """
        Compute the decision function for each sample.
        
        Args:
            X: Feature matrix
            
        Returns:
            Array of decision function values
        """
        pass
        
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict probability of being an anomaly.
        
        Args:
            X: Feature matrix
            
        Returns:
            Array of probabilities
        """
        try:
            if not self.is_fitted:
                raise ValueError("Model must be fitted before making predictions")
                
            # Get decision function values
            decision_scores = self.decision_function(X)
            
            # Convert to probabilities using sigmoid function
            probabilities = 1 / (1 + np.exp(-decision_scores))
            
            # Return probabilities for both classes [normal, anomaly]
            prob_normal = 1 - probabilities
            prob_anomaly = probabilities
            
            return np.column_stack([prob_normal, prob_anomaly])
            
        except Exception as e:
            self.logger.error(f"Error computing prediction probabilities: {e}")
            raise
            
    def score_samples(self, X: pd.DataFrame) -> np.ndarray:
        """
        Compute anomaly scores for each sample.
        
        Args:
            X: Feature matrix
            
        Returns:
            Array of anomaly scores
        """
        try:
            if not self.is_fitted:
                raise ValueError("Model must be fitted before computing scores")
                
            return self.decision_function(X)
            
        except Exception as e:
            self.logger.error(f"Error computing anomaly scores: {e}")
            raise
            
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the model.
        
        Returns:
            Dictionary containing model information
        """
        return {
            'model_name': self.model_name,
            'is_fitted': self.is_fitted,
            'model_params': self.model_params,
            'config': self.config
        }
        
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
                'model_name': self.model_name
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
            
            self.logger.info(f"Model loaded from {file_path}")
            
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
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
            
    def validate_input(self, X: pd.DataFrame) -> None:
        """
        Validate input data.
        
        Args:
            X: Feature matrix to validate
            
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
            
    def set_params(self, **params) -> 'BaseUnsupervisedDetector':
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
