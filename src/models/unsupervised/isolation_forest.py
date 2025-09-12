"""
Isolation Forest implementation for anomaly detection.

Isolation Forest is an unsupervised anomaly detection algorithm that works by
isolating anomalies instead of profiling normal points. It uses the fact that
anomalies are few and different, making them easier to isolate.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
from sklearn.ensemble import IsolationForest
from loguru import logger
from .base_unsupervised import BaseUnsupervisedDetector

class IsolationForestDetector(BaseUnsupervisedDetector):
    """
    Isolation Forest anomaly detector for financial transactions.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the Isolation Forest detector.
        
        Args:
            config: Configuration dictionary containing model parameters
        """
        super().__init__(config, "IsolationForest")
        
        # Get model parameters from config
        model_config = config.get('models', {}).get('unsupervised', {}).get('isolation_forest', {})
        
        self.model_params = {
            'n_estimators': model_config.get('n_estimators', 100),
            'contamination': model_config.get('contamination', 0.1),
            'max_samples': model_config.get('max_samples', 'auto'),
            'random_state': model_config.get('random_state', 42),
            'max_features': model_config.get('max_features', 1.0),
            'bootstrap': model_config.get('bootstrap', False)
        }
        
        self.model = IsolationForest(**self.model_params)
        self.logger.info(f"Initialized Isolation Forest with parameters: {self.model_params}")
        
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'IsolationForestDetector':
        """
        Fit the Isolation Forest model.
        
        Args:
            X: Feature matrix
            y: Target variable (ignored for unsupervised learning)
            
        Returns:
            Self
        """
        try:
            self.validate_input(X)
            self.logger.info(f"Fitting Isolation Forest on {X.shape[0]} samples with {X.shape[1]} features")
            
            # Fit the model
            self.model.fit(X)
            self.is_fitted = True
            
            # Log some statistics
            self.logger.info("Isolation Forest fitted successfully")
            self.logger.info(f"Expected contamination: {self.model_params['contamination']}")
            self.logger.info(f"Number of estimators: {self.model_params['n_estimators']}")
            
            return self
            
        except Exception as e:
            self.logger.error(f"Error fitting Isolation Forest: {e}")
            raise
            
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict anomalies using Isolation Forest.
        
        Args:
            X: Feature matrix
            
        Returns:
            Array of predictions (1 for anomaly, 0 for normal)
        """
        try:
            if not self.is_fitted:
                raise ValueError("Model must be fitted before making predictions")
                
            self.validate_input(X)
            
            # Isolation Forest returns -1 for anomalies, 1 for normal
            predictions = self.model.predict(X)
            
            # Convert to 0/1 format (0 for normal, 1 for anomaly)
            predictions = np.where(predictions == -1, 1, 0)
            
            self.logger.info(f"Predicted {np.sum(predictions)} anomalies out of {len(predictions)} samples")
            
            return predictions
            
        except Exception as e:
            self.logger.error(f"Error making predictions with Isolation Forest: {e}")
            raise
            
    def decision_function(self, X: pd.DataFrame) -> np.ndarray:
        """
        Compute the decision function for each sample.
        
        Args:
            X: Feature matrix
            
        Returns:
            Array of decision function values (higher values = more anomalous)
        """
        try:
            if not self.is_fitted:
                raise ValueError("Model must be fitted before computing decision function")
                
            self.validate_input(X)
            
            # Get decision function values
            decision_scores = self.model.decision_function(X)
            
            # Isolation Forest returns negative values for anomalies
            # We'll negate them to make higher values more anomalous
            decision_scores = -decision_scores
            
            return decision_scores
            
        except Exception as e:
            self.logger.error(f"Error computing decision function: {e}")
            raise
            
    def score_samples(self, X: pd.DataFrame) -> np.ndarray:
        """
        Compute anomaly scores for each sample.
        
        Args:
            X: Feature matrix
            
        Returns:
            Array of anomaly scores (higher values = more anomalous)
        """
        try:
            if not self.is_fitted:
                raise ValueError("Model must be fitted before computing scores")
                
            self.validate_input(X)
            
            # Get the decision function values
            scores = self.decision_function(X)
            
            return scores
            
        except Exception as e:
            self.logger.error(f"Error computing anomaly scores: {e}")
            raise
            
    def get_anomaly_threshold(self, X: pd.DataFrame, contamination: float = None) -> float:
        """
        Get the threshold for anomaly detection.
        
        Args:
            X: Feature matrix
            contamination: Expected contamination rate (if None, uses model's contamination)
            
        Returns:
            Threshold value for anomaly detection
        """
        try:
            if not self.is_fitted:
                raise ValueError("Model must be fitted before computing threshold")
                
            if contamination is None:
                contamination = self.model_params['contamination']
                
            # Get decision function values
            scores = self.decision_function(X)
            
            # Calculate threshold based on contamination rate
            threshold = np.percentile(scores, (1 - contamination) * 100)
            
            self.logger.info(f"Anomaly threshold: {threshold:.4f} (contamination: {contamination})")
            
            return threshold
            
        except Exception as e:
            self.logger.error(f"Error computing anomaly threshold: {e}")
            raise
            
    def predict_with_threshold(self, X: pd.DataFrame, threshold: float = None) -> np.ndarray:
        """
        Predict anomalies using a custom threshold.
        
        Args:
            X: Feature matrix
            threshold: Custom threshold (if None, uses model's default)
            
        Returns:
            Array of predictions (1 for anomaly, 0 for normal)
        """
        try:
            if not self.is_fitted:
                raise ValueError("Model must be fitted before making predictions")
                
            self.validate_input(X)
            
            # Get decision function values
            scores = self.decision_function(X)
            
            # Use custom threshold if provided
            if threshold is None:
                threshold = self.get_anomaly_threshold(X)
                
            # Predict based on threshold
            predictions = (scores > threshold).astype(int)
            
            self.logger.info(f"Predicted {np.sum(predictions)} anomalies using threshold {threshold:.4f}")
            
            return predictions
            
        except Exception as e:
            self.logger.error(f"Error making predictions with threshold: {e}")
            raise
            
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get detailed information about the Isolation Forest model.
        
        Returns:
            Dictionary containing model information
        """
        info = super().get_model_info()
        
        if self.is_fitted:
            info.update({
                'n_estimators': self.model_params['n_estimators'],
                'contamination': self.model_params['contamination'],
                'max_samples': self.model_params['max_samples'],
                'max_features': self.model_params['max_features'],
                'bootstrap': self.model_params['bootstrap']
            })
            
        return info
