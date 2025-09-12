"""
One-Class SVM implementation for anomaly detection.

One-Class SVM is an unsupervised anomaly detection algorithm that learns a
decision function for novelty detection. It finds a hypersphere in the feature
space that encompasses most of the normal data points.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
from sklearn.svm import OneClassSVM
from loguru import logger
from .base_unsupervised import BaseUnsupervisedDetector

class OneClassSVMDetector(BaseUnsupervisedDetector):
    """
    One-Class SVM anomaly detector for financial transactions.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the One-Class SVM detector.
        
        Args:
            config: Configuration dictionary containing model parameters
        """
        super().__init__(config, "OneClassSVM")
        
        # Get model parameters from config
        model_config = config.get('models', {}).get('unsupervised', {}).get('one_class_svm', {})
        
        self.model_params = {
            'kernel': model_config.get('kernel', 'rbf'),
            'gamma': model_config.get('gamma', 'scale'),
            'nu': model_config.get('nu', 0.1),
            'degree': model_config.get('degree', 3),
            'coef0': model_config.get('coef0', 0.0),
            'shrinking': model_config.get('shrinking', True),
            'cache_size': model_config.get('cache_size', 200),
            'max_iter': model_config.get('max_iter', -1),
            'tol': model_config.get('tol', 1e-3)
        }
        
        self.model = OneClassSVM(**self.model_params)
        self.logger.info(f"Initialized One-Class SVM with parameters: {self.model_params}")
        
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'OneClassSVMDetector':
        """
        Fit the One-Class SVM model.
        
        Args:
            X: Feature matrix
            y: Target variable (ignored for unsupervised learning)
            
        Returns:
            Self
        """
        try:
            self.validate_input(X)
            self.logger.info(f"Fitting One-Class SVM on {X.shape[0]} samples with {X.shape[1]} features")
            
            # Fit the model
            self.model.fit(X)
            self.is_fitted = True
            
            # Log some statistics
            self.logger.info("One-Class SVM fitted successfully")
            self.logger.info(f"Kernel: {self.model_params['kernel']}")
            self.logger.info(f"Gamma: {self.model_params['gamma']}")
            self.logger.info(f"Nu: {self.model_params['nu']}")
            
            # Log support vectors info
            if hasattr(self.model, 'n_support_'):
                self.logger.info(f"Number of support vectors: {self.model.n_support_}")
                
            return self
            
        except Exception as e:
            self.logger.error(f"Error fitting One-Class SVM: {e}")
            raise
            
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict anomalies using One-Class SVM.
        
        Args:
            X: Feature matrix
            
        Returns:
            Array of predictions (1 for anomaly, 0 for normal)
        """
        try:
            if not self.is_fitted:
                raise ValueError("Model must be fitted before making predictions")
                
            self.validate_input(X)
            
            # One-Class SVM returns -1 for anomalies, 1 for normal
            predictions = self.model.predict(X)
            
            # Convert to 0/1 format (0 for normal, 1 for anomaly)
            predictions = np.where(predictions == -1, 1, 0)
            
            self.logger.info(f"Predicted {np.sum(predictions)} anomalies out of {len(predictions)} samples")
            
            return predictions
            
        except Exception as e:
            self.logger.error(f"Error making predictions with One-Class SVM: {e}")
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
            
            # One-Class SVM returns negative values for anomalies
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
            
    def get_anomaly_threshold(self, X: pd.DataFrame, contamination: float = 0.1) -> float:
        """
        Get the threshold for anomaly detection.
        
        Args:
            X: Feature matrix
            contamination: Expected contamination rate
            
        Returns:
            Threshold value for anomaly detection
        """
        try:
            if not self.is_fitted:
                raise ValueError("Model must be fitted before computing threshold")
                
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
            threshold: Custom threshold (if None, uses default threshold)
            
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
            
    def get_support_vectors(self) -> Optional[np.ndarray]:
        """
        Get the support vectors of the trained model.
        
        Returns:
            Array of support vectors or None if not available
        """
        try:
            if not self.is_fitted:
                raise ValueError("Model must be fitted before getting support vectors")
                
            if hasattr(self.model, 'support_vectors_'):
                return self.model.support_vectors_
            else:
                self.logger.warning("Support vectors not available for this model")
                return None
                
        except Exception as e:
            self.logger.error(f"Error getting support vectors: {e}")
            return None
            
    def get_dual_coef(self) -> Optional[np.ndarray]:
        """
        Get the dual coefficients of the trained model.
        
        Returns:
            Array of dual coefficients or None if not available
        """
        try:
            if not self.is_fitted:
                raise ValueError("Model must be fitted before getting dual coefficients")
                
            if hasattr(self.model, 'dual_coef_'):
                return self.model.dual_coef_
            else:
                self.logger.warning("Dual coefficients not available for this model")
                return None
                
        except Exception as e:
            self.logger.error(f"Error getting dual coefficients: {e}")
            return None
            
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get detailed information about the One-Class SVM model.
        
        Returns:
            Dictionary containing model information
        """
        info = super().get_model_info()
        
        if self.is_fitted:
            info.update({
                'kernel': self.model_params['kernel'],
                'gamma': self.model_params['gamma'],
                'nu': self.model_params['nu'],
                'degree': self.model_params['degree'],
                'coef0': self.model_params['coef0'],
                'shrinking': self.model_params['shrinking'],
                'cache_size': self.model_params['cache_size'],
                'max_iter': self.model_params['max_iter'],
                'tol': self.model_params['tol']
            })
            
            # Add runtime information
            if hasattr(self.model, 'n_support_'):
                info['n_support_vectors'] = self.model.n_support_
                
        return info
