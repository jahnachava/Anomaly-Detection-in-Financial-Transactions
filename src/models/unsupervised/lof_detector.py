"""
Local Outlier Factor (LOF) implementation for anomaly detection.

LOF is an unsupervised anomaly detection algorithm that computes the local density
deviation of a given data point with respect to its neighbors. It considers as
outliers the samples that have a substantially lower density than their neighbors.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
from sklearn.neighbors import LocalOutlierFactor
from loguru import logger
from .base_unsupervised import BaseUnsupervisedDetector

class LOFDetector(BaseUnsupervisedDetector):
    """
    Local Outlier Factor (LOF) anomaly detector for financial transactions.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the LOF detector.
        
        Args:
            config: Configuration dictionary containing model parameters
        """
        super().__init__(config, "LocalOutlierFactor")
        
        # Get model parameters from config
        model_config = config.get('models', {}).get('unsupervised', {}).get('local_outlier_factor', {})
        
        self.model_params = {
            'n_neighbors': model_config.get('n_neighbors', 20),
            'contamination': model_config.get('contamination', 0.1),
            'algorithm': model_config.get('algorithm', 'auto'),
            'leaf_size': model_config.get('leaf_size', 30),
            'metric': model_config.get('metric', 'minkowski'),
            'p': model_config.get('p', 2),
            'novelty': model_config.get('novelty', False)
        }
        
        self.model = LocalOutlierFactor(**self.model_params)
        self.logger.info(f"Initialized LOF with parameters: {self.model_params}")
        
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'LOFDetector':
        """
        Fit the LOF model.
        
        Args:
            X: Feature matrix
            y: Target variable (ignored for unsupervised learning)
            
        Returns:
            Self
        """
        try:
            self.validate_input(X)
            self.logger.info(f"Fitting LOF on {X.shape[0]} samples with {X.shape[1]} features")
            
            # For LOF, we need to set novelty=True for prediction on new data
            # But for fitting, we use the original model
            self.model.fit(X)
            self.is_fitted = True
            
            # Log some statistics
            self.logger.info("LOF fitted successfully")
            self.logger.info(f"Number of neighbors: {self.model_params['n_neighbors']}")
            self.logger.info(f"Expected contamination: {self.model_params['contamination']}")
            
            return self
            
        except Exception as e:
            self.logger.error(f"Error fitting LOF: {e}")
            raise
            
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict anomalies using LOF.
        
        Args:
            X: Feature matrix
            
        Returns:
            Array of predictions (1 for anomaly, 0 for normal)
        """
        try:
            if not self.is_fitted:
                raise ValueError("Model must be fitted before making predictions")
                
            self.validate_input(X)
            
            # LOF returns -1 for anomalies, 1 for normal
            predictions = self.model.predict(X)
            
            # Convert to 0/1 format (0 for normal, 1 for anomaly)
            predictions = np.where(predictions == -1, 1, 0)
            
            self.logger.info(f"Predicted {np.sum(predictions)} anomalies out of {len(predictions)} samples")
            
            return predictions
            
        except Exception as e:
            self.logger.error(f"Error making predictions with LOF: {e}")
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
            
            # LOF returns negative values for anomalies
            # We'll negate them to make higher values more anomalous
            decision_scores = -decision_scores
            
            return decision_scores
            
        except Exception as e:
            self.logger.error(f"Error computing decision function: {e}")
            raise
            
    def score_samples(self, X: pd.DataFrame) -> np.ndarray:
        """
        Compute LOF scores for each sample.
        
        Args:
            X: Feature matrix
            
        Returns:
            Array of LOF scores (higher values = more anomalous)
        """
        try:
            if not self.is_fitted:
                raise ValueError("Model must be fitted before computing scores")
                
            self.validate_input(X)
            
            # Get the LOF scores
            lof_scores = self.model.negative_outlier_factor_
            
            # For new data, we need to compute LOF scores differently
            if len(lof_scores) != len(X):
                # This is new data, we need to use the decision function
                scores = self.decision_function(X)
            else:
                # This is training data, use the stored scores
                scores = -lof_scores
                
            return scores
            
        except Exception as e:
            self.logger.error(f"Error computing LOF scores: {e}")
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
            
    def get_neighbors_info(self, X: pd.DataFrame, n_samples: int = 5) -> Dict[str, Any]:
        """
        Get information about the neighbors for each sample.
        
        Args:
            X: Feature matrix
            n_samples: Number of samples to analyze
            
        Returns:
            Dictionary containing neighbors information
        """
        try:
            if not self.is_fitted:
                raise ValueError("Model must be fitted before getting neighbors info")
                
            self.validate_input(X)
            
            # Get the k-neighbors graph
            if hasattr(self.model, 'kneighbors_graph'):
                neighbors_graph = self.model.kneighbors_graph(X[:n_samples])
                
                neighbors_info = {}
                for i in range(n_samples):
                    neighbors = neighbors_graph[i].indices[neighbors_graph[i].data > 0]
                    distances = neighbors_graph[i].data[neighbors_graph[i].data > 0]
                    
                    neighbors_info[f'sample_{i}'] = {
                        'neighbors': neighbors.tolist(),
                        'distances': distances.tolist(),
                        'n_neighbors': len(neighbors)
                    }
                    
                return neighbors_info
            else:
                self.logger.warning("Neighbors graph not available for this LOF model")
                return {}
                
        except Exception as e:
            self.logger.error(f"Error getting neighbors info: {e}")
            return {}
            
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get detailed information about the LOF model.
        
        Returns:
            Dictionary containing model information
        """
        info = super().get_model_info()
        
        if self.is_fitted:
            info.update({
                'n_neighbors': self.model_params['n_neighbors'],
                'contamination': self.model_params['contamination'],
                'algorithm': self.model_params['algorithm'],
                'leaf_size': self.model_params['leaf_size'],
                'metric': self.model_params['metric'],
                'p': self.model_params['p']
            })
            
        return info
