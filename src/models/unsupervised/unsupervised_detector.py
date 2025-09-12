"""
Main unsupervised anomaly detector class that combines all unsupervised methods.

This module provides a unified interface for all unsupervised anomaly detection
algorithms and allows for easy comparison and ensemble methods.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from loguru import logger

from .isolation_forest import IsolationForestDetector
from .lof_detector import LOFDetector
from .one_class_svm import OneClassSVMDetector
from .autoencoder import AutoencoderDetector
from .base_unsupervised import BaseUnsupervisedDetector

class UnsupervisedAnomalyDetector:
    """
    Main class for unsupervised anomaly detection that combines multiple algorithms.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the unsupervised anomaly detector.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = logger
        self.detectors = {}
        self.ensemble_weights = {}
        self.is_fitted = False
        
        # Initialize individual detectors
        self._initialize_detectors()
        
    def _initialize_detectors(self) -> None:
        """
        Initialize all available unsupervised detectors.
        """
        try:
            # Initialize Isolation Forest
            self.detectors['isolation_forest'] = IsolationForestDetector(self.config)
            
            # Initialize LOF
            self.detectors['lof'] = LOFDetector(self.config)
            
            # Initialize One-Class SVM
            self.detectors['one_class_svm'] = OneClassSVMDetector(self.config)
            
            # Initialize Autoencoder
            self.detectors['autoencoder'] = AutoencoderDetector(self.config)
            
            # Set default ensemble weights (equal weights)
            self.ensemble_weights = {
                'isolation_forest': 0.25,
                'lof': 0.25,
                'one_class_svm': 0.25,
                'autoencoder': 0.25
            }
            
            self.logger.info(f"Initialized {len(self.detectors)} unsupervised detectors")
            
        except Exception as e:
            self.logger.error(f"Error initializing detectors: {e}")
            raise
            
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None, 
            detectors: Optional[List[str]] = None) -> 'UnsupervisedAnomalyDetector':
        """
        Fit all or selected unsupervised detectors.
        
        Args:
            X: Feature matrix
            y: Target variable (ignored for unsupervised learning)
            detectors: List of detector names to fit (if None, fits all)
            
        Returns:
            Self
        """
        try:
            if detectors is None:
                detectors = list(self.detectors.keys())
                
            self.logger.info(f"Fitting {len(detectors)} unsupervised detectors")
            
            for detector_name in detectors:
                if detector_name in self.detectors:
                    self.logger.info(f"Fitting {detector_name}...")
                    self.detectors[detector_name].fit(X, y)
                else:
                    self.logger.warning(f"Detector {detector_name} not found")
                    
            self.is_fitted = True
            self.logger.info("All selected detectors fitted successfully")
            
            return self
            
        except Exception as e:
            self.logger.error(f"Error fitting detectors: {e}")
            raise
            
    def predict(self, X: pd.DataFrame, method: str = 'ensemble',
                detectors: Optional[List[str]] = None) -> np.ndarray:
        """
        Predict anomalies using individual detectors or ensemble method.
        
        Args:
            X: Feature matrix
            method: Prediction method ('ensemble', 'voting', 'individual')
            detectors: List of detector names to use (if None, uses all fitted detectors)
            
        Returns:
            Array of predictions (1 for anomaly, 0 for normal)
        """
        try:
            if not self.is_fitted:
                raise ValueError("Detectors must be fitted before making predictions")
                
            if detectors is None:
                detectors = [name for name, detector in self.detectors.items() if detector.is_fitted]
                
            if method == 'ensemble':
                return self._ensemble_predict(X, detectors)
            elif method == 'voting':
                return self._voting_predict(X, detectors)
            elif method == 'individual':
                return self._individual_predict(X, detectors)
            else:
                raise ValueError(f"Unknown prediction method: {method}")
                
        except Exception as e:
            self.logger.error(f"Error making predictions: {e}")
            raise
            
    def _ensemble_predict(self, X: pd.DataFrame, detectors: List[str]) -> np.ndarray:
        """
        Make ensemble predictions using weighted average of decision functions.
        
        Args:
            X: Feature matrix
            detectors: List of detector names to use
            
        Returns:
            Array of ensemble predictions
        """
        try:
            decision_scores = []
            weights = []
            
            for detector_name in detectors:
                if detector_name in self.detectors and self.detectors[detector_name].is_fitted:
                    scores = self.detectors[detector_name].decision_function(X)
                    decision_scores.append(scores)
                    weights.append(self.ensemble_weights.get(detector_name, 1.0))
                    
            if not decision_scores:
                raise ValueError("No fitted detectors available for ensemble prediction")
                
            # Normalize weights
            weights = np.array(weights)
            weights = weights / weights.sum()
            
            # Calculate weighted average
            ensemble_scores = np.average(decision_scores, axis=0, weights=weights)
            
            # Calculate threshold (using 90th percentile as default)
            threshold = np.percentile(ensemble_scores, 90)
            
            # Make predictions
            predictions = (ensemble_scores > threshold).astype(int)
            
            self.logger.info(f"Ensemble prediction completed using {len(detectors)} detectors")
            self.logger.info(f"Predicted {np.sum(predictions)} anomalies")
            
            return predictions
            
        except Exception as e:
            self.logger.error(f"Error in ensemble prediction: {e}")
            raise
            
    def _voting_predict(self, X: pd.DataFrame, detectors: List[str]) -> np.ndarray:
        """
        Make voting predictions using majority vote.
        
        Args:
            X: Feature matrix
            detectors: List of detector names to use
            
        Returns:
            Array of voting predictions
        """
        try:
            predictions_list = []
            
            for detector_name in detectors:
                if detector_name in self.detectors and self.detectors[detector_name].is_fitted:
                    pred = self.detectors[detector_name].predict(X)
                    predictions_list.append(pred)
                    
            if not predictions_list:
                raise ValueError("No fitted detectors available for voting prediction")
                
            # Calculate majority vote
            predictions_array = np.array(predictions_list)
            voting_predictions = (np.sum(predictions_array, axis=0) > len(detectors) / 2).astype(int)
            
            self.logger.info(f"Voting prediction completed using {len(detectors)} detectors")
            self.logger.info(f"Predicted {np.sum(voting_predictions)} anomalies")
            
            return voting_predictions
            
        except Exception as e:
            self.logger.error(f"Error in voting prediction: {e}")
            raise
            
    def _individual_predict(self, X: pd.DataFrame, detectors: List[str]) -> Dict[str, np.ndarray]:
        """
        Make individual predictions from each detector.
        
        Args:
            X: Feature matrix
            detectors: List of detector names to use
            
        Returns:
            Dictionary of predictions from each detector
        """
        try:
            individual_predictions = {}
            
            for detector_name in detectors:
                if detector_name in self.detectors and self.detectors[detector_name].is_fitted:
                    pred = self.detectors[detector_name].predict(X)
                    individual_predictions[detector_name] = pred
                    
            self.logger.info(f"Individual predictions completed for {len(individual_predictions)} detectors")
            
            return individual_predictions
            
        except Exception as e:
            self.logger.error(f"Error in individual prediction: {e}")
            raise
            
    def get_anomaly_scores(self, X: pd.DataFrame, 
                          detectors: Optional[List[str]] = None) -> Dict[str, np.ndarray]:
        """
        Get anomaly scores from all detectors.
        
        Args:
            X: Feature matrix
            detectors: List of detector names to use (if None, uses all fitted detectors)
            
        Returns:
            Dictionary of anomaly scores from each detector
        """
        try:
            if not self.is_fitted:
                raise ValueError("Detectors must be fitted before computing scores")
                
            if detectors is None:
                detectors = [name for name, detector in self.detectors.items() if detector.is_fitted]
                
            scores = {}
            
            for detector_name in detectors:
                if detector_name in self.detectors and self.detectors[detector_name].is_fitted:
                    detector_scores = self.detectors[detector_name].score_samples(X)
                    scores[detector_name] = detector_scores
                    
            self.logger.info(f"Anomaly scores computed for {len(scores)} detectors")
            
            return scores
            
        except Exception as e:
            self.logger.error(f"Error computing anomaly scores: {e}")
            raise
            
    def set_ensemble_weights(self, weights: Dict[str, float]) -> None:
        """
        Set ensemble weights for different detectors.
        
        Args:
            weights: Dictionary of detector names and their weights
        """
        try:
            # Validate weights
            total_weight = sum(weights.values())
            if abs(total_weight - 1.0) > 1e-6:
                self.logger.warning(f"Ensemble weights sum to {total_weight}, normalizing to 1.0")
                weights = {k: v/total_weight for k, v in weights.items()}
                
            self.ensemble_weights.update(weights)
            self.logger.info(f"Ensemble weights updated: {self.ensemble_weights}")
            
        except Exception as e:
            self.logger.error(f"Error setting ensemble weights: {e}")
            raise
            
    def get_detector_info(self) -> Dict[str, Any]:
        """
        Get information about all detectors.
        
        Returns:
            Dictionary containing information about each detector
        """
        try:
            info = {
                'fitted_detectors': [],
                'available_detectors': list(self.detectors.keys()),
                'ensemble_weights': self.ensemble_weights.copy(),
                'is_fitted': self.is_fitted
            }
            
            for name, detector in self.detectors.items():
                if detector.is_fitted:
                    info['fitted_detectors'].append(name)
                    info[f'{name}_info'] = detector.get_model_info()
                    
            return info
            
        except Exception as e:
            self.logger.error(f"Error getting detector info: {e}")
            return {}
            
    def save_models(self, save_path: str) -> None:
        """
        Save all fitted models to disk.
        
        Args:
            save_path: Path to save the models
        """
        try:
            import os
            os.makedirs(save_path, exist_ok=True)
            
            for name, detector in self.detectors.items():
                if detector.is_fitted:
                    model_path = f"{save_path}/{name}_model.joblib"
                    detector.save_model(model_path)
                    
            # Save ensemble weights
            import joblib
            weights_path = f"{save_path}/ensemble_weights.joblib"
            joblib.dump(self.ensemble_weights, weights_path)
            
            self.logger.info(f"All models saved to {save_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving models: {e}")
            raise
            
    def load_models(self, load_path: str) -> None:
        """
        Load all models from disk.
        
        Args:
            load_path: Path to load the models from
        """
        try:
            import os
            import joblib
            
            for name, detector in self.detectors.items():
                model_path = f"{load_path}/{name}_model.joblib"
                if os.path.exists(model_path):
                    detector.load_model(model_path)
                    
            # Load ensemble weights
            weights_path = f"{load_path}/ensemble_weights.joblib"
            if os.path.exists(weights_path):
                self.ensemble_weights = joblib.load(weights_path)
                
            self.is_fitted = any(detector.is_fitted for detector in self.detectors.values())
            self.logger.info(f"Models loaded from {load_path}")
            
        except Exception as e:
            self.logger.error(f"Error loading models: {e}")
            raise
