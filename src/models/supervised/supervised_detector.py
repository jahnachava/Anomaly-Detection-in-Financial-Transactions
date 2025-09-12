"""
Main supervised anomaly detector class that combines all supervised methods.

This module provides a unified interface for all supervised anomaly detection
algorithms and allows for easy comparison and ensemble methods.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from loguru import logger

from .random_forest import RandomForestDetector
from .xgboost_detector import XGBoostDetector
from .neural_network import NeuralNetworkDetector
from .base_supervised import BaseSupervisedDetector

class SupervisedAnomalyDetector:
    """
    Main class for supervised anomaly detection that combines multiple algorithms.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the supervised anomaly detector.
        
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
        Initialize all available supervised detectors.
        """
        try:
            # Initialize Random Forest
            self.detectors['random_forest'] = RandomForestDetector(self.config)
            
            # Initialize XGBoost
            self.detectors['xgboost'] = XGBoostDetector(self.config)
            
            # Initialize Neural Network
            self.detectors['neural_network'] = NeuralNetworkDetector(self.config)
            
            # Set default ensemble weights (equal weights)
            self.ensemble_weights = {
                'random_forest': 0.33,
                'xgboost': 0.33,
                'neural_network': 0.34
            }
            
            self.logger.info(f"Initialized {len(self.detectors)} supervised detectors")
            
        except Exception as e:
            self.logger.error(f"Error initializing detectors: {e}")
            raise
            
    def fit(self, X: pd.DataFrame, y: pd.Series, 
            detectors: Optional[List[str]] = None) -> 'SupervisedAnomalyDetector':
        """
        Fit all or selected supervised detectors.
        
        Args:
            X: Feature matrix
            y: Target variable (0 for normal, 1 for anomaly)
            detectors: List of detector names to fit (if None, fits all)
            
        Returns:
            Self
        """
        try:
            if detectors is None:
                detectors = list(self.detectors.keys())
                
            self.logger.info(f"Fitting {len(detectors)} supervised detectors")
            
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
                detectors: Optional[List[str]] = None,
                threshold: float = 0.5) -> np.ndarray:
        """
        Predict anomalies using individual detectors or ensemble method.
        
        Args:
            X: Feature matrix
            method: Prediction method ('ensemble', 'voting', 'individual')
            detectors: List of detector names to use (if None, uses all fitted detectors)
            threshold: Probability threshold for ensemble predictions
            
        Returns:
            Array of predictions (1 for anomaly, 0 for normal)
        """
        try:
            if not self.is_fitted:
                raise ValueError("Detectors must be fitted before making predictions")
                
            if detectors is None:
                detectors = [name for name, detector in self.detectors.items() if detector.is_fitted]
                
            if method == 'ensemble':
                return self._ensemble_predict(X, detectors, threshold)
            elif method == 'voting':
                return self._voting_predict(X, detectors)
            elif method == 'individual':
                return self._individual_predict(X, detectors)
            else:
                raise ValueError(f"Unknown prediction method: {method}")
                
        except Exception as e:
            self.logger.error(f"Error making predictions: {e}")
            raise
            
    def _ensemble_predict(self, X: pd.DataFrame, detectors: List[str], 
                         threshold: float = 0.5) -> np.ndarray:
        """
        Make ensemble predictions using weighted average of probabilities.
        
        Args:
            X: Feature matrix
            detectors: List of detector names to use
            threshold: Probability threshold for predictions
            
        Returns:
            Array of ensemble predictions
        """
        try:
            probabilities_list = []
            weights = []
            
            for detector_name in detectors:
                if detector_name in self.detectors and self.detectors[detector_name].is_fitted:
                    probs = self.detectors[detector_name].predict_proba(X)[:, 1]  # Probability of anomaly
                    probabilities_list.append(probs)
                    weights.append(self.ensemble_weights.get(detector_name, 1.0))
                    
            if not probabilities_list:
                raise ValueError("No fitted detectors available for ensemble prediction")
                
            # Normalize weights
            weights = np.array(weights)
            weights = weights / weights.sum()
            
            # Calculate weighted average of probabilities
            ensemble_probs = np.average(probabilities_list, axis=0, weights=weights)
            
            # Make predictions based on threshold
            predictions = (ensemble_probs > threshold).astype(int)
            
            self.logger.info(f"Ensemble prediction completed using {len(detectors)} detectors")
            self.logger.info(f"Predicted {np.sum(predictions)} anomalies with threshold {threshold}")
            
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
            
    def get_ensemble_probabilities(self, X: pd.DataFrame, 
                                  detectors: Optional[List[str]] = None) -> np.ndarray:
        """
        Get ensemble probabilities from all detectors.
        
        Args:
            X: Feature matrix
            detectors: List of detector names to use (if None, uses all fitted detectors)
            
        Returns:
            Array of ensemble probabilities
        """
        try:
            if not self.is_fitted:
                raise ValueError("Detectors must be fitted before computing probabilities")
                
            if detectors is None:
                detectors = [name for name, detector in self.detectors.items() if detector.is_fitted]
                
            probabilities_list = []
            weights = []
            
            for detector_name in detectors:
                if detector_name in self.detectors and self.detectors[detector_name].is_fitted:
                    probs = self.detectors[detector_name].predict_proba(X)[:, 1]  # Probability of anomaly
                    probabilities_list.append(probs)
                    weights.append(self.ensemble_weights.get(detector_name, 1.0))
                    
            if not probabilities_list:
                raise ValueError("No fitted detectors available for ensemble probabilities")
                
            # Normalize weights
            weights = np.array(weights)
            weights = weights / weights.sum()
            
            # Calculate weighted average of probabilities
            ensemble_probs = np.average(probabilities_list, axis=0, weights=weights)
            
            return ensemble_probs
            
        except Exception as e:
            self.logger.error(f"Error computing ensemble probabilities: {e}")
            raise
            
    def evaluate_ensemble(self, X: pd.DataFrame, y: pd.Series,
                         detectors: Optional[List[str]] = None,
                         threshold: float = 0.5) -> Dict[str, Any]:
        """
        Evaluate the ensemble model performance.
        
        Args:
            X: Feature matrix
            y: True labels
            detectors: List of detector names to use
            threshold: Probability threshold for predictions
            
        Returns:
            Dictionary containing evaluation metrics
        """
        try:
            if not self.is_fitted:
                raise ValueError("Detectors must be fitted before evaluation")
                
            # Get ensemble predictions
            y_pred = self.predict(X, method='ensemble', detectors=detectors, threshold=threshold)
            y_proba = self.get_ensemble_probabilities(X, detectors)
            
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
            
            self.logger.info(f"Ensemble evaluation completed:")
            self.logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
            self.logger.info(f"  Precision: {metrics['precision']:.4f}")
            self.logger.info(f"  Recall: {metrics['recall']:.4f}")
            self.logger.info(f"  F1-Score: {metrics['f1_score']:.4f}")
            self.logger.info(f"  ROC-AUC: {metrics['roc_auc']:.4f}")
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error evaluating ensemble: {e}")
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
