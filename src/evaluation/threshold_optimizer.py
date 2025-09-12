"""
Threshold optimization for anomaly detection models.

This module provides functionality to find optimal thresholds for anomaly detection
models based on different criteria and business requirements.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple, Callable
from sklearn.metrics import precision_recall_curve, roc_curve
from loguru import logger
import warnings
warnings.filterwarnings('ignore')

class ThresholdOptimizer:
    """
    A class to optimize thresholds for anomaly detection models.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the ThresholdOptimizer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = logger
        self.optimization_config = config.get('evaluation', {}).get('threshold_optimization', {})
        
    def find_optimal_threshold(self, y_true: np.ndarray, y_proba: np.ndarray,
                             method: str = 'youden', 
                             custom_criteria: Optional[Callable] = None,
                             **kwargs) -> Dict[str, Any]:
        """
        Find optimal threshold using various methods.
        
        Args:
            y_true: True labels
            y_proba: Predicted probabilities
            method: Optimization method ('youden', 'f1_optimal', 'precision_recall_curve', 'custom')
            custom_criteria: Custom function for threshold optimization
            **kwargs: Additional parameters for specific methods
            
        Returns:
            Dictionary containing optimal threshold and metrics
        """
        try:
            self.logger.info(f"Finding optimal threshold using method: {method}")
            
            if method == 'youden':
                return self._youden_index_optimization(y_true, y_proba)
            elif method == 'f1_optimal':
                return self._f1_score_optimization(y_true, y_proba)
            elif method == 'precision_recall_curve':
                return self._precision_recall_optimization(y_true, y_proba)
            elif method == 'custom' and custom_criteria is not None:
                return self._custom_optimization(y_true, y_proba, custom_criteria, **kwargs)
            else:
                raise ValueError(f"Unknown optimization method: {method}")
                
        except Exception as e:
            self.logger.error(f"Error finding optimal threshold: {e}")
            return {}
            
    def _youden_index_optimization(self, y_true: np.ndarray, y_proba: np.ndarray) -> Dict[str, Any]:
        """
        Find optimal threshold using Youden's J statistic.
        
        Args:
            y_true: True labels
            y_proba: Predicted probabilities
            
        Returns:
            Dictionary containing optimal threshold and metrics
        """
        try:
            # Calculate ROC curve
            fpr, tpr, thresholds = roc_curve(y_true, y_proba)
            
            # Calculate Youden's J statistic (TPR - FPR)
            youden_j = tpr - fpr
            
            # Find optimal threshold
            optimal_idx = np.argmax(youden_j)
            optimal_threshold = thresholds[optimal_idx]
            
            # Calculate metrics at optimal threshold
            y_pred = (y_proba > optimal_threshold).astype(int)
            
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            
            metrics = {
                'threshold': optimal_threshold,
                'youden_j': youden_j[optimal_idx],
                'tpr': tpr[optimal_idx],
                'fpr': fpr[optimal_idx],
                'accuracy': accuracy_score(y_true, y_pred),
                'precision': precision_score(y_true, y_pred, zero_division=0),
                'recall': recall_score(y_true, y_pred, zero_division=0),
                'f1_score': f1_score(y_true, y_pred, zero_division=0)
            }
            
            self.logger.info(f"Youden's J optimization - Threshold: {optimal_threshold:.4f}, J: {metrics['youden_j']:.4f}")
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error in Youden's J optimization: {e}")
            return {}
            
    def _f1_score_optimization(self, y_true: np.ndarray, y_proba: np.ndarray) -> Dict[str, Any]:
        """
        Find optimal threshold by maximizing F1-score.
        
        Args:
            y_true: True labels
            y_proba: Predicted probabilities
            
        Returns:
            Dictionary containing optimal threshold and metrics
        """
        try:
            # Calculate precision-recall curve
            precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
            
            # Calculate F1-score for each threshold
            f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
            
            # Find optimal threshold
            optimal_idx = np.argmax(f1_scores)
            optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5
            
            # Calculate metrics at optimal threshold
            y_pred = (y_proba > optimal_threshold).astype(int)
            
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            
            metrics = {
                'threshold': optimal_threshold,
                'f1_score': f1_scores[optimal_idx],
                'precision': precision[optimal_idx],
                'recall': recall[optimal_idx],
                'accuracy': accuracy_score(y_true, y_pred),
                'precision_at_threshold': precision_score(y_true, y_pred, zero_division=0),
                'recall_at_threshold': recall_score(y_true, y_pred, zero_division=0),
                'f1_score_at_threshold': f1_score(y_true, y_pred, zero_division=0)
            }
            
            self.logger.info(f"F1-score optimization - Threshold: {optimal_threshold:.4f}, F1: {metrics['f1_score']:.4f}")
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error in F1-score optimization: {e}")
            return {}
            
    def _precision_recall_optimization(self, y_true: np.ndarray, y_proba: np.ndarray) -> Dict[str, Any]:
        """
        Find optimal threshold using precision-recall curve analysis.
        
        Args:
            y_true: True labels
            y_proba: Predicted probabilities
            
        Returns:
            Dictionary containing optimal threshold and metrics
        """
        try:
            # Calculate precision-recall curve
            precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
            
            # Find threshold that balances precision and recall
            # Using harmonic mean of precision and recall
            harmonic_mean = 2 * (precision * recall) / (precision + recall + 1e-8)
            
            optimal_idx = np.argmax(harmonic_mean)
            optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5
            
            # Calculate metrics at optimal threshold
            y_pred = (y_proba > optimal_threshold).astype(int)
            
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            
            metrics = {
                'threshold': optimal_threshold,
                'precision': precision[optimal_idx],
                'recall': recall[optimal_idx],
                'harmonic_mean': harmonic_mean[optimal_idx],
                'accuracy': accuracy_score(y_true, y_pred),
                'precision_at_threshold': precision_score(y_true, y_pred, zero_division=0),
                'recall_at_threshold': recall_score(y_true, y_pred, zero_division=0),
                'f1_score_at_threshold': f1_score(y_true, y_pred, zero_division=0)
            }
            
            self.logger.info(f"Precision-Recall optimization - Threshold: {optimal_threshold:.4f}")
            self.logger.info(f"  Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}")
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error in precision-recall optimization: {e}")
            return {}
            
    def _custom_optimization(self, y_true: np.ndarray, y_proba: np.ndarray,
                           custom_criteria: Callable, **kwargs) -> Dict[str, Any]:
        """
        Find optimal threshold using custom criteria.
        
        Args:
            y_true: True labels
            y_proba: Predicted probabilities
            custom_criteria: Custom function for optimization
            **kwargs: Additional parameters
            
        Returns:
            Dictionary containing optimal threshold and metrics
        """
        try:
            # Generate threshold range
            thresholds = np.linspace(0.01, 0.99, 99)
            
            best_score = -np.inf
            best_threshold = 0.5
            best_metrics = {}
            
            for threshold in thresholds:
                y_pred = (y_proba > threshold).astype(int)
                
                # Calculate custom score
                score = custom_criteria(y_true, y_pred, y_proba, threshold, **kwargs)
                
                if score > best_score:
                    best_score = score
                    best_threshold = threshold
                    
                    # Calculate metrics at this threshold
                    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
                    
                    best_metrics = {
                        'threshold': threshold,
                        'custom_score': score,
                        'accuracy': accuracy_score(y_true, y_pred),
                        'precision': precision_score(y_true, y_pred, zero_division=0),
                        'recall': recall_score(y_true, y_pred, zero_division=0),
                        'f1_score': f1_score(y_true, y_pred, zero_division=0)
                    }
                    
            self.logger.info(f"Custom optimization - Threshold: {best_threshold:.4f}, Score: {best_score:.4f}")
            
            return best_metrics
            
        except Exception as e:
            self.logger.error(f"Error in custom optimization: {e}")
            return {}
            
    def optimize_for_business_requirements(self, y_true: np.ndarray, y_proba: np.ndarray,
                                         max_false_positive_rate: float = 0.05,
                                         min_recall: float = 0.8) -> Dict[str, Any]:
        """
        Optimize threshold based on business requirements.
        
        Args:
            y_true: True labels
            y_proba: Predicted probabilities
            max_false_positive_rate: Maximum acceptable false positive rate
            min_recall: Minimum required recall
            
        Returns:
            Dictionary containing optimal threshold and metrics
        """
        try:
            self.logger.info(f"Optimizing for business requirements:")
            self.logger.info(f"  Max FPR: {max_false_positive_rate}")
            self.logger.info(f"  Min Recall: {min_recall}")
            
            # Calculate ROC curve
            fpr, tpr, thresholds = roc_curve(y_true, y_proba)
            
            # Find thresholds that meet business requirements
            valid_indices = (fpr <= max_false_positive_rate) & (tpr >= min_recall)
            
            if not np.any(valid_indices):
                self.logger.warning("No threshold meets the business requirements")
                # Return threshold with minimum FPR
                optimal_idx = np.argmin(fpr)
            else:
                # Among valid thresholds, choose the one with highest TPR
                valid_tpr = tpr[valid_indices]
                valid_fpr = fpr[valid_indices]
                valid_thresholds = thresholds[valid_indices]
                
                optimal_idx = np.argmax(valid_tpr)
                optimal_threshold = valid_thresholds[optimal_idx]
                
            if not np.any(valid_indices):
                optimal_threshold = thresholds[optimal_idx]
                
            # Calculate metrics at optimal threshold
            y_pred = (y_proba > optimal_threshold).astype(int)
            
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
            
            cm = confusion_matrix(y_true, y_pred)
            tn, fp, fn, tp = cm.ravel()
            
            metrics = {
                'threshold': optimal_threshold,
                'accuracy': accuracy_score(y_true, y_pred),
                'precision': precision_score(y_true, y_pred, zero_division=0),
                'recall': recall_score(y_true, y_pred, zero_division=0),
                'f1_score': f1_score(y_true, y_pred, zero_division=0),
                'false_positive_rate': fp / (fp + tn) if (fp + tn) > 0 else 0,
                'true_positive_rate': tp / (tp + fn) if (tp + fn) > 0 else 0,
                'meets_requirements': np.any(valid_indices),
                'business_constraints': {
                    'max_fpr': max_false_positive_rate,
                    'min_recall': min_recall
                }
            }
            
            self.logger.info(f"Business optimization - Threshold: {optimal_threshold:.4f}")
            self.logger.info(f"  FPR: {metrics['false_positive_rate']:.4f}, Recall: {metrics['recall']:.4f}")
            self.logger.info(f"  Meets requirements: {metrics['meets_requirements']}")
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error in business requirements optimization: {e}")
            return {}
            
    def analyze_threshold_sensitivity(self, y_true: np.ndarray, y_proba: np.ndarray,
                                    threshold_range: Tuple[float, float] = (0.1, 0.9),
                                    n_points: int = 20) -> pd.DataFrame:
        """
        Analyze how metrics change with different thresholds.
        
        Args:
            y_true: True labels
            y_proba: Predicted probabilities
            threshold_range: Range of thresholds to analyze
            n_points: Number of threshold points to evaluate
            
        Returns:
            DataFrame with metrics for each threshold
        """
        try:
            thresholds = np.linspace(threshold_range[0], threshold_range[1], n_points)
            
            results = []
            
            for threshold in thresholds:
                y_pred = (y_proba > threshold).astype(int)
                
                from sklearn.metrics import (
                    accuracy_score, precision_score, recall_score, f1_score,
                    confusion_matrix
                )
                
                cm = confusion_matrix(y_true, y_pred)
                tn, fp, fn, tp = cm.ravel()
                
                result = {
                    'threshold': threshold,
                    'accuracy': accuracy_score(y_true, y_pred),
                    'precision': precision_score(y_true, y_pred, zero_division=0),
                    'recall': recall_score(y_true, y_pred, zero_division=0),
                    'f1_score': f1_score(y_true, y_pred, zero_division=0),
                    'true_positives': tp,
                    'false_positives': fp,
                    'true_negatives': tn,
                    'false_negatives': fn,
                    'false_positive_rate': fp / (fp + tn) if (fp + tn) > 0 else 0,
                    'true_positive_rate': tp / (tp + fn) if (tp + fn) > 0 else 0
                }
                
                results.append(result)
                
            sensitivity_df = pd.DataFrame(results)
            
            self.logger.info(f"Threshold sensitivity analysis completed for {n_points} points")
            
            return sensitivity_df
            
        except Exception as e:
            self.logger.error(f"Error in threshold sensitivity analysis: {e}")
            return pd.DataFrame()
            
    def find_multiple_optimal_thresholds(self, y_true: np.ndarray, y_proba: np.ndarray,
                                       methods: List[str] = None) -> Dict[str, Dict[str, Any]]:
        """
        Find optimal thresholds using multiple methods.
        
        Args:
            y_true: True labels
            y_proba: Predicted probabilities
            methods: List of methods to use
            
        Returns:
            Dictionary containing results for each method
        """
        try:
            if methods is None:
                methods = ['youden', 'f1_optimal', 'precision_recall_curve']
                
            results = {}
            
            for method in methods:
                self.logger.info(f"Finding optimal threshold using method: {method}")
                result = self.find_optimal_threshold(y_true, y_proba, method)
                results[method] = result
                
            self.logger.info(f"Multiple threshold optimization completed for {len(methods)} methods")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in multiple threshold optimization: {e}")
            return {}
