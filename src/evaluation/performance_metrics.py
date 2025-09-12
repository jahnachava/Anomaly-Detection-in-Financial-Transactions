"""
Performance metrics for anomaly detection evaluation.

This module provides comprehensive metrics for evaluating anomaly detection
models including accuracy, precision, recall, F1-score, ROC-AUC, and more.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report, precision_recall_curve, roc_curve,
    matthews_corrcoef, cohen_kappa_score
)
from loguru import logger
import warnings
warnings.filterwarnings('ignore')

class PerformanceMetrics:
    """
    A class to compute comprehensive performance metrics for anomaly detection.
    """
    
    def __init__(self):
        """
        Initialize the PerformanceMetrics class.
        """
        self.logger = logger
        
    def compute_basic_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Compute basic classification metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Dictionary containing basic metrics
        """
        try:
            metrics = {
                'accuracy': accuracy_score(y_true, y_pred),
                'precision': precision_score(y_true, y_pred, zero_division=0),
                'recall': recall_score(y_true, y_pred, zero_division=0),
                'f1_score': f1_score(y_true, y_pred, zero_division=0),
                'matthews_corrcoef': matthews_corrcoef(y_true, y_pred),
                'cohen_kappa': cohen_kappa_score(y_true, y_pred)
            }
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error computing basic metrics: {e}")
            return {}
            
    def compute_probability_metrics(self, y_true: np.ndarray, y_proba: np.ndarray) -> Dict[str, float]:
        """
        Compute probability-based metrics.
        
        Args:
            y_true: True labels
            y_proba: Predicted probabilities for positive class
            
        Returns:
            Dictionary containing probability-based metrics
        """
        try:
            metrics = {
                'roc_auc': roc_auc_score(y_true, y_proba),
                'average_precision': average_precision_score(y_true, y_proba)
            }
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error computing probability metrics: {e}")
            return {}
            
    def compute_confusion_matrix_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
        """
        Compute metrics derived from confusion matrix.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Dictionary containing confusion matrix metrics
        """
        try:
            cm = confusion_matrix(y_true, y_pred)
            
            # Extract values from confusion matrix
            tn, fp, fn, tp = cm.ravel()
            
            metrics = {
                'confusion_matrix': cm.tolist(),
                'true_negatives': int(tn),
                'false_positives': int(fp),
                'false_negatives': int(fn),
                'true_positives': int(tp),
                'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
                'sensitivity': tp / (tp + fn) if (tp + fn) > 0 else 0,
                'false_positive_rate': fp / (fp + tn) if (fp + tn) > 0 else 0,
                'false_negative_rate': fn / (fn + tp) if (fn + tp) > 0 else 0,
                'positive_predictive_value': tp / (tp + fp) if (tp + fp) > 0 else 0,
                'negative_predictive_value': tn / (tn + fn) if (tn + fn) > 0 else 0
            }
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error computing confusion matrix metrics: {e}")
            return {}
            
    def compute_anomaly_specific_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                       y_proba: np.ndarray = None) -> Dict[str, Any]:
        """
        Compute metrics specific to anomaly detection.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Predicted probabilities (optional)
            
        Returns:
            Dictionary containing anomaly-specific metrics
        """
        try:
            metrics = {}
            
            # Basic anomaly detection metrics
            cm_metrics = self.compute_confusion_matrix_metrics(y_true, y_pred)
            metrics.update(cm_metrics)
            
            # Anomaly detection rate
            total_anomalies = np.sum(y_true)
            detected_anomalies = np.sum((y_true == 1) & (y_pred == 1))
            metrics['anomaly_detection_rate'] = detected_anomalies / total_anomalies if total_anomalies > 0 else 0
            
            # False alarm rate
            total_normal = np.sum(y_true == 0)
            false_alarms = np.sum((y_true == 0) & (y_pred == 1))
            metrics['false_alarm_rate'] = false_alarms / total_normal if total_normal > 0 else 0
            
            # Precision-Recall curve metrics
            if y_proba is not None:
                precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
                
                # Find optimal threshold (F1-score maximization)
                f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
                optimal_idx = np.argmax(f1_scores)
                
                metrics['optimal_threshold'] = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5
                metrics['optimal_f1_score'] = f1_scores[optimal_idx]
                metrics['optimal_precision'] = precision[optimal_idx]
                metrics['optimal_recall'] = recall[optimal_idx]
                
                # Area under precision-recall curve
                metrics['pr_auc'] = average_precision_score(y_true, y_proba)
                
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error computing anomaly-specific metrics: {e}")
            return {}
            
    def compute_comprehensive_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                    y_proba: np.ndarray = None) -> Dict[str, Any]:
        """
        Compute all available metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Predicted probabilities (optional)
            
        Returns:
            Dictionary containing all metrics
        """
        try:
            metrics = {}
            
            # Basic metrics
            basic_metrics = self.compute_basic_metrics(y_true, y_pred)
            metrics.update(basic_metrics)
            
            # Confusion matrix metrics
            cm_metrics = self.compute_confusion_matrix_metrics(y_true, y_pred)
            metrics.update(cm_metrics)
            
            # Anomaly-specific metrics
            anomaly_metrics = self.compute_anomaly_specific_metrics(y_true, y_pred, y_proba)
            metrics.update(anomaly_metrics)
            
            # Probability-based metrics
            if y_proba is not None:
                prob_metrics = self.compute_probability_metrics(y_true, y_proba)
                metrics.update(prob_metrics)
                
            # Classification report
            try:
                metrics['classification_report'] = classification_report(
                    y_true, y_pred, output_dict=True, zero_division=0
                )
            except Exception as e:
                self.logger.warning(f"Could not generate classification report: {e}")
                
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error computing comprehensive metrics: {e}")
            return {}
            
    def compute_metrics_by_threshold(self, y_true: np.ndarray, y_proba: np.ndarray, 
                                   thresholds: np.ndarray = None) -> pd.DataFrame:
        """
        Compute metrics for different probability thresholds.
        
        Args:
            y_true: True labels
            y_proba: Predicted probabilities
            thresholds: Array of thresholds to evaluate (if None, uses default range)
            
        Returns:
            DataFrame containing metrics for each threshold
        """
        try:
            if thresholds is None:
                thresholds = np.linspace(0.1, 0.9, 9)
                
            results = []
            
            for threshold in thresholds:
                y_pred = (y_proba > threshold).astype(int)
                
                metrics = self.compute_basic_metrics(y_true, y_pred)
                cm_metrics = self.compute_confusion_matrix_metrics(y_true, y_pred)
                
                result = {
                    'threshold': threshold,
                    **metrics,
                    **cm_metrics
                }
                
                results.append(result)
                
            return pd.DataFrame(results)
            
        except Exception as e:
            self.logger.error(f"Error computing metrics by threshold: {e}")
            return pd.DataFrame()
            
    def find_optimal_threshold(self, y_true: np.ndarray, y_proba: np.ndarray, 
                             metric: str = 'f1_score') -> Dict[str, Any]:
        """
        Find the optimal threshold for a given metric.
        
        Args:
            y_true: True labels
            y_proba: Predicted probabilities
            metric: Metric to optimize ('f1_score', 'precision', 'recall', 'accuracy')
            
        Returns:
            Dictionary containing optimal threshold and corresponding metrics
        """
        try:
            thresholds = np.linspace(0.01, 0.99, 99)
            best_score = -1
            best_threshold = 0.5
            best_metrics = {}
            
            for threshold in thresholds:
                y_pred = (y_proba > threshold).astype(int)
                metrics = self.compute_basic_metrics(y_true, y_pred)
                
                if metric in metrics and metrics[metric] > best_score:
                    best_score = metrics[metric]
                    best_threshold = threshold
                    best_metrics = metrics.copy()
                    best_metrics['threshold'] = threshold
                    
            self.logger.info(f"Optimal threshold for {metric}: {best_threshold:.4f} (score: {best_score:.4f})")
            
            return best_metrics
            
        except Exception as e:
            self.logger.error(f"Error finding optimal threshold: {e}")
            return {}
            
    def compute_cost_metrics(self, y_true: np.ndarray, y_pred: np.ndarray,
                           cost_fp: float = 1.0, cost_fn: float = 10.0) -> Dict[str, float]:
        """
        Compute cost-based metrics for anomaly detection.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            cost_fp: Cost of false positive
            cost_fn: Cost of false negative
            
        Returns:
            Dictionary containing cost metrics
        """
        try:
            cm = confusion_matrix(y_true, y_pred)
            tn, fp, fn, tp = cm.ravel()
            
            total_cost = (fp * cost_fp) + (fn * cost_fn)
            total_samples = len(y_true)
            cost_per_sample = total_cost / total_samples
            
            metrics = {
                'total_cost': total_cost,
                'cost_per_sample': cost_per_sample,
                'false_positive_cost': fp * cost_fp,
                'false_negative_cost': fn * cost_fn,
                'cost_fp': cost_fp,
                'cost_fn': cost_fn
            }
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error computing cost metrics: {e}")
            return {}
            
    def generate_metrics_summary(self, metrics: Dict[str, Any]) -> str:
        """
        Generate a formatted summary of metrics.
        
        Args:
            metrics: Dictionary containing metrics
            
        Returns:
            Formatted string summary
        """
        try:
            summary = []
            summary.append("=" * 50)
            summary.append("PERFORMANCE METRICS SUMMARY")
            summary.append("=" * 50)
            
            # Basic metrics
            if 'accuracy' in metrics:
                summary.append(f"Accuracy:           {metrics['accuracy']:.4f}")
            if 'precision' in metrics:
                summary.append(f"Precision:          {metrics['precision']:.4f}")
            if 'recall' in metrics:
                summary.append(f"Recall:             {metrics['recall']:.4f}")
            if 'f1_score' in metrics:
                summary.append(f"F1-Score:           {metrics['f1_score']:.4f}")
                
            # ROC-AUC
            if 'roc_auc' in metrics:
                summary.append(f"ROC-AUC:            {metrics['roc_auc']:.4f}")
            if 'average_precision' in metrics:
                summary.append(f"Average Precision:  {metrics['average_precision']:.4f}")
                
            # Anomaly-specific metrics
            if 'anomaly_detection_rate' in metrics:
                summary.append(f"Anomaly Detection Rate: {metrics['anomaly_detection_rate']:.4f}")
            if 'false_alarm_rate' in metrics:
                summary.append(f"False Alarm Rate:       {metrics['false_alarm_rate']:.4f}")
                
            # Confusion matrix
            if 'confusion_matrix' in metrics:
                cm = metrics['confusion_matrix']
                summary.append("\nConfusion Matrix:")
                summary.append(f"  TN: {cm[0][0]}, FP: {cm[0][1]}")
                summary.append(f"  FN: {cm[1][0]}, TP: {cm[1][1]}")
                
            summary.append("=" * 50)
            
            return "\n".join(summary)
            
        except Exception as e:
            self.logger.error(f"Error generating metrics summary: {e}")
            return "Error generating summary"
