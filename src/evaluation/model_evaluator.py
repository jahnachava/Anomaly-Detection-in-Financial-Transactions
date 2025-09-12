"""
Model evaluator for comprehensive anomaly detection model assessment.

This module provides functionality to evaluate individual models and compare
their performance across different metrics and datasets.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple, Union
from sklearn.model_selection import cross_val_score, StratifiedKFold
from loguru import logger
import warnings
warnings.filterwarnings('ignore')

from .performance_metrics import PerformanceMetrics

class ModelEvaluator:
    """
    A class to evaluate anomaly detection models comprehensively.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the ModelEvaluator.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = logger
        self.metrics_calculator = PerformanceMetrics()
        self.evaluation_config = config.get('evaluation', {})
        
    def evaluate_model(self, model: Any, X: pd.DataFrame, y: pd.Series,
                      X_test: pd.DataFrame = None, y_test: pd.Series = None,
                      threshold: float = 0.5) -> Dict[str, Any]:
        """
        Evaluate a single model comprehensively.
        
        Args:
            model: Trained model object
            X: Training feature matrix
            y: Training target variable
            X_test: Test feature matrix (optional)
            y_test: Test target variable (optional)
            threshold: Probability threshold for predictions
            
        Returns:
            Dictionary containing evaluation results
        """
        try:
            self.logger.info(f"Evaluating model: {getattr(model, 'model_name', 'Unknown')}")
            
            evaluation_results = {
                'model_name': getattr(model, 'model_name', 'Unknown'),
                'model_type': type(model).__name__,
                'is_fitted': getattr(model, 'is_fitted', False)
            }
            
            if not getattr(model, 'is_fitted', False):
                self.logger.warning("Model is not fitted, skipping evaluation")
                return evaluation_results
                
            # Evaluate on training data
            if X is not None and y is not None:
                train_results = self._evaluate_on_data(model, X, y, threshold, 'training')
                evaluation_results['training'] = train_results
                
            # Evaluate on test data
            if X_test is not None and y_test is not None:
                test_results = self._evaluate_on_data(model, X_test, y_test, threshold, 'test')
                evaluation_results['test'] = test_results
                
            # Cross-validation evaluation
            if X is not None and y is not None:
                cv_results = self._cross_validate_model(model, X, y)
                evaluation_results['cross_validation'] = cv_results
                
            self.logger.info("Model evaluation completed successfully")
            
            return evaluation_results
            
        except Exception as e:
            self.logger.error(f"Error evaluating model: {e}")
            return {'error': str(e)}
            
    def _evaluate_on_data(self, model: Any, X: pd.DataFrame, y: pd.Series,
                         threshold: float, data_type: str) -> Dict[str, Any]:
        """
        Evaluate model on specific dataset.
        
        Args:
            model: Trained model
            X: Feature matrix
            y: Target variable
            threshold: Probability threshold
            data_type: Type of data ('training', 'test', 'validation')
            
        Returns:
            Dictionary containing evaluation results
        """
        try:
            self.logger.info(f"Evaluating on {data_type} data: {X.shape[0]} samples")
            
            # Make predictions
            if hasattr(model, 'predict_proba'):
                y_proba = model.predict_proba(X)[:, 1]  # Probability of anomaly
                y_pred = (y_proba > threshold).astype(int)
            else:
                y_pred = model.predict(X)
                y_proba = None
                
            # Compute comprehensive metrics
            if y_proba is not None:
                metrics = self.metrics_calculator.compute_comprehensive_metrics(y, y_pred, y_proba)
            else:
                metrics = self.metrics_calculator.compute_comprehensive_metrics(y, y_pred)
                
            # Add data information
            results = {
                'data_type': data_type,
                'n_samples': len(X),
                'n_features': X.shape[1],
                'class_distribution': y.value_counts().to_dict(),
                'threshold': threshold,
                'metrics': metrics
            }
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error evaluating on {data_type} data: {e}")
            return {'error': str(e)}
            
    def _cross_validate_model(self, model: Any, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """
        Perform cross-validation on the model.
        
        Args:
            model: Model to evaluate
            X: Feature matrix
            y: Target variable
            
        Returns:
            Dictionary containing cross-validation results
        """
        try:
            cv_folds = self.evaluation_config.get('cross_validation', {}).get('cv_folds', 5)
            scoring = self.evaluation_config.get('cross_validation', {}).get('scoring', 'roc_auc')
            
            self.logger.info(f"Performing {cv_folds}-fold cross-validation with {scoring} scoring")
            
            # Create cross-validation strategy
            cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
            
            # Perform cross-validation
            cv_scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
            
            cv_results = {
                'cv_folds': cv_folds,
                'scoring': scoring,
                'cv_scores': cv_scores.tolist(),
                'mean_score': float(cv_scores.mean()),
                'std_score': float(cv_scores.std()),
                'min_score': float(cv_scores.min()),
                'max_score': float(cv_scores.max())
            }
            
            self.logger.info(f"CV Results - Mean: {cv_results['mean_score']:.4f} (+/- {cv_results['std_score']:.4f})")
            
            return cv_results
            
        except Exception as e:
            self.logger.error(f"Error in cross-validation: {e}")
            return {'error': str(e)}
            
    def evaluate_multiple_models(self, models: Dict[str, Any], X: pd.DataFrame, y: pd.Series,
                               X_test: pd.DataFrame = None, y_test: pd.Series = None,
                               threshold: float = 0.5) -> Dict[str, Dict[str, Any]]:
        """
        Evaluate multiple models and compare their performance.
        
        Args:
            models: Dictionary of model names and model objects
            X: Training feature matrix
            y: Training target variable
            X_test: Test feature matrix (optional)
            y_test: Test target variable (optional)
            threshold: Probability threshold for predictions
            
        Returns:
            Dictionary containing evaluation results for all models
        """
        try:
            self.logger.info(f"Evaluating {len(models)} models")
            
            all_results = {}
            
            for model_name, model in models.items():
                self.logger.info(f"Evaluating model: {model_name}")
                
                results = self.evaluate_model(
                    model, X, y, X_test, y_test, threshold
                )
                
                all_results[model_name] = results
                
            self.logger.info("All models evaluated successfully")
            
            return all_results
            
        except Exception as e:
            self.logger.error(f"Error evaluating multiple models: {e}")
            return {}
            
    def compare_models(self, evaluation_results: Dict[str, Dict[str, Any]],
                      metric: str = 'roc_auc', data_type: str = 'test') -> pd.DataFrame:
        """
        Compare models based on a specific metric.
        
        Args:
            evaluation_results: Results from evaluate_multiple_models
            metric: Metric to compare ('roc_auc', 'f1_score', 'precision', 'recall', 'accuracy')
            data_type: Type of data to compare ('training', 'test', 'cross_validation')
            
        Returns:
            DataFrame with model comparison results
        """
        try:
            comparison_data = []
            
            for model_name, results in evaluation_results.items():
                if data_type in results:
                    if data_type == 'cross_validation':
                        # Use CV mean score
                        score = results[data_type].get('mean_score', 0)
                        std_score = results[data_type].get('std_score', 0)
                        comparison_data.append({
                            'model': model_name,
                            'metric': metric,
                            'score': score,
                            'std_score': std_score,
                            'data_type': data_type
                        })
                    else:
                        # Use specific metric from evaluation
                        metrics = results[data_type].get('metrics', {})
                        score = metrics.get(metric, 0)
                        comparison_data.append({
                            'model': model_name,
                            'metric': metric,
                            'score': score,
                            'std_score': 0,
                            'data_type': data_type
                        })
                        
            if not comparison_data:
                self.logger.warning(f"No data found for comparison with metric {metric} and data_type {data_type}")
                return pd.DataFrame()
                
            comparison_df = pd.DataFrame(comparison_data)
            comparison_df = comparison_df.sort_values('score', ascending=False)
            
            self.logger.info(f"Model comparison completed for {metric} on {data_type} data")
            
            return comparison_df
            
        except Exception as e:
            self.logger.error(f"Error comparing models: {e}")
            return pd.DataFrame()
            
    def generate_evaluation_report(self, evaluation_results: Dict[str, Dict[str, Any]]) -> str:
        """
        Generate a comprehensive evaluation report.
        
        Args:
            evaluation_results: Results from evaluate_multiple_models
            
        Returns:
            Formatted evaluation report
        """
        try:
            report = []
            report.append("=" * 80)
            report.append("ANOMALY DETECTION MODEL EVALUATION REPORT")
            report.append("=" * 80)
            
            for model_name, results in evaluation_results.items():
                report.append(f"\nMODEL: {model_name}")
                report.append("-" * 50)
                
                # Model information
                report.append(f"Model Type: {results.get('model_type', 'Unknown')}")
                report.append(f"Is Fitted: {results.get('is_fitted', False)}")
                
                # Training results
                if 'training' in results and 'error' not in results['training']:
                    train_metrics = results['training']['metrics']
                    report.append(f"\nTraining Results ({results['training']['n_samples']} samples):")
                    report.append(f"  Accuracy:  {train_metrics.get('accuracy', 0):.4f}")
                    report.append(f"  Precision: {train_metrics.get('precision', 0):.4f}")
                    report.append(f"  Recall:    {train_metrics.get('recall', 0):.4f}")
                    report.append(f"  F1-Score:  {train_metrics.get('f1_score', 0):.4f}")
                    if 'roc_auc' in train_metrics:
                        report.append(f"  ROC-AUC:   {train_metrics['roc_auc']:.4f}")
                        
                # Test results
                if 'test' in results and 'error' not in results['test']:
                    test_metrics = results['test']['metrics']
                    report.append(f"\nTest Results ({results['test']['n_samples']} samples):")
                    report.append(f"  Accuracy:  {test_metrics.get('accuracy', 0):.4f}")
                    report.append(f"  Precision: {test_metrics.get('precision', 0):.4f}")
                    report.append(f"  Recall:    {test_metrics.get('recall', 0):.4f}")
                    report.append(f"  F1-Score:  {test_metrics.get('f1_score', 0):.4f}")
                    if 'roc_auc' in test_metrics:
                        report.append(f"  ROC-AUC:   {test_metrics['roc_auc']:.4f}")
                        
                # Cross-validation results
                if 'cross_validation' in results and 'error' not in results['cross_validation']:
                    cv_results = results['cross_validation']
                    report.append(f"\nCross-Validation Results ({cv_results['cv_folds']} folds):")
                    report.append(f"  Mean {cv_results['scoring']}: {cv_results['mean_score']:.4f} (+/- {cv_results['std_score']:.4f})")
                    report.append(f"  Min Score:  {cv_results['min_score']:.4f}")
                    report.append(f"  Max Score:  {cv_results['max_score']:.4f}")
                    
            report.append("\n" + "=" * 80)
            
            return "\n".join(report)
            
        except Exception as e:
            self.logger.error(f"Error generating evaluation report: {e}")
            return f"Error generating report: {str(e)}"
            
    def save_evaluation_results(self, evaluation_results: Dict[str, Dict[str, Any]], 
                              file_path: str) -> None:
        """
        Save evaluation results to a file.
        
        Args:
            evaluation_results: Results to save
            file_path: Path to save the results
        """
        try:
            import json
            import os
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            # Convert numpy arrays to lists for JSON serialization
            def convert_numpy(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                return obj
                
            # Recursively convert numpy objects
            def recursive_convert(obj):
                if isinstance(obj, dict):
                    return {k: recursive_convert(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [recursive_convert(item) for item in obj]
                else:
                    return convert_numpy(obj)
                    
            converted_results = recursive_convert(evaluation_results)
            
            with open(file_path, 'w') as f:
                json.dump(converted_results, f, indent=2)
                
            self.logger.info(f"Evaluation results saved to {file_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving evaluation results: {e}")
            raise
