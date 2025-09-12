"""
Model comparison framework for anomaly detection.

This module provides functionality to compare multiple anomaly detection models
and generate comprehensive comparison reports.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from loguru import logger
import warnings
warnings.filterwarnings('ignore')

from .model_evaluator import ModelEvaluator
from .performance_metrics import PerformanceMetrics
from .threshold_optimizer import ThresholdOptimizer

class ModelComparator:
    """
    A class to compare multiple anomaly detection models comprehensively.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the ModelComparator.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = logger
        self.evaluator = ModelEvaluator(config)
        self.metrics_calculator = PerformanceMetrics()
        self.threshold_optimizer = ThresholdOptimizer(config)
        
    def compare_models(self, models: Dict[str, Any], X: pd.DataFrame, y: pd.Series,
                      X_test: pd.DataFrame = None, y_test: pd.Series = None,
                      threshold: float = 0.5) -> Dict[str, Any]:
        """
        Compare multiple models comprehensively.
        
        Args:
            models: Dictionary of model names and model objects
            X: Training feature matrix
            y: Training target variable
            X_test: Test feature matrix (optional)
            y_test: Test target variable (optional)
            threshold: Probability threshold for predictions
            
        Returns:
            Dictionary containing comprehensive comparison results
        """
        try:
            self.logger.info(f"Comparing {len(models)} models")
            
            # Evaluate all models
            evaluation_results = self.evaluator.evaluate_multiple_models(
                models, X, y, X_test, y_test, threshold
            )
            
            # Generate comparison metrics
            comparison_results = {
                'evaluation_results': evaluation_results,
                'summary_statistics': self._generate_summary_statistics(evaluation_results),
                'ranking': self._rank_models(evaluation_results),
                'statistical_tests': self._perform_statistical_tests(evaluation_results),
                'threshold_analysis': self._analyze_thresholds(models, X_test, y_test) if X_test is not None and y_test is not None else None
            }
            
            self.logger.info("Model comparison completed successfully")
            
            return comparison_results
            
        except Exception as e:
            self.logger.error(f"Error comparing models: {e}")
            return {}
            
    def _generate_summary_statistics(self, evaluation_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate summary statistics for all models.
        
        Args:
            evaluation_results: Results from model evaluation
            
        Returns:
            Dictionary containing summary statistics
        """
        try:
            summary = {
                'model_count': len(evaluation_results),
                'metrics_summary': {},
                'best_performers': {}
            }
            
            # Extract metrics for comparison
            metrics_to_compare = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
            
            for metric in metrics_to_compare:
                metric_values = []
                model_names = []
                
                for model_name, results in evaluation_results.items():
                    if 'test' in results and 'error' not in results['test']:
                        metric_value = results['test']['metrics'].get(metric, 0)
                        metric_values.append(metric_value)
                        model_names.append(model_name)
                        
                if metric_values:
                    summary['metrics_summary'][metric] = {
                        'mean': np.mean(metric_values),
                        'std': np.std(metric_values),
                        'min': np.min(metric_values),
                        'max': np.max(metric_values),
                        'median': np.median(metric_values)
                    }
                    
                    # Find best performer
                    best_idx = np.argmax(metric_values)
                    summary['best_performers'][metric] = {
                        'model': model_names[best_idx],
                        'score': metric_values[best_idx]
                    }
                    
            return summary
            
        except Exception as e:
            self.logger.error(f"Error generating summary statistics: {e}")
            return {}
            
    def _rank_models(self, evaluation_results: Dict[str, Dict[str, Any]]) -> Dict[str, List[Tuple[str, float]]]:
        """
        Rank models based on different metrics.
        
        Args:
            evaluation_results: Results from model evaluation
            
        Returns:
            Dictionary containing rankings for each metric
        """
        try:
            rankings = {}
            metrics_to_rank = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
            
            for metric in metrics_to_rank:
                model_scores = []
                
                for model_name, results in evaluation_results.items():
                    if 'test' in results and 'error' not in results['test']:
                        score = results['test']['metrics'].get(metric, 0)
                        model_scores.append((model_name, score))
                        
                # Sort by score (descending)
                model_scores.sort(key=lambda x: x[1], reverse=True)
                rankings[metric] = model_scores
                
            return rankings
            
        except Exception as e:
            self.logger.error(f"Error ranking models: {e}")
            return {}
            
    def _perform_statistical_tests(self, evaluation_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Perform statistical tests to compare models.
        
        Args:
            evaluation_results: Results from model evaluation
            
        Returns:
            Dictionary containing statistical test results
        """
        try:
            from scipy import stats
            
            statistical_tests = {}
            
            # Extract CV scores for statistical comparison
            cv_scores = {}
            for model_name, results in evaluation_results.items():
                if 'cross_validation' in results and 'error' not in results['cross_validation']:
                    cv_scores[model_name] = results['cross_validation']['cv_scores']
                    
            if len(cv_scores) >= 2:
                # Perform pairwise t-tests
                model_names = list(cv_scores.keys())
                pairwise_tests = {}
                
                for i in range(len(model_names)):
                    for j in range(i + 1, len(model_names)):
                        model1, model2 = model_names[i], model_names[j]
                        
                        # Perform t-test
                        t_stat, p_value = stats.ttest_rel(cv_scores[model1], cv_scores[model2])
                        
                        pairwise_tests[f"{model1}_vs_{model2}"] = {
                            't_statistic': t_stat,
                            'p_value': p_value,
                            'significant': p_value < 0.05,
                            'model1_mean': np.mean(cv_scores[model1]),
                            'model2_mean': np.mean(cv_scores[model2])
                        }
                        
                statistical_tests['pairwise_t_tests'] = pairwise_tests
                
                # Perform ANOVA if more than 2 models
                if len(cv_scores) > 2:
                    f_stat, p_value = stats.f_oneway(*cv_scores.values())
                    statistical_tests['anova'] = {
                        'f_statistic': f_stat,
                        'p_value': p_value,
                        'significant': p_value < 0.05
                    }
                    
            return statistical_tests
            
        except Exception as e:
            self.logger.error(f"Error performing statistical tests: {e}")
            return {}
            
    def _analyze_thresholds(self, models: Dict[str, Any], X_test: pd.DataFrame, 
                          y_test: pd.Series) -> Dict[str, Any]:
        """
        Analyze optimal thresholds for all models.
        
        Args:
            models: Dictionary of model names and model objects
            X_test: Test feature matrix
            y_test: Test target variable
            
        Returns:
            Dictionary containing threshold analysis results
        """
        try:
            threshold_analysis = {}
            
            for model_name, model in models.items():
                if hasattr(model, 'predict_proba') and getattr(model, 'is_fitted', False):
                    try:
                        y_proba = model.predict_proba(X_test)[:, 1]
                        
                        # Find optimal thresholds using multiple methods
                        optimal_thresholds = self.threshold_optimizer.find_multiple_optimal_thresholds(
                            y_test, y_proba
                        )
                        
                        threshold_analysis[model_name] = optimal_thresholds
                        
                    except Exception as e:
                        self.logger.warning(f"Could not analyze thresholds for {model_name}: {e}")
                        threshold_analysis[model_name] = {'error': str(e)}
                        
            return threshold_analysis
            
        except Exception as e:
            self.logger.error(f"Error analyzing thresholds: {e}")
            return {}
            
    def generate_comparison_report(self, comparison_results: Dict[str, Any]) -> str:
        """
        Generate a comprehensive comparison report.
        
        Args:
            comparison_results: Results from compare_models
            
        Returns:
            Formatted comparison report
        """
        try:
            report = []
            report.append("=" * 80)
            report.append("ANOMALY DETECTION MODEL COMPARISON REPORT")
            report.append("=" * 80)
            
            # Summary statistics
            if 'summary_statistics' in comparison_results:
                summary = comparison_results['summary_statistics']
                report.append(f"\nSUMMARY STATISTICS")
                report.append("-" * 50)
                report.append(f"Number of models compared: {summary.get('model_count', 0)}")
                
                if 'best_performers' in summary:
                    report.append(f"\nBEST PERFORMERS:")
                    for metric, info in summary['best_performers'].items():
                        report.append(f"  {metric.upper()}: {info['model']} ({info['score']:.4f})")
                        
            # Rankings
            if 'ranking' in comparison_results:
                report.append(f"\nMODEL RANKINGS")
                report.append("-" * 50)
                
                for metric, ranking in comparison_results['ranking'].items():
                    report.append(f"\n{metric.upper()}:")
                    for i, (model_name, score) in enumerate(ranking, 1):
                        report.append(f"  {i}. {model_name}: {score:.4f}")
                        
            # Statistical tests
            if 'statistical_tests' in comparison_results:
                stats = comparison_results['statistical_tests']
                report.append(f"\nSTATISTICAL TESTS")
                report.append("-" * 50)
                
                if 'pairwise_t_tests' in stats:
                    report.append("Pairwise T-tests (p < 0.05 indicates significant difference):")
                    for comparison, result in stats['pairwise_t_tests'].items():
                        significance = "SIGNIFICANT" if result['significant'] else "NOT SIGNIFICANT"
                        report.append(f"  {comparison}: p = {result['p_value']:.4f} ({significance})")
                        
            report.append("\n" + "=" * 80)
            
            return "\n".join(report)
            
        except Exception as e:
            self.logger.error(f"Error generating comparison report: {e}")
            return f"Error generating report: {str(e)}"
            
    def plot_model_comparison(self, comparison_results: Dict[str, Any], 
                            metrics: List[str] = None, save_path: str = None) -> None:
        """
        Create visualization plots for model comparison.
        
        Args:
            comparison_results: Results from compare_models
            metrics: List of metrics to plot
            save_path: Path to save plots (optional)
        """
        try:
            if metrics is None:
                metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
                
            evaluation_results = comparison_results.get('evaluation_results', {})
            
            # Prepare data for plotting
            plot_data = []
            for model_name, results in evaluation_results.items():
                if 'test' in results and 'error' not in results['test']:
                    for metric in metrics:
                        if metric in results['test']['metrics']:
                            plot_data.append({
                                'Model': model_name,
                                'Metric': metric,
                                'Score': results['test']['metrics'][metric]
                            })
                            
            if not plot_data:
                self.logger.warning("No data available for plotting")
                return
                
            df = pd.DataFrame(plot_data)
            
            # Create subplots
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('Model Comparison Results', fontsize=16)
            
            # Bar plot of metrics
            ax1 = axes[0, 0]
            metric_means = df.groupby('Metric')['Score'].mean()
            metric_means.plot(kind='bar', ax=ax1, color='skyblue')
            ax1.set_title('Average Performance by Metric')
            ax1.set_ylabel('Score')
            ax1.tick_params(axis='x', rotation=45)
            
            # Model performance comparison
            ax2 = axes[0, 1]
            model_means = df.groupby('Model')['Score'].mean()
            model_means.plot(kind='bar', ax=ax2, color='lightcoral')
            ax2.set_title('Average Performance by Model')
            ax2.set_ylabel('Score')
            ax2.tick_params(axis='x', rotation=45)
            
            # Heatmap of model vs metric performance
            ax3 = axes[1, 0]
            pivot_df = df.pivot(index='Model', columns='Metric', values='Score')
            sns.heatmap(pivot_df, annot=True, cmap='YlOrRd', ax=ax3, fmt='.3f')
            ax3.set_title('Model Performance Heatmap')
            
            # Box plot of score distribution
            ax4 = axes[1, 1]
            df.boxplot(column='Score', by='Metric', ax=ax4)
            ax4.set_title('Score Distribution by Metric')
            ax4.set_xlabel('Metric')
            ax4.set_ylabel('Score')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                self.logger.info(f"Comparison plots saved to {save_path}")
            else:
                plt.show()
                
        except ImportError:
            self.logger.warning("Matplotlib/Seaborn not available for plotting")
        except Exception as e:
            self.logger.error(f"Error creating comparison plots: {e}")
            
    def save_comparison_results(self, comparison_results: Dict[str, Any], 
                              file_path: str) -> None:
        """
        Save comparison results to a file.
        
        Args:
            comparison_results: Results to save
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
                    
            converted_results = recursive_convert(comparison_results)
            
            with open(file_path, 'w') as f:
                json.dump(converted_results, f, indent=2)
                
            self.logger.info(f"Comparison results saved to {file_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving comparison results: {e}")
            raise
