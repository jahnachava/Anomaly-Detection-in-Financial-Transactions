"""
Utility functions for creating various plots and visualizations.

This module provides helper functions for creating different types of plots
used in anomaly detection visualization and analysis.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union
from loguru import logger
import warnings
warnings.filterwarnings('ignore')

class PlotUtils:
    """
    Utility class for creating various plots and visualizations.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the PlotUtils class.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = logger
        self.plot_config = config.get('dashboard', {}).get('visualization', {})
        
        # Set default style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
    def plot_anomaly_distribution(self, data: pd.DataFrame, anomaly_column: str = 'anomaly',
                                title: str = "Anomaly Distribution") -> go.Figure:
        """
        Create a plot showing the distribution of anomalies.
        
        Args:
            data: DataFrame containing the data
            anomaly_column: Name of the anomaly column
            title: Plot title
            
        Returns:
            Plotly figure object
        """
        try:
            anomaly_counts = data[anomaly_column].value_counts()
            
            fig = px.pie(
                values=anomaly_counts.values,
                names=anomaly_counts.index,
                title=title,
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            
            fig.update_traces(textposition='inside', textinfo='percent+label')
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Error creating anomaly distribution plot: {e}")
            return go.Figure()
            
    def plot_feature_importance(self, feature_names: List[str], importance_scores: np.ndarray,
                              title: str = "Feature Importance", top_n: int = 20) -> go.Figure:
        """
        Create a horizontal bar plot for feature importance.
        
        Args:
            feature_names: List of feature names
            importance_scores: Array of importance scores
            title: Plot title
            top_n: Number of top features to display
            
        Returns:
            Plotly figure object
        """
        try:
            # Sort features by importance
            sorted_indices = np.argsort(importance_scores)[::-1][:top_n]
            top_features = [feature_names[i] for i in sorted_indices]
            top_scores = importance_scores[sorted_indices]
            
            fig = go.Figure(data=go.Bar(
                y=top_features,
                x=top_scores,
                orientation='h',
                marker=dict(color=top_scores, colorscale='Viridis')
            ))
            
            fig.update_layout(
                title=title,
                xaxis_title="Importance Score",
                yaxis_title="Features",
                height=max(400, len(top_features) * 20)
            )
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Error creating feature importance plot: {e}")
            return go.Figure()
            
    def plot_confusion_matrix(self, confusion_matrix: np.ndarray, 
                            class_names: List[str] = None,
                            title: str = "Confusion Matrix") -> go.Figure:
        """
        Create a heatmap for confusion matrix.
        
        Args:
            confusion_matrix: Confusion matrix array
            class_names: List of class names
            title: Plot title
            
        Returns:
            Plotly figure object
        """
        try:
            if class_names is None:
                class_names = ['Normal', 'Anomaly']
                
            fig = px.imshow(
                confusion_matrix,
                text_auto=True,
                aspect="auto",
                title=title,
                labels=dict(x="Predicted", y="Actual"),
                x=class_names,
                y=class_names,
                color_continuous_scale='Blues'
            )
            
            # Add text annotations
            for i in range(len(class_names)):
                for j in range(len(class_names)):
                    fig.add_annotation(
                        x=j, y=i,
                        text=str(confusion_matrix[i, j]),
                        showarrow=False,
                        font=dict(color="white" if confusion_matrix[i, j] > confusion_matrix.max()/2 else "black")
                    )
                    
            return fig
            
        except Exception as e:
            self.logger.error(f"Error creating confusion matrix plot: {e}")
            return go.Figure()
            
    def plot_roc_curve(self, fpr: np.ndarray, tpr: np.ndarray, auc_score: float = None,
                      title: str = "ROC Curve") -> go.Figure:
        """
        Create a ROC curve plot.
        
        Args:
            fpr: False positive rates
            tpr: True positive rates
            auc_score: AUC score (optional)
            title: Plot title
            
        Returns:
            Plotly figure object
        """
        try:
            fig = go.Figure()
            
            # Add ROC curve
            fig.add_trace(go.Scatter(
                x=fpr, y=tpr,
                mode='lines',
                name=f'ROC Curve (AUC = {auc_score:.3f})' if auc_score else 'ROC Curve',
                line=dict(color='blue', width=2)
            ))
            
            # Add diagonal line
            fig.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1],
                mode='lines',
                name='Random',
                line=dict(dash='dash', color='red')
            ))
            
            fig.update_layout(
                title=title,
                xaxis_title="False Positive Rate",
                yaxis_title="True Positive Rate",
                xaxis=dict(range=[0, 1]),
                yaxis=dict(range=[0, 1])
            )
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Error creating ROC curve plot: {e}")
            return go.Figure()
            
    def plot_precision_recall_curve(self, precision: np.ndarray, recall: np.ndarray,
                                  average_precision: float = None,
                                  title: str = "Precision-Recall Curve") -> go.Figure:
        """
        Create a precision-recall curve plot.
        
        Args:
            precision: Precision values
            recall: Recall values
            average_precision: Average precision score (optional)
            title: Plot title
            
        Returns:
            Plotly figure object
        """
        try:
            fig = go.Figure()
            
            # Add precision-recall curve
            fig.add_trace(go.Scatter(
                x=recall, y=precision,
                mode='lines',
                name=f'PR Curve (AP = {average_precision:.3f})' if average_precision else 'PR Curve',
                line=dict(color='green', width=2)
            ))
            
            fig.update_layout(
                title=title,
                xaxis_title="Recall",
                yaxis_title="Precision",
                xaxis=dict(range=[0, 1]),
                yaxis=dict(range=[0, 1])
            )
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Error creating precision-recall curve plot: {e}")
            return go.Figure()
            
    def plot_anomaly_scores(self, scores: np.ndarray, labels: np.ndarray = None,
                          threshold: float = None, title: str = "Anomaly Scores") -> go.Figure:
        """
        Create a plot showing anomaly scores distribution.
        
        Args:
            scores: Array of anomaly scores
            labels: True labels (optional)
            threshold: Threshold line (optional)
            title: Plot title
            
        Returns:
            Plotly figure object
        """
        try:
            fig = go.Figure()
            
            if labels is not None:
                # Separate normal and anomaly scores
                normal_scores = scores[labels == 0]
                anomaly_scores = scores[labels == 1]
                
                # Add normal scores
                fig.add_trace(go.Histogram(
                    x=normal_scores,
                    name='Normal',
                    opacity=0.7,
                    nbinsx=50
                ))
                
                # Add anomaly scores
                fig.add_trace(go.Histogram(
                    x=anomaly_scores,
                    name='Anomaly',
                    opacity=0.7,
                    nbinsx=50
                ))
            else:
                # Single histogram
                fig.add_trace(go.Histogram(
                    x=scores,
                    name='All Scores',
                    nbinsx=50
                ))
                
            # Add threshold line if provided
            if threshold is not None:
                fig.add_vline(
                    x=threshold,
                    line_dash="dash",
                    line_color="red",
                    annotation_text=f"Threshold: {threshold:.3f}"
                )
                
            fig.update_layout(
                title=title,
                xaxis_title="Anomaly Score",
                yaxis_title="Count",
                barmode='overlay'
            )
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Error creating anomaly scores plot: {e}")
            return go.Figure()
            
    def plot_time_series_anomalies(self, timestamps: pd.Series, values: np.ndarray,
                                 anomalies: np.ndarray, title: str = "Time Series with Anomalies") -> go.Figure:
        """
        Create a time series plot with highlighted anomalies.
        
        Args:
            timestamps: Time series timestamps
            values: Time series values
            anomalies: Binary array indicating anomalies
            title: Plot title
            
        Returns:
            Plotly figure object
        """
        try:
            fig = go.Figure()
            
            # Add normal points
            normal_mask = anomalies == 0
            fig.add_trace(go.Scatter(
                x=timestamps[normal_mask],
                y=values[normal_mask],
                mode='markers',
                name='Normal',
                marker=dict(color='blue', size=4)
            ))
            
            # Add anomaly points
            anomaly_mask = anomalies == 1
            fig.add_trace(go.Scatter(
                x=timestamps[anomaly_mask],
                y=values[anomaly_mask],
                mode='markers',
                name='Anomaly',
                marker=dict(color='red', size=8, symbol='x')
            ))
            
            fig.update_layout(
                title=title,
                xaxis_title="Time",
                yaxis_title="Value"
            )
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Error creating time series anomalies plot: {e}")
            return go.Figure()
            
    def plot_model_comparison(self, model_names: List[str], metrics: Dict[str, List[float]],
                            title: str = "Model Comparison") -> go.Figure:
        """
        Create a grouped bar chart for model comparison.
        
        Args:
            model_names: List of model names
            metrics: Dictionary of metric names and their values for each model
            title: Plot title
            
        Returns:
            Plotly figure object
        """
        try:
            fig = go.Figure()
            
            for metric_name, values in metrics.items():
                fig.add_trace(go.Bar(
                    name=metric_name,
                    x=model_names,
                    y=values
                ))
                
            fig.update_layout(
                title=title,
                xaxis_title="Models",
                yaxis_title="Score",
                barmode='group'
            )
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Error creating model comparison plot: {e}")
            return go.Figure()
            
    def plot_correlation_heatmap(self, correlation_matrix: np.ndarray,
                               feature_names: List[str] = None,
                               title: str = "Feature Correlation Heatmap") -> go.Figure:
        """
        Create a correlation heatmap.
        
        Args:
            correlation_matrix: Correlation matrix
            feature_names: List of feature names
            title: Plot title
            
        Returns:
            Plotly figure object
        """
        try:
            if feature_names is None:
                feature_names = [f"Feature_{i}" for i in range(correlation_matrix.shape[0])]
                
            fig = px.imshow(
                correlation_matrix,
                x=feature_names,
                y=feature_names,
                color_continuous_scale='RdBu',
                title=title,
                aspect="auto"
            )
            
            # Add correlation values as text
            for i in range(len(feature_names)):
                for j in range(len(feature_names)):
                    fig.add_annotation(
                        x=j, y=i,
                        text=f"{correlation_matrix[i, j]:.2f}",
                        showarrow=False,
                        font=dict(color="white" if abs(correlation_matrix[i, j]) > 0.5 else "black")
                    )
                    
            return fig
            
        except Exception as e:
            self.logger.error(f"Error creating correlation heatmap: {e}")
            return go.Figure()
            
    def plot_threshold_analysis(self, thresholds: np.ndarray, metrics: Dict[str, np.ndarray],
                              title: str = "Threshold Analysis") -> go.Figure:
        """
        Create a plot showing how metrics change with different thresholds.
        
        Args:
            thresholds: Array of threshold values
            metrics: Dictionary of metric names and their values
            title: Plot title
            
        Returns:
            Plotly figure object
        """
        try:
            fig = go.Figure()
            
            for metric_name, values in metrics.items():
                fig.add_trace(go.Scatter(
                    x=thresholds,
                    y=values,
                    mode='lines+markers',
                    name=metric_name
                ))
                
            fig.update_layout(
                title=title,
                xaxis_title="Threshold",
                yaxis_title="Score"
            )
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Error creating threshold analysis plot: {e}")
            return go.Figure()
            
    def plot_learning_curves(self, train_scores: List[float], val_scores: List[float],
                           train_sizes: List[int] = None,
                           title: str = "Learning Curves") -> go.Figure:
        """
        Create learning curves plot.
        
        Args:
            train_scores: Training scores
            val_scores: Validation scores
            train_sizes: Training set sizes (optional)
            title: Plot title
            
        Returns:
            Plotly figure object
        """
        try:
            if train_sizes is None:
                train_sizes = list(range(1, len(train_scores) + 1))
                
            fig = go.Figure()
            
            # Add training scores
            fig.add_trace(go.Scatter(
                x=train_sizes,
                y=train_scores,
                mode='lines+markers',
                name='Training Score',
                line=dict(color='blue')
            ))
            
            # Add validation scores
            fig.add_trace(go.Scatter(
                x=train_sizes,
                y=val_scores,
                mode='lines+markers',
                name='Validation Score',
                line=dict(color='red')
            ))
            
            fig.update_layout(
                title=title,
                xaxis_title="Training Set Size",
                yaxis_title="Score"
            )
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Error creating learning curves plot: {e}")
            return go.Figure()
            
    def save_plot(self, fig: go.Figure, file_path: str, format: str = 'html') -> None:
        """
        Save a plot to file.
        
        Args:
            fig: Plotly figure object
            file_path: Path to save the file
            format: File format ('html', 'png', 'pdf', 'svg')
        """
        try:
            if format == 'html':
                fig.write_html(file_path)
            elif format == 'png':
                fig.write_image(file_path)
            elif format == 'pdf':
                fig.write_image(file_path)
            elif format == 'svg':
                fig.write_image(file_path)
            else:
                raise ValueError(f"Unsupported format: {format}")
                
            self.logger.info(f"Plot saved to {file_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving plot: {e}")
            raise
