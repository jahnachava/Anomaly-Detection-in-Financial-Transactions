"""
Interactive dashboard for anomaly detection in financial transactions.

This module provides a Streamlit-based dashboard for real-time monitoring
and visualization of anomaly detection results.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, Any, List, Optional
import time
from datetime import datetime, timedelta
from loguru import logger

class AnomalyDetectionDashboard:
    """
    Interactive dashboard for anomaly detection monitoring.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the dashboard.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.dashboard_config = config.get('dashboard', {})
        self.logger = logger
        
        # Set page config
        st.set_page_config(
            page_title=self.dashboard_config.get('title', 'Anomaly Detection Dashboard'),
            page_icon="🔍",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
    def run_dashboard(self):
        """
        Run the main dashboard application.
        """
        try:
            # Main title
            st.title("🔍 Financial Transaction Anomaly Detection Dashboard")
            st.markdown("---")
            
            # Sidebar
            self._create_sidebar()
            
            # Main content
            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "📊 Overview", "🔍 Anomaly Detection", "📈 Model Performance", 
                "⚙️ Model Management", "📋 Reports"
            ])
            
            with tab1:
                self._overview_tab()
                
            with tab2:
                self._anomaly_detection_tab()
                
            with tab3:
                self._model_performance_tab()
                
            with tab4:
                self._model_management_tab()
                
            with tab5:
                self._reports_tab()
                
        except Exception as e:
            st.error(f"Error running dashboard: {e}")
            self.logger.error(f"Dashboard error: {e}")
            
    def _create_sidebar(self):
        """
        Create the sidebar with controls and information.
        """
        st.sidebar.title("🎛️ Controls")
        
        # Date range selector
        st.sidebar.subheader("📅 Date Range")
        col1, col2 = st.sidebar.columns(2)
        
        with col1:
            start_date = st.date_input("Start Date", value=datetime.now() - timedelta(days=30))
        with col2:
            end_date = st.date_input("End Date", value=datetime.now())
            
        # Model selection
        st.sidebar.subheader("🤖 Model Selection")
        selected_models = st.sidebar.multiselect(
            "Select Models",
            ["Random Forest", "XGBoost", "Neural Network", "Isolation Forest", "LOF", "One-Class SVM"],
            default=["Random Forest", "XGBoost"]
        )
        
        # Threshold settings
        st.sidebar.subheader("⚖️ Threshold Settings")
        threshold = st.sidebar.slider("Anomaly Threshold", 0.0, 1.0, 0.5, 0.01)
        
        # Real-time settings
        st.sidebar.subheader("⏱️ Real-time Settings")
        auto_refresh = st.sidebar.checkbox("Auto Refresh", value=True)
        refresh_interval = st.sidebar.selectbox("Refresh Interval (seconds)", [5, 10, 30, 60], index=1)
        
        # Store in session state
        st.session_state.update({
            'start_date': start_date,
            'end_date': end_date,
            'selected_models': selected_models,
            'threshold': threshold,
            'auto_refresh': auto_refresh,
            'refresh_interval': refresh_interval
        })
        
        # Auto refresh
        if auto_refresh:
            time.sleep(refresh_interval)
            st.rerun()
            
    def _overview_tab(self):
        """
        Create the overview tab with key metrics and charts.
        """
        st.header("📊 System Overview")
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="Total Transactions",
                value="1,234,567",
                delta="12.5%"
            )
            
        with col2:
            st.metric(
                label="Anomalies Detected",
                value="2,345",
                delta="8.2%"
            )
            
        with col3:
            st.metric(
                label="Detection Rate",
                value="94.2%",
                delta="2.1%"
            )
            
        with col4:
            st.metric(
                label="False Positive Rate",
                value="3.8%",
                delta="-1.2%"
            )
            
        st.markdown("---")
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📈 Transaction Volume Over Time")
            self._plot_transaction_volume()
            
        with col2:
            st.subheader("🚨 Anomaly Detection Over Time")
            self._plot_anomaly_trends()
            
        # Additional charts
        st.subheader("💰 Transaction Amount Distribution")
        self._plot_amount_distribution()
        
    def _anomaly_detection_tab(self):
        """
        Create the anomaly detection tab with real-time monitoring.
        """
        st.header("🔍 Real-time Anomaly Detection")
        
        # Real-time alerts
        st.subheader("🚨 Recent Anomalies")
        
        # Create sample anomaly data
        anomaly_data = self._generate_sample_anomaly_data()
        
        # Display anomalies in a table
        if not anomaly_data.empty:
            st.dataframe(
                anomaly_data,
                use_container_width=True,
                height=300
            )
            
            # Anomaly details
            st.subheader("📋 Anomaly Details")
            selected_anomaly = st.selectbox("Select Anomaly", anomaly_data['Transaction ID'])
            
            if selected_anomaly:
                self._display_anomaly_details(selected_anomaly, anomaly_data)
                
        else:
            st.info("No anomalies detected in the selected time range.")
            
        # Anomaly visualization
        st.subheader("🎯 Anomaly Visualization")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.plotly_chart(
                self._create_anomaly_scatter_plot(),
                use_container_width=True
            )
            
        with col2:
            st.plotly_chart(
                self._create_anomaly_heatmap(),
                use_container_width=True
            )
            
    def _model_performance_tab(self):
        """
        Create the model performance tab with metrics and comparisons.
        """
        st.header("📈 Model Performance")
        
        # Performance metrics
        st.subheader("📊 Performance Metrics")
        
        # Create sample performance data
        performance_data = self._generate_sample_performance_data()
        
        # Display metrics in columns
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Best Model", "XGBoost", "ROC-AUC: 0.945")
            
        with col2:
            st.metric("Avg Precision", "0.892", "±0.023")
            
        with col3:
            st.metric("Avg Recall", "0.876", "±0.018")
            
        with col4:
            st.metric("Avg F1-Score", "0.884", "±0.020")
            
        # Model comparison chart
        st.subheader("🔄 Model Comparison")
        st.plotly_chart(
            self._create_model_comparison_chart(performance_data),
            use_container_width=True
        )
        
        # ROC curves
        st.subheader("📈 ROC Curves")
        st.plotly_chart(
            self._create_roc_curves(),
            use_container_width=True
        )
        
        # Confusion matrices
        st.subheader("🔢 Confusion Matrices")
        self._display_confusion_matrices()
        
    def _model_management_tab(self):
        """
        Create the model management tab for configuration and training.
        """
        st.header("⚙️ Model Management")
        
        # Model status
        st.subheader("📊 Model Status")
        
        model_status = {
            "Random Forest": {"status": "✅ Trained", "last_update": "2024-01-15 14:30"},
            "XGBoost": {"status": "✅ Trained", "last_update": "2024-01-15 14:25"},
            "Neural Network": {"status": "🔄 Training", "last_update": "2024-01-15 14:20"},
            "Isolation Forest": {"status": "✅ Trained", "last_update": "2024-01-15 14:15"},
            "LOF": {"status": "❌ Error", "last_update": "2024-01-15 14:10"},
            "One-Class SVM": {"status": "✅ Trained", "last_update": "2024-01-15 14:05"}
        }
        
        for model, info in model_status.items():
            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                st.write(f"**{model}**")
            with col2:
                st.write(info["status"])
            with col3:
                st.write(info["last_update"])
                
        st.markdown("---")
        
        # Model training
        st.subheader("🚀 Model Training")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Train New Model**")
            model_type = st.selectbox("Model Type", ["Random Forest", "XGBoost", "Neural Network"])
            
            if st.button("Start Training"):
                with st.spinner("Training model..."):
                    time.sleep(2)  # Simulate training
                    st.success(f"{model_type} training completed!")
                    
        with col2:
            st.write("**Model Configuration**")
            
            if model_type == "Random Forest":
                n_estimators = st.slider("Number of Estimators", 10, 200, 100)
                max_depth = st.slider("Max Depth", 3, 20, 10)
                
            elif model_type == "XGBoost":
                learning_rate = st.slider("Learning Rate", 0.01, 0.3, 0.1)
                max_depth = st.slider("Max Depth", 3, 10, 6)
                
            elif model_type == "Neural Network":
                hidden_layers = st.slider("Hidden Layers", 1, 5, 3)
                learning_rate = st.slider("Learning Rate", 0.001, 0.1, 0.01)
                
        # Model evaluation
        st.subheader("📊 Model Evaluation")
        
        if st.button("Run Evaluation"):
            with st.spinner("Evaluating models..."):
                time.sleep(3)  # Simulate evaluation
                st.success("Model evaluation completed!")
                
                # Display results
                st.write("**Evaluation Results:**")
                eval_results = {
                    "Model": ["Random Forest", "XGBoost", "Neural Network"],
                    "Accuracy": [0.945, 0.952, 0.938],
                    "Precision": [0.892, 0.901, 0.885],
                    "Recall": [0.876, 0.889, 0.871],
                    "F1-Score": [0.884, 0.895, 0.878]
                }
                
                st.dataframe(pd.DataFrame(eval_results), use_container_width=True)
                
    def _reports_tab(self):
        """
        Create the reports tab with downloadable reports and analytics.
        """
        st.header("📋 Reports & Analytics")
        
        # Report generation
        st.subheader("📄 Generate Reports")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📊 Performance Report"):
                st.success("Performance report generated!")
                
        with col2:
            if st.button("🔍 Anomaly Analysis"):
                st.success("Anomaly analysis report generated!")
                
        with col3:
            if st.button("📈 Trend Analysis"):
                st.success("Trend analysis report generated!")
                
        st.markdown("---")
        
        # Sample reports
        st.subheader("📋 Sample Reports")
        
        # Performance summary
        st.write("**Performance Summary Report**")
        performance_summary = {
            "Metric": ["Total Transactions", "Anomalies Detected", "Detection Rate", "False Positive Rate"],
            "Value": ["1,234,567", "2,345", "94.2%", "3.8%"],
            "Change": ["+12.5%", "+8.2%", "+2.1%", "-1.2%"]
        }
        
        st.dataframe(pd.DataFrame(performance_summary), use_container_width=True)
        
        # Download buttons
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.download_button(
                label="📥 Download CSV",
                data=pd.DataFrame(performance_summary).to_csv(index=False),
                file_name="performance_summary.csv",
                mime="text/csv"
            )
            
        with col2:
            st.download_button(
                label="📥 Download JSON",
                data=pd.DataFrame(performance_summary).to_json(orient='records'),
                file_name="performance_summary.json",
                mime="application/json"
            )
            
        with col3:
            if st.button("📧 Email Report"):
                st.success("Report sent to admin@company.com")
                
    def _plot_transaction_volume(self):
        """
        Create a transaction volume chart.
        """
        # Generate sample data
        dates = pd.date_range(start='2024-01-01', end='2024-01-31', freq='D')
        volumes = np.random.normal(1000, 200, len(dates))
        
        fig = px.line(
            x=dates, y=volumes,
            title="Daily Transaction Volume",
            labels={'x': 'Date', 'y': 'Volume'}
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    def _plot_anomaly_trends(self):
        """
        Create an anomaly trends chart.
        """
        # Generate sample data
        dates = pd.date_range(start='2024-01-01', end='2024-01-31', freq='D')
        anomalies = np.random.poisson(50, len(dates))
        
        fig = px.bar(
            x=dates, y=anomalies,
            title="Daily Anomaly Count",
            labels={'x': 'Date', 'y': 'Anomalies'}
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    def _plot_amount_distribution(self):
        """
        Create an amount distribution chart.
        """
        # Generate sample data
        amounts = np.random.lognormal(5, 1, 10000)
        
        fig = px.histogram(
            x=amounts,
            title="Transaction Amount Distribution",
            labels={'x': 'Amount', 'y': 'Count'},
            nbins=50
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    def _generate_sample_anomaly_data(self) -> pd.DataFrame:
        """
        Generate sample anomaly data for demonstration.
        """
        data = {
            'Transaction ID': [f'TXN_{i:06d}' for i in range(1, 21)],
            'Timestamp': pd.date_range(start='2024-01-15', periods=20, freq='H'),
            'Amount': np.random.exponential(100, 20),
            'Merchant': [f'Merchant_{i}' for i in np.random.randint(1, 10, 20)],
            'Anomaly Score': np.random.uniform(0.7, 1.0, 20),
            'Model': np.random.choice(['Random Forest', 'XGBoost', 'Neural Network'], 20)
        }
        
        return pd.DataFrame(data)
        
    def _display_anomaly_details(self, transaction_id: str, anomaly_data: pd.DataFrame):
        """
        Display detailed information about a selected anomaly.
        """
        anomaly = anomaly_data[anomaly_data['Transaction ID'] == transaction_id].iloc[0]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"**Transaction ID:** {anomaly['Transaction ID']}")
            st.write(f"**Timestamp:** {anomaly['Timestamp']}")
            st.write(f"**Amount:** ${anomaly['Amount']:.2f}")
            
        with col2:
            st.write(f"**Merchant:** {anomaly['Merchant']}")
            st.write(f"**Anomaly Score:** {anomaly['Anomaly Score']:.3f}")
            st.write(f"**Detected by:** {anomaly['Model']}")
            
    def _create_anomaly_scatter_plot(self):
        """
        Create a scatter plot for anomaly visualization.
        """
        # Generate sample data
        x = np.random.normal(0, 1, 1000)
        y = np.random.normal(0, 1, 1000)
        colors = np.random.choice(['Normal', 'Anomaly'], 1000, p=[0.95, 0.05])
        
        fig = px.scatter(
            x=x, y=y, color=colors,
            title="Transaction Feature Space",
            labels={'x': 'Feature 1', 'y': 'Feature 2'}
        )
        
        return fig
        
    def _create_anomaly_heatmap(self):
        """
        Create a heatmap for anomaly patterns.
        """
        # Generate sample correlation matrix
        data = np.random.randn(10, 10)
        correlation_matrix = np.corrcoef(data)
        
        fig = px.imshow(
            correlation_matrix,
            title="Feature Correlation Heatmap",
            color_continuous_scale='RdBu'
        )
        
        return fig
        
    def _generate_sample_performance_data(self) -> pd.DataFrame:
        """
        Generate sample performance data for demonstration.
        """
        models = ['Random Forest', 'XGBoost', 'Neural Network', 'Isolation Forest', 'LOF', 'One-Class SVM']
        
        data = {
            'Model': models,
            'Accuracy': np.random.uniform(0.85, 0.95, len(models)),
            'Precision': np.random.uniform(0.80, 0.90, len(models)),
            'Recall': np.random.uniform(0.80, 0.90, len(models)),
            'F1-Score': np.random.uniform(0.80, 0.90, len(models)),
            'ROC-AUC': np.random.uniform(0.85, 0.95, len(models))
        }
        
        return pd.DataFrame(data)
        
    def _create_model_comparison_chart(self, performance_data: pd.DataFrame):
        """
        Create a model comparison chart.
        """
        fig = go.Figure()
        
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
        
        for metric in metrics:
            fig.add_trace(go.Bar(
                name=metric,
                x=performance_data['Model'],
                y=performance_data[metric]
            ))
            
        fig.update_layout(
            title="Model Performance Comparison",
            xaxis_title="Model",
            yaxis_title="Score",
            barmode='group'
        )
        
        return fig
        
    def _create_roc_curves(self):
        """
        Create ROC curves for different models.
        """
        fig = go.Figure()
        
        models = ['Random Forest', 'XGBoost', 'Neural Network']
        colors = ['blue', 'red', 'green']
        
        for model, color in zip(models, colors):
            # Generate sample ROC curve data
            fpr = np.linspace(0, 1, 100)
            tpr = np.random.uniform(0.8, 0.95, 100)
            tpr = np.sort(tpr)
            
            fig.add_trace(go.Scatter(
                x=fpr, y=tpr,
                mode='lines',
                name=model,
                line=dict(color=color)
            ))
            
        # Add diagonal line
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines',
            name='Random',
            line=dict(dash='dash', color='gray')
        ))
        
        fig.update_layout(
            title="ROC Curves",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate"
        )
        
        return fig
        
    def _display_confusion_matrices(self):
        """
        Display confusion matrices for different models.
        """
        models = ['Random Forest', 'XGBoost', 'Neural Network']
        
        for model in models:
            st.write(f"**{model} Confusion Matrix**")
            
            # Generate sample confusion matrix
            cm = np.random.randint(50, 200, (2, 2))
            cm[0, 0] = np.random.randint(800, 1000)  # True Negatives
            cm[1, 1] = np.random.randint(50, 150)    # True Positives
            
            fig = px.imshow(
                cm,
                text_auto=True,
                aspect="auto",
                title=f"{model} Confusion Matrix",
                labels=dict(x="Predicted", y="Actual"),
                x=['Normal', 'Anomaly'],
                y=['Normal', 'Anomaly']
            )
            
            st.plotly_chart(fig, use_container_width=True)
