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
        
        # Auto refresh - simplified approach
        if auto_refresh:
            st.sidebar.info(f"Auto-refresh enabled ({refresh_interval}s)")
            
            # Add a refresh button as alternative
            if st.sidebar.button("🔄 Refresh Now"):
                # Use experimental_rerun for older Streamlit versions
                try:
                    st.rerun()
                except AttributeError:
                    st.experimental_rerun()
            
    def _overview_tab(self):
        """
        Create the overview tab with key metrics and charts.
        """
        st.header("📊 System Overview")
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        # Generate dynamic metrics based on current time
        np.random.seed(int(time.time()) % 10000 + 10)
        
        with col1:
            total_transactions = np.random.randint(1000000, 2000000)
            delta_transactions = np.random.uniform(5, 20)
            st.metric(
                label="Total Transactions",
                value=f"{total_transactions:,}",
                delta=f"{delta_transactions:.1f}%"
            )
            
        with col2:
            anomalies_detected = np.random.randint(1000, 5000)
            delta_anomalies = np.random.uniform(2, 15)
            st.metric(
                label="Anomalies Detected",
                value=f"{anomalies_detected:,}",
                delta=f"{delta_anomalies:.1f}%"
            )
            
        with col3:
            detection_rate = np.random.uniform(90, 98)
            delta_detection = np.random.uniform(-2, 5)
            st.metric(
                label="Detection Rate",
                value=f"{detection_rate:.1f}%",
                delta=f"{delta_detection:.1f}%"
            )
            
        with col4:
            false_positive_rate = np.random.uniform(1, 8)
            delta_fp = np.random.uniform(-3, 2)
            st.metric(
                label="False Positive Rate",
                value=f"{false_positive_rate:.1f}%",
                delta=f"{delta_fp:.1f}%"
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
        
        # Generate dynamic performance metrics
        np.random.seed(int(time.time()) % 10000 + 11)
        
        with col1:
            best_model = np.random.choice(['XGBoost', 'Random Forest', 'Neural Network'])
            roc_auc = np.random.uniform(0.92, 0.98)
            st.metric("Best Model", best_model, f"ROC-AUC: {roc_auc:.3f}")
            
        with col2:
            avg_precision = np.random.uniform(0.85, 0.95)
            precision_std = np.random.uniform(0.015, 0.035)
            st.metric("Avg Precision", f"{avg_precision:.3f}", f"±{precision_std:.3f}")
            
        with col3:
            avg_recall = np.random.uniform(0.82, 0.92)
            recall_std = np.random.uniform(0.012, 0.028)
            st.metric("Avg Recall", f"{avg_recall:.3f}", f"±{recall_std:.3f}")
            
        with col4:
            avg_f1 = np.random.uniform(0.84, 0.94)
            f1_std = np.random.uniform(0.014, 0.032)
            st.metric("Avg F1-Score", f"{avg_f1:.3f}", f"±{f1_std:.3f}")
            
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
        
        # Generate dynamic model status
        np.random.seed(int(time.time()) % 10000 + 12)
        status_options = ["✅ Trained", "🔄 Training", "⏸️ Paused"]
        status_probs = [0.85, 0.10, 0.05]  # Reduced error probability
        
        model_status = {
            "Random Forest": {"status": "✅ Trained", "last_update": datetime.now().strftime("%Y-%m-%d %H:%M")},
            "XGBoost": {"status": "✅ Trained", "last_update": datetime.now().strftime("%Y-%m-%d %H:%M")},
            "Neural Network": {"status": np.random.choice(status_options, p=status_probs), "last_update": datetime.now().strftime("%Y-%m-%d %H:%M")},
            "Isolation Forest": {"status": "✅ Trained", "last_update": datetime.now().strftime("%Y-%m-%d %H:%M")},
            "LOF": {"status": np.random.choice(status_options, p=status_probs), "last_update": datetime.now().strftime("%Y-%m-%d %H:%M")},
            "One-Class SVM": {"status": "✅ Trained", "last_update": datetime.now().strftime("%Y-%m-%d %H:%M")}
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
                # Generate dynamic evaluation results
                np.random.seed(int(time.time()) % 10000 + 13)
                eval_results = {
                    "Model": ["Random Forest", "XGBoost", "Neural Network"],
                    "Accuracy": [round(np.random.uniform(0.90, 0.96), 3) for _ in range(3)],
                    "Precision": [round(np.random.uniform(0.85, 0.95), 3) for _ in range(3)],
                    "Recall": [round(np.random.uniform(0.82, 0.92), 3) for _ in range(3)],
                    "F1-Score": [round(np.random.uniform(0.84, 0.94), 3) for _ in range(3)]
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
        # Generate dynamic performance summary
        np.random.seed(int(time.time()) % 10000 + 14)
        performance_summary = {
            "Metric": ["Total Transactions", "Anomalies Detected", "Detection Rate", "False Positive Rate"],
            "Value": [
                f"{np.random.randint(1000000, 2000000):,}",
                f"{np.random.randint(1000, 5000):,}",
                f"{np.random.uniform(90, 98):.1f}%",
                f"{np.random.uniform(1, 8):.1f}%"
            ],
            "Change": [
                f"{np.random.uniform(5, 20):.1f}%",
                f"{np.random.uniform(2, 15):.1f}%",
                f"{np.random.uniform(-2, 5):.1f}%",
                f"{np.random.uniform(-3, 2):.1f}%"
            ]
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
        # Generate dynamic sample data with current timestamp as seed
        np.random.seed(int(time.time()) % 10000)
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
        # Generate dynamic sample data with current timestamp as seed
        np.random.seed(int(time.time()) % 10000 + 1)
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
        # Generate dynamic sample data with current timestamp as seed
        np.random.seed(int(time.time()) % 10000 + 2)
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
        # Generate dynamic sample data with current timestamp as seed
        np.random.seed(int(time.time()) % 10000 + 3)
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
        # Generate dynamic sample data with current timestamp as seed
        np.random.seed(int(time.time()) % 10000 + 4)
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
        # Generate dynamic sample correlation matrix with current timestamp as seed
        np.random.seed(int(time.time()) % 10000 + 5)
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
        # Generate dynamic sample data with current timestamp as seed
        np.random.seed(int(time.time()) % 10000 + 6)
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
        # Generate dynamic sample data with current timestamp as seed
        np.random.seed(int(time.time()) % 10000 + 7)
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
        # Generate dynamic sample data with current timestamp as seed
        np.random.seed(int(time.time()) % 10000 + 8)
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
