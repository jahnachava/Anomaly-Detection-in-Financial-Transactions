# API Reference

This document provides detailed API reference for the Anomaly Detection in Financial Transactions system.

## Table of Contents

1. [Data Preprocessing](#data-preprocessing)
2. [Models](#models)
3. [Evaluation](#evaluation)
4. [Visualization](#visualization)
5. [Utilities](#utilities)

## Data Preprocessing

### DataLoader

The `DataLoader` class handles loading and basic validation of financial transaction data.

#### Methods

##### `load_csv(file_path, **kwargs)`
Load data from a CSV file.

**Parameters:**
- `file_path` (str or Path): Path to the CSV file
- `**kwargs`: Additional arguments for `pd.read_csv()`

**Returns:**
- `pd.DataFrame`: Loaded DataFrame

**Raises:**
- `FileNotFoundError`: If the file doesn't exist
- `ValueError`: If the data format is invalid

##### `load_credit_card_fraud_data()`
Load the Kaggle Credit Card Fraud Detection dataset.

**Returns:**
- `pd.DataFrame`: DataFrame containing credit card transaction data

##### `validate_data(df, target_column='Class')`
Validate the loaded dataset for common issues.

**Parameters:**
- `df` (pd.DataFrame): DataFrame to validate
- `target_column` (str): Name of the target column

**Returns:**
- `Dict[str, Any]`: Dictionary containing validation results

### FeatureEngineer

The `FeatureEngineer` class handles feature engineering for financial transaction data.

#### Methods

##### `create_time_features(df, time_column='Time')`
Create time-based features from timestamp data.

**Parameters:**
- `df` (pd.DataFrame): DataFrame containing time data
- `time_column` (str): Name of the time column

**Returns:**
- `pd.DataFrame`: DataFrame with additional time features

##### `create_amount_features(df, amount_column='Amount')`
Create amount-based features for transaction analysis.

**Parameters:**
- `df` (pd.DataFrame): DataFrame containing amount data
- `amount_column` (str): Name of the amount column

**Returns:**
- `pd.DataFrame`: DataFrame with additional amount features

##### `select_features(X, y, method='mutual_info', k=20)`
Select the best features using various selection methods.

**Parameters:**
- `X` (pd.DataFrame): Feature matrix
- `y` (pd.Series): Target variable
- `method` (str): Feature selection method ('mutual_info', 'f_score', 'chi2')
- `k` (int): Number of features to select

**Returns:**
- `Tuple[pd.DataFrame, Any]`: Tuple of (selected_features, feature_selector)

### DataScaler

The `DataScaler` class handles scaling and normalization of financial transaction data.

#### Methods

##### `scale_features(df, feature_columns=None, method=None, fit=True)`
Scale specified features in the DataFrame.

**Parameters:**
- `df` (pd.DataFrame): DataFrame containing features to scale
- `feature_columns` (List[str]): List of column names to scale
- `method` (str): Scaling method to use
- `fit` (bool): Whether to fit the scaler or use existing fitted scaler

**Returns:**
- `pd.DataFrame`: DataFrame with scaled features

##### `handle_outliers(df, feature_columns=None, method='iqr', factor=1.5)`
Handle outliers in the dataset using various methods.

**Parameters:**
- `df` (pd.DataFrame): DataFrame containing features
- `feature_columns` (List[str]): List of columns to process
- `method` (str): Outlier handling method ('iqr', 'zscore', 'isolation')
- `factor` (float): Factor for outlier detection (for IQR method)

**Returns:**
- `pd.DataFrame`: DataFrame with outliers handled

### DataSplitter

The `DataSplitter` class handles data splitting and resampling for machine learning.

#### Methods

##### `split_data(X, y, test_size=0.2, validation_size=0.1, random_state=42, stratify=True)`
Split data into train, validation, and test sets.

**Parameters:**
- `X` (pd.DataFrame): Feature matrix
- `y` (pd.Series): Target variable
- `test_size` (float): Proportion of data for test set
- `validation_size` (float): Proportion of data for validation set
- `random_state` (int): Random state for reproducibility
- `stratify` (bool): Whether to use stratified splitting

**Returns:**
- `Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]`: Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)

##### `handle_imbalanced_data(X, y, method='smote', sampling_strategy='auto', random_state=42)`
Handle imbalanced datasets using various resampling techniques.

**Parameters:**
- `X` (pd.DataFrame): Feature matrix
- `y` (pd.Series): Target variable
- `method` (str): Resampling method
- `sampling_strategy` (str): Sampling strategy for resampling
- `random_state` (int): Random state for reproducibility

**Returns:**
- `Tuple[pd.DataFrame, pd.Series]`: Tuple of (X_resampled, y_resampled)

## Models

### SupervisedAnomalyDetector

The `SupervisedAnomalyDetector` class combines multiple supervised learning algorithms.

#### Methods

##### `fit(X, y, detectors=None)`
Fit all or selected supervised detectors.

**Parameters:**
- `X` (pd.DataFrame): Feature matrix
- `y` (pd.Series): Target variable (0 for normal, 1 for anomaly)
- `detectors` (List[str]): List of detector names to fit

**Returns:**
- `SupervisedAnomalyDetector`: Self

##### `predict(X, method='ensemble', detectors=None, threshold=0.5)`
Predict anomalies using individual detectors or ensemble method.

**Parameters:**
- `X` (pd.DataFrame): Feature matrix
- `method` (str): Prediction method ('ensemble', 'voting', 'individual')
- `detectors` (List[str]): List of detector names to use
- `threshold` (float): Probability threshold for ensemble predictions

**Returns:**
- `np.ndarray`: Array of predictions (1 for anomaly, 0 for normal)

### UnsupervisedAnomalyDetector

The `UnsupervisedAnomalyDetector` class combines multiple unsupervised learning algorithms.

#### Methods

##### `fit(X, y=None, detectors=None)`
Fit all or selected unsupervised detectors.

**Parameters:**
- `X` (pd.DataFrame): Feature matrix
- `y` (pd.Series): Target variable (ignored for unsupervised learning)
- `detectors` (List[str]): List of detector names to fit

**Returns:**
- `UnsupervisedAnomalyDetector`: Self

##### `predict(X, method='ensemble', detectors=None)`
Predict anomalies using individual detectors or ensemble method.

**Parameters:**
- `X` (pd.DataFrame): Feature matrix
- `method` (str): Prediction method ('ensemble', 'voting', 'individual')
- `detectors` (List[str]): List of detector names to use

**Returns:**
- `np.ndarray`: Array of predictions (1 for anomaly, 0 for normal)

### Individual Model Classes

#### RandomForestDetector

##### `fit(X, y)`
Fit the Random Forest model.

**Parameters:**
- `X` (pd.DataFrame): Feature matrix
- `y` (pd.Series): Target variable

**Returns:**
- `RandomForestDetector`: Self

##### `predict(X)`
Predict anomalies using Random Forest.

**Parameters:**
- `X` (pd.DataFrame): Feature matrix

**Returns:**
- `np.ndarray`: Array of predictions

#### XGBoostDetector

##### `fit(X, y)`
Fit the XGBoost model.

**Parameters:**
- `X` (pd.DataFrame): Feature matrix
- `y` (pd.Series): Target variable

**Returns:**
- `XGBoostDetector`: Self

##### `predict(X)`
Predict anomalies using XGBoost.

**Parameters:**
- `X` (pd.DataFrame): Feature matrix

**Returns:**
- `np.ndarray`: Array of predictions

#### IsolationForestDetector

##### `fit(X, y=None)`
Fit the Isolation Forest model.

**Parameters:**
- `X` (pd.DataFrame): Feature matrix
- `y` (pd.Series): Target variable (ignored)

**Returns:**
- `IsolationForestDetector`: Self

##### `predict(X)`
Predict anomalies using Isolation Forest.

**Parameters:**
- `X` (pd.DataFrame): Feature matrix

**Returns:**
- `np.ndarray`: Array of predictions

## Evaluation

### ModelEvaluator

The `ModelEvaluator` class evaluates anomaly detection models comprehensively.

#### Methods

##### `evaluate_model(model, X, y, X_test=None, y_test=None, threshold=0.5)`
Evaluate a single model comprehensively.

**Parameters:**
- `model`: Trained model object
- `X` (pd.DataFrame): Training feature matrix
- `y` (pd.Series): Training target variable
- `X_test` (pd.DataFrame): Test feature matrix (optional)
- `y_test` (pd.Series): Test target variable (optional)
- `threshold` (float): Probability threshold for predictions

**Returns:**
- `Dict[str, Any]`: Dictionary containing evaluation results

##### `evaluate_multiple_models(models, X, y, X_test=None, y_test=None, threshold=0.5)`
Evaluate multiple models and compare their performance.

**Parameters:**
- `models` (Dict[str, Any]): Dictionary of model names and model objects
- `X` (pd.DataFrame): Training feature matrix
- `y` (pd.Series): Training target variable
- `X_test` (pd.DataFrame): Test feature matrix (optional)
- `y_test` (pd.Series): Test target variable (optional)
- `threshold` (float): Probability threshold for predictions

**Returns:**
- `Dict[str, Dict[str, Any]]`: Dictionary containing evaluation results for all models

### ModelComparator

The `ModelComparator` class compares multiple anomaly detection models.

#### Methods

##### `compare_models(models, X, y, X_test=None, y_test=None, threshold=0.5)`
Compare multiple models comprehensively.

**Parameters:**
- `models` (Dict[str, Any]): Dictionary of model names and model objects
- `X` (pd.DataFrame): Training feature matrix
- `y` (pd.Series): Training target variable
- `X_test` (pd.DataFrame): Test feature matrix (optional)
- `y_test` (pd.Series): Test target variable (optional)
- `threshold` (float): Probability threshold for predictions

**Returns:**
- `Dict[str, Any]`: Dictionary containing comprehensive comparison results

### PerformanceMetrics

The `PerformanceMetrics` class computes comprehensive performance metrics.

#### Methods

##### `compute_comprehensive_metrics(y_true, y_pred, y_proba=None)`
Compute all available metrics.

**Parameters:**
- `y_true` (np.ndarray): True labels
- `y_pred` (np.ndarray): Predicted labels
- `y_proba` (np.ndarray): Predicted probabilities (optional)

**Returns:**
- `Dict[str, Any]`: Dictionary containing all metrics

### ThresholdOptimizer

The `ThresholdOptimizer` class optimizes thresholds for anomaly detection models.

#### Methods

##### `find_optimal_threshold(y_true, y_proba, method='youden', custom_criteria=None, **kwargs)`
Find optimal threshold using various methods.

**Parameters:**
- `y_true` (np.ndarray): True labels
- `y_proba` (np.ndarray): Predicted probabilities
- `method` (str): Optimization method ('youden', 'f1_optimal', 'precision_recall_curve', 'custom')
- `custom_criteria` (Callable): Custom function for threshold optimization
- `**kwargs`: Additional parameters for specific methods

**Returns:**
- `Dict[str, Any]`: Dictionary containing optimal threshold and metrics

## Visualization

### AnomalyDetectionDashboard

The `AnomalyDetectionDashboard` class provides an interactive dashboard for monitoring.

#### Methods

##### `run_dashboard()`
Run the main dashboard application.

### PlotUtils

The `PlotUtils` class provides utility functions for creating various plots.

#### Methods

##### `plot_anomaly_distribution(data, anomaly_column='anomaly', title='Anomaly Distribution')`
Create a plot showing the distribution of anomalies.

**Parameters:**
- `data` (pd.DataFrame): DataFrame containing the data
- `anomaly_column` (str): Name of the anomaly column
- `title` (str): Plot title

**Returns:**
- `go.Figure`: Plotly figure object

##### `plot_feature_importance(feature_names, importance_scores, title='Feature Importance', top_n=20)`
Create a horizontal bar plot for feature importance.

**Parameters:**
- `feature_names` (List[str]): List of feature names
- `importance_scores` (np.ndarray): Array of importance scores
- `title` (str): Plot title
- `top_n` (int): Number of top features to display

**Returns:**
- `go.Figure`: Plotly figure object

##### `plot_confusion_matrix(confusion_matrix, class_names=None, title='Confusion Matrix')`
Create a heatmap for confusion matrix.

**Parameters:**
- `confusion_matrix` (np.ndarray): Confusion matrix array
- `class_names` (List[str]): List of class names
- `title` (str): Plot title

**Returns:**
- `go.Figure`: Plotly figure object

### RealTimeMonitor

The `RealTimeMonitor` class provides real-time monitoring functionality.

#### Methods

##### `start_monitoring(data_source=None)`
Start real-time monitoring.

**Parameters:**
- `data_source` (Callable): Function that provides new transaction data

##### `stop_monitoring()`
Stop real-time monitoring.

##### `get_latest_data(n_transactions=100)`
Get the latest transaction data.

**Parameters:**
- `n_transactions` (int): Number of recent transactions to return

**Returns:**
- `pd.DataFrame`: DataFrame with recent transactions

##### `get_latest_anomalies(n_anomalies=50)`
Get the latest detected anomalies.

**Parameters:**
- `n_anomalies` (int): Number of recent anomalies to return

**Returns:**
- `pd.DataFrame`: DataFrame with recent anomalies

## Utilities

### ConfigLoader

The `ConfigLoader` class loads and validates configuration files.

#### Methods

##### `load_config()`
Load configuration from file.

**Returns:**
- `Dict[str, Any]`: Dictionary containing configuration

##### `validate_config()`
Validate the loaded configuration.

**Returns:**
- `bool`: True if configuration is valid, False otherwise

##### `get_config_value(key_path, default=None)`
Get a configuration value using dot notation.

**Parameters:**
- `key_path` (str): Dot-separated path to the configuration value
- `default` (Any): Default value if key is not found

**Returns:**
- `Any`: Configuration value or default

### Logger Setup

#### `setup_logger(config)`
Set up logging configuration for the anomaly detection system.

**Parameters:**
- `config` (Dict[str, Any]): Configuration dictionary containing logging settings

#### `get_logger(name=None)`
Get a logger instance.

**Parameters:**
- `name` (str): Logger name (optional)

**Returns:**
- `logger`: Logger instance

## Configuration

The system uses YAML configuration files. The main configuration file is `config/config.yaml` and contains the following sections:

### Data Configuration
- `data.raw_data_path`: Path to raw data
- `data.processed_data_path`: Path to processed data
- `data.dataset`: Dataset-specific settings
- `data.features`: Feature engineering settings

### Model Configuration
- `models.supervised`: Supervised learning model parameters
- `models.unsupervised`: Unsupervised learning model parameters

### Evaluation Configuration
- `evaluation.metrics`: List of metrics to compute
- `evaluation.cross_validation`: Cross-validation settings
- `evaluation.threshold_optimization`: Threshold optimization settings

### Dashboard Configuration
- `dashboard.title`: Dashboard title
- `dashboard.host`: Dashboard host
- `dashboard.port`: Dashboard port
- `dashboard.real_time`: Real-time monitoring settings

### Logging Configuration
- `logging.level`: Logging level
- `logging.format`: Log message format
- `logging.file`: Log file path
- `logging.max_file_size`: Maximum log file size
- `logging.backup_count`: Number of backup files to keep
