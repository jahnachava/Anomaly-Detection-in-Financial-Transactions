# User Guide: Anomaly Detection in Financial Transactions

This comprehensive user guide will help you understand, set up, and use the Anomaly Detection in Financial Transactions system effectively.

## Table of Contents

1. [Getting Started](#getting-started)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [Configuration](#configuration)
5. [Data Preparation](#data-preparation)
6. [Model Training](#model-training)
7. [Model Evaluation](#model-evaluation)
8. [Real-time Monitoring](#real-time-monitoring)
9. [Dashboard Usage](#dashboard-usage)
10. [Troubleshooting](#troubleshooting)
11. [Best Practices](#best-practices)

## Getting Started

### Overview

The Anomaly Detection in Financial Transactions system is a comprehensive machine learning solution designed to identify fraudulent and anomalous patterns in financial transaction data. The system supports both supervised and unsupervised learning approaches and provides real-time monitoring capabilities.

### Key Features

- **Multiple ML Algorithms**: Random Forest, XGBoost, Neural Networks, Isolation Forest, LOF, One-Class SVM, Autoencoders
- **Real-time Monitoring**: Live transaction monitoring with instant anomaly detection
- **Interactive Dashboard**: Web-based interface for visualization and management
- **Comprehensive Evaluation**: Detailed performance metrics and model comparison
- **Scalable Architecture**: Designed for production deployment
- **Explainable AI**: Integration with SHAP and LIME for model interpretability

### System Requirements

- Python 3.8 or higher
- 8GB RAM minimum (16GB recommended)
- 10GB free disk space
- Modern web browser for dashboard access

## Installation

### Step 1: Clone the Repository

```bash
git clone <repository-url>
cd Anomaly-Detection-in-Financial-Transactions
```

### Step 2: Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Verify Installation

```bash
python -c "import src; print('Installation successful!')"
```

## Quick Start

### 1. Basic Usage

The simplest way to get started is using the command-line interface:

```bash
# Run the interactive dashboard
python main.py --mode dashboard

# Train models with default settings
python main.py --mode train

# Evaluate trained models
python main.py --mode evaluate --model models/

# Make predictions on new data
python main.py --mode predict --data data/test.csv --model models/
```

### 2. Using the Dashboard

1. Start the dashboard:
   ```bash
   python main.py --mode dashboard
   ```

2. Open your web browser and navigate to `http://localhost:8501`

3. The dashboard provides:
   - Real-time transaction monitoring
   - Model performance visualization
   - Anomaly detection results
   - System configuration management

### 3. Programmatic Usage

```python
import sys
sys.path.append('src')

from utils.config_loader import ConfigLoader
from data_preprocessing import DataLoader, FeatureEngineer
from models.supervised import SupervisedAnomalyDetector

# Load configuration
config_loader = ConfigLoader('config/config.yaml')
config = config_loader.load_config()

# Load and preprocess data
data_loader = DataLoader(config)
df = data_loader.load_credit_card_fraud_data()

feature_engineer = FeatureEngineer(config)
df = feature_engineer.engineer_all_features(df)

# Train models
X = df.drop('Class', axis=1)
y = df['Class']

detector = SupervisedAnomalyDetector(config)
detector.fit(X, y)

# Make predictions
predictions = detector.predict(X)
```

## Configuration

### Configuration File Structure

The system uses YAML configuration files. The main configuration file is `config/config.yaml`:

```yaml
# Data Configuration
data:
  raw_data_path: "data/raw/"
  processed_data_path: "data/processed/"
  dataset:
    name: "credit_card_fraud"
    target_column: "Class"
    test_size: 0.2
    validation_size: 0.1

# Model Configuration
models:
  supervised:
    random_forest:
      n_estimators: 100
      max_depth: 10
    xgboost:
      n_estimators: 100
      learning_rate: 0.1
  unsupervised:
    isolation_forest:
      n_estimators: 100
      contamination: 0.1

# Dashboard Configuration
dashboard:
  title: "Financial Transaction Anomaly Detection Dashboard"
  host: "localhost"
  port: 8501
```

### Key Configuration Parameters

#### Data Settings
- `data.dataset.test_size`: Proportion of data for testing (default: 0.2)
- `data.dataset.validation_size`: Proportion of data for validation (default: 0.1)
- `data.features.scaling_method`: Feature scaling method ('standard', 'minmax', 'robust')

#### Model Settings
- `models.supervised.random_forest.n_estimators`: Number of trees in Random Forest
- `models.supervised.xgboost.learning_rate`: Learning rate for XGBoost
- `models.unsupervised.isolation_forest.contamination`: Expected anomaly rate

#### Dashboard Settings
- `dashboard.port`: Port for the web dashboard
- `dashboard.real_time.update_interval`: Refresh interval in seconds

## Data Preparation

### Supported Data Formats

The system supports CSV files with the following structure:

```csv
Time,V1,V2,V3,...,V28,Amount,Class
0,-1.359807134,-0.072781173,2.536346738,...,0.133558377,149.62,0
0,1.191857111,0.266150712,0.166480113,...,-0.021053053,2.69,0
```

### Required Columns

- **Time**: Transaction timestamp (in seconds)
- **V1-V28**: PCA-transformed features (numerical)
- **Amount**: Transaction amount (numerical)
- **Class**: Target variable (0 for normal, 1 for fraud)

### Data Quality Requirements

1. **No Missing Values**: All features must have complete data
2. **Proper Data Types**: Numerical features should be numeric
3. **Balanced Classes**: System handles imbalanced datasets automatically
4. **Sufficient Data**: Minimum 1000 samples recommended

### Data Loading Examples

```python
# Load from CSV file
data_loader = DataLoader(config)
df = data_loader.load_csv('data/transactions.csv')

# Load sample dataset
df = data_loader.load_credit_card_fraud_data()

# Validate data quality
validation_results = data_loader.validate_data(df)
if not validation_results['is_valid']:
    print("Data validation failed:", validation_results['issues'])
```

## Model Training

### Supervised Learning Models

#### Random Forest
```python
from models.supervised import RandomForestDetector

detector = RandomForestDetector(config)
detector.fit(X_train, y_train)
predictions = detector.predict(X_test)
```

#### XGBoost
```python
from models.supervised import XGBoostDetector

detector = XGBoostDetector(config)
detector.fit(X_train, y_train)
predictions = detector.predict(X_test)
```

#### Neural Network
```python
from models.supervised import NeuralNetworkDetector

detector = NeuralNetworkDetector(config)
detector.fit(X_train, y_train)
predictions = detector.predict(X_test)
```

### Unsupervised Learning Models

#### Isolation Forest
```python
from models.unsupervised import IsolationForestDetector

detector = IsolationForestDetector(config)
detector.fit(X_train)
predictions = detector.predict(X_test)
```

#### Local Outlier Factor (LOF)
```python
from models.unsupervised import LOFDetector

detector = LOFDetector(config)
detector.fit(X_train)
predictions = detector.predict(X_test)
```

### Ensemble Training

```python
from models.supervised import SupervisedAnomalyDetector

# Train all supervised models
detector = SupervisedAnomalyDetector(config)
detector.fit(X_train, y_train)

# Make ensemble predictions
predictions = detector.predict(X_test, method='ensemble')
```

### Model Persistence

```python
# Save trained models
detector.save_models('models/supervised/')

# Load trained models
detector.load_models('models/supervised/')
```

## Model Evaluation

### Performance Metrics

The system provides comprehensive evaluation metrics:

- **Accuracy**: Overall correctness
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under the ROC curve
- **Average Precision**: Area under the precision-recall curve

### Evaluation Examples

```python
from evaluation import ModelEvaluator

evaluator = ModelEvaluator(config)

# Evaluate single model
results = evaluator.evaluate_model(model, X_train, y_train, X_test, y_test)

# Evaluate multiple models
models = {
    'Random Forest': rf_model,
    'XGBoost': xgb_model,
    'Neural Network': nn_model
}
results = evaluator.evaluate_multiple_models(models, X_train, y_train, X_test, y_test)
```

### Model Comparison

```python
from evaluation import ModelComparator

comparator = ModelComparator(config)
comparison_results = comparator.compare_models(models, X_train, y_train, X_test, y_test)

# Generate comparison report
report = comparator.generate_comparison_report(comparison_results)
print(report)
```

### Threshold Optimization

```python
from evaluation import ThresholdOptimizer

optimizer = ThresholdOptimizer(config)

# Find optimal threshold using Youden's J statistic
optimal_threshold = optimizer.find_optimal_threshold(y_true, y_proba, method='youden')

# Find optimal threshold for business requirements
business_threshold = optimizer.optimize_for_business_requirements(
    y_true, y_proba, 
    max_false_positive_rate=0.05,
    min_recall=0.8
)
```

## Real-time Monitoring

### Setting Up Real-time Monitoring

```python
from visualization import RealTimeMonitor

monitor = RealTimeMonitor(config)

# Set up callbacks
def anomaly_callback(anomaly_data):
    print(f"Anomaly detected: {len(anomaly_data)} transactions")

def alert_callback(alert_info):
    print(f"Alert: {alert_info['message']}")

monitor.set_anomaly_callback(anomaly_callback)
monitor.set_alert_callback(alert_callback)

# Start monitoring
monitor.start_monitoring()
```

### Data Source Integration

```python
def custom_data_source():
    # Your custom data loading logic
    return new_transaction_data

monitor.start_monitoring(data_source=custom_data_source)
```

### Monitoring Statistics

```python
# Get current statistics
stats = monitor.get_statistics()
print(f"Total transactions: {stats['total_transactions']}")
print(f"Anomalies detected: {stats['anomalies_detected']}")
print(f"Anomaly rate: {stats['anomaly_rate']:.2%}")
```

## Dashboard Usage

### Dashboard Features

The interactive dashboard provides:

1. **Overview Tab**: System metrics and key performance indicators
2. **Anomaly Detection Tab**: Real-time anomaly monitoring and visualization
3. **Model Performance Tab**: Model comparison and performance metrics
4. **Model Management Tab**: Model training and configuration
5. **Reports Tab**: Downloadable reports and analytics

### Navigation

1. **Sidebar Controls**:
   - Date range selection
   - Model selection
   - Threshold settings
   - Real-time refresh options

2. **Main Content Areas**:
   - Interactive charts and visualizations
   - Data tables with filtering
   - Configuration panels
   - Download options

### Customization

You can customize the dashboard by modifying the configuration:

```yaml
dashboard:
  title: "Your Custom Dashboard Title"
  theme: "dark"  # or "light"
  real_time:
    update_interval: 10  # seconds
    max_transactions_display: 2000
```

## Troubleshooting

### Common Issues

#### 1. Import Errors
```
ModuleNotFoundError: No module named 'src'
```
**Solution**: Ensure you're running from the project root directory and have added the src directory to your Python path.

#### 2. Configuration Errors
```
FileNotFoundError: Configuration file not found
```
**Solution**: Verify the configuration file exists at `config/config.yaml` and has proper YAML syntax.

#### 3. Memory Issues
```
MemoryError: Unable to allocate array
```
**Solution**: Reduce batch size in model configuration or use data sampling for large datasets.

#### 4. Dashboard Not Loading
```
Connection refused on localhost:8501
```
**Solution**: Check if the port is available and firewall settings allow the connection.

### Debug Mode

Enable verbose logging for debugging:

```bash
python main.py --mode dashboard --verbose
```

### Log Files

Check log files for detailed error information:
- Location: `logs/anomaly_detection.log`
- Format: Timestamp, Level, Module, Message

## Best Practices

### Data Preparation

1. **Data Quality**: Always validate data quality before training
2. **Feature Engineering**: Create meaningful features based on domain knowledge
3. **Data Splitting**: Use stratified splitting for imbalanced datasets
4. **Scaling**: Apply appropriate scaling to numerical features

### Model Training

1. **Cross-Validation**: Use cross-validation for robust model evaluation
2. **Hyperparameter Tuning**: Optimize hyperparameters for better performance
3. **Ensemble Methods**: Combine multiple models for improved accuracy
4. **Regularization**: Use regularization to prevent overfitting

### Model Evaluation

1. **Appropriate Metrics**: Use metrics suitable for imbalanced datasets
2. **Threshold Optimization**: Optimize thresholds based on business requirements
3. **Statistical Testing**: Perform statistical tests for model comparison
4. **Cost Analysis**: Consider the cost of false positives and false negatives

### Production Deployment

1. **Model Versioning**: Keep track of model versions and performance
2. **Monitoring**: Implement continuous monitoring of model performance
3. **Alerting**: Set up alerts for performance degradation
4. **Backup**: Maintain backup models and data

### Security Considerations

1. **Data Privacy**: Ensure compliance with data protection regulations
2. **Access Control**: Implement proper access controls for the system
3. **Audit Logging**: Maintain audit logs for all system activities
4. **Encryption**: Use encryption for sensitive data transmission

### Performance Optimization

1. **Batch Processing**: Process data in batches for large datasets
2. **Caching**: Cache frequently accessed data and models
3. **Parallel Processing**: Use parallel processing for model training
4. **Resource Management**: Monitor and manage system resources

## Support and Resources

### Documentation
- [API Reference](API_REFERENCE.md): Detailed API documentation
- [Configuration Guide](CONFIGURATION.md): Configuration options and examples
- [Examples](examples/): Code examples and tutorials

### Getting Help
- Check the troubleshooting section above
- Review log files for error details
- Consult the API reference for method signatures
- Test with sample data to isolate issues

### Contributing
- Follow the coding standards in the project
- Add tests for new features
- Update documentation for changes
- Submit pull requests for improvements

This user guide should help you get started with the Anomaly Detection in Financial Transactions system. For more detailed information, refer to the API reference and configuration documentation.
