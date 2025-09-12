# Anomaly Detection in Financial Transactions using Machine Learning

**Dissertation Project**  
**Student:** Chava Jahna (ID: 2021HX70019)  
**Supervisor:** Venkatramani Vignesh  
**Institution:** BITS Pilani  

## Abstract

The rapid growth of digital banking, e-commerce, and online payment systems has made financial transactions more convenient, but it has also increased the risk of fraudulent activities. Manual fraud detection techniques are slow and error-prone. This dissertation aims to develop an automated anomaly detection system for financial transactions using Machine Learning (ML) techniques.

The system will process transactional data to identify unusual patterns indicative of potential fraud, money laundering, or policy violations. By leveraging statistical analysis and ML models, the system will detect anomalies in real-time, enabling prompt preventive action.

## Objectives

1. To design and develop an ML-based anomaly detection system for financial transactions
2. To evaluate and compare supervised and unsupervised anomaly detection techniques
3. To minimize false positives while maintaining high fraud detection accuracy
4. To provide real-time transaction monitoring through a visual dashboard

## Project Structure

```
├── data/                          # Dataset storage
│   ├── raw/                       # Original datasets
│   ├── processed/                 # Preprocessed datasets
│   └── external/                  # External data sources
├── src/                          # Source code
│   ├── data_preprocessing/        # Data preprocessing modules
│   ├── models/                   # ML model implementations
│   │   ├── supervised/           # Supervised learning models
│   │   └── unsupervised/         # Unsupervised learning models
│   ├── evaluation/               # Model evaluation modules
│   ├── visualization/            # Dashboard and visualization
│   └── utils/                    # Utility functions
├── notebooks/                    # Jupyter notebooks for analysis
├── tests/                        # Unit tests
├── docs/                         # Documentation
├── config/                       # Configuration files
└── requirements.txt              # Python dependencies
```

## Methodology

### 1. Data Acquisition
- Gathering anonymized financial transaction datasets from open banking repositories
- Using Kaggle Credit Card Fraud Detection Dataset

### 2. Data Preprocessing
- Handling missing values
- Encoding categorical variables
- Scaling numerical values
- Creating time-based features

### 3. Model Development
- **Supervised Learning:** Random Forest, XGBoost, Neural Networks
- **Unsupervised Learning:** Isolation Forest, Local Outlier Factor (LOF), Autoencoders

### 4. Model Evaluation
- Precision, Recall, F1-score
- ROC-AUC
- Confusion Matrix
- False-positive rate analysis

### 5. Deployment
- Real-time transaction monitoring dashboard
- Anomaly visualization interface
- Explainability modules (SHAP, LIME)

## Key Features

- **Feature Engineering:** Transaction amount, frequency, merchant category, geographic location, device fingerprint, time-of-day analysis
- **Real-time Processing:** Live transaction monitoring capabilities
- **Explainable AI:** Integration of SHAP and LIME for model interpretability
- **Scalable Architecture:** Designed for integration into financial institutions' monitoring systems

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd Anomaly-Detection-in-Financial-Transactions
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Data Preprocessing
```bash
python src/data_preprocessing/preprocess.py
```

### Model Training
```bash
python src/models/train_models.py
```

### Dashboard Launch
```bash
python src/visualization/dashboard.py
```

## Technologies Used

- **Python 3.8+**
- **Scikit-learn** - Machine learning algorithms
- **Pandas** - Data manipulation
- **NumPy** - Numerical computing
- **Matplotlib/Seaborn** - Data visualization
- **XGBoost** - Gradient boosting
- **Streamlit** - Dashboard development
- **SHAP/LIME** - Model explainability

## Project Timeline (16 Weeks)

| Week | Task | Deliverable |
|------|------|-------------|
| 1-2 | Literature review | Literature review report |
| 2-3 | Dataset acquisition & exploration | Anonymized dataset |
| 3-5 | Data preprocessing & feature engineering | Preprocessed dataset |
| 5-6 | Baseline model implementation | Initial ML model |
| 6-8 | Supervised learning model training | Trained supervised model |
| 8-9 | Model tuning & ensemble approaches | Optimized models |
| 9-10 | Evaluation & false positive analysis | Evaluation report |
| 10-12 | Dashboard development | Dashboard prototype |
| 12-13 | Integration of ML backend with dashboard | Fully functional system |
| 13-14 | Real-time streaming simulation & testing | Tested system |
| 14-16 | Documentation & final report writing | Dissertation report |

## Contributing

This is a dissertation project. For questions or suggestions, please contact:
- **Student:** Chava Jahna (2021HX70019)
- **Supervisor:** Venkatramani Vignesh (vigneshv2@hexaware.com)

## License

This project is licensed under the MIT License - see the LICENSE file for details.
