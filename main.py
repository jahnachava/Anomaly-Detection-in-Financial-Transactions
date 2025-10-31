"""
Main entry point for the Anomaly Detection in Financial Transactions system.

This script provides a command-line interface for running the anomaly detection
system with various options for training, evaluation, and real-time monitoring.
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from loguru import logger

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.utils.config_loader import ConfigLoader
from src.utils.logger_setup import setup_logger
from src.data_preprocessing import DataLoader, FeatureEngineer, DataScaler, DataSplitter
from src.models.supervised import SupervisedAnomalyDetector
from src.models.unsupervised import UnsupervisedAnomalyDetector
from src.evaluation import ModelEvaluator, ModelComparator
from src.visualization.dashboard import AnomalyDetectionDashboard

def main():
    """
    Main function to run the anomaly detection system.
    """
    parser = argparse.ArgumentParser(description="Anomaly Detection in Financial Transactions")
    parser.add_argument("--config", type=str, default="config/config.yaml", 
                       help="Path to configuration file")
    parser.add_argument("--mode", type=str, choices=["train", "evaluate", "predict", "dashboard"], 
                       default="dashboard", help="Mode to run the system")
    parser.add_argument("--data", type=str, help="Path to input data file")
    parser.add_argument("--model", type=str, help="Path to model file")
    parser.add_argument("--output", type=str, help="Path to output file")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    try:
        # Load configuration
        config_loader = ConfigLoader(args.config)
        config = config_loader.load_config()
        
        # Validate configuration
        if not config_loader.validate_config():
            logger.error("Invalid configuration file")
            sys.exit(1)
            
        # Set up logging
        setup_logger(config)
        
        if args.verbose:
            logger.info("Verbose logging enabled")
            
        logger.info("Starting Anomaly Detection System")
        logger.info(f"Mode: {args.mode}")
        
        # Run based on mode
        if args.mode == "train":
            train_models(config, args)
        elif args.mode == "evaluate":
            evaluate_models(config, args)
        elif args.mode == "predict":
            predict_anomalies(config, args)
        elif args.mode == "dashboard":
            run_dashboard(config, args)
        else:
            logger.error(f"Unknown mode: {args.mode}")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Error in main: {e}")
        sys.exit(1)

def train_models(config, args):
    """
    Train anomaly detection models.
    
    Args:
        config: Configuration dictionary
        args: Command line arguments
    """
    logger.info("Starting model training")
    
    try:
        # Load data
        data_loader = DataLoader(config)
        if args.data:
            df = data_loader.load_csv(args.data)
        else:
            df = data_loader.load_credit_card_fraud_data()
            
        # Validate data
        validation_results = data_loader.validate_data(df)
        if not validation_results['is_valid']:
            logger.error(f"Data validation failed: {validation_results['issues']}")
            return
            
        # Feature engineering
        feature_engineer = FeatureEngineer(config)
        df = feature_engineer.engineer_all_features(df)
        
        # Data scaling
        data_scaler = DataScaler(config)
        df = data_scaler.scale_numerical_features(df, exclude_columns=['Class'])
        
        # Data splitting
        data_splitter = DataSplitter(config)
        X = df.drop('Class', axis=1)
        y = df['Class']
        
        X_train, X_val, X_test, y_train, y_val, y_test = data_splitter.split_data(
            X, y, 
            test_size=config['data']['dataset']['test_size'],
            validation_size=config['data']['dataset']['validation_size']
        )
        
        # Train supervised models
        logger.info("Training supervised models")
        supervised_detector = SupervisedAnomalyDetector(config)
        supervised_detector.fit(X_train, y_train)
        
        # Train unsupervised models
        logger.info("Training unsupervised models")
        unsupervised_detector = UnsupervisedAnomalyDetector(config)
        unsupervised_detector.fit(X_train)
        
        # Save models
        if args.output:
            output_path = Path(args.output)
            output_path.mkdir(parents=True, exist_ok=True)
            
            supervised_detector.save_models(str(output_path / "supervised"))
            unsupervised_detector.save_models(str(output_path / "unsupervised"))
            
            logger.info(f"Models saved to {output_path}")
            
        logger.info("Model training completed successfully")
        
    except Exception as e:
        logger.error(f"Error in model training: {e}")
        raise

def evaluate_models(config, args):
    """
    Evaluate trained models.
    
    Args:
        config: Configuration dictionary
        args: Command line arguments
    """
    logger.info("Starting model evaluation")
    
    try:
        # Load test data
        data_loader = DataLoader(config)
        if args.data:
            df = data_loader.load_csv(args.data)
        else:
            df = data_loader.load_credit_card_fraud_data()
            
        # Preprocess data
        feature_engineer = FeatureEngineer(config)
        df = feature_engineer.engineer_all_features(df)
        
        data_scaler = DataScaler(config)
        df = data_scaler.scale_numerical_features(df, exclude_columns=['Class'])
        
        # Split data
        data_splitter = DataSplitter(config)
        X = df.drop('Class', axis=1)
        y = df['Class']
        
        X_train, X_val, X_test, y_train, y_val, y_test = data_splitter.split_data(
            X, y, 
            test_size=config['data']['dataset']['test_size'],
            validation_size=config['data']['dataset']['validation_size']
        )
        
        # Load models
        if args.model:
            model_path = Path(args.model)
            
            # Load supervised models
            supervised_detector = SupervisedAnomalyDetector(config)
            supervised_detector.load_models(str(model_path / "supervised"))
            
            # Load unsupervised models
            unsupervised_detector = UnsupervisedAnomalyDetector(config)
            unsupervised_detector.load_models(str(model_path / "unsupervised"))
            
        else:
            logger.error("Model path not specified")
            return
            
        # Evaluate models
        evaluator = ModelEvaluator(config)
        
        # Evaluate supervised models
        supervised_models = {
            'Random Forest': supervised_detector.detectors['random_forest'],
            'XGBoost': supervised_detector.detectors['xgboost'],
            'Neural Network': supervised_detector.detectors['neural_network']
        }
        
        supervised_results = evaluator.evaluate_multiple_models(
            supervised_models, X_train, y_train, X_test, y_test
        )
        
        # Evaluate unsupervised models
        unsupervised_models = {
            'Isolation Forest': unsupervised_detector.detectors['isolation_forest'],
            'LOF': unsupervised_detector.detectors['lof'],
            'One-Class SVM': unsupervised_detector.detectors['one_class_svm'],
            'Autoencoder': unsupervised_detector.detectors['autoencoder']
        }
        
        unsupervised_results = evaluator.evaluate_multiple_models(
            unsupervised_models, X_train, y_train, X_test, y_test
        )
        
        # Compare models
        comparator = ModelComparator(config)
        
        # Compare supervised models
        supervised_comparison = comparator.compare_models(
            supervised_models, X_train, y_train, X_test, y_test
        )
        
        # Compare unsupervised models
        unsupervised_comparison = comparator.compare_models(
            unsupervised_models, X_train, y_train, X_test, y_test
        )
        
        # Generate reports
        supervised_report = comparator.generate_comparison_report(supervised_comparison)
        unsupervised_report = comparator.generate_comparison_report(unsupervised_comparison)
        
        # Save results
        if args.output:
            output_path = Path(args.output)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Save evaluation results
            evaluator.save_evaluation_results(supervised_results, str(output_path / "supervised_evaluation.json"))
            evaluator.save_evaluation_results(unsupervised_results, str(output_path / "unsupervised_evaluation.json"))
            
            # Save comparison results
            comparator.save_comparison_results(supervised_comparison, str(output_path / "supervised_comparison.json"))
            comparator.save_comparison_results(unsupervised_comparison, str(output_path / "unsupervised_comparison.json"))
            
            # Save reports
            with open(output_path / "supervised_report.txt", 'w') as f:
                f.write(supervised_report)
            with open(output_path / "unsupervised_report.txt", 'w') as f:
                f.write(unsupervised_report)
                
            logger.info(f"Evaluation results saved to {output_path}")
            
        # Print reports
        print("\n" + "="*80)
        print("SUPERVISED MODELS EVALUATION REPORT")
        print("="*80)
        print(supervised_report)
        
        print("\n" + "="*80)
        print("UNSUPERVISED MODELS EVALUATION REPORT")
        print("="*80)
        print(unsupervised_report)
        
        logger.info("Model evaluation completed successfully")
        
    except Exception as e:
        logger.error(f"Error in model evaluation: {e}")
        raise

def predict_anomalies(config, args):
    """
    Predict anomalies using trained models.
    
    Args:
        config: Configuration dictionary
        args: Command line arguments
    """
    logger.info("Starting anomaly prediction")
    
    try:
        # Load data
        if not args.data:
            logger.error("Input data file not specified")
            return
            
        data_loader = DataLoader(config)
        df = data_loader.load_csv(args.data)
        
        # Preprocess data
        feature_engineer = FeatureEngineer(config)
        df = feature_engineer.engineer_all_features(df)
        
        data_scaler = DataScaler(config)
        df = data_scaler.scale_numerical_features(df, exclude_columns=['Class'] if 'Class' in df.columns else [])
        
        # Load models
        if not args.model:
            logger.error("Model path not specified")
            return
            
        model_path = Path(args.model)
        
        # Load supervised models
        supervised_detector = SupervisedAnomalyDetector(config)
        supervised_detector.load_models(str(model_path / "supervised"))
        
        # Load unsupervised models
        unsupervised_detector = UnsupervisedAnomalyDetector(config)
        unsupervised_detector.load_models(str(model_path / "unsupervised"))
        
        # Make predictions
        X = df.drop('Class', axis=1) if 'Class' in df.columns else df
        
        # Supervised predictions
        supervised_predictions = supervised_detector.predict(X, method='ensemble')
        supervised_probabilities = supervised_detector.get_ensemble_probabilities(X)
        
        # Unsupervised predictions
        unsupervised_predictions = unsupervised_detector.predict(X, method='ensemble')
        unsupervised_scores = unsupervised_detector.get_anomaly_scores(X)
        
        # Combine results
        results_df = df.copy()
        results_df['supervised_prediction'] = supervised_predictions
        results_df['supervised_probability'] = supervised_probabilities
        results_df['unsupervised_prediction'] = unsupervised_predictions
        results_df['unsupervised_score'] = unsupervised_scores
        
        # Ensemble prediction (simple voting)
        ensemble_prediction = ((supervised_predictions + unsupervised_predictions) > 1).astype(int)
        results_df['ensemble_prediction'] = ensemble_prediction
        
        # Save results
        if args.output:
            results_df.to_csv(args.output, index=False)
            logger.info(f"Predictions saved to {args.output}")
        else:
            # Print summary
            print(f"Total samples: {len(results_df)}")
            print(f"Supervised anomalies: {supervised_predictions.sum()}")
            print(f"Unsupervised anomalies: {unsupervised_predictions.sum()}")
            print(f"Ensemble anomalies: {ensemble_prediction.sum()}")
            
        logger.info("Anomaly prediction completed successfully")
        
    except Exception as e:
        logger.error(f"Error in anomaly prediction: {e}")
        raise

def run_dashboard(config, args):
    """
    Run the interactive dashboard.
    
    Args:
        config: Configuration dictionary
        args: Command line arguments
    """
    logger.info("Starting interactive dashboard")
    
    try:
        # Create and run dashboard
        dashboard = AnomalyDetectionDashboard(config)
        dashboard.run_dashboard()
        
    except Exception as e:
        logger.error(f"Error running dashboard: {e}")
        raise

if __name__ == "__main__":
    main()
