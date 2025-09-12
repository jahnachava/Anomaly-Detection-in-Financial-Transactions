"""
Data loading utilities for financial transaction datasets.

This module provides functionality to load and validate financial transaction data
from various sources including CSV files, databases, and APIs.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Union, Optional, Dict, Any
import logging
from loguru import logger

class DataLoader:
    """
    A class to handle loading and basic validation of financial transaction data.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the DataLoader with configuration.
        
        Args:
            config: Configuration dictionary containing data paths and settings
        """
        self.config = config
        self.data_paths = config.get('data', {})
        self.logger = logger
        
    def load_csv(self, file_path: Union[str, Path], **kwargs) -> pd.DataFrame:
        """
        Load data from a CSV file.
        
        Args:
            file_path: Path to the CSV file
            **kwargs: Additional arguments for pd.read_csv()
            
        Returns:
            Loaded DataFrame
            
        Raises:
            FileNotFoundError: If the file doesn't exist
            ValueError: If the data format is invalid
        """
        try:
            file_path = Path(file_path)
            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")
                
            self.logger.info(f"Loading data from {file_path}")
            df = pd.read_csv(file_path, **kwargs)
            
            self.logger.info(f"Data loaded successfully. Shape: {df.shape}")
            return df
            
        except Exception as e:
            self.logger.error(f"Error loading CSV file: {e}")
            raise
            
    def load_credit_card_fraud_data(self) -> pd.DataFrame:
        """
        Load the Kaggle Credit Card Fraud Detection dataset.
        
        Returns:
            DataFrame containing credit card transaction data
        """
        try:
            # This would typically load from the actual dataset
            # For now, we'll create a placeholder structure
            self.logger.info("Loading Credit Card Fraud Detection dataset")
            
            # In a real implementation, you would load the actual dataset
            # For demonstration, we'll create a sample structure
            n_samples = 10000
            n_features = 30
            
            # Create sample data structure matching the Kaggle dataset
            data = {
                'Time': np.random.uniform(0, 172792, n_samples),
                'Amount': np.random.exponential(88, n_samples),
                'Class': np.random.choice([0, 1], n_samples, p=[0.998, 0.002])
            }
            
            # Add V1-V28 features (PCA transformed features in the original dataset)
            for i in range(1, 29):
                data[f'V{i}'] = np.random.normal(0, 1, n_samples)
                
            df = pd.DataFrame(data)
            
            self.logger.info(f"Sample dataset created. Shape: {df.shape}")
            self.logger.info(f"Fraud rate: {df['Class'].mean():.4f}")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error loading credit card fraud data: {e}")
            raise
            
    def validate_data(self, df: pd.DataFrame, target_column: str = 'Class') -> Dict[str, Any]:
        """
        Validate the loaded dataset for common issues.
        
        Args:
            df: DataFrame to validate
            target_column: Name of the target column
            
        Returns:
            Dictionary containing validation results
        """
        validation_results = {
            'is_valid': True,
            'issues': [],
            'summary': {}
        }
        
        try:
            # Check for missing values
            missing_values = df.isnull().sum()
            if missing_values.any():
                validation_results['issues'].append(f"Missing values found: {missing_values[missing_values > 0].to_dict()}")
                validation_results['is_valid'] = False
                
            # Check for duplicate rows
            duplicates = df.duplicated().sum()
            if duplicates > 0:
                validation_results['issues'].append(f"Duplicate rows found: {duplicates}")
                
            # Check target column
            if target_column not in df.columns:
                validation_results['issues'].append(f"Target column '{target_column}' not found")
                validation_results['is_valid'] = False
            else:
                # Check for class imbalance
                class_counts = df[target_column].value_counts()
                validation_results['summary']['class_distribution'] = class_counts.to_dict()
                validation_results['summary']['class_balance_ratio'] = class_counts.min() / class_counts.max()
                
            # Check data types
            validation_results['summary']['dtypes'] = df.dtypes.to_dict()
            validation_results['summary']['shape'] = df.shape
            validation_results['summary']['memory_usage'] = df.memory_usage(deep=True).sum()
            
            self.logger.info(f"Data validation completed. Valid: {validation_results['is_valid']}")
            if validation_results['issues']:
                self.logger.warning(f"Issues found: {validation_results['issues']}")
                
            return validation_results
            
        except Exception as e:
            self.logger.error(f"Error during data validation: {e}")
            validation_results['is_valid'] = False
            validation_results['issues'].append(f"Validation error: {str(e)}")
            return validation_results
            
    def get_data_info(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Get comprehensive information about the dataset.
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Dictionary containing dataset information
        """
        info = {
            'shape': df.shape,
            'columns': list(df.columns),
            'dtypes': df.dtypes.to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'duplicate_rows': df.duplicated().sum(),
            'memory_usage': df.memory_usage(deep=True).sum(),
            'numeric_columns': df.select_dtypes(include=[np.number]).columns.tolist(),
            'categorical_columns': df.select_dtypes(include=['object']).columns.tolist()
        }
        
        # Add statistical summary for numeric columns
        if info['numeric_columns']:
            info['numeric_summary'] = df[info['numeric_columns']].describe().to_dict()
            
        # Add value counts for categorical columns
        if info['categorical_columns']:
            info['categorical_summary'] = {}
            for col in info['categorical_columns']:
                info['categorical_summary'][col] = df[col].value_counts().to_dict()
                
        return info
