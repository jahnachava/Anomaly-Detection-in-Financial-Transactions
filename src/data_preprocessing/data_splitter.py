"""
Data splitting utilities for train-test-validation splits.

This module provides functionality to split datasets for machine learning
with proper handling of imbalanced datasets and time series data.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, Optional, List
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.model_selection import TimeSeriesSplit
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
from imblearn.under_sampling import RandomUnderSampler, EditedNearestNeighbours
from imblearn.combine import SMOTETomek, SMOTEENN
from loguru import logger
import warnings
warnings.filterwarnings('ignore')

class DataSplitter:
    """
    A class to handle data splitting and resampling for machine learning.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the DataSplitter with configuration.
        
        Args:
            config: Configuration dictionary containing splitting settings
        """
        self.config = config
        self.data_config = config.get('data', {})
        self.logger = logger
        
    def split_data(self, X: pd.DataFrame, y: pd.Series,
                   test_size: float = 0.2, 
                   validation_size: float = 0.1,
                   random_state: int = 42,
                   stratify: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, 
                                                  pd.Series, pd.Series, pd.Series]:
        """
        Split data into train, validation, and test sets.
        
        Args:
            X: Feature matrix
            y: Target variable
            test_size: Proportion of data for test set
            validation_size: Proportion of data for validation set
            random_state: Random state for reproducibility
            stratify: Whether to use stratified splitting
            
        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        try:
            self.logger.info(f"Splitting data with test_size={test_size}, validation_size={validation_size}")
            
            # First split: separate test set
            stratify_param = y if stratify else None
            X_temp, X_test, y_temp, y_test = train_test_split(
                X, y, 
                test_size=test_size, 
                random_state=random_state,
                stratify=stratify_param
            )
            
            # Second split: separate train and validation sets
            val_size_adjusted = validation_size / (1 - test_size)
            stratify_param = y_temp if stratify else None
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp,
                test_size=val_size_adjusted,
                random_state=random_state,
                stratify=stratify_param
            )
            
            self.logger.info(f"Data split completed:")
            self.logger.info(f"  Train: {X_train.shape[0]} samples")
            self.logger.info(f"  Validation: {X_val.shape[0]} samples")
            self.logger.info(f"  Test: {X_test.shape[0]} samples")
            
            # Log class distribution
            for split_name, y_split in [('Train', y_train), ('Validation', y_val), ('Test', y_test)]:
                class_counts = y_split.value_counts()
                self.logger.info(f"  {split_name} class distribution: {class_counts.to_dict()}")
                
            return X_train, X_val, X_test, y_train, y_val, y_test
            
        except Exception as e:
            self.logger.error(f"Error splitting data: {e}")
            raise
            
    def time_series_split(self, X: pd.DataFrame, y: pd.Series,
                         n_splits: int = 5,
                         test_size: float = 0.2) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        """
        Create time series splits for temporal data.
        
        Args:
            X: Feature matrix
            y: Target variable
            n_splits: Number of splits
            test_size: Proportion of data for test set
            
        Returns:
            List of (train_idx, test_idx) tuples
        """
        try:
            tscv = TimeSeriesSplit(n_splits=n_splits)
            splits = []
            
            for train_idx, test_idx in tscv.split(X):
                splits.append((train_idx, test_idx))
                
            self.logger.info(f"Time series splits created: {len(splits)} splits")
            return splits
            
        except Exception as e:
            self.logger.error(f"Error creating time series splits: {e}")
            return []
            
    def handle_imbalanced_data(self, X: pd.DataFrame, y: pd.Series,
                              method: str = 'smote',
                              sampling_strategy: str = 'auto',
                              random_state: int = 42) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Handle imbalanced datasets using various resampling techniques.
        
        Args:
            X: Feature matrix
            y: Target variable
            method: Resampling method ('smote', 'adasyn', 'borderline_smote', 
                    'random_undersample', 'edited_nn', 'smote_tomek', 'smote_enn')
            sampling_strategy: Sampling strategy for resampling
            random_state: Random state for reproducibility
            
        Returns:
            Tuple of (X_resampled, y_resampled)
        """
        try:
            self.logger.info(f"Handling imbalanced data using {method}")
            
            # Get original class distribution
            original_dist = y.value_counts()
            self.logger.info(f"Original class distribution: {original_dist.to_dict()}")
            
            # Initialize resampler
            resamplers = {
                'smote': SMOTE(sampling_strategy=sampling_strategy, random_state=random_state),
                'adasyn': ADASYN(sampling_strategy=sampling_strategy, random_state=random_state),
                'borderline_smote': BorderlineSMOTE(sampling_strategy=sampling_strategy, random_state=random_state),
                'random_undersample': RandomUnderSampler(sampling_strategy=sampling_strategy, random_state=random_state),
                'edited_nn': EditedNearestNeighbours(sampling_strategy=sampling_strategy),
                'smote_tomek': SMOTETomek(sampling_strategy=sampling_strategy, random_state=random_state),
                'smote_enn': SMOTEENN(sampling_strategy=sampling_strategy, random_state=random_state)
            }
            
            if method not in resamplers:
                self.logger.error(f"Unknown resampling method: {method}")
                return X, y
                
            resampler = resamplers[method]
            X_resampled, y_resampled = resampler.fit_resample(X, y)
            
            # Convert back to DataFrame/Series if needed
            if isinstance(X, pd.DataFrame):
                X_resampled = pd.DataFrame(X_resampled, columns=X.columns, index=range(len(X_resampled)))
            if isinstance(y, pd.Series):
                y_resampled = pd.Series(y_resampled, name=y.name, index=range(len(y_resampled)))
                
            # Log new class distribution
            new_dist = pd.Series(y_resampled).value_counts()
            self.logger.info(f"New class distribution: {new_dist.to_dict()}")
            self.logger.info(f"Resampling completed: {len(X)} -> {len(X_resampled)} samples")
            
            return X_resampled, y_resampled
            
        except Exception as e:
            self.logger.error(f"Error handling imbalanced data: {e}")
            return X, y
            
    def cross_validation_split(self, X: pd.DataFrame, y: pd.Series,
                              cv_folds: int = 5,
                              stratify: bool = True,
                              random_state: int = 42) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Create cross-validation splits.
        
        Args:
            X: Feature matrix
            y: Target variable
            cv_folds: Number of CV folds
            stratify: Whether to use stratified splitting
            random_state: Random state for reproducibility
            
        Returns:
            List of (train_idx, test_idx) tuples
        """
        try:
            if stratify:
                cv = StratifiedShuffleSplit(n_splits=cv_folds, test_size=0.2, random_state=random_state)
            else:
                from sklearn.model_selection import ShuffleSplit
                cv = ShuffleSplit(n_splits=cv_folds, test_size=0.2, random_state=random_state)
                
            splits = []
            for train_idx, test_idx in cv.split(X, y):
                splits.append((train_idx, test_idx))
                
            self.logger.info(f"Cross-validation splits created: {len(splits)} folds")
            return splits
            
        except Exception as e:
            self.logger.error(f"Error creating CV splits: {e}")
            return []
            
    def get_split_summary(self, X_train: pd.DataFrame, X_val: pd.DataFrame, X_test: pd.DataFrame,
                         y_train: pd.Series, y_val: pd.Series, y_test: pd.Series) -> Dict[str, Any]:
        """
        Get a summary of the data splits.
        
        Args:
            X_train, X_val, X_test: Feature matrices
            y_train, y_val, y_test: Target variables
            
        Returns:
            Dictionary containing split summary
        """
        try:
            summary = {
                'splits': {
                    'train': {'samples': len(X_train), 'features': X_train.shape[1]},
                    'validation': {'samples': len(X_val), 'features': X_val.shape[1]},
                    'test': {'samples': len(X_test), 'features': X_test.shape[1]}
                },
                'class_distributions': {
                    'train': y_train.value_counts().to_dict(),
                    'validation': y_val.value_counts().to_dict(),
                    'test': y_test.value_counts().to_dict()
                },
                'class_ratios': {
                    'train': y_train.value_counts(normalize=True).to_dict(),
                    'validation': y_val.value_counts(normalize=True).to_dict(),
                    'test': y_test.value_counts(normalize=True).to_dict()
                }
            }
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error generating split summary: {e}")
            return {}
            
    def save_splits(self, X_train: pd.DataFrame, X_val: pd.DataFrame, X_test: pd.DataFrame,
                   y_train: pd.Series, y_val: pd.Series, y_test: pd.Series,
                   save_path: str = "data/processed/") -> None:
        """
        Save the data splits to files.
        
        Args:
            X_train, X_val, X_test: Feature matrices
            y_train, y_val, y_test: Target variables
            save_path: Path to save the files
        """
        try:
            import os
            os.makedirs(save_path, exist_ok=True)
            
            # Save features
            X_train.to_csv(f"{save_path}/X_train.csv", index=False)
            X_val.to_csv(f"{save_path}/X_val.csv", index=False)
            X_test.to_csv(f"{save_path}/X_test.csv", index=False)
            
            # Save targets
            y_train.to_csv(f"{save_path}/y_train.csv", index=False)
            y_val.to_csv(f"{save_path}/y_val.csv", index=False)
            y_test.to_csv(f"{save_path}/y_test.csv", index=False)
            
            self.logger.info(f"Data splits saved to {save_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving data splits: {e}")
            
    def load_splits(self, load_path: str = "data/processed/") -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame,
                                                                     pd.Series, pd.Series, pd.Series]:
        """
        Load previously saved data splits.
        
        Args:
            load_path: Path to load the files from
            
        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        try:
            # Load features
            X_train = pd.read_csv(f"{load_path}/X_train.csv")
            X_val = pd.read_csv(f"{load_path}/X_val.csv")
            X_test = pd.read_csv(f"{load_path}/X_test.csv")
            
            # Load targets
            y_train = pd.read_csv(f"{load_path}/y_train.csv").squeeze()
            y_val = pd.read_csv(f"{load_path}/y_val.csv").squeeze()
            y_test = pd.read_csv(f"{load_path}/y_test.csv").squeeze()
            
            self.logger.info(f"Data splits loaded from {load_path}")
            return X_train, X_val, X_test, y_train, y_val, y_test
            
        except Exception as e:
            self.logger.error(f"Error loading data splits: {e}")
            raise
