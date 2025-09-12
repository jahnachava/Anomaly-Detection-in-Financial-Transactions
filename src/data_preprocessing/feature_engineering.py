"""
Feature engineering module for financial transaction data.

This module provides functionality to create, transform, and select features
for anomaly detection in financial transactions.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from loguru import logger
import warnings
warnings.filterwarnings('ignore')

class FeatureEngineer:
    """
    A class to handle feature engineering for financial transaction data.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the FeatureEngineer with configuration.
        
        Args:
            config: Configuration dictionary containing feature engineering settings
        """
        self.config = config
        self.feature_config = config.get('features', {})
        self.logger = logger
        self.encoders = {}
        self.feature_selector = None
        
    def create_time_features(self, df: pd.DataFrame, time_column: str = 'Time') -> pd.DataFrame:
        """
        Create time-based features from timestamp data.
        
        Args:
            df: DataFrame containing time data
            time_column: Name of the time column
            
        Returns:
            DataFrame with additional time features
        """
        try:
            df = df.copy()
            
            if time_column in df.columns:
                # Convert time to datetime if it's in seconds
                if df[time_column].dtype in ['int64', 'float64']:
                    # Assuming time is in seconds since epoch or relative time
                    df['hour'] = (df[time_column] % (24 * 3600)) // 3600
                    df['day_of_week'] = (df[time_column] // (24 * 3600)) % 7
                    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
                    df['is_night'] = df['hour'].isin([22, 23, 0, 1, 2, 3, 4, 5]).astype(int)
                    df['is_business_hours'] = df['hour'].between(9, 17).astype(int)
                else:
                    # If it's already datetime
                    df['hour'] = pd.to_datetime(df[time_column]).dt.hour
                    df['day_of_week'] = pd.to_datetime(df[time_column]).dt.dayofweek
                    df['is_weekend'] = pd.to_datetime(df[time_column]).dt.dayofweek.isin([5, 6]).astype(int)
                    df['is_night'] = df['hour'].isin([22, 23, 0, 1, 2, 3, 4, 5]).astype(int)
                    df['is_business_hours'] = df['hour'].between(9, 17).astype(int)
                    
                self.logger.info("Time-based features created successfully")
            else:
                self.logger.warning(f"Time column '{time_column}' not found")
                
            return df
            
        except Exception as e:
            self.logger.error(f"Error creating time features: {e}")
            return df
            
    def create_amount_features(self, df: pd.DataFrame, amount_column: str = 'Amount') -> pd.DataFrame:
        """
        Create amount-based features for transaction analysis.
        
        Args:
            df: DataFrame containing amount data
            amount_column: Name of the amount column
            
        Returns:
            DataFrame with additional amount features
        """
        try:
            df = df.copy()
            
            if amount_column in df.columns:
                # Log transformation to handle skewed amounts
                df['amount_log'] = np.log1p(df[amount_column])
                
                # Amount categories
                df['amount_category'] = pd.cut(
                    df[amount_column], 
                    bins=[0, 10, 100, 1000, float('inf')], 
                    labels=['small', 'medium', 'large', 'very_large']
                )
                
                # High-value transaction flag
                df['is_high_value'] = (df[amount_column] > df[amount_column].quantile(0.95)).astype(int)
                
                # Amount relative to user's typical spending (if user_id available)
                # This would require grouping by user_id in real scenarios
                
                self.logger.info("Amount-based features created successfully")
            else:
                self.logger.warning(f"Amount column '{amount_column}' not found")
                
            return df
            
        except Exception as e:
            self.logger.error(f"Error creating amount features: {e}")
            return df
            
    def create_frequency_features(self, df: pd.DataFrame, 
                                group_columns: List[str] = None) -> pd.DataFrame:
        """
        Create frequency-based features for transaction patterns.
        
        Args:
            df: DataFrame containing transaction data
            group_columns: Columns to group by for frequency calculation
            
        Returns:
            DataFrame with additional frequency features
        """
        try:
            df = df.copy()
            
            # If no group columns specified, use time-based grouping
            if group_columns is None:
                group_columns = ['hour', 'day_of_week'] if 'hour' in df.columns else []
                
            if group_columns and all(col in df.columns for col in group_columns):
                # Transaction frequency by time periods
                for col in group_columns:
                    freq_col = f'{col}_frequency'
                    df[freq_col] = df.groupby(col)[col].transform('count')
                    
                self.logger.info("Frequency-based features created successfully")
            else:
                self.logger.warning("Group columns not found for frequency features")
                
            return df
            
        except Exception as e:
            self.logger.error(f"Error creating frequency features: {e}")
            return df
            
    def create_interaction_features(self, df: pd.DataFrame, 
                                  feature_pairs: List[Tuple[str, str]] = None) -> pd.DataFrame:
        """
        Create interaction features between pairs of variables.
        
        Args:
            df: DataFrame containing features
            feature_pairs: List of tuples containing feature pairs to interact
            
        Returns:
            DataFrame with additional interaction features
        """
        try:
            df = df.copy()
            
            if feature_pairs is None:
                # Default interaction pairs for financial data
                feature_pairs = [
                    ('Amount', 'hour'),
                    ('Amount', 'day_of_week'),
                    ('V1', 'V2'),
                    ('V3', 'V4')
                ]
                
            for feat1, feat2 in feature_pairs:
                if feat1 in df.columns and feat2 in df.columns:
                    # Multiplication interaction
                    interaction_name = f'{feat1}_x_{feat2}'
                    df[interaction_name] = df[feat1] * df[feat2]
                    
                    # Division interaction (avoid division by zero)
                    if df[feat2].abs().min() > 0:
                        division_name = f'{feat1}_div_{feat2}'
                        df[division_name] = df[feat1] / (df[feat2] + 1e-8)
                        
            self.logger.info("Interaction features created successfully")
            return df
            
        except Exception as e:
            self.logger.error(f"Error creating interaction features: {e}")
            return df
            
    def encode_categorical_features(self, df: pd.DataFrame, 
                                  categorical_columns: List[str] = None,
                                  encoding_method: str = 'label') -> pd.DataFrame:
        """
        Encode categorical features using various encoding methods.
        
        Args:
            df: DataFrame containing categorical features
            categorical_columns: List of categorical column names
            encoding_method: Method to use ('label', 'onehot', 'target')
            
        Returns:
            DataFrame with encoded categorical features
        """
        try:
            df = df.copy()
            
            if categorical_columns is None:
                categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
                
            for col in categorical_columns:
                if col in df.columns:
                    if encoding_method == 'label':
                        if col not in self.encoders:
                            self.encoders[col] = LabelEncoder()
                        df[col] = self.encoders[col].fit_transform(df[col].astype(str))
                        
                    elif encoding_method == 'onehot':
                        # Create one-hot encoded columns
                        dummies = pd.get_dummies(df[col], prefix=col)
                        df = pd.concat([df, dummies], axis=1)
                        df = df.drop(col, axis=1)
                        
            self.logger.info(f"Categorical features encoded using {encoding_method} encoding")
            return df
            
        except Exception as e:
            self.logger.error(f"Error encoding categorical features: {e}")
            return df
            
    def select_features(self, X: pd.DataFrame, y: pd.Series, 
                       method: str = 'mutual_info', k: int = 20) -> Tuple[pd.DataFrame, Any]:
        """
        Select the best features using various selection methods.
        
        Args:
            X: Feature matrix
            y: Target variable
            method: Feature selection method ('mutual_info', 'f_score', 'chi2')
            k: Number of features to select
            
        Returns:
            Tuple of (selected_features, feature_selector)
        """
        try:
            if method == 'mutual_info':
                selector = SelectKBest(score_func=mutual_info_classif, k=k)
            elif method == 'f_score':
                selector = SelectKBest(score_func=f_classif, k=k)
            else:
                raise ValueError(f"Unknown feature selection method: {method}")
                
            X_selected = selector.fit_transform(X, y)
            selected_features = X.columns[selector.get_support()].tolist()
            
            self.feature_selector = selector
            self.logger.info(f"Selected {len(selected_features)} features using {method}")
            self.logger.info(f"Selected features: {selected_features}")
            
            return pd.DataFrame(X_selected, columns=selected_features, index=X.index), selector
            
        except Exception as e:
            self.logger.error(f"Error in feature selection: {e}")
            return X, None
            
    def engineer_all_features(self, df: pd.DataFrame, target_column: str = 'Class') -> pd.DataFrame:
        """
        Apply all feature engineering steps to the dataset.
        
        Args:
            df: Input DataFrame
            target_column: Name of the target column
            
        Returns:
            DataFrame with all engineered features
        """
        try:
            self.logger.info("Starting comprehensive feature engineering")
            
            # Create time features
            df = self.create_time_features(df)
            
            # Create amount features
            df = self.create_amount_features(df)
            
            # Create frequency features
            df = self.create_frequency_features(df)
            
            # Create interaction features
            df = self.create_interaction_features(df)
            
            # Encode categorical features
            df = self.encode_categorical_features(df)
            
            # Handle any remaining missing values
            df = df.fillna(df.median())
            
            self.logger.info(f"Feature engineering completed. Final shape: {df.shape}")
            return df
            
        except Exception as e:
            self.logger.error(f"Error in comprehensive feature engineering: {e}")
            return df
