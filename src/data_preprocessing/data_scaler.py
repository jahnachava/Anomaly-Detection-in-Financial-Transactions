"""
Data scaling and normalization utilities for financial transaction data.

This module provides functionality to scale and normalize features
for machine learning models.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.preprocessing import PowerTransformer, QuantileTransformer
from loguru import logger
import warnings
warnings.filterwarnings('ignore')

class DataScaler:
    """
    A class to handle scaling and normalization of financial transaction data.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the DataScaler with configuration.
        
        Args:
            config: Configuration dictionary containing scaling settings
        """
        self.config = config
        self.scaling_config = config.get('features', {}).get('scaling_method', 'standard')
        self.logger = logger
        self.scalers = {}
        self.fitted_scalers = {}
        
    def get_scaler(self, method: str = None):
        """
        Get the appropriate scaler based on the method.
        
        Args:
            method: Scaling method ('standard', 'minmax', 'robust', 'power', 'quantile')
            
        Returns:
            Scaler object
        """
        if method is None:
            method = self.scaling_config
            
        scalers = {
            'standard': StandardScaler(),
            'minmax': MinMaxScaler(),
            'robust': RobustScaler(),
            'power': PowerTransformer(method='yeo-johnson'),
            'quantile': QuantileTransformer(output_distribution='normal')
        }
        
        if method not in scalers:
            self.logger.warning(f"Unknown scaling method: {method}. Using standard scaling.")
            method = 'standard'
            
        return scalers[method]
        
    def scale_features(self, df: pd.DataFrame, 
                      feature_columns: List[str] = None,
                      method: str = None,
                      fit: bool = True) -> pd.DataFrame:
        """
        Scale specified features in the DataFrame.
        
        Args:
            df: DataFrame containing features to scale
            feature_columns: List of column names to scale
            method: Scaling method to use
            fit: Whether to fit the scaler or use existing fitted scaler
            
        Returns:
            DataFrame with scaled features
        """
        try:
            df_scaled = df.copy()
            
            if feature_columns is None:
                # Scale all numeric columns
                feature_columns = df.select_dtypes(include=[np.number]).columns.tolist()
                
            if not feature_columns:
                self.logger.warning("No numeric features found to scale")
                return df_scaled
                
            scaler = self.get_scaler(method)
            scaler_key = f"{method}_{'_'.join(feature_columns)}"
            
            if fit:
                # Fit and transform
                scaled_data = scaler.fit_transform(df[feature_columns])
                self.fitted_scalers[scaler_key] = scaler
                self.logger.info(f"Scaler fitted and applied to {len(feature_columns)} features using {method} method")
            else:
                # Use existing fitted scaler
                if scaler_key in self.fitted_scalers:
                    scaler = self.fitted_scalers[scaler_key]
                    scaled_data = scaler.transform(df[feature_columns])
                    self.logger.info(f"Applied existing scaler to {len(feature_columns)} features")
                else:
                    self.logger.error(f"No fitted scaler found for key: {scaler_key}")
                    return df_scaled
                    
            # Replace original columns with scaled data
            df_scaled[feature_columns] = scaled_data
            
            return df_scaled
            
        except Exception as e:
            self.logger.error(f"Error scaling features: {e}")
            return df
            
    def scale_numerical_features(self, df: pd.DataFrame, 
                                exclude_columns: List[str] = None,
                                method: str = None) -> pd.DataFrame:
        """
        Scale all numerical features in the DataFrame.
        
        Args:
            df: DataFrame containing features
            exclude_columns: List of columns to exclude from scaling
            method: Scaling method to use
            
        Returns:
            DataFrame with scaled numerical features
        """
        try:
            if exclude_columns is None:
                exclude_columns = []
                
            # Get numerical columns excluding specified columns
            numerical_columns = df.select_dtypes(include=[np.number]).columns.tolist()
            feature_columns = [col for col in numerical_columns if col not in exclude_columns]
            
            if not feature_columns:
                self.logger.warning("No numerical features found to scale")
                return df
                
            return self.scale_features(df, feature_columns, method, fit=True)
            
        except Exception as e:
            self.logger.error(f"Error scaling numerical features: {e}")
            return df
            
    def handle_outliers(self, df: pd.DataFrame, 
                       feature_columns: List[str] = None,
                       method: str = 'iqr',
                       factor: float = 1.5) -> pd.DataFrame:
        """
        Handle outliers in the dataset using various methods.
        
        Args:
            df: DataFrame containing features
            feature_columns: List of columns to process
            method: Outlier handling method ('iqr', 'zscore', 'isolation')
            factor: Factor for outlier detection (for IQR method)
            
        Returns:
            DataFrame with outliers handled
        """
        try:
            df_processed = df.copy()
            
            if feature_columns is None:
                feature_columns = df.select_dtypes(include=[np.number]).columns.tolist()
                
            if method == 'iqr':
                for col in feature_columns:
                    if col in df.columns:
                        Q1 = df[col].quantile(0.25)
                        Q3 = df[col].quantile(0.75)
                        IQR = Q3 - Q1
                        lower_bound = Q1 - factor * IQR
                        upper_bound = Q3 + factor * IQR
                        
                        # Cap outliers
                        df_processed[col] = df_processed[col].clip(lower=lower_bound, upper=upper_bound)
                        
            elif method == 'zscore':
                for col in feature_columns:
                    if col in df.columns:
                        z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                        # Cap values with z-score > 3
                        df_processed[col] = np.where(z_scores > 3, 
                                                   df[col].median(), 
                                                   df[col])
                                                   
            self.logger.info(f"Outliers handled using {method} method for {len(feature_columns)} features")
            return df_processed
            
        except Exception as e:
            self.logger.error(f"Error handling outliers: {e}")
            return df
            
    def normalize_distribution(self, df: pd.DataFrame, 
                             feature_columns: List[str] = None,
                             method: str = 'yeo-johnson') -> pd.DataFrame:
        """
        Normalize the distribution of features to be more Gaussian-like.
        
        Args:
            df: DataFrame containing features
            feature_columns: List of columns to normalize
            method: Normalization method ('yeo-johnson', 'box-cox', 'quantile')
            
        Returns:
            DataFrame with normalized distributions
        """
        try:
            df_normalized = df.copy()
            
            if feature_columns is None:
                feature_columns = df.select_dtypes(include=[np.number]).columns.tolist()
                
            if method in ['yeo-johnson', 'box-cox']:
                transformer = PowerTransformer(method=method)
                normalized_data = transformer.fit_transform(df[feature_columns])
                df_normalized[feature_columns] = normalized_data
                
            elif method == 'quantile':
                transformer = QuantileTransformer(output_distribution='normal')
                normalized_data = transformer.fit_transform(df[feature_columns])
                df_normalized[feature_columns] = normalized_data
                
            self.logger.info(f"Distribution normalized using {method} method for {len(feature_columns)} features")
            return df_normalized
            
        except Exception as e:
            self.logger.error(f"Error normalizing distribution: {e}")
            return df
            
    def inverse_transform(self, df: pd.DataFrame, 
                         feature_columns: List[str],
                         scaler_key: str) -> pd.DataFrame:
        """
        Inverse transform scaled features back to original scale.
        
        Args:
            df: DataFrame with scaled features
            feature_columns: List of columns to inverse transform
            scaler_key: Key of the fitted scaler to use
            
        Returns:
            DataFrame with inverse transformed features
        """
        try:
            df_inverse = df.copy()
            
            if scaler_key in self.fitted_scalers:
                scaler = self.fitted_scalers[scaler_key]
                inverse_data = scaler.inverse_transform(df[feature_columns])
                df_inverse[feature_columns] = inverse_data
                self.logger.info(f"Inverse transform applied to {len(feature_columns)} features")
            else:
                self.logger.error(f"No fitted scaler found for key: {scaler_key}")
                
            return df_inverse
            
        except Exception as e:
            self.logger.error(f"Error in inverse transform: {e}")
            return df
            
    def get_scaling_summary(self, df_original: pd.DataFrame, 
                           df_scaled: pd.DataFrame,
                           feature_columns: List[str]) -> Dict[str, Any]:
        """
        Get a summary of the scaling transformation.
        
        Args:
            df_original: Original DataFrame
            df_scaled: Scaled DataFrame
            feature_columns: List of columns that were scaled
            
        Returns:
            Dictionary containing scaling summary statistics
        """
        try:
            summary = {}
            
            for col in feature_columns:
                if col in df_original.columns and col in df_scaled.columns:
                    summary[col] = {
                        'original_mean': df_original[col].mean(),
                        'original_std': df_original[col].std(),
                        'scaled_mean': df_scaled[col].mean(),
                        'scaled_std': df_scaled[col].std(),
                        'original_min': df_original[col].min(),
                        'original_max': df_original[col].max(),
                        'scaled_min': df_scaled[col].min(),
                        'scaled_max': df_scaled[col].max()
                    }
                    
            return summary
            
        except Exception as e:
            self.logger.error(f"Error generating scaling summary: {e}")
            return {}
