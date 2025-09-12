"""
XGBoost implementation for supervised anomaly detection.

XGBoost (Extreme Gradient Boosting) is an optimized gradient boosting framework
that is highly effective for classification tasks, especially with imbalanced datasets.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
import xgboost as xgb
from sklearn.model_selection import cross_val_score, GridSearchCV
from loguru import logger
from .base_supervised import BaseSupervisedDetector

class XGBoostDetector(BaseSupervisedDetector):
    """
    XGBoost anomaly detector for financial transactions.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the XGBoost detector.
        
        Args:
            config: Configuration dictionary containing model parameters
        """
        super().__init__(config, "XGBoost")
        
        # Get model parameters from config
        model_config = config.get('models', {}).get('supervised', {}).get('xgboost', {})
        
        self.model_params = {
            'n_estimators': model_config.get('n_estimators', 100),
            'max_depth': model_config.get('max_depth', 6),
            'learning_rate': model_config.get('learning_rate', 0.1),
            'subsample': model_config.get('subsample', 0.8),
            'colsample_bytree': model_config.get('colsample_bytree', 0.8),
            'colsample_bylevel': model_config.get('colsample_bylevel', 1.0),
            'colsample_bynode': model_config.get('colsample_bynode', 1.0),
            'reg_alpha': model_config.get('reg_alpha', 0),
            'reg_lambda': model_config.get('reg_lambda', 1),
            'gamma': model_config.get('gamma', 0),
            'min_child_weight': model_config.get('min_child_weight', 1),
            'random_state': model_config.get('random_state', 42),
            'n_jobs': model_config.get('n_jobs', -1),
            'scale_pos_weight': model_config.get('scale_pos_weight', 1),
            'eval_metric': model_config.get('eval_metric', 'logloss')
        }
        
        self.model = xgb.XGBClassifier(**self.model_params)
        self.feature_names_ = None
        self.logger.info(f"Initialized XGBoost with parameters: {self.model_params}")
        
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'XGBoostDetector':
        """
        Fit the XGBoost model.
        
        Args:
            X: Feature matrix
            y: Target variable (0 for normal, 1 for anomaly)
            
        Returns:
            Self
        """
        try:
            self.validate_input(X, y)
            self.feature_names_ = X.columns.tolist()
            
            self.logger.info(f"Fitting XGBoost on {X.shape[0]} samples with {X.shape[1]} features")
            
            # Log class distribution
            class_counts = y.value_counts()
            self.logger.info(f"Class distribution: {class_counts.to_dict()}")
            
            # Calculate scale_pos_weight for imbalanced data
            if self.model_params['scale_pos_weight'] == 1:
                pos_count = class_counts.get(1, 0)
                neg_count = class_counts.get(0, 0)
                if pos_count > 0 and neg_count > 0:
                    scale_pos_weight = neg_count / pos_count
                    self.model.set_params(scale_pos_weight=scale_pos_weight)
                    self.logger.info(f"Auto-calculated scale_pos_weight: {scale_pos_weight:.2f}")
            
            # Fit the model
            self.model.fit(X, y)
            self.is_fitted = True
            
            # Store feature importance
            self.feature_importance = self.model.feature_importances_
            
            # Log some statistics
            self.logger.info("XGBoost fitted successfully")
            self.logger.info(f"Number of estimators: {self.model_params['n_estimators']}")
            self.logger.info(f"Max depth: {self.model_params['max_depth']}")
            self.logger.info(f"Learning rate: {self.model_params['learning_rate']}")
            self.logger.info(f"Feature importance available: {self.feature_importance is not None}")
            
            return self
            
        except Exception as e:
            self.logger.error(f"Error fitting XGBoost: {e}")
            raise
            
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict anomalies using XGBoost.
        
        Args:
            X: Feature matrix
            
        Returns:
            Array of predictions (1 for anomaly, 0 for normal)
        """
        try:
            if not self.is_fitted:
                raise ValueError("Model must be fitted before making predictions")
                
            self.validate_input(X)
            
            predictions = self.model.predict(X)
            
            self.logger.info(f"Predicted {np.sum(predictions)} anomalies out of {len(predictions)} samples")
            
            return predictions
            
        except Exception as e:
            self.logger.error(f"Error making predictions with XGBoost: {e}")
            raise
            
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict probability of being an anomaly.
        
        Args:
            X: Feature matrix
            
        Returns:
            Array of probabilities [prob_normal, prob_anomaly]
        """
        try:
            if not self.is_fitted:
                raise ValueError("Model must be fitted before making predictions")
                
            self.validate_input(X)
            
            probabilities = self.model.predict_proba(X)
            
            return probabilities
            
        except Exception as e:
            self.logger.error(f"Error computing prediction probabilities: {e}")
            raise
            
    def get_feature_importance(self, importance_type: str = 'weight') -> Optional[np.ndarray]:
        """
        Get feature importance from the XGBoost model.
        
        Args:
            importance_type: Type of importance ('weight', 'gain', 'cover')
            
        Returns:
            Array of feature importance scores
        """
        try:
            if not self.is_fitted:
                raise ValueError("Model must be fitted before getting feature importance")
                
            return self.model.get_booster().get_score(importance_type=importance_type)
            
        except Exception as e:
            self.logger.error(f"Error getting feature importance: {e}")
            return None
            
    def get_top_features(self, n_features: int = 10, importance_type: str = 'weight') -> list:
        """
        Get top N most important features.
        
        Args:
            n_features: Number of top features to return
            importance_type: Type of importance ('weight', 'gain', 'cover')
            
        Returns:
            List of tuples (feature_name, importance_score)
        """
        try:
            if not self.is_fitted:
                raise ValueError("Model must be fitted before getting top features")
                
            importance_dict = self.get_feature_importance(importance_type)
            if not importance_dict:
                return []
                
            # Convert to list and sort
            feature_importance_pairs = list(importance_dict.items())
            feature_importance_pairs.sort(key=lambda x: x[1], reverse=True)
            
            top_features = feature_importance_pairs[:n_features]
            
            self.logger.info(f"Top {n_features} features ({importance_type}):")
            for i, (feature, importance) in enumerate(top_features, 1):
                self.logger.info(f"  {i}. {feature}: {importance:.4f}")
                
            return top_features
            
        except Exception as e:
            self.logger.error(f"Error getting top features: {e}")
            return []
            
    def cross_validate(self, X: pd.DataFrame, y: pd.Series, 
                      cv_folds: int = 5, scoring: str = 'roc_auc') -> Dict[str, Any]:
        """
        Perform cross-validation on the model.
        
        Args:
            X: Feature matrix
            y: Target variable
            cv_folds: Number of CV folds
            scoring: Scoring metric
            
        Returns:
            Dictionary containing CV results
        """
        try:
            if not self.is_fitted:
                raise ValueError("Model must be fitted before cross-validation")
                
            self.logger.info(f"Performing {cv_folds}-fold cross-validation with {scoring} scoring")
            
            # Perform cross-validation
            cv_scores = cross_val_score(self.model, X, y, cv=cv_folds, scoring=scoring)
            
            cv_results = {
                'cv_scores': cv_scores.tolist(),
                'mean_score': cv_scores.mean(),
                'std_score': cv_scores.std(),
                'cv_folds': cv_folds,
                'scoring': scoring
            }
            
            self.logger.info(f"Cross-validation results:")
            self.logger.info(f"  Mean {scoring}: {cv_results['mean_score']:.4f} (+/- {cv_results['std_score']:.4f})")
            
            return cv_results
            
        except Exception as e:
            self.logger.error(f"Error in cross-validation: {e}")
            raise
            
    def hyperparameter_tuning(self, X: pd.DataFrame, y: pd.Series,
                             param_grid: Optional[Dict[str, list]] = None,
                             cv_folds: int = 3, scoring: str = 'roc_auc') -> Dict[str, Any]:
        """
        Perform hyperparameter tuning using GridSearchCV.
        
        Args:
            X: Feature matrix
            y: Target variable
            param_grid: Parameter grid for tuning
            cv_folds: Number of CV folds
            scoring: Scoring metric
            
        Returns:
            Dictionary containing tuning results
        """
        try:
            self.validate_input(X, y)
            
            if param_grid is None:
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [3, 6, 9],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'subsample': [0.8, 0.9, 1.0],
                    'colsample_bytree': [0.8, 0.9, 1.0]
                }
                
            self.logger.info("Starting hyperparameter tuning with GridSearchCV")
            self.logger.info(f"Parameter grid: {param_grid}")
            
            # Create a new model for tuning
            tuning_model = xgb.XGBClassifier(
                random_state=self.model_params['random_state'],
                n_jobs=self.model_params['n_jobs'],
                eval_metric=self.model_params['eval_metric']
            )
            
            # Perform grid search
            grid_search = GridSearchCV(
                tuning_model,
                param_grid,
                cv=cv_folds,
                scoring=scoring,
                n_jobs=-1,
                verbose=1
            )
            
            grid_search.fit(X, y)
            
            # Update model with best parameters
            self.model_params.update(grid_search.best_params_)
            self.model = grid_search.best_estimator_
            self.is_fitted = True
            self.feature_importance = self.model.feature_importances_
            
            tuning_results = {
                'best_params': grid_search.best_params_,
                'best_score': grid_search.best_score_,
                'cv_results': grid_search.cv_results_,
                'param_grid': param_grid
            }
            
            self.logger.info("Hyperparameter tuning completed")
            self.logger.info(f"Best parameters: {tuning_results['best_params']}")
            self.logger.info(f"Best {scoring} score: {tuning_results['best_score']:.4f}")
            
            return tuning_results
            
        except Exception as e:
            self.logger.error(f"Error in hyperparameter tuning: {e}")
            raise
            
    def plot_feature_importance(self, n_features: int = 20, importance_type: str = 'weight') -> None:
        """
        Plot feature importance.
        
        Args:
            n_features: Number of top features to plot
            importance_type: Type of importance ('weight', 'gain', 'cover')
        """
        try:
            import matplotlib.pyplot as plt
            
            if not self.is_fitted:
                raise ValueError("Model must be fitted before plotting feature importance")
                
            top_features = self.get_top_features(n_features, importance_type)
            if not top_features:
                self.logger.warning("No feature importance data available for plotting")
                return
                
            features, importances = zip(*top_features)
            
            plt.figure(figsize=(10, 8))
            plt.barh(range(len(features)), importances)
            plt.yticks(range(len(features)), features)
            plt.xlabel(f'Feature Importance ({importance_type})')
            plt.title(f'Top {n_features} Feature Importance - XGBoost')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.show()
            
        except ImportError:
            self.logger.warning("Matplotlib not available for plotting")
        except Exception as e:
            self.logger.error(f"Error plotting feature importance: {e}")
            
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get detailed information about the XGBoost model.
        
        Returns:
            Dictionary containing model information
        """
        info = super().get_model_info()
        
        if self.is_fitted:
            info.update({
                'n_estimators': self.model_params['n_estimators'],
                'max_depth': self.model_params['max_depth'],
                'learning_rate': self.model_params['learning_rate'],
                'subsample': self.model_params['subsample'],
                'colsample_bytree': self.model_params['colsample_bytree'],
                'reg_alpha': self.model_params['reg_alpha'],
                'reg_lambda': self.model_params['reg_lambda'],
                'gamma': self.model_params['gamma'],
                'min_child_weight': self.model_params['min_child_weight'],
                'scale_pos_weight': self.model_params['scale_pos_weight'],
                'eval_metric': self.model_params['eval_metric'],
                'n_features': len(self.feature_names_) if self.feature_names_ else 0
            })
            
        return info
