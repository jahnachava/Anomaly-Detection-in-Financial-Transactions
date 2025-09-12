"""
Random Forest implementation for supervised anomaly detection.

Random Forest is an ensemble learning method that constructs multiple decision trees
and outputs the class that is the mode of the classes of the individual trees.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
from loguru import logger
from .base_supervised import BaseSupervisedDetector

class RandomForestDetector(BaseSupervisedDetector):
    """
    Random Forest anomaly detector for financial transactions.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the Random Forest detector.
        
        Args:
            config: Configuration dictionary containing model parameters
        """
        super().__init__(config, "RandomForest")
        
        # Get model parameters from config
        model_config = config.get('models', {}).get('supervised', {}).get('random_forest', {})
        
        self.model_params = {
            'n_estimators': model_config.get('n_estimators', 100),
            'max_depth': model_config.get('max_depth', 10),
            'min_samples_split': model_config.get('min_samples_split', 5),
            'min_samples_leaf': model_config.get('min_samples_leaf', 2),
            'max_features': model_config.get('max_features', 'sqrt'),
            'bootstrap': model_config.get('bootstrap', True),
            'random_state': model_config.get('random_state', 42),
            'n_jobs': model_config.get('n_jobs', -1),
            'class_weight': model_config.get('class_weight', 'balanced')
        }
        
        self.model = RandomForestClassifier(**self.model_params)
        self.feature_names_ = None
        self.logger.info(f"Initialized Random Forest with parameters: {self.model_params}")
        
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'RandomForestDetector':
        """
        Fit the Random Forest model.
        
        Args:
            X: Feature matrix
            y: Target variable (0 for normal, 1 for anomaly)
            
        Returns:
            Self
        """
        try:
            self.validate_input(X, y)
            self.feature_names_ = X.columns.tolist()
            
            self.logger.info(f"Fitting Random Forest on {X.shape[0]} samples with {X.shape[1]} features")
            
            # Log class distribution
            class_counts = y.value_counts()
            self.logger.info(f"Class distribution: {class_counts.to_dict()}")
            
            # Fit the model
            self.model.fit(X, y)
            self.is_fitted = True
            
            # Store feature importance
            self.feature_importance = self.model.feature_importances_
            
            # Log some statistics
            self.logger.info("Random Forest fitted successfully")
            self.logger.info(f"Number of estimators: {self.model_params['n_estimators']}")
            self.logger.info(f"Max depth: {self.model_params['max_depth']}")
            self.logger.info(f"Feature importance available: {self.feature_importance is not None}")
            
            return self
            
        except Exception as e:
            self.logger.error(f"Error fitting Random Forest: {e}")
            raise
            
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict anomalies using Random Forest.
        
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
            self.logger.error(f"Error making predictions with Random Forest: {e}")
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
            
    def get_feature_importance(self) -> Optional[np.ndarray]:
        """
        Get feature importance from the Random Forest model.
        
        Returns:
            Array of feature importance scores
        """
        try:
            if not self.is_fitted:
                raise ValueError("Model must be fitted before getting feature importance")
                
            return self.model.feature_importances_
            
        except Exception as e:
            self.logger.error(f"Error getting feature importance: {e}")
            return None
            
    def get_top_features(self, n_features: int = 10) -> list:
        """
        Get top N most important features.
        
        Args:
            n_features: Number of top features to return
            
        Returns:
            List of tuples (feature_name, importance_score)
        """
        try:
            if not self.is_fitted:
                raise ValueError("Model must be fitted before getting top features")
                
            importance = self.get_feature_importance()
            if importance is None:
                return []
                
            # Create feature importance pairs
            feature_importance_pairs = list(zip(self.feature_names_, importance))
            feature_importance_pairs.sort(key=lambda x: x[1], reverse=True)
            
            top_features = feature_importance_pairs[:n_features]
            
            self.logger.info(f"Top {n_features} features:")
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
                    'max_depth': [5, 10, 15, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }
                
            self.logger.info("Starting hyperparameter tuning with GridSearchCV")
            self.logger.info(f"Parameter grid: {param_grid}")
            
            # Create a new model for tuning
            tuning_model = RandomForestClassifier(
                random_state=self.model_params['random_state'],
                n_jobs=self.model_params['n_jobs'],
                class_weight=self.model_params['class_weight']
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
            
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get detailed information about the Random Forest model.
        
        Returns:
            Dictionary containing model information
        """
        info = super().get_model_info()
        
        if self.is_fitted:
            info.update({
                'n_estimators': self.model_params['n_estimators'],
                'max_depth': self.model_params['max_depth'],
                'min_samples_split': self.model_params['min_samples_split'],
                'min_samples_leaf': self.model_params['min_samples_leaf'],
                'max_features': self.model_params['max_features'],
                'bootstrap': self.model_params['bootstrap'],
                'class_weight': self.model_params['class_weight'],
                'n_features': len(self.feature_names_) if self.feature_names_ else 0
            })
            
            # Add feature importance statistics
            if self.feature_importance is not None:
                info['feature_importance_stats'] = {
                    'mean': float(np.mean(self.feature_importance)),
                    'std': float(np.std(self.feature_importance)),
                    'min': float(np.min(self.feature_importance)),
                    'max': float(np.max(self.feature_importance))
                }
                
        return info
