"""
Neural Network implementation for supervised anomaly detection.

This module implements a deep neural network for anomaly detection using TensorFlow/Keras.
The network is designed to handle imbalanced datasets and provide probability outputs.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks
from sklearn.model_selection import cross_val_score
from loguru import logger
from .base_supervised import BaseSupervisedDetector

class NeuralNetworkDetector(BaseSupervisedDetector):
    """
    Neural Network anomaly detector for financial transactions.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the Neural Network detector.
        
        Args:
            config: Configuration dictionary containing model parameters
        """
        super().__init__(config, "NeuralNetwork")
        
        # Get model parameters from config
        model_config = config.get('models', {}).get('supervised', {}).get('neural_network', {})
        
        self.model_params = {
            'hidden_layers': model_config.get('hidden_layers', [64, 32, 16]),
            'activation': model_config.get('activation', 'relu'),
            'optimizer': model_config.get('optimizer', 'adam'),
            'learning_rate': model_config.get('learning_rate', 0.001),
            'epochs': model_config.get('epochs', 100),
            'batch_size': model_config.get('batch_size', 32),
            'validation_split': model_config.get('validation_split', 0.2),
            'dropout_rate': model_config.get('dropout_rate', 0.2),
            'l2_reg': model_config.get('l2_reg', 0.001),
            'class_weight': model_config.get('class_weight', None)
        }
        
        self.model = None
        self.history = None
        self.feature_names_ = None
        self.logger.info(f"Initialized Neural Network with parameters: {self.model_params}")
        
    def _build_model(self, input_dim: int) -> keras.Model:
        """
        Build the neural network architecture.
        
        Args:
            input_dim: Input dimension (number of features)
            
        Returns:
            Compiled Keras model
        """
        try:
            # Input layer
            input_layer = layers.Input(shape=(input_dim,), name='input')
            
            # Hidden layers
            x = input_layer
            for i, hidden_dim in enumerate(self.model_params['hidden_layers']):
                x = layers.Dense(
                    hidden_dim,
                    activation=self.model_params['activation'],
                    kernel_regularizer=keras.regularizers.l2(self.model_params['l2_reg']),
                    name=f'hidden_{i+1}'
                )(x)
                x = layers.Dropout(
                    self.model_params['dropout_rate'],
                    name=f'dropout_{i+1}'
                )(x)
                x = layers.BatchNormalization(name=f'batch_norm_{i+1}')(x)
                
            # Output layer
            output = layers.Dense(
                1,
                activation='sigmoid',
                name='output'
            )(x)
            
            # Create model
            model = keras.Model(input_layer, output, name='neural_network')
            
            # Compile model
            if self.model_params['optimizer'] == 'adam':
                optimizer = keras.optimizers.Adam(learning_rate=self.model_params['learning_rate'])
            elif self.model_params['optimizer'] == 'sgd':
                optimizer = keras.optimizers.SGD(learning_rate=self.model_params['learning_rate'])
            else:
                optimizer = self.model_params['optimizer']
                
            model.compile(
                optimizer=optimizer,
                loss='binary_crossentropy',
                metrics=['accuracy', 'precision', 'recall']
            )
            
            self.logger.info(f"Neural Network architecture built:")
            self.logger.info(f"  Input dimension: {input_dim}")
            self.logger.info(f"  Hidden layers: {self.model_params['hidden_layers']}")
            self.logger.info(f"  Activation: {self.model_params['activation']}")
            self.logger.info(f"  Optimizer: {self.model_params['optimizer']}")
            self.logger.info(f"  Learning rate: {self.model_params['learning_rate']}")
            
            return model
            
        except Exception as e:
            self.logger.error(f"Error building neural network architecture: {e}")
            raise
            
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'NeuralNetworkDetector':
        """
        Fit the Neural Network model.
        
        Args:
            X: Feature matrix
            y: Target variable (0 for normal, 1 for anomaly)
            
        Returns:
            Self
        """
        try:
            self.validate_input(X, y)
            self.feature_names_ = X.columns.tolist()
            
            self.logger.info(f"Fitting Neural Network on {X.shape[0]} samples with {X.shape[1]} features")
            
            # Convert to numpy arrays
            X_array = X.values.astype(np.float32)
            y_array = y.values.astype(np.float32)
            
            # Log class distribution
            class_counts = y.value_counts()
            self.logger.info(f"Class distribution: {class_counts.to_dict()}")
            
            # Build the model
            self.model = self._build_model(X.shape[1])
            
            # Calculate class weights if not provided
            class_weight = self.model_params['class_weight']
            if class_weight is None:
                pos_count = class_counts.get(1, 0)
                neg_count = class_counts.get(0, 0)
                if pos_count > 0 and neg_count > 0:
                    class_weight = {0: 1.0, 1: neg_count / pos_count}
                    self.logger.info(f"Auto-calculated class weights: {class_weight}")
                    
            # Define callbacks
            callbacks_list = [
                callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=10,
                    restore_best_weights=True,
                    verbose=1
                ),
                callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=5,
                    min_lr=1e-7,
                    verbose=1
                )
            ]
            
            # Train the model
            self.history = self.model.fit(
                X_array, y_array,
                epochs=self.model_params['epochs'],
                batch_size=self.model_params['batch_size'],
                validation_split=self.model_params['validation_split'],
                class_weight=class_weight,
                callbacks=callbacks_list,
                verbose=1
            )
            
            self.is_fitted = True
            
            # Log training results
            final_loss = self.history.history['loss'][-1]
            final_val_loss = self.history.history['val_loss'][-1]
            final_accuracy = self.history.history['accuracy'][-1]
            final_val_accuracy = self.history.history['val_accuracy'][-1]
            
            self.logger.info("Neural Network fitted successfully")
            self.logger.info(f"Final training loss: {final_loss:.6f}")
            self.logger.info(f"Final validation loss: {final_val_loss:.6f}")
            self.logger.info(f"Final training accuracy: {final_accuracy:.4f}")
            self.logger.info(f"Final validation accuracy: {final_val_accuracy:.4f}")
            
            return self
            
        except Exception as e:
            self.logger.error(f"Error fitting Neural Network: {e}")
            raise
            
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict anomalies using Neural Network.
        
        Args:
            X: Feature matrix
            
        Returns:
            Array of predictions (1 for anomaly, 0 for normal)
        """
        try:
            if not self.is_fitted:
                raise ValueError("Model must be fitted before making predictions")
                
            self.validate_input(X)
            
            X_array = X.values.astype(np.float32)
            probabilities = self.model.predict(X_array, verbose=0)
            predictions = (probabilities > 0.5).astype(int).flatten()
            
            self.logger.info(f"Predicted {np.sum(predictions)} anomalies out of {len(predictions)} samples")
            
            return predictions
            
        except Exception as e:
            self.logger.error(f"Error making predictions with Neural Network: {e}")
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
            
            X_array = X.values.astype(np.float32)
            prob_anomaly = self.model.predict(X_array, verbose=0).flatten()
            prob_normal = 1 - prob_anomaly
            
            probabilities = np.column_stack([prob_normal, prob_anomaly])
            
            return probabilities
            
        except Exception as e:
            self.logger.error(f"Error computing prediction probabilities: {e}")
            raise
            
    def get_feature_importance(self) -> Optional[np.ndarray]:
        """
        Get feature importance using permutation importance.
        
        Returns:
            Array of feature importance scores
        """
        try:
            if not self.is_fitted:
                raise ValueError("Model must be fitted before getting feature importance")
                
            # This is a placeholder - in practice, you would implement
            # permutation importance or other methods for neural networks
            self.logger.warning("Feature importance not directly available for neural networks")
            self.logger.warning("Consider using permutation importance or SHAP values")
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting feature importance: {e}")
            return None
            
    def get_training_history(self) -> Optional[Dict[str, List[float]]]:
        """
        Get the training history.
        
        Returns:
            Dictionary containing training history or None if not available
        """
        try:
            if self.history is not None:
                return self.history.history
            else:
                return None
                
        except Exception as e:
            self.logger.error(f"Error getting training history: {e}")
            return None
            
    def plot_training_history(self) -> None:
        """
        Plot the training history.
        """
        try:
            import matplotlib.pyplot as plt
            
            if self.history is None:
                self.logger.warning("No training history available for plotting")
                return
                
            history = self.history.history
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # Plot loss
            axes[0, 0].plot(history['loss'], label='Training Loss')
            axes[0, 0].plot(history['val_loss'], label='Validation Loss')
            axes[0, 0].set_title('Model Loss')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].legend()
            
            # Plot accuracy
            axes[0, 1].plot(history['accuracy'], label='Training Accuracy')
            axes[0, 1].plot(history['val_accuracy'], label='Validation Accuracy')
            axes[0, 1].set_title('Model Accuracy')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Accuracy')
            axes[0, 1].legend()
            
            # Plot precision
            if 'precision' in history:
                axes[1, 0].plot(history['precision'], label='Training Precision')
                axes[1, 0].plot(history['val_precision'], label='Validation Precision')
                axes[1, 0].set_title('Model Precision')
                axes[1, 0].set_xlabel('Epoch')
                axes[1, 0].set_ylabel('Precision')
                axes[1, 0].legend()
            
            # Plot recall
            if 'recall' in history:
                axes[1, 1].plot(history['recall'], label='Training Recall')
                axes[1, 1].plot(history['val_recall'], label='Validation Recall')
                axes[1, 1].set_title('Model Recall')
                axes[1, 1].set_xlabel('Epoch')
                axes[1, 1].set_ylabel('Recall')
                axes[1, 1].legend()
            
            plt.tight_layout()
            plt.show()
            
        except ImportError:
            self.logger.warning("Matplotlib not available for plotting")
        except Exception as e:
            self.logger.error(f"Error plotting training history: {e}")
            
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get detailed information about the Neural Network model.
        
        Returns:
            Dictionary containing model information
        """
        info = super().get_model_info()
        
        if self.is_fitted:
            info.update({
                'hidden_layers': self.model_params['hidden_layers'],
                'activation': self.model_params['activation'],
                'optimizer': self.model_params['optimizer'],
                'learning_rate': self.model_params['learning_rate'],
                'epochs': self.model_params['epochs'],
                'batch_size': self.model_params['batch_size'],
                'dropout_rate': self.model_params['dropout_rate'],
                'l2_reg': self.model_params['l2_reg'],
                'n_features': len(self.feature_names_) if self.feature_names_ else 0
            })
            
            # Add model summary
            if self.model is not None:
                info['total_params'] = self.model.count_params()
                
            # Add training history summary
            if self.history is not None:
                history = self.history.history
                info['training_history'] = {
                    'final_loss': history['loss'][-1],
                    'final_val_loss': history['val_loss'][-1],
                    'final_accuracy': history['accuracy'][-1],
                    'final_val_accuracy': history['val_accuracy'][-1],
                    'epochs_trained': len(history['loss'])
                }
                
        return info
