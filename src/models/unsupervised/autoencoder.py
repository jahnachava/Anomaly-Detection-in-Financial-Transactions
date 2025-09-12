"""
Autoencoder implementation for anomaly detection.

Autoencoders are neural networks that learn to reconstruct their input data.
Anomalies are detected by measuring the reconstruction error - normal data
should have low reconstruction error while anomalies should have high error.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks
from loguru import logger
from .base_unsupervised import BaseUnsupervisedDetector

class AutoencoderDetector(BaseUnsupervisedDetector):
    """
    Autoencoder anomaly detector for financial transactions.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the Autoencoder detector.
        
        Args:
            config: Configuration dictionary containing model parameters
        """
        super().__init__(config, "Autoencoder")
        
        # Get model parameters from config
        model_config = config.get('models', {}).get('unsupervised', {}).get('autoencoder', {})
        
        self.model_params = {
            'encoding_dim': model_config.get('encoding_dim', 32),
            'hidden_layers': model_config.get('hidden_layers', [64, 32]),
            'activation': model_config.get('activation', 'relu'),
            'optimizer': model_config.get('optimizer', 'adam'),
            'learning_rate': model_config.get('learning_rate', 0.001),
            'epochs': model_config.get('epochs', 100),
            'batch_size': model_config.get('batch_size', 32),
            'validation_split': model_config.get('validation_split', 0.2),
            'dropout_rate': model_config.get('dropout_rate', 0.2),
            'l2_reg': model_config.get('l2_reg', 0.001)
        }
        
        self.autoencoder = None
        self.encoder = None
        self.decoder = None
        self.history = None
        self.threshold = None
        
        self.logger.info(f"Initialized Autoencoder with parameters: {self.model_params}")
        
    def _build_autoencoder(self, input_dim: int) -> Tuple[keras.Model, keras.Model, keras.Model]:
        """
        Build the autoencoder architecture.
        
        Args:
            input_dim: Input dimension (number of features)
            
        Returns:
            Tuple of (autoencoder, encoder, decoder) models
        """
        try:
            # Input layer
            input_layer = layers.Input(shape=(input_dim,), name='input')
            
            # Encoder
            encoder_layers = [input_layer]
            current_dim = input_dim
            
            for i, hidden_dim in enumerate(self.model_params['hidden_layers']):
                encoder_layers.append(
                    layers.Dense(
                        hidden_dim,
                        activation=self.model_params['activation'],
                        kernel_regularizer=keras.regularizers.l2(self.model_params['l2_reg']),
                        name=f'encoder_dense_{i+1}'
                    )(encoder_layers[-1])
                )
                encoder_layers.append(
                    layers.Dropout(
                        self.model_params['dropout_rate'],
                        name=f'encoder_dropout_{i+1}'
                    )(encoder_layers[-1])
                )
                current_dim = hidden_dim
                
            # Bottleneck layer
            encoded = layers.Dense(
                self.model_params['encoding_dim'],
                activation=self.model_params['activation'],
                name='encoded'
            )(encoder_layers[-1])
            
            # Decoder
            decoder_layers = [encoded]
            current_dim = self.model_params['encoding_dim']
            
            # Reverse the hidden layers for decoder
            for i, hidden_dim in enumerate(reversed(self.model_params['hidden_layers'])):
                decoder_layers.append(
                    layers.Dense(
                        hidden_dim,
                        activation=self.model_params['activation'],
                        kernel_regularizer=keras.regularizers.l2(self.model_params['l2_reg']),
                        name=f'decoder_dense_{i+1}'
                    )(decoder_layers[-1])
                )
                decoder_layers.append(
                    layers.Dropout(
                        self.model_params['dropout_rate'],
                        name=f'decoder_dropout_{i+1}'
                    )(decoder_layers[-1])
                )
                current_dim = hidden_dim
                
            # Output layer
            decoded = layers.Dense(
                input_dim,
                activation='linear',
                name='decoded'
            )(decoder_layers[-1])
            
            # Create models
            autoencoder = keras.Model(input_layer, decoded, name='autoencoder')
            encoder = keras.Model(input_layer, encoded, name='encoder')
            
            # Decoder model
            encoded_input = layers.Input(shape=(self.model_params['encoding_dim'],))
            decoder_input = encoded_input
            
            for layer in autoencoder.layers[len(encoder.layers):]:
                decoder_input = layer(decoder_input)
                
            decoder = keras.Model(encoded_input, decoder_input, name='decoder')
            
            self.logger.info(f"Autoencoder architecture built:")
            self.logger.info(f"  Input dimension: {input_dim}")
            self.logger.info(f"  Encoding dimension: {self.model_params['encoding_dim']}")
            self.logger.info(f"  Hidden layers: {self.model_params['hidden_layers']}")
            
            return autoencoder, encoder, decoder
            
        except Exception as e:
            self.logger.error(f"Error building autoencoder architecture: {e}")
            raise
            
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'AutoencoderDetector':
        """
        Fit the Autoencoder model.
        
        Args:
            X: Feature matrix
            y: Target variable (ignored for unsupervised learning)
            
        Returns:
            Self
        """
        try:
            self.validate_input(X)
            self.logger.info(f"Fitting Autoencoder on {X.shape[0]} samples with {X.shape[1]} features")
            
            # Convert to numpy array
            X_array = X.values.astype(np.float32)
            
            # Build the autoencoder
            self.autoencoder, self.encoder, self.decoder = self._build_autoencoder(X.shape[1])
            
            # Compile the model
            optimizer = keras.optimizers.Adam(learning_rate=self.model_params['learning_rate'])
            self.autoencoder.compile(
                optimizer=optimizer,
                loss='mse',
                metrics=['mae']
            )
            
            # Define callbacks
            callbacks_list = [
                callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=10,
                    restore_best_weights=True
                ),
                callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=5,
                    min_lr=1e-7
                )
            ]
            
            # Train the model
            self.history = self.autoencoder.fit(
                X_array, X_array,
                epochs=self.model_params['epochs'],
                batch_size=self.model_params['batch_size'],
                validation_split=self.model_params['validation_split'],
                callbacks=callbacks_list,
                verbose=1
            )
            
            self.is_fitted = True
            
            # Calculate threshold based on reconstruction error
            self._calculate_threshold(X_array)
            
            self.logger.info("Autoencoder fitted successfully")
            self.logger.info(f"Final training loss: {self.history.history['loss'][-1]:.6f}")
            self.logger.info(f"Final validation loss: {self.history.history['val_loss'][-1]:.6f}")
            self.logger.info(f"Anomaly threshold: {self.threshold:.6f}")
            
            return self
            
        except Exception as e:
            self.logger.error(f"Error fitting Autoencoder: {e}")
            raise
            
    def _calculate_threshold(self, X: np.ndarray, contamination: float = 0.1) -> None:
        """
        Calculate the anomaly threshold based on reconstruction error.
        
        Args:
            X: Training data
            contamination: Expected contamination rate
        """
        try:
            # Get reconstruction errors for training data
            reconstructions = self.autoencoder.predict(X, verbose=0)
            reconstruction_errors = np.mean(np.square(X - reconstructions), axis=1)
            
            # Calculate threshold based on contamination rate
            self.threshold = np.percentile(reconstruction_errors, (1 - contamination) * 100)
            
            self.logger.info(f"Threshold calculated: {self.threshold:.6f} (contamination: {contamination})")
            
        except Exception as e:
            self.logger.error(f"Error calculating threshold: {e}")
            self.threshold = None
            
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict anomalies using Autoencoder.
        
        Args:
            X: Feature matrix
            
        Returns:
            Array of predictions (1 for anomaly, 0 for normal)
        """
        try:
            if not self.is_fitted:
                raise ValueError("Model must be fitted before making predictions")
                
            self.validate_input(X)
            
            # Get reconstruction errors
            reconstruction_errors = self._get_reconstruction_errors(X)
            
            # Predict based on threshold
            predictions = (reconstruction_errors > self.threshold).astype(int)
            
            self.logger.info(f"Predicted {np.sum(predictions)} anomalies out of {len(predictions)} samples")
            
            return predictions
            
        except Exception as e:
            self.logger.error(f"Error making predictions with Autoencoder: {e}")
            raise
            
    def decision_function(self, X: pd.DataFrame) -> np.ndarray:
        """
        Compute the decision function for each sample.
        
        Args:
            X: Feature matrix
            
        Returns:
            Array of decision function values (higher values = more anomalous)
        """
        try:
            if not self.is_fitted:
                raise ValueError("Model must be fitted before computing decision function")
                
            self.validate_input(X)
            
            # Get reconstruction errors
            reconstruction_errors = self._get_reconstruction_errors(X)
            
            return reconstruction_errors
            
        except Exception as e:
            self.logger.error(f"Error computing decision function: {e}")
            raise
            
    def _get_reconstruction_errors(self, X: pd.DataFrame) -> np.ndarray:
        """
        Get reconstruction errors for the input data.
        
        Args:
            X: Feature matrix
            
        Returns:
            Array of reconstruction errors
        """
        try:
            X_array = X.values.astype(np.float32)
            
            # Get reconstructions
            reconstructions = self.autoencoder.predict(X_array, verbose=0)
            
            # Calculate reconstruction errors (MSE)
            reconstruction_errors = np.mean(np.square(X_array - reconstructions), axis=1)
            
            return reconstruction_errors
            
        except Exception as e:
            self.logger.error(f"Error computing reconstruction errors: {e}")
            raise
            
    def score_samples(self, X: pd.DataFrame) -> np.ndarray:
        """
        Compute anomaly scores for each sample.
        
        Args:
            X: Feature matrix
            
        Returns:
            Array of anomaly scores (higher values = more anomalous)
        """
        try:
            if not self.is_fitted:
                raise ValueError("Model must be fitted before computing scores")
                
            self.validate_input(X)
            
            # Get reconstruction errors
            scores = self._get_reconstruction_errors(X)
            
            return scores
            
        except Exception as e:
            self.logger.error(f"Error computing anomaly scores: {e}")
            raise
            
    def encode(self, X: pd.DataFrame) -> np.ndarray:
        """
        Encode input data to the latent space.
        
        Args:
            X: Feature matrix
            
        Returns:
            Array of encoded representations
        """
        try:
            if not self.is_fitted:
                raise ValueError("Model must be fitted before encoding")
                
            self.validate_input(X)
            
            X_array = X.values.astype(np.float32)
            encoded = self.encoder.predict(X_array, verbose=0)
            
            return encoded
            
        except Exception as e:
            self.logger.error(f"Error encoding data: {e}")
            raise
            
    def decode(self, encoded_data: np.ndarray) -> np.ndarray:
        """
        Decode encoded data back to original space.
        
        Args:
            encoded_data: Encoded representations
            
        Returns:
            Array of decoded data
        """
        try:
            if not self.is_fitted:
                raise ValueError("Model must be fitted before decoding")
                
            decoded = self.decoder.predict(encoded_data, verbose=0)
            
            return decoded
            
        except Exception as e:
            self.logger.error(f"Error decoding data: {e}")
            raise
            
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
            
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get detailed information about the Autoencoder model.
        
        Returns:
            Dictionary containing model information
        """
        info = super().get_model_info()
        
        if self.is_fitted:
            info.update({
                'encoding_dim': self.model_params['encoding_dim'],
                'hidden_layers': self.model_params['hidden_layers'],
                'activation': self.model_params['activation'],
                'optimizer': self.model_params['optimizer'],
                'learning_rate': self.model_params['learning_rate'],
                'epochs': self.model_params['epochs'],
                'batch_size': self.model_params['batch_size'],
                'dropout_rate': self.model_params['dropout_rate'],
                'l2_reg': self.model_params['l2_reg'],
                'threshold': self.threshold
            })
            
            # Add model summary
            if self.autoencoder is not None:
                info['total_params'] = self.autoencoder.count_params()
                
        return info
