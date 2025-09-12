"""
Configuration loader utility for the anomaly detection system.

This module provides functionality to load and validate configuration
files for the anomaly detection system.
"""

import yaml
import json
from pathlib import Path
from typing import Dict, Any, Optional
from loguru import logger

class ConfigLoader:
    """
    A class to load and validate configuration files.
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize the ConfigLoader.
        
        Args:
            config_path: Path to the configuration file
        """
        self.config_path = Path(config_path)
        self.config = {}
        self.logger = logger
        
    def load_config(self) -> Dict[str, Any]:
        """
        Load configuration from file.
        
        Returns:
            Dictionary containing configuration
        """
        try:
            if not self.config_path.exists():
                raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
                
            if self.config_path.suffix.lower() == '.yaml' or self.config_path.suffix.lower() == '.yml':
                with open(self.config_path, 'r') as f:
                    self.config = yaml.safe_load(f)
            elif self.config_path.suffix.lower() == '.json':
                with open(self.config_path, 'r') as f:
                    self.config = json.load(f)
            else:
                raise ValueError(f"Unsupported configuration file format: {self.config_path.suffix}")
                
            self.logger.info(f"Configuration loaded from {self.config_path}")
            return self.config
            
        except Exception as e:
            self.logger.error(f"Error loading configuration: {e}")
            raise
            
    def validate_config(self) -> bool:
        """
        Validate the loaded configuration.
        
        Returns:
            True if configuration is valid, False otherwise
        """
        try:
            required_sections = ['data', 'models', 'evaluation', 'dashboard']
            
            for section in required_sections:
                if section not in self.config:
                    self.logger.error(f"Missing required configuration section: {section}")
                    return False
                    
            # Validate data configuration
            data_config = self.config.get('data', {})
            if 'dataset' not in data_config:
                self.logger.error("Missing dataset configuration")
                return False
                
            # Validate models configuration
            models_config = self.config.get('models', {})
            if 'supervised' not in models_config and 'unsupervised' not in models_config:
                self.logger.error("Missing model configurations")
                return False
                
            self.logger.info("Configuration validation passed")
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating configuration: {e}")
            return False
            
    def get_config_value(self, key_path: str, default: Any = None) -> Any:
        """
        Get a configuration value using dot notation.
        
        Args:
            key_path: Dot-separated path to the configuration value
            default: Default value if key is not found
            
        Returns:
            Configuration value or default
        """
        try:
            keys = key_path.split('.')
            value = self.config
            
            for key in keys:
                if isinstance(value, dict) and key in value:
                    value = value[key]
                else:
                    return default
                    
            return value
            
        except Exception as e:
            self.logger.error(f"Error getting config value for {key_path}: {e}")
            return default
            
    def update_config(self, key_path: str, value: Any) -> None:
        """
        Update a configuration value using dot notation.
        
        Args:
            key_path: Dot-separated path to the configuration value
            value: New value to set
        """
        try:
            keys = key_path.split('.')
            config = self.config
            
            for key in keys[:-1]:
                if key not in config:
                    config[key] = {}
                config = config[key]
                
            config[keys[-1]] = value
            self.logger.info(f"Configuration updated: {key_path} = {value}")
            
        except Exception as e:
            self.logger.error(f"Error updating config value for {key_path}: {e}")
            raise
            
    def save_config(self, output_path: Optional[str] = None) -> None:
        """
        Save configuration to file.
        
        Args:
            output_path: Path to save the configuration (optional)
        """
        try:
            save_path = Path(output_path) if output_path else self.config_path
            
            if save_path.suffix.lower() == '.yaml' or save_path.suffix.lower() == '.yml':
                with open(save_path, 'w') as f:
                    yaml.dump(self.config, f, default_flow_style=False, indent=2)
            elif save_path.suffix.lower() == '.json':
                with open(save_path, 'w') as f:
                    json.dump(self.config, f, indent=2)
            else:
                raise ValueError(f"Unsupported configuration file format: {save_path.suffix}")
                
            self.logger.info(f"Configuration saved to {save_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving configuration: {e}")
            raise
