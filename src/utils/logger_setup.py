"""
Logger setup utility for the anomaly detection system.

This module provides functionality to set up logging for the entire
anomaly detection system with proper formatting and file handling.
"""

import sys
from pathlib import Path
from loguru import logger
from typing import Dict, Any, Optional

def setup_logger(config: Dict[str, Any]) -> None:
    """
    Set up logging configuration for the anomaly detection system.
    
    Args:
        config: Configuration dictionary containing logging settings
    """
    try:
        # Remove default logger
        logger.remove()
        
        # Get logging configuration
        log_config = config.get('logging', {})
        log_level = log_config.get('level', 'INFO')
        log_format = log_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        log_file = log_config.get('file', 'logs/anomaly_detection.log')
        max_file_size = log_config.get('max_file_size', '10MB')
        backup_count = log_config.get('backup_count', 5)
        
        # Create logs directory if it doesn't exist
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Add console handler
        logger.add(
            sys.stdout,
            level=log_level,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
            colorize=True
        )
        
        # Add file handler
        logger.add(
            log_file,
            level=log_level,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            rotation=max_file_size,
            retention=backup_count,
            compression="zip"
        )
        
        logger.info("Logging system initialized successfully")
        
    except Exception as e:
        print(f"Error setting up logger: {e}")
        raise

def get_logger(name: str = None) -> logger:
    """
    Get a logger instance.
    
    Args:
        name: Logger name (optional)
        
    Returns:
        Logger instance
    """
    if name:
        return logger.bind(name=name)
    return logger
