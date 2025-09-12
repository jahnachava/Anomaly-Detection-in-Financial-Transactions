"""
Machine learning models for anomaly detection in financial transactions.

This module contains implementations of both supervised and unsupervised
anomaly detection algorithms.
"""

from .supervised import SupervisedAnomalyDetector
from .unsupervised import UnsupervisedAnomalyDetector

__all__ = [
    'SupervisedAnomalyDetector',
    'UnsupervisedAnomalyDetector'
]
