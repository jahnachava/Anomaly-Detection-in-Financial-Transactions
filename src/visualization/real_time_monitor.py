"""
Real-time monitoring system for anomaly detection.

This module provides functionality for real-time monitoring of financial
transactions and anomaly detection results.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Callable
import threading
import time
from datetime import datetime, timedelta
from queue import Queue
from loguru import logger
import json

class RealTimeMonitor:
    """
    Real-time monitoring system for anomaly detection.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the real-time monitor.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = logger
        self.monitor_config = config.get('dashboard', {}).get('real_time', {})
        
        # Monitoring state
        self.is_monitoring = False
        self.monitor_thread = None
        self.data_queue = Queue()
        self.anomaly_queue = Queue()
        
        # Callbacks
        self.anomaly_callback = None
        self.alert_callback = None
        
        # Statistics
        self.stats = {
            'total_transactions': 0,
            'anomalies_detected': 0,
            'false_positives': 0,
            'processing_time': 0,
            'last_update': None
        }
        
        # Configuration
        self.update_interval = self.monitor_config.get('update_interval', 5)
        self.max_transactions_display = self.monitor_config.get('max_transactions_display', 1000)
        self.alert_threshold = self.monitor_config.get('alert_threshold', 0.8)
        
    def start_monitoring(self, data_source: Callable = None) -> None:
        """
        Start real-time monitoring.
        
        Args:
            data_source: Function that provides new transaction data
        """
        try:
            if self.is_monitoring:
                self.logger.warning("Monitoring is already running")
                return
                
            self.is_monitoring = True
            self.logger.info("Starting real-time monitoring")
            
            # Start monitoring thread
            self.monitor_thread = threading.Thread(
                target=self._monitoring_loop,
                args=(data_source,),
                daemon=True
            )
            self.monitor_thread.start()
            
        except Exception as e:
            self.logger.error(f"Error starting monitoring: {e}")
            raise
            
    def stop_monitoring(self) -> None:
        """
        Stop real-time monitoring.
        """
        try:
            if not self.is_monitoring:
                self.logger.warning("Monitoring is not running")
                return
                
            self.is_monitoring = False
            self.logger.info("Stopping real-time monitoring")
            
            # Wait for thread to finish
            if self.monitor_thread and self.monitor_thread.is_alive():
                self.monitor_thread.join(timeout=5)
                
        except Exception as e:
            self.logger.error(f"Error stopping monitoring: {e}")
            
    def _monitoring_loop(self, data_source: Callable = None) -> None:
        """
        Main monitoring loop.
        
        Args:
            data_source: Function that provides new transaction data
        """
        try:
            while self.is_monitoring:
                start_time = time.time()
                
                # Get new data
                if data_source:
                    new_data = data_source()
                    if new_data is not None:
                        self._process_new_data(new_data)
                        
                # Update statistics
                self._update_statistics()
                
                # Check for alerts
                self._check_alerts()
                
                # Sleep for update interval
                elapsed_time = time.time() - start_time
                sleep_time = max(0, self.update_interval - elapsed_time)
                time.sleep(sleep_time)
                
        except Exception as e:
            self.logger.error(f"Error in monitoring loop: {e}")
            self.is_monitoring = False
            
    def _process_new_data(self, data: pd.DataFrame) -> None:
        """
        Process new transaction data.
        
        Args:
            data: New transaction data
        """
        try:
            # Add to data queue
            self.data_queue.put(data)
            
            # Update statistics
            self.stats['total_transactions'] += len(data)
            self.stats['last_update'] = datetime.now()
            
            # Simulate anomaly detection (in real implementation, this would use actual models)
            anomalies = self._simulate_anomaly_detection(data)
            
            if anomalies.any():
                anomaly_data = data[anomalies].copy()
                anomaly_data['detection_time'] = datetime.now()
                anomaly_data['anomaly_score'] = np.random.uniform(0.7, 1.0, len(anomaly_data))
                
                # Add to anomaly queue
                self.anomaly_queue.put(anomaly_data)
                
                # Update statistics
                self.stats['anomalies_detected'] += len(anomaly_data)
                
                # Trigger anomaly callback
                if self.anomaly_callback:
                    self.anomaly_callback(anomaly_data)
                    
        except Exception as e:
            self.logger.error(f"Error processing new data: {e}")
            
    def _simulate_anomaly_detection(self, data: pd.DataFrame) -> np.ndarray:
        """
        Simulate anomaly detection (for demonstration purposes).
        
        Args:
            data: Transaction data
            
        Returns:
            Boolean array indicating anomalies
        """
        try:
            # Simple simulation based on amount and time
            anomalies = np.zeros(len(data), dtype=bool)
            
            # High amount transactions
            if 'Amount' in data.columns:
                high_amount_threshold = data['Amount'].quantile(0.95)
                anomalies |= data['Amount'] > high_amount_threshold
                
            # Unusual time patterns
            if 'Time' in data.columns:
                # Simulate time-based anomalies
                time_anomalies = np.random.random(len(data)) < 0.02  # 2% anomaly rate
                anomalies |= time_anomalies
                
            return anomalies
            
        except Exception as e:
            self.logger.error(f"Error in anomaly simulation: {e}")
            return np.zeros(len(data), dtype=bool)
            
    def _update_statistics(self) -> None:
        """
        Update monitoring statistics.
        """
        try:
            # Calculate processing time
            if self.stats['total_transactions'] > 0:
                self.stats['processing_time'] = time.time() - self.stats.get('start_time', time.time())
                
        except Exception as e:
            self.logger.error(f"Error updating statistics: {e}")
            
    def _check_alerts(self) -> None:
        """
        Check for alert conditions.
        """
        try:
            # Check for high anomaly rate
            if self.stats['total_transactions'] > 0:
                anomaly_rate = self.stats['anomalies_detected'] / self.stats['total_transactions']
                
                if anomaly_rate > self.alert_threshold:
                    alert_message = f"High anomaly rate detected: {anomaly_rate:.2%}"
                    self.logger.warning(alert_message)
                    
                    if self.alert_callback:
                        self.alert_callback({
                            'type': 'high_anomaly_rate',
                            'message': alert_message,
                            'anomaly_rate': anomaly_rate,
                            'timestamp': datetime.now()
                        })
                        
        except Exception as e:
            self.logger.error(f"Error checking alerts: {e}")
            
    def get_latest_data(self, n_transactions: int = 100) -> pd.DataFrame:
        """
        Get the latest transaction data.
        
        Args:
            n_transactions: Number of recent transactions to return
            
        Returns:
            DataFrame with recent transactions
        """
        try:
            # In a real implementation, this would retrieve from a database
            # For now, return sample data
            return self._generate_sample_data(n_transactions)
            
        except Exception as e:
            self.logger.error(f"Error getting latest data: {e}")
            return pd.DataFrame()
            
    def get_latest_anomalies(self, n_anomalies: int = 50) -> pd.DataFrame:
        """
        Get the latest detected anomalies.
        
        Args:
            n_anomalies: Number of recent anomalies to return
            
        Returns:
            DataFrame with recent anomalies
        """
        try:
            anomalies = []
            
            # Collect anomalies from queue
            while not self.anomaly_queue.empty() and len(anomalies) < n_anomalies:
                try:
                    anomaly_data = self.anomaly_queue.get_nowait()
                    anomalies.append(anomaly_data)
                except:
                    break
                    
            if anomalies:
                return pd.concat(anomalies, ignore_index=True)
            else:
                # Return sample data if no real anomalies
                return self._generate_sample_anomaly_data(n_anomalies)
                
        except Exception as e:
            self.logger.error(f"Error getting latest anomalies: {e}")
            return pd.DataFrame()
            
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get current monitoring statistics.
        
        Returns:
            Dictionary with monitoring statistics
        """
        try:
            stats = self.stats.copy()
            
            # Calculate additional metrics
            if stats['total_transactions'] > 0:
                stats['anomaly_rate'] = stats['anomalies_detected'] / stats['total_transactions']
                stats['false_positive_rate'] = stats['false_positives'] / stats['total_transactions']
            else:
                stats['anomaly_rate'] = 0
                stats['false_positive_rate'] = 0
                
            return stats
            
        except Exception as e:
            self.logger.error(f"Error getting statistics: {e}")
            return {}
            
    def set_anomaly_callback(self, callback: Callable) -> None:
        """
        Set callback function for anomaly detection events.
        
        Args:
            callback: Function to call when anomalies are detected
        """
        self.anomaly_callback = callback
        
    def set_alert_callback(self, callback: Callable) -> None:
        """
        Set callback function for alert events.
        
        Args:
            callback: Function to call when alerts are triggered
        """
        self.alert_callback = callback
        
    def _generate_sample_data(self, n_transactions: int) -> pd.DataFrame:
        """
        Generate sample transaction data for demonstration.
        
        Args:
            n_transactions: Number of transactions to generate
            
        Returns:
            DataFrame with sample transaction data
        """
        try:
            data = {
                'Transaction_ID': [f'TXN_{i:06d}' for i in range(1, n_transactions + 1)],
                'Timestamp': pd.date_range(start=datetime.now() - timedelta(hours=1), 
                                         periods=n_transactions, freq='1min'),
                'Amount': np.random.exponential(100, n_transactions),
                'Merchant': [f'Merchant_{i}' for i in np.random.randint(1, 20, n_transactions)],
                'Category': np.random.choice(['Food', 'Gas', 'Shopping', 'Entertainment', 'Other'], n_transactions),
                'Location': [f'City_{i}' for i in np.random.randint(1, 10, n_transactions)]
            }
            
            return pd.DataFrame(data)
            
        except Exception as e:
            self.logger.error(f"Error generating sample data: {e}")
            return pd.DataFrame()
            
    def _generate_sample_anomaly_data(self, n_anomalies: int) -> pd.DataFrame:
        """
        Generate sample anomaly data for demonstration.
        
        Args:
            n_anomalies: Number of anomalies to generate
            
        Returns:
            DataFrame with sample anomaly data
        """
        try:
            data = {
                'Transaction_ID': [f'ANOM_{i:06d}' for i in range(1, n_anomalies + 1)],
                'Timestamp': pd.date_range(start=datetime.now() - timedelta(hours=1), 
                                         periods=n_anomalies, freq='5min'),
                'Amount': np.random.exponential(500, n_anomalies),  # Higher amounts for anomalies
                'Merchant': [f'Suspicious_Merchant_{i}' for i in np.random.randint(1, 5, n_anomalies)],
                'Category': np.random.choice(['Unknown', 'High_Risk', 'Suspicious'], n_anomalies),
                'Location': [f'Unknown_Location_{i}' for i in np.random.randint(1, 3, n_anomalies)],
                'Anomaly_Score': np.random.uniform(0.7, 1.0, n_anomalies),
                'Detection_Time': pd.date_range(start=datetime.now() - timedelta(hours=1), 
                                              periods=n_anomalies, freq='5min')
            }
            
            return pd.DataFrame(data)
            
        except Exception as e:
            self.logger.error(f"Error generating sample anomaly data: {e}")
            return pd.DataFrame()
            
    def export_data(self, file_path: str, data_type: str = 'all') -> None:
        """
        Export monitoring data to file.
        
        Args:
            file_path: Path to save the file
            data_type: Type of data to export ('all', 'transactions', 'anomalies', 'statistics')
        """
        try:
            if data_type in ['all', 'transactions']:
                transactions = self.get_latest_data(1000)
                if not transactions.empty:
                    transactions.to_csv(f"{file_path}_transactions.csv", index=False)
                    
            if data_type in ['all', 'anomalies']:
                anomalies = self.get_latest_anomalies(500)
                if not anomalies.empty:
                    anomalies.to_csv(f"{file_path}_anomalies.csv", index=False)
                    
            if data_type in ['all', 'statistics']:
                stats = self.get_statistics()
                with open(f"{file_path}_statistics.json", 'w') as f:
                    json.dump(stats, f, indent=2, default=str)
                    
            self.logger.info(f"Data exported to {file_path}")
            
        except Exception as e:
            self.logger.error(f"Error exporting data: {e}")
            raise
