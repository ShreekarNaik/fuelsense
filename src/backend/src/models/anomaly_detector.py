# anomaly_detector.py
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import joblib
import os

class FuelAnomalyDetector:
    """Detects anomalies in fuel consumption and equipment operation."""
    
    def __init__(self, method='isolation_forest'):
        """Initialize the anomaly detector.
        
        Args:
            method (str): Detection method ('isolation_forest', 'statistical', or 'combined')
        """
        self.method = method
        self.models = {}
        self.scalers = {}
        self.thresholds = {}
        self.equipment_types = None
    
    def preprocess_data(self, data):
        """Preprocess input data for anomaly detection."""
        # Convert timestamp to datetime if not already
        if 'Timestamp' in data.columns and not pd.api.types.is_datetime64_dtype(data['Timestamp']):
            data['Timestamp'] = pd.to_datetime(data['Timestamp'])
        
        return data
    
    def get_features(self):
        """Return the list of features used for anomaly detection."""
        # Base features for anomaly detection
        features = [
            'Fuel_Consumed (L/hr)', 'Load (%)', 'RPM', 'Speed (km/h)',
            'Idle_Time (min)', 'Temperature (°C)', 'Engine_Temperature (°C)',
            'Oil_Pressure (psi)', 'Fuel_Efficiency (km/L)', 
            'Emission_Rate (g/km)'
        ]
        
        # Additional features if available
        optional_features = [
            'Fuel_Level_Change', 'Engine_Ambient_Temp_Diff',
            'Consumption_Per_Load', 'Battery_Voltage (V)',
            'Throttle_Position (%)'
        ]
        
        return features, optional_features
    
    def prepare_features(self, data):
        """Prepare feature set for training or prediction."""
        data = self.preprocess_data(data)
        base_features, optional_features = self.get_features()
        
        # Check which optional features are available
        available_features = base_features + [f for f in optional_features if f in data.columns]
        
        # Ensure all features are present with numeric values
        feature_data = data[available_features].copy()
        
        # Replace infinities with NaN
        feature_data.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        # Replace NaN with 0 (or could use other imputation strategies)
        feature_data.fillna(0, inplace=True)
        
        return feature_data
    
    def _extract_features(self, data):
        """Extract and prepare features for anomaly detection.
        
        Args:
            data (DataFrame): Input data
            
        Returns:
            ndarray: Prepared feature matrix
        """
        # Prepare features
        X = self.prepare_features(data)
        
        # Scale features if model exists for this equipment type
        eq_type = data['Equipment_Type'].iloc[0] if len(data) > 0 else None
        
        if eq_type in self.scalers:
            # Convert to numpy array
            X_values = X.values
            
            # Scale features
            return self.scalers[eq_type].transform(X_values)
        else:
            return X.values
    
    def train(self, training_data):
        """Train the anomaly detection model.
        
        Args:
            training_data (DataFrame): Historical equipment data with normal operation
        """
        print("Training anomaly detection model...")
        training_data = self.preprocess_data(training_data)
        
        # Store equipment types for later use
        self.equipment_types = training_data['Equipment_Type'].unique()
        
        # Train a separate model for each equipment type for better accuracy
        for eq_type in self.equipment_types:
            print(f"Training anomaly detection for equipment type: {eq_type}")
            
            # Get data for this equipment type, filtering out known anomalies for training
            eq_data = training_data[
                (training_data['Equipment_Type'] == eq_type) & 
                (training_data['Event_Notes'].str.len() == 0 | training_data['Event_Notes'].isna())
            ].copy()
            
            # Skip if not enough data
            if len(eq_data) < 50:
                print(f"Not enough normal data for {eq_type}. Skipping.")
                continue
            
            # Prepare features
            X = self.prepare_features(eq_data)
            
            # Create and fit scaler
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Train Isolation Forest for anomaly detection
            if self.method in ['isolation_forest', 'combined']:
                model = IsolationForest(contamination=0.05, random_state=42)
                model.fit(X_scaled)
                
                # Store model and scaler
                self.models[eq_type] = model
                self.scalers[eq_type] = scaler
            
            # If using statistical method, calculate thresholds for key metrics
            if self.method in ['statistical', 'combined']:
                # Define key metrics for statistical anomaly detection
                key_metrics = [
                    'Fuel_Consumed (L/hr)', 'Fuel_Efficiency (km/L)', 
                    'Engine_Temperature (°C)', 'Oil_Pressure (psi)'
                ]
                
                # Calculate thresholds (mean ± 3*std) for each metric
                self.thresholds[eq_type] = {}
                
                for metric in key_metrics:
                    if metric in eq_data.columns:
                        values = eq_data[metric].dropna()
                        if len(values) > 0:
                            mean = values.mean()
                            std = values.std()
                            self.thresholds[eq_type][metric] = {
                                'mean': mean,
                                'std': std,
                                'lower': mean - 3 * std,
                                'upper': mean + 3 * std
                            }
            
            print(f"Anomaly detection for {eq_type} trained successfully.")
        
        print("All anomaly detection models trained successfully.")
    
    def detect(self, data):
        """Detect anomalies in the given data.
        
        Args:
            data (DataFrame): Data to analyze for anomalies
            
        Returns:
            DataFrame: Original data with anomaly scores and detection results
        """
        results = []
        
        # If no equipment types were trained, return empty DataFrame
        if not hasattr(self, 'equipment_types') or len(self.equipment_types) == 0:
            return pd.DataFrame()
            
        # Process each equipment type
        for eq_type in self.equipment_types:
            if eq_type not in self.models:
                continue
                
            # Get data for this equipment type
            eq_data = data[data['Equipment_Type'] == eq_type].copy()
            if len(eq_data) == 0:
                continue
            
            # Extract features
            X = self._extract_features(eq_data)
            
            # Get anomaly scores
            anomaly_scores = self.models[eq_type].decision_function(X)
            
            # Add results
            eq_data['Anomaly_Score'] = anomaly_scores
            eq_data['Anomaly_Detected'] = anomaly_scores < self.thresholds.get(eq_type, -0.5)
            results.append(eq_data)
        
        # Combine results
        if results:
            return pd.concat(results, axis=0)
        return pd.DataFrame()
    
    def save_models(self, directory='models'):
        """Save trained models to disk."""
        os.makedirs(directory, exist_ok=True)
        
        # Save ML models and scalers
        for eq_type, model in self.models.items():
            # Save model
            model_path = os.path.join(directory, f"anomaly_model_{eq_type}.joblib")
            joblib.dump(model, model_path)
            
            # Save scaler
            scaler_path = os.path.join(directory, f"anomaly_scaler_{eq_type}.joblib")
            joblib.dump(self.scalers[eq_type], scaler_path)
            
            print(f"Anomaly model for {eq_type} saved to {model_path}")
        
        # Save statistical thresholds
        if self.thresholds:
            threshold_path = os.path.join(directory, "anomaly_thresholds.joblib")
            joblib.dump(self.thresholds, threshold_path)
            print(f"Anomaly thresholds saved to {threshold_path}")
    
    def load_models(self, directory='models'):
        """Load trained models from disk."""
        # Get all model files
        model_files = [f for f in os.listdir(directory) if f.startswith("anomaly_model_") and f.endswith(".joblib")]
        
        for model_file in model_files:
            # Extract equipment type from filename
            eq_type = model_file.replace("anomaly_model_", "").replace(".joblib", "")
            
            # Load model
            model_path = os.path.join(directory, model_file)
            self.models[eq_type] = joblib.load(model_path)
            
            # Load corresponding scaler
            scaler_path = os.path.join(directory, f"anomaly_scaler_{eq_type}.joblib")
            if os.path.exists(scaler_path):
                self.scalers[eq_type] = joblib.load(scaler_path)
            
            print(f"Anomaly model for {eq_type} loaded from {model_path}")
        
        # Load thresholds if available
        threshold_path = os.path.join(directory, "anomaly_thresholds.joblib")
        if os.path.exists(threshold_path):
            self.thresholds = joblib.load(threshold_path)
            print(f"Anomaly thresholds loaded from {threshold_path}")
        
        # Set equipment types based on loaded models
        self.equipment_types = list(self.models.keys())
