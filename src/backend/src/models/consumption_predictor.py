# consumption_predictor.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
import joblib
import os
from datetime import datetime, timedelta

class FuelConsumptionPredictor:
    """Predicts future fuel consumption based on historical data and equipment parameters."""
    
    def __init__(self, model_type='xgboost'):
        """Initialize the consumption predictor.
        
        Args:
            model_type (str): Type of model to use ('rf', 'xgboost', or 'ensemble')
        """
        self.model_type = model_type
        self.models = {}
        self.scalers = {}
        self.equipment_types = None
        self.feature_importances = {}
        self.model_features = {}
        self.categorical_values = {}  # Store possible values for categorical features
    
    def preprocess_data(self, data):
        """Preprocess input data for prediction."""
        # Convert timestamp to datetime if it's not already
        if 'Timestamp' in data.columns and not pd.api.types.is_datetime64_dtype(data['Timestamp']):
            data['Timestamp'] = pd.to_datetime(data['Timestamp'])
        
        # Create time-based features if not present
        if 'Hour' not in data.columns:
            data['Hour'] = data['Timestamp'].dt.hour
        if 'Weekday' not in data.columns:
            data['Weekday'] = data['Timestamp'].dt.weekday
        if 'Is_Weekend' not in data.columns:
            data['Is_Weekend'] = (data['Weekday'] >= 5).astype(int)
        
        return data
    
    def get_features(self):
        """Return the list of features used by the model."""
        # Base features for consumption prediction
        features = [
            'Hour', 'Weekday', 'Is_Weekend', 
            'Load (%)', 'RPM', 'Speed (km/h)', 'Temperature (°C)',
            'Equipment_Type_Bulldozer', 'Equipment_Type_Excavator', 
            'Equipment_Type_Forklift', 'Equipment_Type_Loader',
            'Equipment_Type_Truck', 'Operational_Status_Running'
        ]
        
        # Add rolling statistics if available
        optional_features = [
            'Avg_Load_12h', 'Avg_Speed_12h', 'Avg_Consumption_12h',
            'Engine_Temperature (°C)', 'Maintenance_Status_Normal'
        ]
        
        return features, optional_features
    
    def prepare_features(self, data, is_training=False):
        """Prepare features for prediction."""
        # Create a copy of the data to avoid modifying the original
        data = data.copy()
        
        # Drop timestamp column as it's not needed for prediction
        if 'Timestamp' in data.columns:
            data = data.drop('Timestamp', axis=1)
        
        # Handle categorical variables
        categorical_features = ['Equipment_Type', 'Operational_Status', 'Maintenance_Status']
        
        # During training phase, record all possible categorical values
        if is_training:
            for feature in categorical_features:
                if feature in data.columns:
                    self.categorical_values[feature] = sorted(data[feature].unique().tolist())
        
        # One-hot encode categorical variables with consistent columns
        for feature in categorical_features:
            if feature in data.columns:
                # Use get_dummies with specific categories if we have them
                if feature in self.categorical_values:
                    # Get dummies with consistent categories
                    dummies = pd.get_dummies(
                        data[feature], 
                        prefix=feature,
                        dummy_na=False  # Don't create column for NaN values
                    )
                    
                    # Add missing columns with zeros
                    for value in self.categorical_values[feature]:
                        dummy_col = f"{feature}_{value}"
                        if dummy_col not in dummies.columns:
                            dummies[dummy_col] = 0
                            
                    # Drop any extra columns not seen during training
                    expected_cols = [f"{feature}_{value}" for value in self.categorical_values[feature]]
                    extra_cols = [col for col in dummies.columns if col not in expected_cols]
                    if extra_cols:
                        dummies = dummies.drop(columns=extra_cols)
                        
                else:
                    # Simple get_dummies during initial training
                    dummies = pd.get_dummies(data[feature], prefix=feature)
                
                # Combine with main dataframe
                data = pd.concat([data, dummies], axis=1)
                data.drop(feature, axis=1, inplace=True)
        
        # Ensure all required features exist and are numeric
        required_features = [
            'Hour', 'Weekday', 'Is_Weekend', 'Load (%)', 'RPM', 'Speed (km/h)',
            'Temperature (°C)', 'Engine_Temperature (°C)', 'Avg_Load_12h',
            'Avg_Speed_12h', 'Avg_Consumption_12h'
        ]
        
        for feature in required_features:
            if feature not in data.columns:
                data.loc[:, feature] = 0
            else:
                # Convert to numeric, replacing any non-numeric values with 0
                data.loc[:, feature] = pd.to_numeric(data[feature], errors='coerce').fillna(0)
        
        # Drop any remaining non-numeric columns
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        data = data[numeric_cols]
        
        return data
    
    def train(self, training_data):
        """Train the consumption prediction model.
        
        Args:
            training_data (DataFrame): Historical equipment data
        """
        print("Training fuel consumption prediction model...")
        training_data = self.preprocess_data(training_data)
        
        # Store equipment types for later use
        self.equipment_types = training_data['Equipment_Type'].unique()
        self.model_features = {}  # Store features for each model
        
        # Train a separate model for each equipment type for better accuracy
        for eq_type in self.equipment_types:
            print(f"Training model for equipment type: {eq_type}")
            
            # Get data for this equipment type
            eq_data = training_data[training_data['Equipment_Type'] == eq_type].copy()
            
            # Skip if not enough data
            if len(eq_data) < 100:
                print(f"Not enough data for {eq_type}. Skipping.")
                continue
            
            # Prepare features and target
            X = self.prepare_features(eq_data, is_training=True)
            y = eq_data['Fuel_Consumed (L/hr)']
            
            # Store the exact feature columns used for this model
            self.model_features[eq_type] = X.columns.tolist()
            
            # Create pipeline with preprocessing and model
            if self.model_type == 'rf':
                model = Pipeline([
                    ('scaler', StandardScaler()),
                    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
                ])
            else:  # Default to xgboost
                from xgboost import XGBRegressor
                model = Pipeline([
                    ('scaler', StandardScaler()),
                    ('regressor', XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42))
                ])
            
            # Train the model
            model.fit(X, y)
            
            # Store model and scaler
            self.models[eq_type] = model
            self.scalers[eq_type] = model.named_steps['scaler']
            
            # Store feature importances
            if hasattr(model.named_steps['regressor'], 'feature_importances_'):
                self.feature_importances[eq_type] = dict(zip(
                    X.columns, 
                    model.named_steps['regressor'].feature_importances_
                ))
            
            print(f"Model for {eq_type} trained successfully.")
        
        print("All models trained successfully.")
    
    def predict(self, input_data, future_hours=24):
        """Predict future fuel consumption.
        
        Args:
            input_data (DataFrame): Current equipment data
            future_hours (int): Number of hours to predict into the future
            
        Returns:
            DataFrame: Predictions with equipment ID, timestamp, and predicted consumption
        """
        input_data = self.preprocess_data(input_data)
        
        # Group by equipment ID to make predictions for each piece of equipment
        predictions = []
        
        # Check we have equipment type in model features
        equipment_type = input_data['Equipment_Type'].iloc[0] if len(input_data) > 0 else None
        if equipment_type and equipment_type not in self.model_features:
            print(f"Warning: No model features stored for {equipment_type}")
            return pd.DataFrame(columns=['Equipment_ID', 'Equipment_Type', 'Timestamp', 'Predicted_Consumption'])
        
        for eq_id, eq_data in input_data.groupby('Equipment_ID'):
            # Get the equipment type
            eq_type = eq_data['Equipment_Type'].iloc[0]
            
            # Skip if we don't have a model for this equipment type
            if eq_type not in self.models:
                print(f"No model available for equipment type: {eq_type}")
                continue
            
            # Get the latest data point for this equipment
            latest_data = eq_data.sort_values('Timestamp').iloc[-1:].copy()
            latest_time = latest_data['Timestamp'].iloc[0]
            
            # Make predictions for future hours
            for hour in range(1, future_hours + 1):
                # Create a copy of the latest data for prediction
                future_data = latest_data.copy()
                
                # Update timestamp and time-based features
                future_time = latest_time + timedelta(hours=hour)
                future_data['Timestamp'] = future_time
                future_data['Hour'] = future_time.hour
                future_data['Weekday'] = future_time.weekday()
                future_data['Is_Weekend'] = int(future_time.weekday() >= 5)
                
                try:
                    # Prepare features for prediction using cached categorical values
                    X_future = self.prepare_features(future_data)
                    
                    # Ensure feature consistency with training data
                    if eq_type in self.model_features:
                        needed_features = self.model_features[eq_type]
                        
                        # Debug information
                        if len(X_future.columns) != len(needed_features):
                            missing = set(needed_features) - set(X_future.columns)
                            extra = set(X_future.columns) - set(needed_features)
                            if missing:
                                print(f"Missing features for {eq_type}: {missing}")
                            if extra:
                                print(f"Extra features for {eq_type} that weren't in training: {extra}")
                        
                        # Add any missing features with zeros
                        for col in needed_features:
                            if col not in X_future.columns:
                                X_future[col] = 0
                        
                        # Select only the features used during training, in the exact same order
                        X_future = X_future[needed_features]
                    
                    # Make prediction
                    pred_consumption = self.models[eq_type].predict(X_future)[0]
                    
                    # Store prediction
                    predictions.append({
                        'Equipment_ID': eq_id,
                        'Equipment_Type': eq_type,
                        'Timestamp': future_time,
                        'Predicted_Consumption': max(0, pred_consumption)  # Ensure non-negative
                    })
                except Exception as e:
                    print(f"Error predicting for {eq_type} (hour {hour}): {str(e)}")
                    continue
        
        # Create DataFrame from predictions
        if predictions:
            return pd.DataFrame(predictions)
        else:
            return pd.DataFrame(columns=['Equipment_ID', 'Equipment_Type', 'Timestamp', 'Predicted_Consumption'])
    
    def save_models(self, directory='models'):
        """Save trained models to disk."""
        os.makedirs(directory, exist_ok=True)
        
        # Save models and metadata for each equipment type
        for eq_type, model in self.models.items():
            # Save model
            model_path = os.path.join(directory, f"consumption_model_{eq_type}.joblib")
            joblib.dump(model, model_path)
            
            # Save model features if available
            if eq_type in self.model_features:
                features_path = os.path.join(directory, f"consumption_features_{eq_type}.joblib")
                joblib.dump(self.model_features[eq_type], features_path)
            
            print(f"Model for {eq_type} saved to {model_path}")
        
        # Save categorical values
        if self.categorical_values:
            categorical_path = os.path.join(directory, "consumption_categorical_values.joblib")
            joblib.dump(self.categorical_values, categorical_path)
    
    def load_models(self, directory='models'):
        """Load trained models from disk."""
        # Get all model files
        model_files = [f for f in os.listdir(directory) if f.startswith("consumption_model_") and f.endswith(".joblib")]
        
        for model_file in model_files:
            # Extract equipment type from filename
            eq_type = model_file.replace("consumption_model_", "").replace(".joblib", "")
            
            # Load model
            model_path = os.path.join(directory, model_file)
            self.models[eq_type] = joblib.load(model_path)
            
            # Load feature list if available
            features_path = os.path.join(directory, f"consumption_features_{eq_type}.joblib")
            if os.path.exists(features_path):
                self.model_features[eq_type] = joblib.load(features_path)
            
            # Extract scaler from pipeline
            self.scalers[eq_type] = self.models[eq_type].named_steps['scaler']
            
            print(f"Model for {eq_type} loaded from {model_path}")
        
        # Load categorical values if available
        categorical_path = os.path.join(directory, "consumption_categorical_values.joblib")
        if os.path.exists(categorical_path):
            self.categorical_values = joblib.load(categorical_path)
