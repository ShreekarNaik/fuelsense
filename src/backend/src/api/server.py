# api/server.py
from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import json
import os
import joblib
import sys
from datetime import datetime, timedelta
from flasgger import Swagger, swag_from

# Add the parent directory (src) to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import our models
from models.consumption_predictor import FuelConsumptionPredictor
from models.anomaly_detector import FuelAnomalyDetector
from models.insight_generator import InsightGenerator

app = Flask(__name__)

# Configure Swagger UI
swagger_config = {
    "headers": [],
    "specs": [
        {
            "endpoint": "apispec",
            "route": "/apispec.json",
            "rule_filter": lambda rule: True,
            "model_filter": lambda tag: True,
        }
    ],
    "static_url_path": "/flasgger_static",
    "swagger_ui": True,
    "specs_route": "/docs/"
}

swagger_template = {
    "info": {
        "title": "Fuelsense Backend API",
        "description": "API for monitoring and predicting fuel consumption for various equipment types",
        "version": "1.0.0",
        "contact": {
            "email": "support@fuelintelligence.com"
        }
    },
    "tags": [
        {
            "name": "Health",
            "description": "System health endpoints"
        },
        {
            "name": "Equipment",
            "description": "Equipment data and management"
        },
        {
            "name": "Analytics",
            "description": "Predictions, anomalies, and insights"
        },
        {
            "name": "Dashboard",
            "description": "Dashboard data and summaries"
        }
    ]
}

swagger = Swagger(app, config=swagger_config, template=swagger_template)

# Load models
consumption_predictor = FuelConsumptionPredictor()
anomaly_detector = FuelAnomalyDetector()
insight_generator = InsightGenerator()

# Load models if they exist
try:
    consumption_predictor.load_models(directory='models')
    anomaly_detector.load_models(directory='models')
    print("Models loaded successfully")
except Exception as e:
    print(f"Error loading models: {e}")
    print("Models will be initialized with default parameters")

# Define dataset path once
DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data', 'FuelIntel_Final_Dataset.csv')

@app.route('/', methods=['GET'])
@swag_from({
    "tags": ["Documentation"],
    "summary": "API Documentation",
    "description": "Returns basic information about the API endpoints or redirects to Swagger UI documentation",
    "responses": {
        "200": {
            "description": "API Information",
            "content": {
                "application/json": {
                    "schema": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string", "example": "Fuelsense Backend API"},
                            "version": {"type": "string", "example": "1.0.0"},
                            "description": {"type": "string"},
                            "endpoints": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "path": {"type": "string", "example": "/api/health"},
                                        "description": {"type": "string", "example": "Health check endpoint"}
                                    }
                                }
                            }
                        }
                    }
                }
            }
        },
        "302": {
            "description": "Redirect to Swagger UI documentation"
        }
    }
})
def index():
    """Documentation page for the Fuelsense Backend API."""
    # Redirect to Swagger UI documentation
    if request.headers.get('Accept', '').find('application/json') < 0:
        return app.redirect('/docs/')
        
    # Otherwise return API information as JSON
    docs = {
        "name": "Fuelsense Backend API",
        "version": "1.0.0",
        "description": "API for monitoring and predicting fuel consumption for various equipment types",
        "endpoints": [
            {"path": "/api/health", "description": "Health check endpoint"},
            {"path": "/api/equipment", "description": "Get list of all equipment"},
            {"path": "/api/equipment/<equipment_id>", "description": "Get detailed data for a specific equipment"},
            {"path": "/api/predictions", "description": "Get fuel consumption predictions for all equipment"},
            {"path": "/api/anomalies", "description": "Get detected anomalies for all equipment"},
            {"path": "/api/insights", "description": "Get actionable insights for all equipment"},
            {"path": "/api/dashboard", "description": "Get summary data for the dashboard"},
            {"path": "/api/consumption/history", "description": "Get historical consumption data for all equipment"}
        ]
    }
    return jsonify(docs), 200

@app.route('/api/health', methods=['GET'])
@swag_from({
    "tags": ["Health"],
    "summary": "Health check endpoint",
    "description": "Returns the current status of the API and timestamp",
    "responses": {
        "200": {
            "description": "API is healthy",
            "content": {
                "application/json": {
                    "schema": {
                        "type": "object",
                        "properties": {
                            "status": {"type": "string", "example": "ok"},
                            "timestamp": {"type": "string", "format": "date-time", "example": "2023-04-01T12:00:00Z"}
                        }
                    }
                }
            }
        }
    }
})
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'ok',
        'timestamp': datetime.now().isoformat()
    }), 200

@app.route('/api/equipment', methods=['GET'])
@swag_from({
    "tags": ["Equipment"],
    "summary": "Get list of all equipment",
    "description": "Returns a list of all equipment with their basic information",
    "responses": {
        "200": {
            "description": "List of equipment",
            "content": {
                "application/json": {
                    "schema": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "Equipment_ID": {"type": "string", "example": "Forklift_001"},
                                "Equipment_Type": {"type": "string", "example": "Forklift"},
                                "Driver_ID": {"type": "string", "example": "D001"},
                                "Fuel_Level": {"type": "number", "example": 35.5},
                                "Fuel_Capacity": {"type": "number", "example": 50.0},
                                "Operational_Status": {"type": "string", "example": "Running"},
                                "Maintenance_Status": {"type": "string", "example": "Normal"},
                                "Location": {"type": "string", "example": "Warehouse A"}
                            }
                        }
                    }
                }
            }
        },
        "500": {
            "description": "Internal server error"
        }
    }
})
def get_equipment_list():
    """Get list of equipment."""
    # Load the latest data
    try:
        data = pd.read_csv(DATA_PATH)
        
        # Extract unique equipment data
        equipment_list = []
        for eq_id, eq_data in data.groupby('Equipment_ID'):
            latest = eq_data.iloc[-1]
            equipment_list.append({
                'Equipment_ID': eq_id,
                'Equipment_Type': latest['Equipment_Type'],
                'Driver_ID': latest['Driver_ID'],
                'Fuel_Level': latest['Fuel_Level (L)'],
                'Fuel_Capacity': latest['Fuel_Tank_Capacity (L)'],
                'Operational_Status': latest['Operational_Status'],
                'Maintenance_Status': latest['Maintenance_Status'],
                'Location': latest['Location']
            })
        
        # Replace NaNs with "Data Not Available"
        for equipment in equipment_list:
            for key, value in equipment.items():
                if pd.isna(value) or pd.isnull(value):
                    equipment[key] = "Data Not Available"

        
        return jsonify(equipment_list), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/equipment/<equipment_id>', methods=['GET'])
@swag_from({
    "tags": ["Equipment"],
    "summary": "Get detailed data for a specific equipment",
    "description": "Returns detailed information and historical data for the specified equipment",
    "parameters": [
        {
            "name": "equipment_id",
            "in": "path",
            "required": True,
            "description": "Unique identifier for the equipment",
            "schema": {
                "type": "string"
            },
            "example": "Forklift_001"
        }
    ],
    "responses": {
        "200": {
            "description": "Detailed equipment data",
            "content": {
                "application/json": {
                    "schema": {
                        "type": "object",
                        "properties": {
                            "latest": {
                                "type": "object",
                                "description": "Latest status of the equipment"
                            },
                            "history": {
                                "type": "array",
                                "description": "Historical data points",
                                "items": {
                                    "type": "object"
                                }
                            }
                        }
                    }
                }
            }
        },
        "404": {
            "description": "Equipment not found"
        },
        "500": {
            "description": "Internal server error"
        }
    }
})
def get_equipment_details(equipment_id):
    """Get detailed data for a specific equipment."""
    try:
        data = pd.read_csv(DATA_PATH)
        
        # Filter data for this equipment
        eq_data = data[data['Equipment_ID'] == equipment_id]
        
        if eq_data.empty:
            return jsonify({'error': 'Equipment ID not found'}), 404
        
        # Get latest record
        latest = eq_data.iloc[-1].to_dict()
        
        # Get historical data
        history = eq_data[['Timestamp', 'Fuel_Consumed (L/hr)', 'Fuel_Level (L)', 
                          'Fuel_Efficiency (km/L)', 'Load (%)', 'Speed (km/h)']].to_dict('records')
        
        # Combine data
        result = {
            'latest': latest,
            'history': history
        }
        


        # Replace NaNs with "Data Not Available"
        for key, value in result['latest'].items():
            if pd.isna(value) or pd.isnull(value):
                result['latest'][key] = "Data Not Available"
        
        
        return jsonify(result), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/predictions', methods=['GET'])
@swag_from({
    "tags": ["Analytics"],
    "summary": "Get fuel consumption predictions",
    "description": "Returns predicted fuel consumption for all equipment for the next 24 hours",
    "responses": {
        "200": {
            "description": "Fuel consumption predictions",
            "content": {
                "application/json": {
                    "schema": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "Equipment_ID": {"type": "string", "example": "Forklift_001"},
                                "Equipment_Type": {"type": "string", "example": "Forklift"},
                                "predictions": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "Timestamp": {"type": "string", "format": "date-time"},
                                            "Predicted_Consumption": {"type": "number", "example": 6.5}
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        },
        "500": {
            "description": "Internal server error"
        }
    }
})
def get_predictions():
    """Get fuel consumption predictions for all equipment."""
    try:
        # Load the latest data
        data = pd.read_csv(DATA_PATH)
        data['Timestamp'] = pd.to_datetime(data['Timestamp'])
        
        # Get the latest data point for each equipment
        latest_data = data.groupby('Equipment_ID').apply(
            lambda x: x.sort_values('Timestamp').iloc[-1]
        ).reset_index(drop=True)
        
        # Generate predictions
        all_predictions = []
        
        # For each equipment ID in the data
        for _, row in latest_data.iterrows():
            equipment_id = row['Equipment_ID']
            equipment_type = row['Equipment_Type']
            base_time = row['Timestamp']
            
            # Create simulated predictions for next 24 hours
            pred_list = []
            for hour in range(1, 25):
                # Simulate predictions for each hour
                future_time = base_time + pd.Timedelta(hours=hour)
                
                # Different consumption values for each equipment type
                if equipment_type == 'Generator':
                    consumption = 7.5
                elif equipment_type == 'Truck':
                    consumption = 12.0
                elif equipment_type == 'Excavator':
                    consumption = 5.0
                else:  # Forklift or any other
                    # Apply some variability for Forklift
                    base_consumption = 6.5
                    # Add some randomness
                    consumption = base_consumption + np.random.normal(0, 0.5)
                
                pred_list.append({
                    'Equipment_ID': equipment_id,
                    'Equipment_Type': equipment_type,
                    'Timestamp': future_time.isoformat(),
                    'Predicted_Consumption': round(consumption, 2)
                })
            
            # Add to all predictions
            all_predictions.extend(pred_list)
        
        # Convert to DataFrame
        predictions = pd.DataFrame(all_predictions)
        
        # Format predictions for response
        result = []
        if not predictions.empty:
            for eq_id, pred_data in predictions.groupby('Equipment_ID'):
                result.append({
                    'Equipment_ID': eq_id,
                    'Equipment_Type': pred_data['Equipment_Type'].iloc[0],
                    'predictions': pred_data[['Timestamp', 'Predicted_Consumption']].to_dict('records')
                })
        
        return jsonify(result), 200
    except Exception as e:
        print(f"Error in predictions: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/anomalies', methods=['GET'])
@swag_from({
    "tags": ["Analytics"],
    "summary": "Get detected anomalies",
    "description": "Returns detected anomalies for all equipment from the past 48 hours",
    "responses": {
        "200": {
            "description": "Detected anomalies",
            "content": {
                "application/json": {
                    "schema": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "Equipment_ID": {"type": "string", "example": "Forklift_001"},
                                "Equipment_Type": {"type": "string", "example": "Forklift"},
                                "anomalies": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "Timestamp": {"type": "string", "format": "date-time"},
                                            "Anomaly_Score": {"type": "number", "example": 0.85},
                                            "Anomaly_Reason": {"type": "string", "example": "Unusual fuel consumption spike"},
                                            "Fuel_Consumed (L/hr)": {"type": "number", "example": 12.5},
                                            "Fuel_Efficiency (km/L)": {"type": "number", "example": 3.2},
                                            "Location": {"type": "string", "example": "Warehouse A"}
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        },
        "500": {
            "description": "Internal server error"
        }
    }
})
def get_anomalies():
    """Get detected anomalies for all equipment."""
    try:
        # Load the latest data
        data = pd.read_csv(DATA_PATH)
        data['Timestamp'] = pd.to_datetime(data['Timestamp'])
        
        # Get recent data (last 48 hours)
        max_time = data['Timestamp'].max()
        recent_data = data[data['Timestamp'] >= max_time - timedelta(hours=48)]
        
        # Create simulated anomalies
        np.random.seed(42)  # For reproducibility
        
        # Create a copy of the data for anomalies
        anomaly_results = recent_data.copy()
        
        # Randomly mark ~5% of data points as anomalies
        anomaly_results['Anomaly_Detected'] = np.random.choice(
            [0, 1], 
            size=len(anomaly_results), 
            p=[0.95, 0.05]
        )
        
        # Add anomaly scores
        anomaly_results['Anomaly_Score'] = np.random.uniform(0, 1, size=len(anomaly_results))
        anomaly_results.loc[anomaly_results['Anomaly_Detected'] == 1, 'Anomaly_Score'] += 0.5
        anomaly_results.loc[anomaly_results['Anomaly_Detected'] == 0, 'Anomaly_Score'] *= 0.7
        anomaly_results['Anomaly_Score'] = anomaly_results['Anomaly_Score'].clip(0, 1)
        anomaly_results['Anomaly_Score'] = anomaly_results['Anomaly_Score'].round(2)
        
        # Add reasons
        reasons = [
            "Unusual fuel consumption spike",
            "Irregular usage pattern",
            "Potential leak detected",
            "Efficiency below normal range",
            "Unusual operational hours"
        ]
        
        anomaly_results['Anomaly_Reason'] = ""
        for idx in anomaly_results[anomaly_results['Anomaly_Detected'] == 1].index:
            anomaly_results.at[idx, 'Anomaly_Reason'] = np.random.choice(reasons)
        
        # Filter to only anomalies
        anomalies = anomaly_results[anomaly_results['Anomaly_Detected'] == 1]
        
        # Format anomalies for response
        result = []
        for eq_id, anomaly_data in anomalies.groupby('Equipment_ID'):
            result.append({
                'Equipment_ID': eq_id,
                'Equipment_Type': anomaly_data['Equipment_Type'].iloc[0],
                'anomalies': anomaly_data[[
                    'Timestamp', 'Anomaly_Score', 'Anomaly_Reason',
                    'Fuel_Consumed (L/hr)', 'Fuel_Efficiency (km/L)', 
                    'Location'
                ]].to_dict('records')
            })
        
        return jsonify(result), 200
    except Exception as e:
        print(f"Error in anomalies: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/insights', methods=['GET'])
@swag_from({
    "tags": ["Analytics"],
    "summary": "Get actionable insights",
    "description": "Returns actionable insights for all equipment based on consumption patterns and anomalies",
    "responses": {
        "200": {
            "description": "Equipment insights",
            "content": {
                "application/json": {
                    "schema": {
                        "type": "object",
                        "additionalProperties": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "type": {"type": "string", "example": "consumption_trend"},
                                    "equipment_id": {"type": "string", "example": "Forklift_001"},
                                    "equipment_type": {"type": "string", "example": "Forklift"},
                                    "message": {"type": "string", "example": "Forklift_001 shows stable fuel consumption patterns over the observed period."},
                                    "timestamp": {"type": "string", "format": "date-time"},
                                    "priority": {"type": "string", "enum": ["low", "medium", "high"], "example": "medium"}
                                }
                            }
                        }
                    }
                }
            }
        },
        "500": {
            "description": "Internal server error"
        }
    }
})
def get_insights():
    """Get actionable insights for all equipment."""
    try:
        # Load the latest data
        data = pd.read_csv(DATA_PATH)
        data['Timestamp'] = pd.to_datetime(data['Timestamp'])
        
        # Create simulated insights
        insights = []
        
        # Get unique equipment IDs
        equipment_ids = data['Equipment_ID'].unique()
        
        # Create different types of insights
        insight_types = [
            {
                "type": "consumption_trend",
                "message_template": "{equipment_id} shows {trend} fuel consumption patterns over the observed period."
            },
            {
                "type": "anomaly_alert",
                "message_template": "Potential fuel efficiency issue detected for {equipment_id}."
            },
            {
                "type": "efficiency_recommendation",
                "message_template": "Consider scheduling maintenance for {equipment_id} to improve fuel efficiency."
            },
            {
                "type": "refueling_prediction",
                "message_template": "{equipment_id} will need refueling within the next {hours} hours."
            },
            {
                "type": "maintenance_reminder",
                "message_template": "{equipment_id} is due for scheduled maintenance in {days} days."
            }
        ]
        
        # Generate insights for each equipment
        for equipment_id in equipment_ids:
            # Get equipment type
            equipment_type = data[data['Equipment_ID'] == equipment_id]['Equipment_Type'].iloc[0]
            
            # Generate 1-3 random insights per equipment
            num_insights = np.random.randint(1, 4)
            selected_types = np.random.choice(range(len(insight_types)), num_insights, replace=False)
            
            for type_idx in selected_types:
                insight_type = insight_types[type_idx]
                
                # Create insight with random parameters
                if insight_type["type"] == "consumption_trend":
                    trend = np.random.choice(["stable", "increasing", "decreasing"])
                    message = insight_type["message_template"].format(equipment_id=equipment_id, trend=trend)
                elif insight_type["type"] == "refueling_prediction":
                    hours = np.random.randint(4, 24)
                    message = insight_type["message_template"].format(equipment_id=equipment_id, hours=hours)
                elif insight_type["type"] == "maintenance_reminder":
                    days = np.random.randint(1, 14)
                    message = insight_type["message_template"].format(equipment_id=equipment_id, days=days)
                else:
                    message = insight_type["message_template"].format(equipment_id=equipment_id)
                
                insights.append({
                    "type": insight_type["type"],
                    "equipment_id": equipment_id,
                    "equipment_type": equipment_type,
                    "message": message,
                    "timestamp": datetime.now().isoformat(),
                    "priority": np.random.choice(["low", "medium", "high"])
                })
        
        # Group insights by equipment
        result = {}
        for insight in insights:
            eq_id = insight['equipment_id']
            if eq_id not in result:
                result[eq_id] = []
            result[eq_id].append(insight)
        
        return jsonify(result), 200
    except Exception as e:
        print(f"Error in insights: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/dashboard', methods=['GET'])
@swag_from({
    "tags": ["Dashboard"],
    "summary": "Get dashboard summary data",
    "description": "Returns summary statistics and equipment status for the dashboard",
    "responses": {
        "200": {
            "description": "Dashboard data",
            "content": {
                "application/json": {
                    "schema": {
                        "type": "object",
                        "properties": {
                            "summary": {
                                "type": "object",
                                "properties": {
                                    "total_equipment": {"type": "integer", "example": 10},
                                    "active_equipment": {"type": "integer", "example": 7},
                                    "maintenance_needed": {"type": "integer", "example": 2},
                                    "low_fuel_count": {"type": "integer", "example": 3},
                                    "total_fuel_consumed": {"type": "number", "example": 520.5},
                                    "anomaly_count_24h": {"type": "integer", "example": 5}
                                }
                            },
                            "equipment": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "Equipment_ID": {"type": "string", "example": "Forklift_001"},
                                        "Equipment_Type": {"type": "string", "example": "Forklift"},
                                        "Operational_Status": {"type": "string", "example": "Running"},
                                        "Fuel_Level_Pct": {"type": "number", "example": 72.5},
                                        "Location": {"type": "string", "example": "Warehouse A"},
                                        "Maintenance_Status": {"type": "string", "example": "Normal"}
                                    }
                                }
                            },
                            "last_updated": {"type": "string", "format": "date-time"}
                        }
                    }
                }
            }
        },
        "500": {
            "description": "Internal server error"
        }
    }
})
def get_dashboard_data():
    """Get summary data for the dashboard."""
    try:
        # Load the latest data
        data = pd.read_csv(DATA_PATH)
        data['Timestamp'] = pd.to_datetime(data['Timestamp'])
        
        # Get the most recent timestamp
        max_time = data['Timestamp'].max()
        
        # Get the latest data for each equipment
        latest_data = data.groupby('Equipment_ID').apply(
            lambda x: x.sort_values('Timestamp').iloc[-1]
        ).reset_index(drop=True)
        
        # Calculate summary statistics
        total_equipment = len(latest_data)
        active_equipment = sum(latest_data['Operational_Status'] == 'Running')
        maintenance_needed = sum(latest_data['Maintenance_Status'] == 'Needs Service')
        low_fuel = sum(latest_data['Fuel_Level (L)'] / latest_data['Fuel_Tank_Capacity (L)'] < 0.2)
        
        # Calculate total fuel consumed in the last 24 hours
        recent_data = data[data['Timestamp'] >= max_time - timedelta(hours=24)]
        total_fuel_consumed = recent_data['Fuel_Consumed (L/hr)'].sum()
        
        # Simulate anomalies
        np.random.seed(42)  # For reproducibility
        num_anomalies = round(len(recent_data) * 0.05)  # About 5% of data points
        
        # Format equipment summary data
        equipment_summary = []
        for _, row in latest_data.iterrows():
            equipment_summary.append({
                'Equipment_ID': row['Equipment_ID'],
                'Equipment_Type': row['Equipment_Type'],
                'Operational_Status': row['Operational_Status'],
                'Fuel_Level_Pct': round(row['Fuel_Level (L)'] / row['Fuel_Tank_Capacity (L)'] * 100, 1),
                'Location': row['Location'],
                'Maintenance_Status': row['Maintenance_Status']
            })
        
        # Create response
        result = {
            'summary': {
                'total_equipment': int(total_equipment),
                'active_equipment': int(active_equipment),
                'maintenance_needed': int(maintenance_needed),
                'low_fuel_count': int(low_fuel),
                'total_fuel_consumed': round(float(total_fuel_consumed), 2),
                'anomaly_count_24h': num_anomalies
            },
            'equipment': equipment_summary,
            'last_updated': max_time.isoformat()
        }
        
        return jsonify(result), 200
    except Exception as e:
        print(f"Error in dashboard: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/consumption/history', methods=['GET'])
@swag_from({
    "tags": ["Analytics"],
    "summary": "Get historical consumption data",
    "description": "Returns historical fuel consumption data for all equipment types",
    "parameters": [
        {
            "name": "days",
            "in": "query",
            "required": False,
            "description": "Number of days of history to return (default: 7)",
            "schema": {
                "type": "integer",
                "minimum": 1,
                "maximum": 30,
                "default": 7
            }
        }
    ],
    "responses": {
        "200": {
            "description": "Historical consumption data",
            "content": {
                "application/json": {
                    "schema": {
                        "type": "object",
                        "additionalProperties": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "Date": {"type": "string", "format": "date"},
                                    "Fuel_Consumed (L/hr)": {"type": "number", "example": 5.2}
                                }
                            }
                        }
                    }
                }
            }
        },
        "500": {
            "description": "Internal server error"
        }
    }
})
def get_consumption_history():
    """Get historical consumption data for all equipment."""
    try:
        # Load the data
        data = pd.read_csv(DATA_PATH)
        data['Timestamp'] = pd.to_datetime(data['Timestamp'])
        
        # Get the time range parameter
        days = int(request.args.get('days', 7))
        max_time = data['Timestamp'].max()
        filter_time = max_time - timedelta(days=days)
        
        # Filter data by time
        filtered_data = data[data['Timestamp'] >= filter_time]
        
        # Group and aggregate by equipment type and day
        filtered_data['Date'] = filtered_data['Timestamp'].dt.date
        consumption_by_type = filtered_data.groupby(['Equipment_Type', 'Date'])['Fuel_Consumed (L/hr)'].mean().reset_index()
        
        # Format for response
        result = {}
        for eq_type, type_data in consumption_by_type.groupby('Equipment_Type'):
            result[eq_type] = type_data[['Date', 'Fuel_Consumed (L/hr)']].to_dict('records')
        
        return jsonify(result), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
