# insight_generator.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os

class InsightGenerator:
    """Generates actionable insights from equipment data and predictions."""
    
    def __init__(self, config_file=None):
        """Initialize the insight generator.
        
        Args:
            config_file (str): Path to configuration file with thresholds and rules
        """
        self.config = self.load_config(config_file)
        self.equipment_data = {}
        self.insights = []
    
    def load_config(self, config_file):
        """Load configuration from file or use defaults."""
        if config_file and os.path.exists(config_file):
            with open(config_file, 'r') as f:
                return json.load(f)
        else:
            # Default configuration
            return {
                'efficiency_thresholds': {
                    'Bulldozer': {'good': 0.8, 'average': 0.6, 'poor': 0.4},
                    'Excavator': {'good': 0.8, 'average': 0.6, 'poor': 0.4},
                    'Forklift': {'good': 0.85, 'average': 0.7, 'poor': 0.5},
                    'Loader': {'good': 0.8, 'average': 0.6, 'poor': 0.4},
                    'Truck': {'good': 0.85, 'average': 0.7, 'poor': 0.5},
                    'default': {'good': 0.8, 'average': 0.6, 'poor': 0.4}
                },
                'maintenance_thresholds': {
                    'service_due_hours': 500,
                    'service_warning_hours': 450
                },
                'idle_thresholds': {
                    'excessive_idle_minutes': 20,
                    'daily_idle_ratio': 0.3
                },
                'refuel_thresholds': {
                    'low_fuel_percentage': 20,
                    'optimal_refuel_percentage': 25
                }
            }
    
    def process_data(self, current_data, historical_data=None, predictions=None, anomalies=None):
        """Process current and historical data to update equipment information.
        
        Args:
            current_data (DataFrame): Current equipment data
            historical_data (DataFrame): Historical equipment data (optional)
            predictions (DataFrame): Fuel consumption predictions (optional)
            anomalies (DataFrame): Detected anomalies (optional)
        """
        # Process current data
        for eq_id, eq_data in current_data.groupby('Equipment_ID'):
            # Get the latest data point for this equipment
            latest = eq_data.sort_values('Timestamp').iloc[-1]
            
            # Initialize equipment record if not exists
            if eq_id not in self.equipment_data:
                self.equipment_data[eq_id] = {
                    'Equipment_ID': eq_id,
                    'Equipment_Type': latest['Equipment_Type'],
                    'last_update': latest['Timestamp'],
                    'fuel_level': latest['Fuel_Level (L)'],
                    'fuel_capacity': latest['Fuel_Tank_Capacity (L)'],
                    'fuel_level_percentage': (latest['Fuel_Level (L)'] / latest['Fuel_Tank_Capacity (L)'] * 100),
                    'operational_status': latest['Operational_Status'],
                    'maintenance_status': latest['Maintenance_Status'],
                    'time_since_service': latest['Time_Since_Last_Service (hrs)'],
                    'total_idle_time': 0,
                    'total_active_time': 0,
                    'consumption_history': [],
                    'efficiency_history': [],
                    'anomaly_history': [],
                    'last_refuel': None,
                    'predicted_consumption': None,
                    'estimated_runtime': None,
                    'insights': []
                }
            
            # Update equipment data
            self.equipment_data[eq_id]['last_update'] = latest['Timestamp']
            self.equipment_data[eq_id]['fuel_level'] = latest['Fuel_Level (L)']
            self.equipment_data[eq_id]['fuel_level_percentage'] = (
                latest['Fuel_Level (L)'] / latest['Fuel_Tank_Capacity (L)'] * 100
            )
            self.equipment_data[eq_id]['operational_status'] = latest['Operational_Status']
            self.equipment_data[eq_id]['maintenance_status'] = latest['Maintenance_Status']
            self.equipment_data[eq_id]['time_since_service'] = latest['Time_Since_Last_Service (hrs)']
            
            # Track idle and active time
            if latest['Operational_Status'] == 'Running':
                if latest['Idle_Time (min)'] > 0:
                    self.equipment_data[eq_id]['total_idle_time'] += 1
                else:
                    self.equipment_data[eq_id]['total_active_time'] += 1
            
            # Track consumption and efficiency
            if 'Fuel_Consumed (L/hr)' in latest and latest['Fuel_Consumed (L/hr)'] > 0:
                self.equipment_data[eq_id]['consumption_history'].append({
                    'timestamp': latest['Timestamp'],
                    'consumption': latest['Fuel_Consumed (L/hr)']
                })
            
            if 'Fuel_Efficiency (km/L)' in latest and latest['Fuel_Efficiency (km/L)'] > 0:
                self.equipment_data[eq_id]['efficiency_history'].append({
                    'timestamp': latest['Timestamp'],
                    'efficiency': latest['Fuel_Efficiency (km/L)']
                })
            
            # Track refueling events
            if 'Refuel_Event' in latest and latest['Refuel_Event'] == 'Yes':
                self.equipment_data[eq_id]['last_refuel'] = latest['Timestamp']
        
        # Process historical data
        if historical_data is not None:
            # Add insights based on historical trends
            pass
        
        # Process predictions
        if predictions is not None:
            for eq_id, pred_data in predictions.groupby('Equipment_ID'):
                if eq_id in self.equipment_data:
                    # Store predicted consumption
                    self.equipment_data[eq_id]['predicted_consumption'] = pred_data.to_dict('records')
                    
                    # Calculate estimated runtime based on fuel level and predicted consumption
                    if len(pred_data) > 0:
                        avg_consumption = pred_data['Predicted_Consumption'].mean()
                        if avg_consumption > 0:
                            hours_remaining = self.equipment_data[eq_id]['fuel_level'] / avg_consumption
                            self.equipment_data[eq_id]['estimated_runtime'] = hours_remaining
        
        # Process anomalies
        if anomalies is not None:
            for eq_id, anomaly_data in anomalies.groupby('Equipment_ID'):
                if eq_id in self.equipment_data:
                    # Store detected anomalies
                    detected = anomaly_data[anomaly_data['Anomaly_Detected']].to_dict('records')
                    self.equipment_data[eq_id]['anomaly_history'].extend(detected)
    
    def generate_efficiency_insights(self):
        """Generate insights related to fuel efficiency."""
        for eq_id, equipment in self.equipment_data.items():
            # Skip if not enough efficiency data
            if len(equipment['efficiency_history']) < 5:
                continue
            
            # Get equipment type and relevant thresholds
            eq_type = equipment['Equipment_Type']
            thresholds = self.config['efficiency_thresholds'].get(
                eq_type, self.config['efficiency_thresholds']['default']
            )
            
            # Calculate average efficiency
            efficiencies = [e['efficiency'] for e in equipment['efficiency_history']]
            avg_efficiency = sum(efficiencies) / len(efficiencies)
            
            # Get baseline efficiency for this equipment type (can be extended with more sophisticated baseline)
            baseline_efficiency = 5.0  # Default baseline (km/L)
            if eq_type == 'Truck':
                baseline_efficiency = 6.0
            elif eq_type == 'Bulldozer':
                baseline_efficiency = 3.0
            elif eq_type == 'Forklift':
                baseline_efficiency = 8.0
            
            # Calculate efficiency ratio compared to baseline
            efficiency_ratio = avg_efficiency / baseline_efficiency
            
            # Generate insight based on efficiency ratio
            if efficiency_ratio >= thresholds['good']:
                insight = {
                    'type': 'efficiency',
                    'priority': 'low',
                    'equipment_id': eq_id,
                    'timestamp': datetime.now(),
                    'message': f"Good fuel efficiency. Current average: {avg_efficiency:.2f} km/L.",
                    'details': {
                        'avg_efficiency': avg_efficiency,
                        'baseline': baseline_efficiency,
                        'ratio': efficiency_ratio
                    },
                    'actions': [
                        "Maintain current operating practices."
                    ]
                }
            elif efficiency_ratio >= thresholds['average']:
                insight = {
                    'type': 'efficiency',
                    'priority': 'medium',
                    'equipment_id': eq_id,
                    'timestamp': datetime.now(),
                    'message': f"Average fuel efficiency. Current average: {avg_efficiency:.2f} km/L.",
                    'details': {
                        'avg_efficiency': avg_efficiency,
                        'baseline': baseline_efficiency,
                        'ratio': efficiency_ratio
                    },
                    'actions': [
                        "Consider reducing idle time",
                        "Check for proper maintenance"
                    ]
                }
            else:
                insight = {
                    'type': 'efficiency',
                    'priority': 'high',
                    'equipment_id': eq_id,
                    'timestamp': datetime.now(),
                    'message': f"Poor fuel efficiency. Current average: {avg_efficiency:.2f} km/L.",
                    'details': {
                        'avg_efficiency': avg_efficiency,
                        'baseline': baseline_efficiency,
                        'ratio': efficiency_ratio
                    },
                    'actions': [
                        "Schedule maintenance check",
                        "Review operator behavior",
                        "Check for mechanical issues"
                    ]
                }
            
            # Add insight to equipment and global lists
            equipment['insights'].append(insight)
            self.insights.append(insight)
    
    def generate_maintenance_insights(self):
        """Generate insights related to maintenance."""
        for eq_id, equipment in self.equipment_data.items():
            # Get maintenance thresholds
            service_due = self.config['maintenance_thresholds']['service_due_hours']
            service_warning = self.config['maintenance_thresholds']['service_warning_hours']
            
            # Generate insight based on time since last service
            hours_since_service = equipment['time_since_service']
            
            if hours_since_service >= service_due:
                insight = {
                    'type': 'maintenance',
                    'priority': 'high',
                    'equipment_id': eq_id,
                    'timestamp': datetime.now(),
                    'message': f"Maintenance overdue. Hours since last service: {hours_since_service:.1f}",
                    'details': {
                        'hours_since_service': hours_since_service,
                        'threshold': service_due
                    },
                    'actions': [
                        "Schedule immediate maintenance",
                        "Inspect for potential issues"
                    ]
                }
            elif hours_since_service >= service_warning:
                insight = {
                    'type': 'maintenance',
                    'priority': 'medium',
                    'equipment_id': eq_id,
                    'timestamp': datetime.now(),
                    'message': f"Maintenance due soon. Hours since last service: {hours_since_service:.1f}",
                    'details': {
                        'hours_since_service': hours_since_service,
                        'threshold': service_due,
                        'hours_remaining': service_due - hours_since_service
                    },
                    'actions': [
                        "Plan for upcoming maintenance",
                        "Schedule service in the next few days"
                    ]
                }
            else:
                # No immediate maintenance needed
                continue
            
            # Add insight to equipment and global lists
            equipment['insights'].append(insight)
            self.insights.append(insight)
    
    def generate_refueling_insights(self):
        """Generate insights related to refueling."""
        for eq_id, equipment in self.equipment_data.items():
            # Skip if estimated runtime not available
            if not equipment['estimated_runtime']:
                continue
            
            # Get refueling thresholds
            low_fuel_pct = self.config['refuel_thresholds']['low_fuel_percentage']
            optimal_refuel_pct = self.config['refuel_thresholds']['optimal_refuel_percentage']
            
            # Generate insight based on fuel level and estimated runtime
            fuel_pct = equipment['fuel_level_percentage']
            runtime_hrs = equipment['estimated_runtime']
            
            if fuel_pct <= low_fuel_pct:
                insight = {
                    'type': 'refueling',
                    'priority': 'high',
                    'equipment_id': eq_id,
                    'timestamp': datetime.now(),
                    'message': f"Low fuel alert. Estimated runtime: {runtime_hrs:.1f} hours.",
                    'details': {
                        'fuel_percentage': fuel_pct,
                        'estimated_runtime': runtime_hrs,
                        'fuel_level': equipment['fuel_level'],
                        'fuel_capacity': equipment['fuel_capacity']
                    },
                    'actions': [
                        "Refuel immediately",
                        "Plan route to nearest refueling station"
                    ]
                }
                
                # Add insight to equipment and global lists
                equipment['insights'].append(insight)
                self.insights.append(insight)
            elif fuel_pct <= optimal_refuel_pct:
                # Calculate optimal time to refuel
                if runtime_hrs < 8:  # If less than a workday remaining
                    insight = {
                        'type': 'refueling',
                        'priority': 'medium',
                        'equipment_id': eq_id,
                        'timestamp': datetime.now(),
                        'message': f"Plan refueling soon. Estimated runtime: {runtime_hrs:.1f} hours.",
                        'details': {
                            'fuel_percentage': fuel_pct,
                            'estimated_runtime': runtime_hrs,
                            'fuel_level': equipment['fuel_level'],
                            'fuel_capacity': equipment['fuel_capacity']
                        },
                        'actions': [
                            "Schedule refueling today",
                            "Consider refueling at the end of the shift"
                        ]
                    }
                    
                    # Add insight to equipment and global lists
                    equipment['insights'].append(insight)
                    self.insights.append(insight)
    
    def generate_idle_time_insights(self):
        """Generate insights related to idle time."""
        for eq_id, equipment in self.equipment_data.items():
            # Skip if not enough activity data
            total_time = equipment['total_idle_time'] + equipment['total_active_time']
            if total_time < 5:
                continue
            
            # Get idle thresholds
            excessive_idle = self.config['idle_thresholds']['excessive_idle_minutes']
            daily_idle_ratio = self.config['idle_thresholds']['daily_idle_ratio']
            
            # Calculate idle ratio
            idle_ratio = equipment['total_idle_time'] / max(1, total_time)
            
            # Generate insight based on idle ratio
            if idle_ratio > daily_idle_ratio:
                insight = {
                    'type': 'idle',
                    'priority': 'medium',
                    'equipment_id': eq_id,
                    'timestamp': datetime.now(),
                    'message': f"Excessive idle time detected. Idle ratio: {idle_ratio:.1%}",
                    'details': {
                        'idle_time': equipment['total_idle_time'],
                        'active_time': equipment['total_active_time'],
                        'idle_ratio': idle_ratio
                    },
                    'actions': [
                        "Review operator behavior",
                        "Consider auto-shutdown when idle",
                        "Optimize work scheduling to reduce idle time"
                    ]
                }
                
                # Add insight to equipment and global lists
                equipment['insights'].append(insight)
                self.insights.append(insight)
    
    def generate_anomaly_insights(self):
        """Generate insights based on detected anomalies."""
        for eq_id, equipment in self.equipment_data.items():
            # Skip if no anomalies detected
            if not equipment['anomaly_history']:
                continue
            
            # Group anomalies by type
            anomaly_types = {}
            for anomaly in equipment['anomaly_history']:
                anomaly_type = anomaly.get('Anomaly_Type', 'Unknown')
                if anomaly_type not in anomaly_types:
                    anomaly_types[anomaly_type] = []
                anomaly_types[anomaly_type].append(anomaly)
            
            # Generate insights for each anomaly type
            for anomaly_type, anomalies in anomaly_types.items():
                # Skip if no specific anomaly type
                if not anomaly_type or anomaly_type == 'Unknown':
                    continue
                
                # Generate specific insight based on anomaly type
                if 'Fuel Theft' in anomaly_type:
                    total_missing = sum([abs(a.get('Fuel_Level_Change', 0)) for a in anomalies])
                    insight = {
                        'type': 'anomaly',
                        'subtype': 'fuel_theft',
                        'priority': 'high',
                        'equipment_id': eq_id,
                        'timestamp': datetime.now(),
                        'message': f"Potential fuel theft detected. Total missing: {total_missing:.1f} L",
                        'details': {
                            'anomaly_count': len(anomalies),
                            'total_missing': total_missing,
                            'locations': list(set([a.get('Location', 'Unknown') for a in anomalies]))
                        },
                        'actions': [
                            "Investigate fuel usage patterns",
                            "Check equipment security",
                            "Review refueling procedures"
                        ]
                    }
                elif 'Efficiency' in anomaly_type:
                    insight = {
                        'type': 'anomaly',
                        'subtype': 'efficiency',
                        'priority': 'medium',
                        'equipment_id': eq_id,
                        'timestamp': datetime.now(),
                        'message': f"Abnormal fuel efficiency detected. {len(anomalies)} instances.",
                        'details': {
                            'anomaly_count': len(anomalies),
                            'affected_timestamps': [a.get('Timestamp') for a in anomalies]
                        },
                        'actions': [
                            "Check for mechanical issues",
                            "Inspect fuel quality",
                            "Review operator behavior"
                        ]
                    }
                elif 'Idle' in anomaly_type:
                    insight = {
                        'type': 'anomaly',
                        'subtype': 'idle',
                        'priority': 'medium',
                        'equipment_id': eq_id,
                        'timestamp': datetime.now(),
                        'message': f"Excessive idle time detected. {len(anomalies)} instances.",
                        'details': {
                            'anomaly_count': len(anomalies),
                            'affected_timestamps': [a.get('Timestamp') for a in anomalies]
                        },
                        'actions': [
                            "Implement idle reduction policies",
                            "Train operators on efficient practices",
                            "Consider auto-shutdown systems"
                        ]
                    }
                elif 'Engine' in anomaly_type or 'Oil' in anomaly_type or 'Sensor' in anomaly_type:
                    insight = {
                        'type': 'anomaly',
                        'subtype': 'mechanical',
                        'priority': 'high',
                        'equipment_id': eq_id,
                        'timestamp': datetime.now(),
                        'message': f"{anomaly_type} detected. {len(anomalies)} instances.",
                        'details': {
                            'anomaly_count': len(anomalies),
                            'affected_timestamps': [a.get('Timestamp') for a in anomalies]
                        },
                        'actions': [
                            "Schedule diagnostic check",
                            "Inspect equipment for issues",
                            "Prepare maintenance"
                        ]
                    }
                else:
                    # Generic anomaly insight
                    insight = {
                        'type': 'anomaly',
                        'subtype': 'general',
                        'priority': 'medium',
                        'equipment_id': eq_id,
                        'timestamp': datetime.now(),
                        'message': f"{anomaly_type} anomaly detected. {len(anomalies)} instances.",
                        'details': {
                            'anomaly_count': len(anomalies),
                            'affected_timestamps': [a.get('Timestamp') for a in anomalies]
                        },
                        'actions': [
                            "Investigate anomaly pattern",
                            "Monitor equipment performance",
                            "Document occurrences"
                        ]
                    }
                
                # Add insight to equipment and global lists
                equipment['insights'].append(insight)
                self.insights.append(insight)
    
    def generate_all_insights(self, current_data, historical_data=None, predictions=None, anomalies=None):
        """Generate all types of insights based on available data.
        
        Args:
            current_data (DataFrame): Current equipment data
            historical_data (DataFrame): Historical equipment data (optional)
            predictions (DataFrame): Fuel consumption predictions (optional)
            anomalies (DataFrame): Detected anomalies (optional)
            
        Returns:
            list: Generated insights
        """
        # Reset insights list
        self.insights = []
        
        # Process input data
        self.process_data(current_data, historical_data, predictions, anomalies)
        
        # Generate different types of insights
        self.generate_efficiency_insights()
        self.generate_maintenance_insights()
        self.generate_refueling_insights()
        self.generate_idle_time_insights()
        self.generate_anomaly_insights()
        
        # Sort insights by priority
        priority_order = {'high': 0, 'medium': 1, 'low': 2}
        self.insights.sort(key=lambda x: (priority_order.get(x['priority'], 3), x['timestamp']), reverse=True)
        
        return self.insights
