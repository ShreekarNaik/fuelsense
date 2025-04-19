"""Utility functions for models."""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json

def normalize_equipment_type(eq_type):
    """Normalize equipment type strings for consistent mapping."""
    eq_type = str(eq_type).lower()
    
    if 'bulldozer' in eq_type:
        return 'Bulldozer'
    elif 'excavator' in eq_type:
        return 'Excavator'
    elif 'forklift' in eq_type:
        return 'Forklift'
    elif 'loader' in eq_type:
        return 'Loader'
    elif 'truck' in eq_type:
        return 'Truck'
    elif 'crane' in eq_type:
        return 'Crane'
    elif 'generator' in eq_type:
        return 'Generator'
    elif 'tractor' in eq_type:
        return 'Tractor'
    elif 'compactor' in eq_type:
        return 'Compactor'
    else:
        return 'Other'

def calculate_fuel_efficiency(distance, consumption):
    """Calculate fuel efficiency in km/L."""
    if consumption > 0 and distance > 0:
        return distance / consumption
    else:
        return 0

def estimate_runtime(fuel_level, consumption_rate):
    """Estimate runtime in hours based on fuel level and consumption rate."""
    if consumption_rate > 0:
        return fuel_level / consumption_rate
    else:
        return float('inf')  # Infinite runtime if consumption is zero

def categorize_anomaly(anomaly_type, anomaly_score):
    """Categorize anomaly severity based on type and score."""
    if anomaly_score >= 0.8:
        severity = 'critical'
    elif anomaly_score >= 0.6:
        severity = 'high'
    elif anomaly_score >= 0.4:
        severity = 'medium'
    else:
        severity = 'low'
    
    # Adjust based on anomaly type
    if 'Theft' in anomaly_type or 'Oil' in anomaly_type or 'Overheat' in anomaly_type:
        severity = max(severity, 'high')  # These types are at least high severity
    
    return severity

def format_insight_message(insight_type, data):
    """Format insight message based on type and data."""
    if insight_type == 'efficiency':
        efficiency = data.get('avg_efficiency', 0)
        ratio = data.get('ratio', 0)
        
        if ratio >= 0.8:
            return f"Good fuel efficiency. Current average: {efficiency:.2f} km/L."
        elif ratio >= 0.6:
            return f"Average fuel efficiency. Current average: {efficiency:.2f} km/L."
        else:
            return f"Poor fuel efficiency. Current average: {efficiency:.2f} km/L."
    
    elif insight_type == 'maintenance':
        hours = data.get('hours_since_service', 0)
        threshold = data.get('threshold', 500)
        remaining = data.get('hours_remaining', threshold - hours)
        
        if hours >= threshold:
            return f"Maintenance overdue. Hours since last service: {hours:.1f}"
        else:
            return f"Maintenance due soon. Hours remaining: {remaining:.1f}"
    
    elif insight_type == 'refueling':
        fuel_pct = data.get('fuel_percentage', 0)
        runtime = data.get('estimated_runtime', 0)
        
        if fuel_pct <= 10:
            return f"Critical fuel level. Estimated runtime: {runtime:.1f} hours."
        elif fuel_pct <= 20:
            return f"Low fuel alert. Estimated runtime: {runtime:.1f} hours."
        else:
            return f"Plan refueling soon. Estimated runtime: {runtime:.1f} hours."
    
    elif insight_type == 'anomaly':
        subtype = data.get('subtype', 'general')
        count = data.get('anomaly_count', 1)
        
        if subtype == 'fuel_theft':
            total = data.get('total_missing', 0)
            return f"Potential fuel theft detected. Total missing: {total:.1f} L"
        elif subtype == 'efficiency':
            return f"Abnormal fuel efficiency detected. {count} instances."
        elif subtype == 'idle':
            return f"Excessive idle time detected. {count} instances."
        elif subtype == 'mechanical':
            return f"Mechanical issue detected. {count} instances."
        else:
            return f"Anomaly detected. {count} instances."
    
    else:
        return "Insight generated."

def get_trend_direction(values, min_change=0.05):
    """Determine trend direction from a series of values."""
    if not values or len(values) < 2:
        return "stable"
    
    first = values[0]
    last = values[-1]
    
    if abs(last - first) < min_change * first:
        return "stable"
    elif last > first:
        return "increasing"
    else:
        return "decreasing"
