import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import zipfile
import io
import os
import json
import random
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import IsolationForest

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

# Create directory for datasets if it doesn't exist
os.makedirs('src/datasets', exist_ok=True)
os.makedirs('data', exist_ok=True)

#######################################################
# PART 1: LOAD REFERENCE DATASETS FROM LOCAL FILES
#######################################################

def load_epa_fuel_economy_data():
    """Load the EPA Fuel Economy dataset from local file."""
    print("Loading EPA Fuel Economy data...")
    
    # Path to the local file
    file_path = "src/datasets/epa_fuel_economy/vehicles.csv"
    
    if os.path.exists(file_path):
        try:
            # Load the CSV file
            epa_data = pd.read_csv(file_path, low_memory=False)
            
            print(f"EPA Fuel Economy data loaded. Shape: {epa_data.shape}")
            return epa_data
        except Exception as e:
            print(f"Error loading EPA data: {e}")
            return create_simulated_epa_data()
    else:
        print(f"EPA Fuel Economy data file not found at {file_path}")
        return create_simulated_epa_data()

def create_simulated_epa_data():
    """Create a simulated EPA Fuel Economy dataset if local file is not available."""
    print("Creating simulated EPA Fuel Economy data...")
    
    # [Same implementation as before]
    # Define vehicle classes and their efficiency characteristics
    vehicle_classes = ['Small Cars', 'Midsize Cars', 'Large Cars', 
                      'Small Pickup Trucks', 'Standard Pickup Trucks', 
                      'Vans, Passenger Type', 'Vans, Cargo Type', 
                      'SUVs', 'Special Purpose Vehicles']
    
    # Base MPG values for each class
    base_mpg = {
        'Small Cars': 40,
        'Midsize Cars': 35,
        'Large Cars': 30,
        'Small Pickup Trucks': 25,
        'Standard Pickup Trucks': 20,
        'Vans, Passenger Type': 22,
        'Vans, Cargo Type': 18,
        'SUVs': 23,
        'Special Purpose Vehicles': 15
    }
    
    # Create rows for the dataset
    rows = []
    model_years = list(range(2015, 2026))
    
    for vclass in vehicle_classes:
        for year in model_years:
            # Number of entries per class/year combination
            num_entries = random.randint(5, 15)
            
            for _ in range(num_entries):
                # Base MPG with some variation
                city_mpg = base_mpg[vclass] * random.uniform(0.85, 1.15) * (1 + 0.01 * (year - 2015))
                hwy_mpg = city_mpg * random.uniform(1.2, 1.4)
                combined_mpg = (city_mpg * 0.55 + hwy_mpg * 0.45)
                
                # Cylinder variations
                if 'Small' in vclass:
                    cylinders = random.choice([3, 4])
                elif 'Midsize' in vclass:
                    cylinders = random.choice([4, 6])
                else:
                    cylinders = random.choice([6, 8])
                
                # Displacement variations
                if cylinders == 3:
                    displacement = random.uniform(1.0, 1.5)
                elif cylinders == 4:
                    displacement = random.uniform(1.5, 2.5)
                elif cylinders == 6:
                    displacement = random.uniform(2.5, 4.0)
                else:  # 8 cylinders
                    displacement = random.uniform(4.0, 6.0)
                
                # Transmission type
                transmission = random.choice(['Automatic', 'Manual', 'CVT'])
                
                # Add row to dataset
                rows.append({
                    'year': year,
                    'make': f"Manufacturer_{random.randint(1, 10)}",
                    'model': f"Model_{random.randint(1, 100)}",
                    'VClass': vclass,
                    'displ': round(displacement, 1),
                    'cylinders': cylinders,
                    'trany': transmission,
                    'city08': round(city_mpg, 1),
                    'highway08': round(hwy_mpg, 1),
                    'comb08': round(combined_mpg, 1),
                    'fuelType': random.choice(['Regular Gasoline', 'Premium Gasoline', 'Diesel'])
                })
    
    # Create dataframe
    epa_data = pd.DataFrame(rows)
    
    # Save simulated dataset
    epa_data.to_csv("src/datasets/epa_fuel_economy_simulated.csv", index=False)
    
    print(f"Simulated EPA Fuel Economy data created. Shape: {epa_data.shape}")
    return epa_data

def load_auto_mpg_dataset():
    """Load the UCI Auto MPG dataset from local file."""
    print("Loading UCI Auto MPG dataset...")
    
    # Path to the local file
    file_path = "src/datasets/auto_mpg/auto-mpg.data"
    
    if os.path.exists(file_path):
        try:
            # Define column names
            columns = ['mpg', 'cylinders', 'displacement', 'horsepower', 'weight',
                      'acceleration', 'model_year', 'origin', 'car_name']
            
            # Load data
            auto_mpg = pd.read_csv(file_path, delim_whitespace=True, names=columns, na_values='?')
            
            # Clean data
            auto_mpg['horsepower'] = pd.to_numeric(auto_mpg['horsepower'], errors='coerce')
            auto_mpg.dropna(inplace=True)
            
            # Add calculated features
            auto_mpg['fuel_efficiency_km_per_l'] = auto_mpg['mpg'] * 0.425  # Convert MPG to km/L
            auto_mpg['weight_kg'] = auto_mpg['weight'] * 0.453592  # Convert lb to kg
            
            # Calculate fuel consumption rate (L/hr) assuming average speed of 60 km/h
            avg_speed = 60  # km/h
            auto_mpg['fuel_consumption_l_per_hr'] = avg_speed / auto_mpg['fuel_efficiency_km_per_l']
            
            print(f"Auto MPG dataset loaded. Shape: {auto_mpg.shape}")
            return auto_mpg
        except Exception as e:
            print(f"Error loading Auto MPG data: {e}")
            return create_simulated_auto_mpg_data()
    else:
        print(f"Auto MPG data file not found at {file_path}")
        return create_simulated_auto_mpg_data()

def create_simulated_auto_mpg_data():
    """Create a simulated Auto MPG dataset if local file is not available."""
    print("Creating simulated Auto MPG data...")
    
    # [Same implementation as before]
    # Define ranges for each feature
    mpg_range = (10, 45)
    cylinders_options = [4, 6, 8]
    displacement_range = (50, 500)
    horsepower_range = (50, 300)
    weight_range = (1500, 5000)
    acceleration_range = (8, 25)
    model_years = list(range(70, 83))
    origin_options = [1, 2, 3]
    
    # Create car names
    manufacturers = ['ford', 'chevrolet', 'plymouth', 'dodge', 'buick', 'pontiac', 
                     'toyota', 'honda', 'datsun', 'volkswagen', 'audi', 'volvo']
    
    # Create rows
    rows = []
    for _ in range(400):  # Create 400 rows to match original dataset size
        cylinders = random.choice(cylinders_options)
        
        # Create relationships between variables
        displacement = cylinders * random.uniform(50, 80)
        horsepower = displacement * random.uniform(0.4, 0.6)
        weight = 1500 + cylinders * random.uniform(300, 500)
        
        # MPG has inverse relationship with weight and cylinders
        base_mpg = 45 - (cylinders - 4) * 5 - (weight - 2000) / 300
        mpg = max(10, min(45, base_mpg * random.uniform(0.85, 1.15)))
        
        # Acceleration has inverse relationship with weight and positive with mpg
        acceleration = 25 - (weight - 1500) / 500 + (mpg - 10) / 10
        acceleration = max(8, min(25, acceleration * random.uniform(0.9, 1.1)))
        
        # Create row
        make = random.choice(manufacturers)
        model = f"model_{random.randint(1, 50)}"
        
        row = {
            'mpg': round(mpg, 1),
            'cylinders': cylinders,
            'displacement': round(displacement, 1),
            'horsepower': round(horsepower, 1),
            'weight': round(weight),
            'acceleration': round(acceleration, 1),
            'model_year': random.choice(model_years),
            'origin': random.choice(origin_options),
            'car_name': f"{make} {model}"
        }
        
        # Add calculated features
        row['fuel_efficiency_km_per_l'] = row['mpg'] * 0.425
        row['weight_kg'] = row['weight'] * 0.453592
        row['fuel_consumption_l_per_hr'] = 60 / row['fuel_efficiency_km_per_l']
        
        rows.append(row)
    
    # Create dataframe
    auto_mpg = pd.DataFrame(rows)
    
    # Save simulated dataset
    auto_mpg.to_csv('src/datasets/auto_mpg_simulated.csv', index=False)
    
    print(f"Simulated Auto MPG data created. Shape: {auto_mpg.shape}")
    return auto_mpg

def load_energy_consumption_dataset():
    """Load the Kaggle Hourly Energy Consumption dataset from local file."""
    print("Loading Energy Consumption dataset...")
    
    # Path to the local file
    file_path = "src/datasets/energy_consumption/AEP_hourly.csv"
    
    if os.path.exists(file_path):
        try:
            # Load the CSV file
            energy_df = pd.read_csv(file_path)
            
            # Process the dataset
            energy_df['Datetime'] = pd.to_datetime(energy_df['Datetime'])
            energy_df['Hour'] = energy_df['Datetime'].dt.hour
            energy_df['Day'] = energy_df['Datetime'].dt.day
            energy_df['Month'] = energy_df['Datetime'].dt.month
            energy_df['Weekday'] = energy_df['Datetime'].dt.weekday
            energy_df['Is_Weekend'] = energy_df['Weekday'].apply(lambda x: 1 if x >= 5 else 0)
            
            # Calculate hourly patterns - normalize by daily max to get percentage patterns
            energy_df['Date'] = energy_df['Datetime'].dt.date
            
            # Group by date and find daily maximum
            daily_max = energy_df.groupby('Date')['AEP_MW'].max().reset_index()
            daily_max.columns = ['Date', 'Daily_Max_MW']
            
            # Merge back with original dataframe
            energy_df = pd.merge(energy_df, daily_max, on='Date')
            
            # Calculate percentage of daily maximum
            energy_df['Consumption_Pct'] = energy_df['AEP_MW'] / energy_df['Daily_Max_MW'] * 100
            
            # Calculate hourly patterns
            hourly_patterns = energy_df.groupby('Hour')['Consumption_Pct'].agg(['mean', 'std']).reset_index()
            hourly_patterns.columns = ['Hour', 'Mean_Consumption_Pct', 'Std_Consumption_Pct']
            
            print(f"Energy Consumption dataset loaded. Shape: {energy_df.shape}")
            return energy_df, hourly_patterns
        except Exception as e:
            print(f"Error loading Energy Consumption data: {e}")
            return create_simulated_energy_data()
    else:
        print(f"Energy Consumption data file not found at {file_path}")
        return create_simulated_energy_data()

def create_simulated_energy_data():
    """Create a simulated hourly energy consumption dataset if local file is not available."""
    print("Creating simulated hourly energy consumption data...")
    
    # [Same implementation as before]
    # Create hourly patterns based on typical daily usage curves
    hours = list(range(24))
    
    # Typical hourly consumption patterns (percentage of peak)
    # These values simulate a typical daily load curve
    hourly_consumption = [
        65, 60, 58, 55, 58, 65,     # 0-5 (early morning)
        75, 85, 95, 98, 100, 98,    # 6-11 (morning to noon)
        95, 93, 90, 92, 95, 98,     # 12-17 (afternoon)
        100, 98, 95, 90, 80, 70     # 18-23 (evening)
    ]
    
    # Standard deviations (higher during transition periods)
    hourly_std = [
        5, 5, 5, 5, 6, 8,           # 0-5
        10, 8, 7, 5, 5, 5,          # 6-11
        5, 5, 5, 7, 9, 10,          # 12-17
        10, 8, 7, 7, 6, 5           # 18-23
    ]
    
    # Create hourly patterns dataframe
    hourly_patterns = pd.DataFrame({
        'Hour': hours,
        'Mean_Consumption_Pct': hourly_consumption,
        'Std_Consumption_Pct': hourly_std
    })
    
    # Generate daily data for a year
    start_date = datetime(2022, 1, 1)
    end_date = datetime(2022, 12, 31, 23)
    
    # Generate hourly timestamps
    dates = pd.date_range(start=start_date, end=end_date, freq='H')
    
    # Base load values with seasonal and weekly patterns
    rows = []
    
    for date in dates:
        hour = date.hour
        day = date.day
        month = date.month
        weekday = date.weekday()
        
        # Base consumption from hourly pattern
        base_hourly = hourly_consumption[hour]
        
        # Add weekday factor (weekends have lower consumption)
        weekday_factor = 1.0 if weekday < 5 else 0.85
        
        # Add seasonal factor (higher in winter and summer)
        if month in [12, 1, 2]:  # Winter
            seasonal_factor = 1.15
        elif month in [6, 7, 8]:  # Summer
            seasonal_factor = 1.10
        else:  # Spring/Fall
            seasonal_factor = 0.95
        
        # Calculate consumption with random variation
        consumption = base_hourly * weekday_factor * seasonal_factor * random.uniform(0.95, 1.05)
        
        # Convert to MW scale (arbitrary scale for simulation)
        aep_mw = consumption * 150
        
        rows.append({
            'Datetime': date,
            'AEP_MW': aep_mw,
            'Hour': hour,
            'Day': day,
            'Month': month,
            'Weekday': weekday,
            'Is_Weekend': 1 if weekday >= 5 else 0,
            'Consumption_Pct': consumption
        })
    
    # Create dataframe
    energy_df = pd.DataFrame(rows)
    
    # Save simulated dataset
    energy_df.to_csv('src/datasets/energy_consumption_simulated.csv', index=False)
    hourly_patterns.to_csv('src/datasets/hourly_patterns.csv', index=False)
    
    print(f"Simulated energy consumption data created. Shape: {energy_df.shape}")
    return energy_df, hourly_patterns

def load_canada_fuel_ratings():
    """Load Canada Fuel Ratings dataset from local file."""
    print("Loading Canada Fuel Ratings data...")
    
    # Path to the local file - try multiple potential filenames
    potential_paths = [
        "src/datasets/canada_fuel_ratings/fuel-consumption-ratings.csv",
        "src/datasets/canada_fuel_ratings/2022-fuel-consumption-ratings.csv",
        "src/datasets/canada_fuel_ratings/2023-fuel-consumption-ratings.csv",
        "src/datasets/canada_fuel_ratings/2024-fuel-consumption-ratings.csv"
    ]
    
    # Try each potential file path
    for file_path in potential_paths:
        if os.path.exists(file_path):
            try:
                # Load CSV file
                fuel_ratings_df = pd.read_csv(file_path)
                
                print(f"Canada Fuel Ratings data loaded from {file_path}. Shape: {fuel_ratings_df.shape}")
                return fuel_ratings_df
            except Exception as e:
                print(f"Error loading Canada Fuel Ratings data from {file_path}: {e}")
                continue
    
    # If no files found or loaded successfully, create simulated data
    print("No valid Canada Fuel Ratings data file found. Creating simulated data.")
    return create_simulated_fuel_ratings_data()

def create_simulated_fuel_ratings_data():
    """Create a simulated fuel ratings dataset if local file is not available."""
    print("Creating simulated fuel ratings data...")
    
    # [Same implementation as before]
    # Define vehicle classes, engine types, and emission characteristics
    vehicle_classes = ['COMPACT', 'MID-SIZE', 'FULL-SIZE', 'SUV - SMALL', 'SUV - STANDARD', 
                     'PICKUP TRUCK - SMALL', 'PICKUP TRUCK - STANDARD', 'MINIVAN', 'VAN - CARGO']
    
    fuel_types = ['X', 'D', 'E', 'N']  # X = Gasoline, D = Diesel, E = Electric, N = Natural Gas
    fuel_type_names = {
        'X': 'Regular Gasoline',
        'D': 'Diesel',
        'E': 'Electric',
        'N': 'Natural Gas'
    }
    
    # Base CO2 emissions (g/km) for each fuel type
    base_emissions = {
        'X': 220,
        'D': 180,
        'E': 0,
        'N': 150
    }
    
    # Create rows for the dataset
    rows = []
    model_years = [2021, 2022, 2023, 2024]
    
    for vclass in vehicle_classes:
        for fuel in fuel_types:
            # Skip unrealistic combinations
            if fuel == 'E' and 'PICKUP TRUCK' in vclass:
                continue  # Few electric pickup trucks in the dataset timeframe
                
            if fuel == 'N' and ('VAN' in vclass or 'MINIVAN' in vclass):
                continue  # Few natural gas vans
            
            # Number of entries per class/fuel combination
            num_entries = random.randint(2, 10)
            
            for year in model_years:
                for _ in range(num_entries):
                    # Engine size
                    if fuel == 'E':
                        engine_size = 0
                    elif 'COMPACT' in vclass:
                        engine_size = round(random.uniform(1.0, 2.0), 1)
                    elif 'MID-SIZE' in vclass:
                        engine_size = round(random.uniform(1.8, 2.5), 1)
                    elif 'TRUCK' in vclass or 'SUV' in vclass:
                        engine_size = round(random.uniform(2.5, 5.5), 1)
                    else:
                        engine_size = round(random.uniform(2.0, 3.5), 1)
                    
                    # Fuel consumption rates
                    if fuel == 'E':
                        city_consumption = round(random.uniform(15, 20), 1)  # kWh/100 km
                        hwy_consumption = round(random.uniform(13, 18), 1)   # kWh/100 km
                        combined_consumption = round((city_consumption * 0.55 + hwy_consumption * 0.45), 1)
                        consumption_unit = 'kWh/100 km'
                    else:
                        # L/100 km values based on vehicle class and fuel type
                        if 'COMPACT' in vclass:
                            base_consumption = 7.5
                        elif 'MID-SIZE' in vclass:
                            base_consumption = 9.0
                        elif 'FULL-SIZE' in vclass:
                            base_consumption = 11.0
                        elif 'SUV - SMALL' in vclass:
                            base_consumption = 10.0
                        elif 'SUV - STANDARD' in vclass:
                            base_consumption = 12.5
                        elif 'PICKUP TRUCK - SMALL' in vclass:
                            base_consumption = 11.5
                        elif 'PICKUP TRUCK - STANDARD' in vclass:
                            base_consumption = 14.0
                        else:  # Vans
                            base_consumption = 12.0
                        
                        # Adjust for fuel type
                        if fuel == 'D':  # Diesel is more efficient
                            fuel_factor = 0.85
                        elif fuel == 'N':  # Natural gas
                            fuel_factor = 1.05
                        else:  # Gasoline
                            fuel_factor = 1.0
                        
                        city_consumption = round(base_consumption * fuel_factor * random.uniform(0.95, 1.05), 1)
                        hwy_consumption = round(city_consumption * random.uniform(0.65, 0.75), 1)
                        combined_consumption = round((city_consumption * 0.55 + hwy_consumption * 0.45), 1)
                        consumption_unit = 'L/100 km'
                    
                    # CO2 emissions
                    if fuel == 'E':
                        co2_emissions = 0
                    else:
                        co2_base = base_emissions[fuel]
                        co2_emissions = round(co2_base * (combined_consumption / 10) * random.uniform(0.95, 1.05))
                    
                    # Create the row
                    row = {
                        'Model Year': year,
                        'Make': f"Manufacturer_{random.randint(1, 15)}",
                        'Model': f"Model_{random.randint(1, 100)}",
                        'Vehicle Class': vclass,
                        'Engine Size (L)': engine_size,
                        'Cylinders': 0 if fuel == 'E' else random.choice([3, 4, 6, 8]),
                        'Transmission': 'A' if fuel == 'E' else random.choice(['A', 'AM', 'M']),
                        'Fuel Type': fuel_type_names[fuel],
                        'Fuel Consumption City (' + consumption_unit + ')': city_consumption,
                        'Fuel Consumption Hwy (' + consumption_unit + ')': hwy_consumption,
                        'Fuel Consumption Comb (' + consumption_unit + ')': combined_consumption,
                        'CO2 Emissions(g/km)': co2_emissions,
                        'CO2 Rating': 10 if fuel == 'E' else max(1, 10 - int(co2_emissions / 50)),
                        'Smog Rating': 10 if fuel == 'E' else random.randint(3, 10)
                    }
                    
                    rows.append(row)
    
    # Create dataframe
    fuel_ratings_df = pd.DataFrame(rows)
    
    # Save simulated dataset
    fuel_ratings_df.to_csv('src/datasets/canada_fuel_ratings_simulated.csv', index=False)
    
    print(f"Simulated fuel ratings data created. Shape: {fuel_ratings_df.shape}")
    return fuel_ratings_df

#######################################################
# PART 2: PROCESS DATASETS AND EXTRACT PATTERNS
#######################################################

def process_epa_data(epa_data):
    """Extract useful patterns from EPA Fuel Economy data."""
    print("Processing EPA Fuel Economy data to extract patterns...")
    
    # [Same implementation as before, but with added class for JSON serialization]
    # Select relevant columns based on the actual dataset structure
    # First, let's identify which columns exist in the dataset
    expected_cols = [
        'year', 'make', 'model', 'VClass', 'displ', 'cylinders', 
        'trany', 'city08', 'highway08', 'comb08', 'fuelType'
    ]
    
    # Select columns that exist in the dataset
    existing_cols = [col for col in expected_cols if col in epa_data.columns]
    
    if not existing_cols:
        print("No expected columns found in EPA data. Using all columns.")
        processed_epa = epa_data.copy()
    else:
        processed_epa = epa_data[existing_cols].copy()
    
    # Create a mapping from vehicle classes to equipment types
    vclass_col = None
    for possible_col in ['VClass', 'Vehicle Class', 'vehicle_class']:
        if possible_col in processed_epa.columns:
            vclass_col = possible_col
            break
    
    if vclass_col:
        # Create a mapping dictionary
        vehicle_to_equipment = {
            'Small Cars': 'Forklift',
            'Midsize Cars': 'Forklift',
            'Large Cars': 'Forklift',
            'Small Pickup Trucks': 'Loader',
            'Pickup Trucks': 'Loader',
            'Standard Pickup Trucks': 'Loader',
            'Vans, Passenger Type': 'Truck',
            'Vans': 'Truck',
            'Vans, Cargo Type': 'Truck',
            'SUVs': 'Truck',
            'SUV': 'Truck',
            'Special Purpose Vehicles': 'Bulldozer',
            'Special Purpose': 'Bulldozer'
        }
        
        # Apply a default mapping for any unmatched classes
        processed_epa['equipment_type'] = processed_epa[vclass_col].astype(str).apply(
            lambda x: next((v for k, v in vehicle_to_equipment.items() if k.lower() in x.lower()), 'Truck')
        )
    else:
        # If no vehicle class column exists, create a random mapping
        processed_epa['equipment_type'] = np.random.choice(
            ['Forklift', 'Loader', 'Truck', 'Bulldozer'], 
            size=len(processed_epa)
        )
    
    # Check for fuel efficiency columns
    city_col = next((col for col in processed_epa.columns if 'city' in col.lower()), None)
    hwy_col = next((col for col in processed_epa.columns if 'highway' in col.lower() or 'hwy' in col.lower()), None)
    combined_col = next((col for col in processed_epa.columns if 'comb' in col.lower()), None)
    
    if city_col and hwy_col and combined_col:
        # Check if values are in MPG or L/100km
        # If median value is high (e.g., > 20), it's likely MPG
        is_mpg = processed_epa[combined_col].median() > 15
        
        if is_mpg:
            # Convert MPG to km/L (MPG * 0.425)
            processed_epa['city_efficiency_km_per_l'] = processed_epa[city_col] * 0.425
            processed_epa['highway_efficiency_km_per_l'] = processed_epa[hwy_col] * 0.425
            processed_epa['combined_efficiency_km_per_l'] = processed_epa[combined_col] * 0.425
            
            # Calculate fuel consumption rates (L/100km)
            processed_epa['city_consumption_l_per_100km'] = 100 / processed_epa['city_efficiency_km_per_l']
            processed_epa['highway_consumption_l_per_100km'] = 100 / processed_epa['highway_efficiency_km_per_l']
            processed_epa['combined_consumption_l_per_100km'] = 100 / processed_epa['combined_efficiency_km_per_l']
        else:
            # Already in L/100km, convert to km/L
            processed_epa['city_consumption_l_per_100km'] = processed_epa[city_col]
            processed_epa['highway_consumption_l_per_100km'] = processed_epa[hwy_col]
            processed_epa['combined_consumption_l_per_100km'] = processed_epa[combined_col]
            
            processed_epa['city_efficiency_km_per_l'] = 100 / processed_epa[city_col]
            processed_epa['highway_efficiency_km_per_l'] = 100 / processed_epa[hwy_col]
            processed_epa['combined_efficiency_km_per_l'] = 100 / processed_epa[combined_col]
    else:
        # Create placeholder efficiency metrics if no appropriate columns exist
        print("No fuel efficiency columns found in EPA data. Creating placeholder values.")
        
        # Base values by equipment type
        efficiency_map = {
            'Forklift': 12,
            'Loader': 5,
            'Truck': 8,
            'Bulldozer': 3,
        }
        
        processed_epa['combined_efficiency_km_per_l'] = processed_epa['equipment_type'].map(
            lambda x: efficiency_map.get(x, 6) * random.uniform(0.9, 1.1)
        )
        processed_epa['city_efficiency_km_per_l'] = processed_epa['combined_efficiency_km_per_l'] * random.uniform(0.8, 0.9)
        processed_epa['highway_efficiency_km_per_l'] = processed_epa['combined_efficiency_km_per_l'] * random.uniform(1.1, 1.3)
        
        processed_epa['city_consumption_l_per_100km'] = 100 / processed_epa['city_efficiency_km_per_l']
        processed_epa['highway_consumption_l_per_100km'] = 100 / processed_epa['highway_efficiency_km_per_l']
        processed_epa['combined_consumption_l_per_100km'] = 100 / processed_epa['combined_efficiency_km_per_l']
    
    # Group by equipment type to get average metrics
    equipment_efficiency = processed_epa.groupby('equipment_type').agg({
        'city_efficiency_km_per_l': 'mean',
        'highway_efficiency_km_per_l': 'mean',
        'combined_efficiency_km_per_l': 'mean',
        'city_consumption_l_per_100km': 'mean',
        'highway_consumption_l_per_100km': 'mean',
        'combined_consumption_l_per_100km': 'mean'
    }).reset_index()
    
    # Save equipment efficiency patterns
    equipment_efficiency.to_csv('src/datasets/equipment_efficiency_patterns.csv', index=False)
    
    print(f"EPA data processed. Equipment efficiency patterns extracted.")
    return equipment_efficiency

def process_auto_mpg_data(auto_mpg):
    """Extract useful patterns from Auto MPG dataset."""
    print("Processing Auto MPG data to extract patterns...")
    
    # [Same implementation as before]
    # Group by cylinders to find relationship between cylinders and fuel efficiency
    if 'cylinders' in auto_mpg.columns and 'mpg' in auto_mpg.columns:
        cylinder_efficiency = auto_mpg.groupby('cylinders').agg({
            'mpg': ['mean', 'std'],
            'fuel_efficiency_km_per_l': ['mean', 'std']
        }).reset_index()
        
        # Flatten multi-index columns
        cylinder_efficiency.columns = [
            'cylinders', 'mpg_mean', 'mpg_std', 
            'fuel_efficiency_km_per_l_mean', 'fuel_efficiency_km_per_l_std'
        ]
    else:
        print("Cylinders or MPG columns not found in Auto MPG data.")
        # Create a placeholder if columns don't exist
        cylinder_efficiency = pd.DataFrame({
            'cylinders': [4, 6, 8],
            'mpg_mean': [28, 20, 15],
            'mpg_std': [5, 4, 3],
            'fuel_efficiency_km_per_l_mean': [12, 8.5, 6.4],
            'fuel_efficiency_km_per_l_std': [2, 1.7, 1.3]
        })
    
    # Analyze relationship between weight and fuel consumption
    if 'weight_kg' in auto_mpg.columns and 'fuel_efficiency_km_per_l' in auto_mpg.columns:
        from sklearn.linear_model import LinearRegression
        
        # Prepare data
        X = auto_mpg[['weight_kg']].values.reshape(-1, 1)
        y = 100 / auto_mpg['fuel_efficiency_km_per_l'].values  # Convert to L/100km
        
        # Fit model
        model = LinearRegression()
        model.fit(X, y)
        
        print(f"Weight-consumption relationship: Consumption = {model.intercept_:.4f} + {model.coef_[0]:.6f} * Weight(kg)")
        
        # Create lookup dictionary for weight ranges and expected consumption
        weight_ranges = [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000]
        consumption_lookup = {}
        
        for i in range(len(weight_ranges)-1):
            min_weight = weight_ranges[i]
            max_weight = weight_ranges[i+1]
            
            # Calculate average consumption for this weight range
            mask = (auto_mpg['weight_kg'] >= min_weight) & (auto_mpg['weight_kg'] < max_weight)
            if mask.sum() > 0:
                avg_consumption = float((100 / auto_mpg.loc[mask, 'fuel_efficiency_km_per_l']).mean())
                consumption_lookup[f"{min_weight}-{max_weight}"] = avg_consumption
            else:
                # If no data in this range, estimate using the linear model
                mid_weight = (min_weight + max_weight) / 2
                consumption_lookup[f"{min_weight}-{max_weight}"] = float(model.intercept_ + model.coef_[0] * mid_weight)
    else:
        print("Weight or fuel efficiency columns not found in Auto MPG data.")
        # Create a placeholder lookup based on typical values
        consumption_lookup = {
            "500-1000": 5.0,
            "1000-1500": 7.0,
            "1500-2000": 9.0,
            "2000-2500": 11.0,
            "2500-3000": 13.0,
            "3000-3500": 15.0,
            "3500-4000": 17.0
        }
    
    # Save patterns
    with open('src/datasets/weight_consumption_patterns.json', 'w') as f:
        json.dump(consumption_lookup, f)
    
    cylinder_efficiency.to_csv('src/datasets/cylinder_efficiency_patterns.csv', index=False)
    
    print("Auto MPG data processed. Efficiency patterns extracted.")
    return cylinder_efficiency, consumption_lookup

def process_energy_consumption_data(energy_df, hourly_patterns):
    """Extract useful patterns from Energy Consumption dataset."""
    print("Processing Energy Consumption data to extract time-based patterns...")
    
    # [Same implementation as before]
    # Check if the required columns exist
    if not all(col in energy_df.columns for col in ['Weekday', 'Month', 'Is_Weekend', 'Consumption_Pct']):
        print("Some required columns missing from energy consumption data.")
        return create_default_time_patterns()
    
    # Extract day-of-week patterns
    weekday_patterns = energy_df.groupby('Weekday')['Consumption_Pct'].agg(['mean', 'std']).reset_index()
    weekday_patterns.columns = ['Weekday', 'Mean_Consumption_Pct', 'Std_Consumption_Pct']
    
    # Extract monthly patterns
    monthly_patterns = energy_df.groupby('Month')['Consumption_Pct'].agg(['mean', 'std']).reset_index()
    monthly_patterns.columns = ['Month', 'Mean_Consumption_Pct', 'Std_Consumption_Pct']
    
    # Extract weekend vs weekday patterns
    weekend_patterns = energy_df.groupby('Is_Weekend')['Consumption_Pct'].agg(['mean', 'std']).reset_index()
    weekend_patterns.columns = ['Is_Weekend', 'Mean_Consumption_Pct', 'Std_Consumption_Pct']
    
    # Save patterns
    weekday_patterns.to_csv('src/datasets/weekday_consumption_patterns.csv', index=False)
    monthly_patterns.to_csv('src/datasets/monthly_consumption_patterns.csv', index=False)
    weekend_patterns.to_csv('src/datasets/weekend_consumption_patterns.csv', index=False)
    
    # Create time-based activity factors dictionary
    time_activity_factors = {
        'hourly': {str(hour): float(row['Mean_Consumption_Pct'] / 100) 
                  for _, row in hourly_patterns.iterrows() 
                  for hour in [row['Hour']]},
        'weekday': {str(day): float(row['Mean_Consumption_Pct'] / 100) 
                   for _, row in weekday_patterns.iterrows() 
                   for day in [row['Weekday']]},
        'monthly': {str(month): float(row['Mean_Consumption_Pct'] / 100) 
                   for _, row in monthly_patterns.iterrows() 
                   for month in [row['Month']]}
    }
    
    # Save the dictionary
    with open('src/datasets/time_activity_factors.json', 'w') as f:
        json.dump(time_activity_factors, f)
    
    print("Energy Consumption data processed. Time-based patterns extracted.")
    return time_activity_factors

def create_default_time_patterns():
    """Create default time-based patterns if energy consumption data is unavailable."""
    print("Creating default time-based patterns...")
    
    # [Same implementation as before]
    # Hourly patterns (percentage of peak)
    hourly_patterns = {
        '0': 0.65, '1': 0.60, '2': 0.58, '3': 0.55, '4': 0.58, '5': 0.65,
        '6': 0.75, '7': 0.85, '8': 0.95, '9': 0.98, '10': 1.00, '11': 0.98,
        '12': 0.95, '13': 0.93, '14': 0.90, '15': 0.92, '16': 0.95, '17': 0.98,
        '18': 1.00, '19': 0.98, '20': 0.95, '21': 0.90, '22': 0.80, '23': 0.70
    }
    
    # Weekday patterns (relative to average)
    weekday_patterns = {
        '0': 1.05, '1': 1.05, '2': 1.05, '3': 1.05, '4': 1.00, '5': 0.90, '6': 0.85
    }
    
    # Monthly patterns (relative to average)
    monthly_patterns = {
        '1': 1.10, '2': 1.05, '3': 1.00, '4': 0.95, '5': 0.90, '6': 0.85,
        '7': 0.85, '8': 0.90, '9': 0.95, '10': 1.00, '11': 1.05, '12': 1.10
    }
    
    # Combine into a single dictionary
    time_activity_factors = {
        'hourly': hourly_patterns,
        'weekday': weekday_patterns,
        'monthly': monthly_patterns
    }
    
    # Save the patterns
    with open('src/datasets/time_activity_factors.json', 'w') as f:
        json.dump(time_activity_factors, f)
    
    # Create and save DataFrame versions for consistency
    hourly_df = pd.DataFrame([
        {'Hour': int(h), 'Mean_Consumption_Pct': v * 100, 'Std_Consumption_Pct': 5} 
        for h, v in hourly_patterns.items()
    ])
    
    weekday_df = pd.DataFrame([
        {'Weekday': int(d), 'Mean_Consumption_Pct': v * 100, 'Std_Consumption_Pct': 5} 
        for d, v in weekday_patterns.items()
    ])
    
    monthly_df = pd.DataFrame([
        {'Month': int(m), 'Mean_Consumption_Pct': v * 100, 'Std_Consumption_Pct': 5} 
        for m, v in monthly_patterns.items()
    ])
    
    hourly_df.to_csv('src/datasets/hourly_patterns.csv', index=False)
    weekday_df.to_csv('src/datasets/weekday_consumption_patterns.csv', index=False)
    monthly_df.to_csv('src/datasets/monthly_consumption_patterns.csv', index=False)
    
    print("Default time-based patterns created.")
    return time_activity_factors

def process_canada_fuel_ratings(fuel_ratings_df):
    """Extract useful patterns from Canada Fuel Ratings data."""
    print("Processing Canada Fuel Ratings data to extract emissions patterns...")
    
    # [Same implementation as before]
    # Try to identify the emissions column
    emissions_col = None
    for possible_col in ['CO2 Emissions(g/km)', 'CO2 Emissions (g/km)', 'CO2 Rating', 'Emissions']:
        if possible_col in fuel_ratings_df.columns:
            emissions_col = possible_col
            break
    
    # Try to identify the fuel type column
    fuel_type_col = None
    for possible_col in ['Fuel Type', 'Fuel', 'fuelType']:
        if possible_col in fuel_ratings_df.columns:
            fuel_type_col = possible_col
            break
    
    if emissions_col and fuel_type_col:
        # Group by fuel type to get average emissions
        emissions_by_fuel = fuel_ratings_df.groupby(fuel_type_col)[emissions_col].agg(['mean', 'std']).reset_index()
        emissions_by_fuel.columns = [fuel_type_col, 'Mean_Emissions', 'Std_Emissions']
        
        # Create a more flexible mapping that looks for keywords
        def map_fuel_type(fuel_str):
            fuel_str = str(fuel_str).lower()
            if 'gasoline' in fuel_str or 'gas' in fuel_str or fuel_str == 'x':
                return 'Gasoline'
            elif 'diesel' in fuel_str or fuel_str == 'd':
                return 'Diesel'
            elif 'electric' in fuel_str or 'battery' in fuel_str or fuel_str == 'e':
                return 'Electric'
            elif 'natural' in fuel_str or 'cng' in fuel_str or fuel_str == 'n':
                return 'Natural Gas'
            else:
                return 'Other'
        
        # Apply mapping to create a standardized fuel type
        emissions_by_fuel['Engine_Type'] = emissions_by_fuel[fuel_type_col].apply(map_fuel_type)
        
        # Group by standardized engine type
        emissions_by_engine = emissions_by_fuel.groupby('Engine_Type').agg({
            'Mean_Emissions': 'mean',
            'Std_Emissions': 'mean'
        }).reset_index()
        
        # Create mapping dictionary for emissions by engine type
        emissions_lookup = {}
        for _, row in emissions_by_engine.iterrows():
            engine_type = row['Engine_Type']
            emissions_lookup[engine_type] = {
                'mean_emissions': float(row['Mean_Emissions']),
                'std_emissions': float(row['Std_Emissions'])
            }
        
        # Save patterns
        emissions_by_fuel.to_csv('src/datasets/emissions_by_fuel_type.csv', index=False)
        emissions_by_engine.to_csv('src/datasets/emissions_by_engine_type.csv', index=False)
        
        with open('src/datasets/emissions_lookup.json', 'w') as f:
            json.dump(emissions_lookup, f)
        
        print("Canada Fuel Ratings data processed. Emissions patterns extracted.")
        return emissions_lookup
    else:
        print("Could not find appropriate columns in Canada Fuel Ratings data.")
        return create_default_emissions_data()

def create_default_emissions_data():
    """Create default emissions data if Canada Fuel Ratings data is unavailable."""
    print("Creating default emissions data...")
    
    # [Same implementation as before]
    # Default emissions by engine type
    emissions_data = {
        'Gasoline': {'mean_emissions': 220, 'std_emissions': 30},
        'Diesel': {'mean_emissions': 180, 'std_emissions': 25},
        'Electric': {'mean_emissions': 0, 'std_emissions': 0},
        'Natural Gas': {'mean_emissions': 150, 'std_emissions': 20},
        'Other': {'mean_emissions': 200, 'std_emissions': 30}
    }
    
    # Save the data
    with open('src/datasets/emissions_lookup.json', 'w') as f:
        json.dump(emissions_data, f)
    
    # Create DataFrame version for consistency
    emissions_df = pd.DataFrame([
        {'Engine_Type': k, 'Mean_Emissions': v['mean_emissions'], 'Std_Emissions': v['std_emissions']}
        for k, v in emissions_data.items()
    ])
    
    emissions_df.to_csv('src/datasets/emissions_by_engine_type.csv', index=False)
    
    print("Default emissions data created.")
    return emissions_data

#######################################################
# PART 3: ANALYZE BASE DATASET
#######################################################

def analyze_base_dataset(df):
    """Analyze the base FuelIntel dataset and extract patterns."""
    print("Analyzing base FuelIntel dataset...")
    
    # Basic statistics and patterns
    print(f"Number of rows: {len(df)}")
    print(f"Number of columns: {len(df.columns)}")
    print(f"Equipment types: {df['Equipment_Type'].unique()}")
    
    # Create mapping dictionaries for different equipment types
    equipment_patterns = {}
    
    for eq_type in df['Equipment_Type'].unique():
        # Get subset for this equipment type
        subset = df[df['Equipment_Type'] == eq_type]
        
        # Extract key metrics with error handling
        patterns = {}
        
        # For each metric, check if column exists and has valid data
        for col, default in [
            ('Fuel_Consumed (L/hr)', 10),
            ('Load (%)', 50),
            ('RPM', 1500),
            ('Speed (km/h)', 20),
            ('Fuel_Efficiency (km/L)', 5),
            ('Emission_Rate (g/km)', 200),
            ('Idle_Time (min)', 10),
            ('Fuel_Tank_Capacity (L)', 100)
        ]:
            if col in subset.columns and not subset[col].isna().all():
                # Convert NumPy types to Python native types
                patterns[col.split(' ')[0].lower()] = {
                    'mean': float(subset[col].mean()),
                    'std': float(max(0.1, subset[col].std())),
                    'min': float(subset[col].min()),
                    'max': float(subset[col].max())
                }
            else:
                patterns[col.split(' ')[0].lower()] = {
                    'mean': float(default),
                    'std': float(default * 0.1),
                    'min': float(default * 0.5),
                    'max': float(default * 1.5)
                }
        
        equipment_patterns[eq_type] = patterns
    
    # Save equipment patterns
    with open('src/datasets/equipment_patterns.json', 'w') as f:
        json.dump(equipment_patterns, f, indent=2)
    
    print("Base dataset analysis complete. Equipment patterns extracted.")
    return equipment_patterns

#######################################################
# PART 4: EXPAND DATASET WITH REAL PATTERNS
#######################################################

def map_equipment_to_vehicle_type(equipment_type):
    """Map equipment types to vehicle types."""
    mapping = {
        'Bulldozer': 'Special Purpose Vehicles',
        'Excavator': 'Special Purpose Vehicles',
        'Loader': 'Standard Pickup Trucks',
        'Truck': 'Standard Pickup Trucks',
        'Crane': 'Special Purpose Vehicles',
        'Forklift': 'Small Pickup Trucks',
        'Generator': 'Special Purpose Vehicles',
        'Tractor': 'Standard Pickup Trucks',
        'Compactor': 'Special Purpose Vehicles'
    }
    
    # Default to standard pickup if not found
    return mapping.get(equipment_type, 'Standard Pickup Trucks')

def map_equipment_to_engine_type(equipment_type):
    """Map equipment types to engine types."""
    mapping = {
        'Bulldozer': 'Diesel',
        'Excavator': 'Diesel',
        'Loader': 'Diesel',
        'Truck': 'Diesel',
        'Crane': 'Diesel',
        'Forklift': 'Gasoline',
        'Generator': 'Diesel',
        'Tractor': 'Diesel',
        'Compactor': 'Diesel'
    }
    
    # Default to diesel if not found
    return mapping.get(equipment_type, 'Diesel')

def expand_dataset_with_real_patterns(
    base_df, 
    equipment_patterns, 
    equipment_efficiency, 
    weight_consumption_lookup,
    time_activity_factors, 
    emissions_lookup,
    num_equipment=15, 
    days=30
):
    """Expand the dataset using patterns extracted from real-world datasets."""
    print("Expanding dataset using real-world patterns...")
    
    # [Rest of the implementation remains the same, with appropriate type conversions for JSON serialization]
    # Create mapping from equipment types to reference data
    equipment_mapping = {}
    for eq_type in base_df['Equipment_Type'].unique():
        # Find matching equipment in efficiency data
        efficiency_data = None
        if equipment_efficiency is not None and 'equipment_type' in equipment_efficiency.columns:
            matches = equipment_efficiency[equipment_efficiency['equipment_type'] == eq_type]
            if not matches.empty:
                efficiency_data = matches.iloc[0].to_dict()
        
        if efficiency_data is None:
            # Use default if no match
            efficiency_data = {
                'combined_efficiency_km_per_l': 5.0,
                'city_efficiency_km_per_l': 4.0,
                'highway_efficiency_km_per_l': 6.0
            }
        
        # Map to engine type for emissions
        engine_type = map_equipment_to_engine_type(eq_type)
        
        # Get emissions data
        if emissions_lookup and engine_type in emissions_lookup:
            emissions_data = emissions_lookup[engine_type]
        else:
            # Default emissions if no match
            emissions_data = {'mean_emissions': 200, 'std_emissions': 20}
        
        # Estimate weight for consumption lookup
        if eq_type in ['Bulldozer', 'Excavator', 'Crane']:
            weight_range = (3000, 4000)
        elif eq_type in ['Loader', 'Truck', 'Tractor']:
            weight_range = (2000, 3000)
        else:  # Forklift, Generator, etc.
            weight_range = (1000, 2000)
        
        # Get consumption from weight-based lookup
        consumption = None
        weight_range_str = f"{weight_range[0]}-{weight_range[1]}"
        consumption = weight_consumption_lookup.get(weight_range_str, 20.0)
        
        equipment_mapping[eq_type] = {
            'efficiency_city': float(efficiency_data.get('city_efficiency_km_per_l', 5.0)),
            'efficiency_highway': float(efficiency_data.get('highway_efficiency_km_per_l', 7.0)),
            'efficiency_combined': float(efficiency_data.get('combined_efficiency_km_per_l', 6.0)),
            'consumption_per_100km': float(consumption),
            'engine_type': engine_type,
            'mean_emissions': float(emissions_data.get('mean_emissions', 200)),
            'std_emissions': float(emissions_data.get('std_emissions', 20))
        }
    
    # Generate expanded equipment list
    equipment_list = []
    for eq_type in base_df['Equipment_Type'].unique():
        # Get subset for this equipment type
        subset = base_df[base_df['Equipment_Type'] == eq_type]
        
        # Create multiple instances of each type
        type_count = max(1, int(num_equipment * (1 / len(base_df['Equipment_Type'].unique()))))
        
        for i in range(type_count):
            # Sample a row from the subset as base configuration
            if len(subset) > 0:
                base_row = subset.sample(1).iloc[0]
                tank_capacity = base_row['Fuel_Tank_Capacity (L)']
            else:
                # Default values if no sample is available
                tank_capacity = 100
                
            # Create equipment instance
            equipment = {
                'Equipment_ID': f"{eq_type}_{i+1:03d}",
                'Equipment_Type': eq_type,
                'Driver_ID': f"D{random.randint(1, 50):03d}",
                'Fuel_Tank_Capacity (L)': float(tank_capacity),
                'base_efficiency': equipment_mapping[eq_type]['efficiency_combined'],
                'base_emissions': equipment_mapping[eq_type]['mean_emissions'],
                'engine_type': equipment_mapping[eq_type]['engine_type']
            }
            
            equipment_list.append(equipment)
    
    # Generate time series data
    all_rows = []
    start_date = datetime(2024, 1, 1)
    
    for equipment in equipment_list:
        # Get equipment type and associated patterns
        eq_type = equipment['Equipment_Type']
        
        # Get base patterns from the analysis of the original dataset
        base_patterns = equipment_patterns.get(eq_type, {})
        
        # Initial state
        fuel_level = equipment['Fuel_Tank_Capacity (L)'] * random.uniform(0.5, 0.9)
        time_since_service = random.uniform(0, 500)
        
        # Generate data for each hour
        for day in range(days):
            for hour in range(24):
                current_date = start_date + timedelta(days=day, hours=hour)
                
                # Get time-based activity factors
                hour_factor = float(time_activity_factors['hourly'].get(str(hour), 0.5))
                weekday_factor = float(time_activity_factors['weekday'].get(str(current_date.weekday()), 0.5))
                month_factor = float(time_activity_factors['monthly'].get(str(current_date.month), 0.5))
                
                # Combined activity factor
                activity_factor = hour_factor * weekday_factor * month_factor
                
                # Determine if equipment is active based on time factors
                is_active = random.random() < activity_factor
                
                if is_active:
                    # Generate operating parameters based on real patterns
                    # Load percentage represents work intensity
                    load_pct = random.uniform(20, 90) * activity_factor
                    
                    # Speed depends on the type of equipment
                    if eq_type in ['Truck', 'Tractor']:
                        max_speed = base_patterns.get('speed', {}).get('max', 60)
                        speed = random.uniform(10, max_speed) * activity_factor
                    else:
                        max_speed = base_patterns.get('speed', {}).get('max', 15)
                        speed = random.uniform(0, max_speed) * activity_factor
                    
                    # Determine if idle
                    is_idle = speed < 3 and random.random() < 0.3
                    
                    if is_idle:
                        idle_time = random.uniform(5, 30)
                        speed = 0
                    else:
                        idle_time = 0
                    
                    # RPM based on equipment type and activity
                    if is_idle:
                        rpm = random.uniform(600, 800)
                    else:
                        min_rpm = base_patterns.get('rpm', {}).get('min', 600)
                        max_rpm = base_patterns.get('rpm', {}).get('max', 2500)
                        rpm_range = max_rpm - min_rpm
                        rpm = min_rpm + (rpm_range * load_pct / 100)
                    
                    # Calculate fuel consumption based on patterns
                    base_consumption_rate = base_patterns.get('fuel_consumed', {}).get('mean', 10)
                    
                    if is_idle:
                        fuel_consumed = base_consumption_rate * 0.3  # 30% of base at idle
                    else:
                        # Scale by load and speed
                        load_factor = 0.5 + 0.5 * load_pct/100
                        speed_factor = 0.7 + 0.3 * speed/30 if speed > 0 else 0.7
                        fuel_consumed = base_consumption_rate * load_factor * speed_factor
                    
                    # Temperature affects efficiency
                    # Simulate ambient temperature based on month
                    month = current_date.month
                    if 1 <= month <= 3 or 11 <= month <= 12:  # Winter
                        ambient_temp = random.uniform(-10, 10)
                        temp_factor = 0.9  # 10% less efficient in cold
                    elif 4 <= month <= 5 or 9 <= month <= 10:  # Spring/Fall
                        ambient_temp = random.uniform(10, 25)
                        temp_factor = 1.0
                    else:  # Summer
                        ambient_temp = random.uniform(20, 35)
                        temp_factor = 0.95  # 5% less efficient in extreme heat
                    
                    # Apply temperature factor
                    fuel_consumed *= temp_factor
                    
                    # Calculate distance traveled
                    distance = speed * (1 - idle_time/60)  # km in one hour
                    
                    # Calculate fuel efficiency
                    if distance > 0:
                        fuel_efficiency = distance / fuel_consumed
                    else:
                        fuel_efficiency = 0
                    
                    # Engine temperature based on ambient and load
                    engine_temp = ambient_temp + 70 + load_pct * 0.3
                    
                    # Oil pressure based on load and engine health
                    oil_pressure = 40 + load_pct * 0.2
                    
                    # Emissions based on real data
                    emission_rate = equipment['base_emissions'] * (0.7 + 0.3 * load_pct/100) * temp_factor
                    
                    # Operational status
                    operational_status = "Running"
                else:
                    # Equipment not active
                    load_pct = 0
                    speed = 0
                    idle_time = 0
                    rpm = 0
                    fuel_consumed = 0
                    distance = 0
                    fuel_efficiency = 0
                    emission_rate = 0
                    
                    # Ambient temperature
                    month = current_date.month
                    if 1 <= month <= 3 or 11 <= month <= 12:  # Winter
                        ambient_temp = random.uniform(-10, 10)
                    elif 4 <= month <= 5 or 9 <= month <= 10:  # Spring/Fall
                        ambient_temp = random.uniform(10, 25)
                    else:  # Summer
                        ambient_temp = random.uniform(20, 35)
                    
                    engine_temp = ambient_temp + random.uniform(5, 15)
                    oil_pressure = random.uniform(10, 20)
                    
                    # Operational status
                    operational_status = "Stopped"
                
                # Update fuel level
                fuel_level -= fuel_consumed
                refuel_event = "No"
                
                # Refueling logic
                if fuel_level < equipment['Fuel_Tank_Capacity (L)'] * 0.15 and random.random() < 0.8:
                    refill_amount = equipment['Fuel_Tank_Capacity (L)'] * random.uniform(0.7, 0.95)
                    fuel_level += refill_amount
                    refuel_event = "Yes"
                
                # Ensure fuel level doesn't go negative
                fuel_level = max(0, fuel_level)
                
                # Update maintenance time
                if is_active:
                    time_since_service += 1
                    
                # Maintenance status
                if time_since_service > 500:
                    maintenance_status = "Needs Service"
                elif time_since_service > 400:
                    maintenance_status = "Service Due Soon"
                else:
                    maintenance_status = "Normal"
                
                # Battery voltage
                battery_voltage = 12 + random.uniform(-0.5, 0.5)
                
                # Sensor fault (rare)
                sensor_fault = 0
                if random.random() < 0.01:
                    sensor_fault = random.choice([101, 102, 201, 202, 301, 302])
                
                # Gear position and driving mode
                if is_active and not is_idle:
                    driving_mode = random.choice(["Eco", "Normal", "Power"])
                    
                    if speed < 10:
                        gear_position = "1"
                    elif speed < 20:
                        gear_position = "2"
                    elif speed < 30:
                        gear_position = "3"
                    elif speed < 40:
                        gear_position = "4"
                    else:
                        gear_position = "5"
                else:
                    driving_mode = "N/A"
                    gear_position = "N"
                
                # Location
                location = f"Site_{random.randint(1, 5)}"
                
                # Average load over last 3 hours (simplified)
                avg_load = load_pct * random.uniform(0.8, 1.2)
                avg_load = min(100, max(0, avg_load))  # Ensure it's within 0-100 range
                
                # Throttle position
                throttle_position = load_pct * random.uniform(0.8, 1.2)
                throttle_position = min(100, max(0, throttle_position))  # Ensure it's within 0-100 range
                
                # Event notes
                event_notes = ""
                if refuel_event == "Yes":
                    event_notes = "Refueling event"
                elif sensor_fault > 0:
                    event_notes = f"Sensor fault detected: {sensor_fault}"
                elif maintenance_status == "Needs Service":
                    event_notes = "Maintenance needed"
                elif fuel_level < equipment['Fuel_Tank_Capacity (L)'] * 0.1:
                    event_notes = "Low fuel warning"
                
                # Create the row with all parameters matching your original dataset
                row = {
                    'Timestamp': current_date,
                    'Equipment_ID': equipment['Equipment_ID'],
                    'Driver_ID': equipment['Driver_ID'],
                    'Equipment_Type': equipment['Equipment_Type'],
                    'Fuel_Level (L)': fuel_level,
                    'Fuel_Consumed (L/hr)': fuel_consumed,
                    'Fuel_Tank_Capacity (L)': equipment['Fuel_Tank_Capacity (L)'],
                    'Idle_Time (min)': idle_time,
                    'Load (%)': load_pct,
                    'RPM': rpm,
                    'Speed (km/h)': speed,
                    'Distance_Travelled (km)': distance,
                    'Fuel_Price (₹)': random.uniform(90, 110),  # Random fuel price
                    'Temperature (°C)': ambient_temp,
                    'Engine_Temperature (°C)': engine_temp,
                    'Oil_Pressure (psi)': oil_pressure,
                    'Throttle_Position (%)': throttle_position,
                    'Gear_Position': gear_position,
                    'Driving_Mode': driving_mode,
                    'Battery_Voltage (V)': battery_voltage,
                    'Sensor_Fault_Code': sensor_fault,
                    'Fuel_Efficiency (km/L)': fuel_efficiency,
                    'Emission_Rate (g/km)': emission_rate,
                    'Maintenance_Status': maintenance_status,
                    'Time_Since_Last_Service (hrs)': time_since_service,
                    'Average_Load_Last_3hr (%)': avg_load,
                    'Operational_Status': operational_status,
                    'Refuel_Event': refuel_event,
                    'Event_Notes': event_notes,
                    'Location': location
                }
                
                all_rows.append(row)
    
    # Create the expanded dataframe
    expanded_df = pd.DataFrame(all_rows)
    
    # Save expanded dataset
    expanded_df.to_csv('data/FuelIntel_Expanded_Real_Data.csv', index=False)
    
    print(f"Dataset expansion complete. Expanded to {len(expanded_df)} rows.")
    return expanded_df

#######################################################
# PART 5: INJECT ANOMALIES
#######################################################

def inject_anomalies(df, anomaly_percentage=0.03):
    """Inject realistic anomalies into the dataset."""
    print("Injecting anomalies...")
    
    # [Same implementation as before]
    anomaly_df = df.copy()
    rows_count = len(anomaly_df)
    anomaly_count = int(rows_count * anomaly_percentage)
    
    # Select random rows for anomalies, excluding rows with existing anomalies
    existing_anomalies = anomaly_df[anomaly_df['Sensor_Fault_Code'] > 0].index
    valid_indices = anomaly_df.index.difference(existing_anomalies)
    
    # Ensure we don't select more indices than available
    anomaly_count = min(anomaly_count, len(valid_indices))
    anomaly_indices = np.random.choice(valid_indices, anomaly_count, replace=False)
    
    anomaly_types = [
        'fuel_theft', 
        'inefficiency', 
        'high_idle', 
        'sensor_error', 
        'overheating', 
        'excessive_load',
        'low_oil_pressure'
    ]
    
    for idx in anomaly_indices:
        anomaly_type = random.choice(anomaly_types)
        row = anomaly_df.loc[idx]
        
        if anomaly_type == 'fuel_theft':
            # Sudden drop in fuel level without corresponding consumption
            original_level = row['Fuel_Level (L)']
            theft_amount = original_level * random.uniform(0.1, 0.3)
            anomaly_df.loc[idx, 'Fuel_Level (L)'] = original_level - theft_amount
            anomaly_df.loc[idx, 'Event_Notes'] = 'Potential fuel theft detected'
            
        elif anomaly_type == 'inefficiency':
            # Higher consumption than expected for the conditions
            if row['Fuel_Consumed (L/hr)'] > 0:
                anomaly_df.loc[idx, 'Fuel_Consumed (L/hr)'] *= random.uniform(1.3, 1.8)
                if row['Fuel_Efficiency (km/L)'] > 0:
                    anomaly_df.loc[idx, 'Fuel_Efficiency (km/L)'] /= random.uniform(1.3, 1.8)
                anomaly_df.loc[idx, 'Event_Notes'] = 'Unexpected fuel inefficiency'
            
        elif anomaly_type == 'high_idle':
            # Excessive idle time
            anomaly_df.loc[idx, 'Idle_Time (min)'] = random.uniform(45, 60)
            anomaly_df.loc[idx, 'Speed (km/h)'] = 0
            anomaly_df.loc[idx, 'Event_Notes'] = 'Excessive idle time detected'
            
        elif anomaly_type == 'sensor_error':
            # Implausible sensor values
            anomaly_df.loc[idx, 'Sensor_Fault_Code'] = random.choice([401, 402, 403])
            
            # Pick a sensor to corrupt
            corrupt_sensor = random.choice([
                'Engine_Temperature (°C)', 
                'Oil_Pressure (psi)', 
                'Battery_Voltage (V)',
                'RPM'
            ])
            
            if corrupt_sensor == 'Engine_Temperature (°C)':
                anomaly_df.loc[idx, corrupt_sensor] = random.choice([
                    random.uniform(-50, -10),  # Too cold
                    random.uniform(150, 200)   # Too hot
                ])
            elif corrupt_sensor == 'Oil_Pressure (psi)':
                anomaly_df.loc[idx, corrupt_sensor] = random.choice([
                    random.uniform(-20, 0),    # Negative (impossible)
                    random.uniform(100, 150)   # Too high
                ])
            elif corrupt_sensor == 'Battery_Voltage (V)':
                anomaly_df.loc[idx, corrupt_sensor] = random.choice([
                    random.uniform(0, 5),      # Too low
                    random.uniform(18, 24)     # Too high
                ])
            elif corrupt_sensor == 'RPM':
                if row['Operational_Status'] == 'Running':
                    anomaly_df.loc[idx, corrupt_sensor] = random.uniform(7000, 9000)  # Extremely high RPM
            
            anomaly_df.loc[idx, 'Event_Notes'] = f'Sensor error: abnormal {corrupt_sensor.split(" ")[0]}'
            
        elif anomaly_type == 'overheating':
            # Engine overheating
            anomaly_df.loc[idx, 'Engine_Temperature (°C)'] = random.uniform(105, 125)
            anomaly_df.loc[idx, 'Event_Notes'] = 'Engine overheating detected'
            
        elif anomaly_type == 'excessive_load':
            # Equipment operated under excessive load
            anomaly_df.loc[idx, 'Load (%)'] = random.uniform(95, 110)
            # RPM drops under excessive load
            if row['RPM'] > 0:
                anomaly_df.loc[idx, 'RPM'] *= 0.8
            # Higher consumption under load
            if row['Fuel_Consumed (L/hr)'] > 0:
                anomaly_df.loc[idx, 'Fuel_Consumed (L/hr)'] *= 1.3
            anomaly_df.loc[idx, 'Event_Notes'] = 'Excessive load detected'
            
        elif anomaly_type == 'low_oil_pressure':
            # Low oil pressure warning
            anomaly_df.loc[idx, 'Oil_Pressure (psi)'] = random.uniform(5, 15)
            anomaly_df.loc[idx, 'Event_Notes'] = 'Low oil pressure warning'
    
    print(f"Injected {anomaly_count} anomalies into the dataset")
    return anomaly_df

#######################################################
# PART 6: FEATURE ENGINEERING
#######################################################

def engineer_features(df):
    """Create additional features to improve model performance."""
    print("Engineering features...")
    
    # [Same implementation as before]
    # Make a copy to avoid modifying original
    feature_df = df.copy()
    
    # 1. Time-based features
    feature_df['Hour'] = pd.to_datetime(feature_df['Timestamp']).dt.hour
    feature_df['Day'] = pd.to_datetime(feature_df['Timestamp']).dt.day
    feature_df['Weekday'] = pd.to_datetime(feature_df['Timestamp']).dt.weekday
    feature_df['Is_Weekend'] = feature_df['Weekday'].apply(lambda x: 1 if x >= 5 else 0)
    feature_df['Is_WorkHours'] = ((feature_df['Hour'] >= 9) & (feature_df['Hour'] < 17)).astype(int)
    
    # 2. Equipment specific features
    feature_df['Fuel_Level_Pct'] = feature_df['Fuel_Level (L)'] / feature_df['Fuel_Tank_Capacity (L)'] * 100
    feature_df['Is_Low_Fuel'] = (feature_df['Fuel_Level_Pct'] < 20).astype(int)
    
    # 3. Operational features
    feature_df['Is_Idle'] = (feature_df['Idle_Time (min)'] > 0).astype(int)
    feature_df['Is_Active'] = (feature_df['Operational_Status'] == 'Running').astype(int)
    
    # 4. Efficiency features
    # Handle potential division by zero
    feature_df['Consumption_Per_Load'] = feature_df.apply(
        lambda x: x['Fuel_Consumed (L/hr)'] / max(1, x['Load (%)']) if x['Load (%)'] > 0 else 0, 
        axis=1
    )
    
    feature_df['Engine_Ambient_Temp_Diff'] = feature_df['Engine_Temperature (°C)'] - feature_df['Temperature (°C)']
    
    # 5. Maintenance features
    feature_df['Maintenance_Due_Soon'] = (feature_df['Time_Since_Last_Service (hrs)'] > 400).astype(int)
    
    # 6. Previous state features (using shift to get previous hour's values)
    for equipment in feature_df['Equipment_ID'].unique():
        equipment_data = feature_df[feature_df['Equipment_ID'] == equipment].sort_values('Timestamp')
        
        if len(equipment_data) > 1:  # Only if there's more than one record
            # Previous hour features
            feature_df.loc[equipment_data.index, 'Prev_Fuel_Level'] = equipment_data['Fuel_Level (L)'].shift(1)
            feature_df.loc[equipment_data.index, 'Prev_Load'] = equipment_data['Load (%)'].shift(1)
            feature_df.loc[equipment_data.index, 'Prev_RPM'] = equipment_data['RPM'].shift(1)
            
            # Calculate rate of change features
            feature_df.loc[equipment_data.index, 'Fuel_Level_Change'] = (
                equipment_data['Fuel_Level (L)'] - equipment_data['Fuel_Level (L)'].shift(1)
            )
            
            # Calculate rolling statistics (last 12 hours)
            for window in [6, 12, 24]:
                if len(equipment_data) >= window:
                    # Consumption rolling statistics
                    feature_df.loc[equipment_data.index, f'Avg_Consumption_{window}h'] = (
                        equipment_data['Fuel_Consumed (L/hr)'].rolling(window, min_periods=1).mean()
                    )
                    
                    # Load rolling statistics
                    feature_df.loc[equipment_data.index, f'Avg_Load_{window}h'] = (
                        equipment_data['Load (%)'].rolling(window, min_periods=1).mean()
                    )
                    
                    # Speed rolling statistics
                    feature_df.loc[equipment_data.index, f'Avg_Speed_{window}h'] = (
                        equipment_data['Speed (km/h)'].rolling(window, min_periods=1).mean()
                    )
    
    # 7. Anomaly indicators
    feature_df['Has_Sensor_Fault'] = (feature_df['Sensor_Fault_Code'] > 0).astype(int)
    feature_df['Engine_Overheating'] = (feature_df['Engine_Temperature (°C)'] > 100).astype(int)
    feature_df['Oil_Pressure_Warning'] = (feature_df['Oil_Pressure (psi)'] < 20).astype(int)
    
    # 8. Fill NAs in newly created columns
    na_columns = [col for col in feature_df.columns if feature_df[col].isna().any()]
    for col in na_columns:
        if col.startswith('Prev_'):
            # Fill with the current value for the first record
            base_col = col.replace('Prev_', '')
            if base_col in feature_df.columns:
                feature_df[col].fillna(feature_df[base_col], inplace=True)
            else:
                feature_df[col].fillna(0, inplace=True)
        elif col == 'Fuel_Level_Change':
            feature_df[col].fillna(0, inplace=True)
        elif col.startswith('Avg_'):
            # Fill rolling averages with the current value
            if col.endswith('_Consumption_6h') or col.endswith('_Consumption_12h') or col.endswith('_Consumption_24h'):
                feature_df[col].fillna(feature_df['Fuel_Consumed (L/hr)'], inplace=True)
            elif col.endswith('_Load_6h') or col.endswith('_Load_12h') or col.endswith('_Load_24h'):
                feature_df[col].fillna(feature_df['Load (%)'], inplace=True)
            elif col.endswith('_Speed_6h') or col.endswith('_Speed_12h') or col.endswith('_Speed_24h'):
                feature_df[col].fillna(feature_df['Speed (km/h)'], inplace=True)
            else:
                feature_df[col].fillna(0, inplace=True)
        else:
            feature_df[col].fillna(0, inplace=True)
    
    # 9. One-hot encoding for categorical features
    cat_columns = ['Equipment_Type', 'Gear_Position', 'Driving_Mode', 'Operational_Status', 'Maintenance_Status']
    for col in cat_columns:
        if col in feature_df.columns:
            dummies = pd.get_dummies(feature_df[col], prefix=col, drop_first=False)
            feature_df = pd.concat([feature_df, dummies], axis=1)
    
    print(f"Feature engineering complete. New shape: {feature_df.shape}")
    return feature_df

#######################################################
# PART 7: COMPLETE DATASET PIPELINE
#######################################################

def complete_dataset_pipeline(input_file, num_equipment=15, days=30):
    """Run the complete dataset processing pipeline."""
    print("\n====== STARTING COMPLETE DATASET PROCESSING PIPELINE ======\n")
    
    # Step 1: Load original dataset
    base_df = pd.read_csv(input_file)
    print(f"Loaded base dataset with shape: {base_df.shape}")
    
    # Step 2: Load and process reference datasets
    epa_data = load_epa_fuel_economy_data()
    auto_mpg = load_auto_mpg_dataset()
    energy_df, hourly_patterns = load_energy_consumption_dataset()
    fuel_ratings_df = load_canada_fuel_ratings()
    
    # Step 3: Extract patterns from reference datasets
    equipment_efficiency = process_epa_data(epa_data)
    cylinder_efficiency, weight_consumption_lookup = process_auto_mpg_data(auto_mpg)
    time_activity_factors = process_energy_consumption_data(energy_df, hourly_patterns)
    emissions_lookup = process_canada_fuel_ratings(fuel_ratings_df)
    
    # Step 4: Analyze base dataset
    equipment_patterns = analyze_base_dataset(base_df)
    
    # Step 5: Expand dataset using real-world patterns
    expanded_df = expand_dataset_with_real_patterns(
        base_df,
        equipment_patterns,
        equipment_efficiency,
        weight_consumption_lookup,
        time_activity_factors,
        emissions_lookup,
        num_equipment=num_equipment,
        days=days
    )
    
    # Step 6: Inject anomalies for detection
    anomaly_df = inject_anomalies(expanded_df)
    
    # Step 7: Engineer features for model training
    final_df = engineer_features(anomaly_df)
    
    # Save dataset to CSV
    final_file = 'data/FuelIntel_Final_Dataset.csv'
    final_df.to_csv(final_file, index=False)
    
    print(f"\nComplete dataset processing pipeline finished.")
    print(f"Final dataset saved to '{final_file}' with shape: {final_df.shape}")
    
    # Create a simple split for training and testing
    split_point = int(len(final_df) * 0.8)
    train_df = final_df.iloc[:split_point]
    test_df = final_df.iloc[split_point:]
    
    train_df.to_csv('data/FuelIntel_Train_Dataset.csv', index=False)
    test_df.to_csv('data/FuelIntel_Test_Dataset.csv', index=False)
    
    print(f"Training set ({train_df.shape[0]} rows) saved to 'data/FuelIntel_Train_Dataset.csv'")
    print(f"Test set ({test_df.shape[0]} rows) saved to 'data/FuelIntel_Test_Dataset.csv'")
    
    return final_df

#######################################################
# PART 8: EXECUTE PIPELINE
#######################################################

if __name__ == "__main__":
    # Run the complete pipeline
    final_dataset = complete_dataset_pipeline(
        'data/FuelIntel_Dataset.csv',  # Your original dataset
        num_equipment=15,         # Number of equipment to simulate
        days=30                   # Number of days of data to generate
    )
    
    # Basic statistics of the final dataset
    print("\nFinal Dataset Statistics:")
    print(f"Total rows: {len(final_dataset)}")
    print(f"Unique equipment: {final_dataset['Equipment_ID'].nunique()}")
    print(f"Equipment types: {final_dataset['Equipment_Type'].unique()}")
    print(f"Date range: {final_dataset['Timestamp'].min()} to {final_dataset['Timestamp'].max()}")
    
    # Count anomalies
    anomaly_keywords = ['theft', 'error', 'warning', 'Excessive', 'Unexpected', 'overheating']
    anomaly_count = len(final_dataset[
        final_dataset['Event_Notes'].str.contains('|'.join(anomaly_keywords), case=False, na=False)
    ])
    print(f"Anomaly count: {anomaly_count}")
    
    print("\nDataset Processing Complete!")