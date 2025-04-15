import pandas as pd
import numpy as np
from datetime import datetime

def modify_dataset():
    # Read the original dataset
    df = pd.read_csv('Occupancy_with_appliances.csv')
    
    # Make a copy of the dataframe
    modified_df = df.copy()
    
    # 1. Make Light threshold more lenient (turn on lights at higher light levels)
    # Original threshold seemed to be around 450, let's increase it to 500
    modified_df.loc[modified_df['Light'] < 350, 'Lights_On'] = 1
    
    # 2. Make Temperature threshold more lenient (turn on AC at lower temperatures)
    # Original seemed to activate at ~23.5°C, let's decrease to 23.0°C
    modified_df.loc[modified_df['Temperature'] > 21.0, 'AC_On'] = 1
    
    # 3. Make Humidity threshold more lenient (turn on humidity machine at lower humidity)
    # Original seemed to activate at ~26.5%, let's decrease to 25.5%
    modified_df.loc[modified_df['Humidity'] > 25, 'Humidity_Machine_On'] = 1
    
    # 4. Improve occupancy detection (make it more sensitive)
    # If CO2 is above 600 or there's significant change in sensor readings, mark as occupied
    co2_threshold = 600  # Lower threshold from original
    
    # Calculate changes in sensor readings
    modified_df['light_change'] = modified_df['Light'].diff().abs().fillna(0)
    modified_df['temp_change'] = modified_df['Temperature'].diff().abs().fillna(0)
    modified_df['co2_change'] = modified_df['CO2'].diff().abs().fillna(0)
    
    # Mark as occupied if CO2 is high or there are significant changes
    modified_df.loc[
        (modified_df['CO2'] > co2_threshold) | 
        (modified_df['light_change'] > 15) |
        (modified_df['temp_change'] > 0.1) |
        (modified_df['co2_change'] > 20), 
        'Occupancy'] = 1
    
    # 5. Ensure when room is occupied, appropriate devices are on
    occupied_indices = modified_df[modified_df['Occupancy'] == 1].index
    for idx in occupied_indices:
        # If no appliance is on, turn on one based on conditions
        if (modified_df.loc[idx, 'Lights_On'] == 0 and 
            modified_df.loc[idx, 'AC_On'] == 0 and 
            modified_df.loc[idx, 'Humidity_Machine_On'] == 0):
            
            # Decide which appliance to turn on based on conditions
            if modified_df.loc[idx, 'Light'] < 330:
                modified_df.loc[idx, 'Lights_On'] = 1
            elif modified_df.loc[idx, 'Temperature'] > 20.8:
                modified_df.loc[idx, 'AC_On'] = 1
            elif modified_df.loc[idx, 'Humidity'] > 25.0:
                modified_df.loc[idx, 'Humidity_Machine_On'] = 1
            else:
                # Default to turning on lights if no condition is met
                modified_df.loc[idx, 'Lights_On'] = 1
    
    # Drop temporary columns
    modified_df = modified_df.drop(['light_change', 'temp_change', 'co2_change'], axis=1)
    
    # Save the modified dataset
    modified_df.to_csv('Modified_Occupancy_with_appliances.csv', index=False)
    
    # Print statistics to show changes
    print("Original Dataset Statistics:")
    print(f"Occupancy = 1: {df['Occupancy'].sum()} records ({df['Occupancy'].mean()*100:.2f}%)")
    print(f"Lights_On = 1: {df['Lights_On'].sum()} records ({df['Lights_On'].mean()*100:.2f}%)")
    print(f"AC_On = 1: {df['AC_On'].sum()} records ({df['AC_On'].mean()*100:.2f}%)")
    print(f"Humidity_Machine_On = 1: {df['Humidity_Machine_On'].sum()} records ({df['Humidity_Machine_On'].mean()*100:.2f}%)")
    
    print("\nModified Dataset Statistics:")
    print(f"Occupancy = 1: {modified_df['Occupancy'].sum()} records ({modified_df['Occupancy'].mean()*100:.2f}%)")
    print(f"Lights_On = 1: {modified_df['Lights_On'].sum()} records ({modified_df['Lights_On'].mean()*100:.2f}%)")
    print(f"AC_On = 1: {modified_df['AC_On'].sum()} records ({modified_df['AC_On'].mean()*100:.2f}%)")
    print(f"Humidity_Machine_On = 1: {modified_df['Humidity_Machine_On'].sum()} records ({modified_df['Humidity_Machine_On'].mean()*100:.2f}%)")
    
    return modified_df

if __name__ == "__main__":
    modified_dataset = modify_dataset()