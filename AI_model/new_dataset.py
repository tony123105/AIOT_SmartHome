import pandas as pd
import numpy as np

def modify_dataset(input_file='Occupancy_with_appliances.csv', output_file='Modified_Occupancy_with_appliances.csv'):
    # Read the original dataset
    df = pd.read_csv(input_file)
    
    # Make a copy of the dataframe
    modified_df = df.copy()
    
    # 1. Remove CO2 column
    modified_df = modified_df.drop('CO2', axis=1)
    
    # 2. Make Light threshold more lenient (turn on lights when light level is below 480 instead of 450)
    # Original: ~450, New: ~480
    light_threshold = 480
    modified_df.loc[modified_df['Light'] < light_threshold, 'Lights_On'] = 1
    
    # 3. Make Temperature threshold more lenient (turn on AC when temperature is above 22.8 instead of 23.2)
    # Original: ~23.2°C, New: ~22.8°C
    temp_threshold = 22.8
    modified_df.loc[modified_df['Temperature'] > temp_threshold, 'AC_On'] = 1
    
    # 4. Make Humidity threshold more lenient (turn on humidity machine when humidity is above 25.0 instead of 26.0)
    # Original: ~26%, New: ~25%
    humidity_threshold = 25.0
    modified_df.loc[modified_df['Humidity'] > humidity_threshold, 'Humidity_Machine_On'] = 1
    
    # Save the modified dataset
    modified_df.to_csv(output_file, index=False)
    
    # Print statistics to show changes
    print("Original Dataset Statistics:")
    print(f"Lights_On = 1: {df['Lights_On'].sum()} records ({df['Lights_On'].mean()*100:.2f}%)")
    print(f"AC_On = 1: {df['AC_On'].sum()} records ({df['AC_On'].mean()*100:.2f}%)")
    print(f"Humidity_Machine_On = 1: {df['Humidity_Machine_On'].sum()} records ({df['Humidity_Machine_On'].mean()*100:.2f}%)")
    
    print("\nModified Dataset Statistics:")
    print(f"Lights_On = 1: {modified_df['Lights_On'].sum()} records ({modified_df['Lights_On'].mean()*100:.2f}%)")
    print(f"AC_On = 1: {modified_df['AC_On'].sum()} records ({modified_df['AC_On'].mean()*100:.2f}%)")
    print(f"Humidity_Machine_On = 1: {modified_df['Humidity_Machine_On'].sum()} records ({modified_df['Humidity_Machine_On'].mean()*100:.2f}%)")
    
    return modified_df

if __name__ == "__main__":
    modified_dataset = modify_dataset()
    print("\nDataset modified successfully: CO2 removed and appliance thresholds adjusted.")