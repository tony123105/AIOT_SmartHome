import pandas as pd
import numpy as np
import time
import datetime
import requests
import argparse
import joblib
import os
import json
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model

class SmartController:
    def __init__(self, read_api_key, write_api_key, channel_id):
        """
        Initialize the Smart Controller with both occupancy prediction and appliance control
        
        Args:
            read_api_key (str): ThingSpeak Read API Key
            write_api_key (str): ThingSpeak Write API Key
            channel_id (str): ThingSpeak Channel ID
        """
        # ThingSpeak configuration
        self.read_api_key = read_api_key
        self.write_api_key = write_api_key
        self.channel_id = channel_id
        self.read_url = f"https://api.thingspeak.com/channels/{channel_id}/feeds/last.json"
        self.write_url = f"https://api.thingspeak.com/update"
        
        # ThingSpeak field mappings
        self.field_mapping = {
            'read': {
                'Occupancy': 1,  # Real-time occupancy (will be overridden by prediction if using schedule)
                'Temperature': 6,
                'Humidity': 7,
                'Light': 8
            },
            'write': {
                'AC_On': 3,
                'Humidity_Machine_On': 4,
                'Lights_On': 5
            }
        }
        
        # Load occupancy prediction model
        try:
            print("Loading occupancy prediction model...")
            self.occupancy_model = load_model('occupancy_model/occupancy_predictor.h5')
            self.time_scaler = joblib.load('occupancy_model/time_scaler.pkl')
            self.time_features = joblib.load('occupancy_model/time_features.pkl')
            print("Occupancy prediction model loaded successfully!")
        except Exception as e:
            print(f"Error loading occupancy prediction model: {e}")
            self.occupancy_model = None
        
        # Load appliance control model
        try:
            print("Loading appliance control model...")
            self.appliance_model = load_model('model_output/appliance_control_model.h5')
            self.scaler = joblib.load('model_output/appliance_scaler.pkl')
            print("Appliance control model loaded successfully!")
        except Exception as e:
            print(f"Error loading appliance control model: {e}")
            self.appliance_model = None
        
        # Check if models loaded successfully
        if self.occupancy_model is None or self.appliance_model is None:
            print("ERROR: Models could not be loaded! Please ensure training has been completed.")
            raise Exception("Models not found or could not be loaded")
    
    def read_thingspeak_data(self):
        """
        Read the latest data from ThingSpeak
        Returns a dictionary of sensor values
        """
        try:
            # Prepare request parameters
            params = {
                'api_key': self.read_api_key,
                'results': 1
            }
            
            # Make the request
            response = requests.get(self.read_url, params=params)
            response.raise_for_status()
            
            # Parse the response
            data = response.json()
            
            # Extract values for each field
            sensor_data = {}
            field_mapping = self.field_mapping['read']
            
            for sensor, field_num in field_mapping.items():
                field_key = f'field{field_num}'
                if field_key in data and data[field_key] is not None:
                    try:
                        sensor_data[sensor] = float(data[field_key])
                    except (ValueError, TypeError):
                        print(f"Warning: Invalid value for {field_key} in ThingSpeak response")
                        # Set default values based on sensor type
                        if sensor == 'Occupancy':
                            sensor_data[sensor] = 0
                        elif sensor == 'Temperature':
                            sensor_data[sensor] = 22.0
                        elif sensor == 'Humidity':
                            sensor_data[sensor] = 50.0
                        elif sensor == 'Light':
                            sensor_data[sensor] = 300.0
                        else:
                            sensor_data[sensor] = 0.0
                else:
                    print(f"Warning: {field_key} not found or is None in ThingSpeak response")
                    # Set default values
                    if sensor == 'Occupancy':
                        sensor_data[sensor] = 0
                    elif sensor == 'Temperature':
                        sensor_data[sensor] = 22.0
                    elif sensor == 'Humidity':
                        sensor_data[sensor] = 50.0
                    elif sensor == 'Light':
                        sensor_data[sensor] = 300.0
                    else:
                        sensor_data[sensor] = 0.0
            
            # Adjust light reading (divide by 12)
            if 'Light' in sensor_data:
                sensor_data['Light'] = sensor_data['Light'] / 12
            
            # Convert occupancy to integer (0 or 1)
            if 'Occupancy' in sensor_data:
                sensor_data['Occupancy'] = int(sensor_data['Occupancy'])
            
            # Print read values
            print(f"Read from ThingSpeak: {sensor_data}")
            return sensor_data
            
        except requests.exceptions.RequestException as e:
            print(f"Error reading from ThingSpeak: {e}")
            # Return default values
            return {
                'Occupancy': 0,
                'Temperature': 22.0,
                'Humidity': 50.0,
                'Light': 300.0
            }
    
    def predict_occupancy(self, datetime_input=None):
        """
        Predict occupancy based on time and schedule patterns
        
        Args:
            datetime_input (datetime or str, optional): Custom datetime for prediction
                                                      If None, use current time
        
        Returns:
            dict: Dictionary with occupancy prediction and details
        """
        try:
            # Use current time if no datetime is provided
            if datetime_input is None:
                dt = datetime.datetime.now()
                time_str = dt.strftime('%Y-%m-%d %H:%M:%S')
                print(f"Predicting occupancy for current time: {time_str}")
            else:
                # Convert string to datetime if needed
                if isinstance(datetime_input, str):
                    dt = pd.to_datetime(datetime_input)
                else:
                    dt = datetime_input
                print(f"Predicting occupancy for custom time: {dt}")
            
            # Extract time features
            hour = dt.hour
            minute = dt.minute
            day_of_week = dt.weekday()
            is_weekend = 1 if day_of_week >= 5 else 0
            time_decimal = hour + minute/60
            
            # Create cyclic features
            hour_sin = np.sin(2 * np.pi * hour / 24)
            hour_cos = np.cos(2 * np.pi * hour / 24)
            weekday_sin = np.sin(2 * np.pi * day_of_week / 7)
            weekday_cos = np.cos(2 * np.pi * day_of_week / 7)
            
            # Build feature vector to match training features
            features = [hour_sin, hour_cos, weekday_sin, weekday_cos, is_weekend, time_decimal]
            
            # Scale features
            features_scaled = self.time_scaler.transform([features])
            
            # Predict
            prediction_prob = self.occupancy_model.predict(features_scaled)[0][0]
            prediction = 1 if prediction_prob > 0.5 else 0
            
            day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            result = {
                'occupancy': prediction,
                'probability': float(prediction_prob),
                'time': f"{hour:02d}:{minute:02d}",
                'day': day_names[day_of_week],
                'is_weekend': is_weekend
            }
            
            print(f"Occupancy prediction: {'Occupied' if prediction == 1 else 'Unoccupied'} (Probability: {prediction_prob:.4f})")
            print(f"Day: {result['day']}, Time: {result['time']}")
            
            return result
            
        except Exception as e:
            print(f"Error predicting occupancy: {e}")
            # Return default value (assume unoccupied in case of error)
            return {
                'occupancy': 0,
                'probability': 0.0,
                'time': 'unknown',
                'day': 'unknown',
                'is_weekend': 0
            }
    
    def predict_appliance_states(self, sensor_data, predicted_occupancy):
        """
        Predict appliance states using sensor data and predicted occupancy
        
        Args:
            sensor_data (dict): Dictionary of sensor readings
            predicted_occupancy (dict): Dictionary with occupancy prediction
        
        Returns:
            dict: Dictionary with predicted states for each appliance
        """
        try:
            # Validate sensor data
            required_sensors = ['Temperature', 'Humidity', 'Light', 'Occupancy']
            for sensor in required_sensors:
                if sensor not in sensor_data or sensor_data[sensor] is None:
                    print(f"Warning: Missing or None value for {sensor}, using default value")
                    if sensor == 'Temperature':
                        sensor_data[sensor] = 22.0
                    elif sensor == 'Humidity':
                        sensor_data[sensor] = 50.0
                    elif sensor == 'Light':
                        sensor_data[sensor] = 300.0
                    elif sensor == 'Occupancy':
                        sensor_data[sensor] = 0
            
            # Update the occupancy value in sensor data with our prediction
            sensor_data['Occupancy'] = predicted_occupancy['occupancy']
            
            # Prepare input for the appliance model
            input_data = np.array([
                [
                    float(sensor_data['Temperature']), 
                    float(sensor_data['Humidity']), 
                    float(sensor_data['Light']), 
                    float(sensor_data['Occupancy'])
                ]
            ])
            
            # Scale the inputs
            input_scaled = self.scaler.transform(input_data)
            
            # Make prediction
            prediction_probs = self.appliance_model.predict(input_scaled)[0]
            
            # Process predictions
            appliance_names = ['AC_On', 'Humidity_Machine_On', 'Lights_On']
            results = {}
            
            for i, appliance in enumerate(appliance_names):
                try:
                    prob = float(prediction_probs[i])
                    decision = int(prob > 0.5)
                    results[appliance] = {
                        'probability': prob,
                        'decision': decision
                    }
                except (ValueError, TypeError, IndexError) as e:
                    print(f"Warning: Error processing prediction for {appliance}: {e}")
                    results[appliance] = {
                        'probability': 0.0,
                        'decision': 0
                    }
            
            return results
            
        except Exception as e:
            print(f"Error predicting appliance states: {e}")
            # Return default values (all off)
            return {
                'AC_On': {'probability': 0.0, 'decision': 0},
                'Humidity_Machine_On': {'probability': 0.0, 'decision': 0},
                'Lights_On': {'probability': 0.0, 'decision': 0}
            }
    
    def write_to_thingspeak(self, appliance_states):
        """
        Write appliance states to ThingSpeak
        
        Args:
            appliance_states (dict): Dictionary with appliance states
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Prepare data for ThingSpeak
            params = {
                'api_key': self.write_api_key,
            }
            
            # Map appliance states to ThingSpeak fields
            for appliance, field_num in self.field_mapping['write'].items():
                if appliance in appliance_states:
                    params[f'field{field_num}'] = appliance_states[appliance]['decision']
            
            # Make the request
            response = requests.post(self.write_url, params=params)
            response.raise_for_status()
            
            # Check if the update was successful
            if response.text.strip().isdigit():
                print(f"Successfully wrote to ThingSpeak: Entry ID {response.text}")
                print(f"Appliance states: {json.dumps({k: v['decision'] for k, v in appliance_states.items()})}")
                return True
            else:
                print(f"Error writing to ThingSpeak: {response.text}")
                return False
                
        except requests.exceptions.RequestException as e:
            print(f"Error writing to ThingSpeak: {e}")
            return False
    
    def run_control_cycle(self, custom_time=None):
        """
        Run a single control cycle: predict occupancy, read sensors, 
        predict appliance states, update outputs
        
        Args:
            custom_time (datetime or str, optional): Custom time for occupancy prediction
        
        Returns:
            tuple: (predicted_occupancy, appliance_states) or (None, None) if error
        """
        print("\n===== Starting control cycle =====")
        current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f"Current time: {current_time}")
        
        # 1. Predict occupancy based on time (real or custom)
        predicted_occupancy = self.predict_occupancy(custom_time)
        
        # 2. Read sensor data from ThingSpeak
        sensor_data = self.read_thingspeak_data()
        if not sensor_data:
            print("Failed to read sensor data. Aborting control cycle.")
            return None, None
        
        # 3. Predict appliance states
        appliance_states = self.predict_appliance_states(sensor_data, predicted_occupancy)
        if not appliance_states:
            print("Failed to predict appliance states. Aborting control cycle.")
            return predicted_occupancy, None
        
        # 4. Display results
        print("\n----- Prediction Results -----")
        print(f"Schedule-based occupancy: {'Occupied' if predicted_occupancy['occupancy'] == 1 else 'Unoccupied'} " +
              f"(Probability: {predicted_occupancy['probability']:.4f})")
        print(f"Actual occupancy sensor reading: {'Occupied' if sensor_data['Occupancy'] == 1 else 'Unoccupied'}")
        
        for appliance, state in appliance_states.items():
            status = "ON" if state['decision'] == 1 else "OFF"
            print(f"{appliance}: {status} (probability: {state['probability']:.4f})")
        
        # 5. Write appliance states to ThingSpeak
        write_success = self.write_to_thingspeak(appliance_states)
        if not write_success:
            print("Warning: Failed to write to ThingSpeak")
        
        print("===== Control cycle completed =====")
        return predicted_occupancy, appliance_states
    
    def run_continuous(self, interval=60, max_cycles=None):
        """
        Run the control system continuously
        
        Args:
            interval (int): Interval between control cycles in seconds
            max_cycles (int, optional): Maximum number of cycles to run (None for infinite)
        """
        print(f"Starting continuous control with {interval} second interval")
        
        try:
            cycle_count = 0
            while True:
                self.run_control_cycle()
                
                cycle_count += 1
                if max_cycles is not None and cycle_count >= max_cycles:
                    print(f"\nCompleted {max_cycles} control cycles. Stopping.")
                    break
                
                print(f"\nWaiting {interval} seconds until next cycle...")
                time.sleep(interval)
                
        except KeyboardInterrupt:
            print("\nControl system stopped by user")
        except Exception as e:
            print(f"Error in continuous control: {e}")
    
    def simulate_time_range(self, start_time, end_time, step_minutes=60, write_to_thingspeak=False):
        """
        Simulate the system's behavior across a range of times
        
        Args:
            start_time (str): Start time in format 'YYYY-MM-DD HH:MM:SS'
            end_time (str): End time in format 'YYYY-MM-DD HH:MM:SS'
            step_minutes (int): Time step in minutes
            write_to_thingspeak (bool): Whether to write results to ThingSpeak
        
        Returns:
            pd.DataFrame: Results of the simulation
        """
        print(f"\n===== Starting time simulation from {start_time} to {end_time} =====")
        
        # Convert strings to datetime objects
        start_dt = pd.to_datetime(start_time)
        end_dt = pd.to_datetime(end_time)
        
        # Create time range
        current_dt = start_dt
        time_delta = datetime.timedelta(minutes=step_minutes)
        
        # Read sensor data once (will be reused for all time points)
        sensor_data = self.read_thingspeak_data()
        if not sensor_data:
            print("Failed to read sensor data. Aborting simulation.")
            return None
        
        # Store results
        results = []
        
        while current_dt <= end_dt:
            time_str = current_dt.strftime('%Y-%m-%d %H:%M:%S')
            print(f"\n----- Simulating time: {time_str} -----")
            
            # Predict occupancy for this time
            occupancy = self.predict_occupancy(current_dt)
            
            # Predict appliance states
            appliance_states = self.predict_appliance_states(sensor_data, occupancy)
            
            # Store results
            result = {
                'datetime': time_str,
                'day': occupancy['day'],
                'time': occupancy['time'],
                'predicted_occupancy': occupancy['occupancy'],
                'occupancy_probability': occupancy['probability'],
                'AC_On': appliance_states['AC_On']['decision'],
                'Humidity_Machine_On': appliance_states['Humidity_Machine_On']['decision'],
                'Lights_On': appliance_states['Lights_On']['decision'],
                'AC_probability': appliance_states['AC_On']['probability'],
                'Humidity_Machine_probability': appliance_states['Humidity_Machine_On']['probability'],
                'Lights_probability': appliance_states['Lights_On']['probability'],
            }
            
            results.append(result)
            
            # Write to ThingSpeak if enabled
            if write_to_thingspeak:
                self.write_to_thingspeak(appliance_states)
                # Add a delay to avoid rate limiting
                time.sleep(15)
            
            # Move to next time step
            current_dt += time_delta
        
        # Convert results to DataFrame
        results_df = pd.DataFrame(results)
        
        # Save results to CSV
        output_file = 'time_simulation_results.csv'
        results_df.to_csv(output_file, index=False)
        print(f"\nSimulation results saved to {output_file}")
        
        # Create visualization
        self.visualize_simulation_results(results_df)
        
        return results_df
    
    def visualize_simulation_results(self, results_df):
        """
        Create visualizations of the simulation results
        
        Args:
            results_df (pd.DataFrame): Simulation results
        """
        # Extract hour from time for grouping
        results_df['hour'] = results_df['time'].apply(lambda x: int(x.split(':')[0]))
        
        # Create figure
        plt.figure(figsize=(15, 10))
        
        # Plot 1: Occupancy prediction by hour
        plt.subplot(2, 2, 1)
        results_df.groupby('hour')['predicted_occupancy'].mean().plot(kind='bar')
        plt.title('Predicted Occupancy by Hour')
        plt.xlabel('Hour of Day')
        plt.ylabel('Occupancy Rate')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        # Plot 2: Appliance activation by hour
        plt.subplot(2, 2, 2)
        results_df.groupby('hour')[['AC_On', 'Humidity_Machine_On', 'Lights_On']].mean().plot(kind='bar')
        plt.title('Appliance Activation by Hour')
        plt.xlabel('Hour of Day')
        plt.ylabel('Activation Rate')
        plt.xticks(rotation=45)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 3: Timeline of predictions
        plt.subplot(2, 1, 2)
        timeline_df = results_df.set_index('datetime')
        timeline_df[['predicted_occupancy', 'AC_On', 'Humidity_Machine_On', 'Lights_On']].plot(figsize=(15, 5))
        plt.title('Timeline of Predictions')
        plt.xlabel('Date and Time')
        plt.ylabel('State (0=Off, 1=On)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('time_simulation_visualization.png')
        print("Simulation visualization saved to time_simulation_visualization.png")

def test_custom_time_prediction(controller, time_str):
    """
    Test the occupancy prediction with a custom time
    
    Args:
        controller (SmartController): Initialized controller
        time_str (str): Time string in format 'YYYY-MM-DD HH:MM:SS'
    """
    print(f"\n===== Testing with custom time: {time_str} =====")
    
    # Predict occupancy for the custom time
    predicted_occupancy = controller.predict_occupancy(time_str)
    
    # Read current sensor data
    sensor_data = controller.read_thingspeak_data()
    
    # Update the sensor data with our prediction
    if sensor_data:
        sensor_data['Occupancy'] = predicted_occupancy['occupancy']
        
        # Predict appliance states
        appliance_states = controller.predict_appliance_states(sensor_data, predicted_occupancy)
        
        # Display results
        print("\n----- Custom Time Test Results -----")
        print(f"Time: {time_str} ({predicted_occupancy['day']} at {predicted_occupancy['time']})")
        print(f"Predicted occupancy: {'Occupied' if predicted_occupancy['occupancy'] == 1 else 'Unoccupied'} " +
              f"(Probability: {predicted_occupancy['probability']:.4f})")
        
        for appliance, state in appliance_states.items():
            status = "ON" if state['decision'] == 1 else "OFF"
            print(f"{appliance}: {status} (probability: {state['probability']:.4f})")
        
        # Ask if the user wants to write these states to ThingSpeak
        if input("\nWrite these states to ThingSpeak? (y/n): ").lower() == 'y':
            controller.write_to_thingspeak(appliance_states)
            print("States written to ThingSpeak")
        else:
            print("Not writing to ThingSpeak")

def main():
    """Main function to run the smart controller"""
    parser = argparse.ArgumentParser(description='Smart Home Control System')
    parser.add_argument('--read-key', required=True, help='ThingSpeak Read API Key')
    parser.add_argument('--write-key', required=True, help='ThingSpeak Write API Key')
    parser.add_argument('--channel-id', required=True, help='ThingSpeak Channel ID')
    parser.add_argument('--interval', type=int, default=60, help='Control cycle interval in seconds')
    parser.add_argument('--max-cycles', type=int, help='Maximum number of control cycles to run')
    parser.add_argument('--custom-time', help='Test with custom time (format: YYYY-MM-DD HH:MM:SS)')
    parser.add_argument('--simulate', action='store_true', help='Run time simulation mode')
    parser.add_argument('--start-time', help='Simulation start time (format: YYYY-MM-DD HH:MM:SS)')
    parser.add_argument('--end-time', help='Simulation end time (format: YYYY-MM-DD HH:MM:SS)')
    parser.add_argument('--step', type=int, default=60, help='Simulation time step in minutes')
    parser.add_argument('--write-simulation', action='store_true', help='Write simulation results to ThingSpeak')
    
    args = parser.parse_args()
    
    # Create controller
    controller = SmartController(
        read_api_key=args.read_key,
        write_api_key=args.write_key,
        channel_id=args.channel_id
    )
    
    # Handle different run modes
    if args.custom_time:
        # Test with custom time
        test_custom_time_prediction(controller, args.custom_time)
    elif args.simulate:
        # Run time simulation
        if not args.start_time or not args.end_time:
            print("Error: --start-time and --end-time are required for simulation mode")
            return
        
        controller.simulate_time_range(
            start_time=args.start_time,
            end_time=args.end_time,
            step_minutes=args.step,
            write_to_thingspeak=args.write_simulation
        )
    else:
        # Run continuous control
        controller.run_continuous(
            interval=args.interval,
            max_cycles=args.max_cycles
        )

if __name__ == "__main__":
    main()