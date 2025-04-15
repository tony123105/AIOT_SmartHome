import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
import inspect
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Create output directory
os.makedirs('occupancy_model', exist_ok=True)

def load_and_prepare_data(file_path):
    """
    Load and prepare the occupancy dataset for training the occupancy predictor
    """
    print(f"Loading data from {file_path}...")
    data = pd.read_csv(file_path)
    
    # Convert date string to datetime
    data['date'] = pd.to_datetime(data['date'])
    
    # Extract time features
    data['hour'] = data['date'].dt.hour
    data['minute'] = data['date'].dt.minute
    data['day_of_week'] = data['date'].dt.dayofweek  # Monday=0, Sunday=6
    data['is_weekend'] = data['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
    data['month'] = data['date'].dt.month
    data['day'] = data['date'].dt.day
    
    # Create time decimal feature (hour + minute/60)
    data['time_decimal'] = data['hour'] + data['minute']/60
    
    # Create cyclic time features (sin/cos transformations)
    # These help the model understand the cyclic nature of time
    data['hour_sin'] = np.sin(2 * np.pi * data['hour'] / 24)
    data['hour_cos'] = np.cos(2 * np.pi * data['hour'] / 24)
    data['weekday_sin'] = np.sin(2 * np.pi * data['day_of_week'] / 7)
    data['weekday_cos'] = np.cos(2 * np.pi * data['day_of_week'] / 7)
    
    print("Data preparation complete.")
    return data

def analyze_occupancy_patterns(data):
    """
    Analyze and visualize occupancy patterns by time
    """
    print("Analyzing occupancy patterns...")
    
    # Group data by hour and day of week to see patterns
    hourly_occupancy = data.groupby('hour')['Occupancy'].mean()
    weekly_occupancy = data.groupby('day_of_week')['Occupancy'].mean()
    
    # Hour of day vs. occupancy
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    sns.lineplot(x=hourly_occupancy.index, y=hourly_occupancy.values)
    plt.title('Occupancy by Hour of Day')
    plt.xlabel('Hour')
    plt.ylabel('Occupancy Rate')
    plt.xticks(range(0, 24, 2))
    plt.grid(True, alpha=0.3)
    
    # Day of week vs. occupancy
    plt.subplot(1, 2, 2)
    sns.barplot(x=weekly_occupancy.index, y=weekly_occupancy.values)
    plt.title('Occupancy by Day of Week')
    plt.xlabel('Day (0=Monday, 6=Sunday)')
    plt.ylabel('Occupancy Rate')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('occupancy_model/occupancy_patterns.png')
    
    # Create heatmap of hour vs. day of week
    pivot = data.pivot_table(
        index='day_of_week', 
        columns='hour', 
        values='Occupancy', 
        aggfunc='mean'
    )
    
    plt.figure(figsize=(15, 6))
    sns.heatmap(pivot, cmap='viridis', annot=False, fmt='.2f', cbar_kws={'label': 'Occupancy Rate'})
    plt.title('Occupancy Rate by Day of Week and Hour')
    plt.xlabel('Hour of Day')
    plt.ylabel('Day of Week (0=Monday, 6=Sunday)')
    plt.tight_layout()
    plt.savefig('occupancy_model/occupancy_heatmap.png')
    
    print("Occupancy pattern analysis complete. Visualizations saved.")

def build_occupancy_predictor(input_shape):
    """
    Build a neural network model to predict occupancy based on time features
    """
    model = Sequential([
        Dense(64, activation='relu', input_shape=(input_shape,)),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid', name='occupancy_output')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    print(f"Built occupancy prediction model with input shape: {input_shape}")
    return model

def train_occupancy_model(data):
    """
    Train the occupancy prediction model and save it
    """
    # Define features for occupancy prediction
    time_features = [
        'hour_sin', 'hour_cos', 'weekday_sin', 'weekday_cos', 
        'is_weekend', 'time_decimal'
    ]
    
    # Split data
    X = data[time_features]
    y = data['Occupancy']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save scaler and feature list
    joblib.dump(scaler, 'occupancy_model/time_scaler.pkl')
    joblib.dump(time_features, 'occupancy_model/time_features.pkl')
    
    # Build model
    model = build_occupancy_predictor(X_train_scaled.shape[1])
    
    # Callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        ModelCheckpoint('occupancy_model/best_occupancy_model.h5', save_best_only=True)
    ]
    
    # Train model
    print("Training occupancy prediction model...")
    history = model.fit(
        X_train_scaled, y_train,
        epochs=50,
        batch_size=32,
        validation_split=0.2,
        callbacks=callbacks,
        verbose=1
    )
    
    # Save the final model
    model.save('occupancy_model/occupancy_predictor.h5')
    
    # Save training history
    pd.DataFrame(history.history).to_csv('occupancy_model/occupancy_training_history.csv', index=False)
    
    # Evaluate model
    y_pred_probs = model.predict(X_test_scaled)
    y_pred = (y_pred_probs > 0.5).astype(int)
    
    # Print evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nOccupancy Prediction Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Create confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
              xticklabels=['Predicted Unoccupied', 'Predicted Occupied'],
              yticklabels=['Actually Unoccupied', 'Actually Occupied'])
    plt.title('Occupancy Prediction Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('occupancy_model/occupancy_confusion_matrix.png')
    
    # Plot training history
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('occupancy_model/occupancy_training_history.png')
    
    print("Model training and evaluation complete!")
    return model, scaler, time_features

def create_time_simulation_utility():
    """
    Create a utility function that will be used for time simulation
    """
    def predict_occupancy_for_time(model, scaler, time_features, datetime_input):
        """
        Predict occupancy for a specific datetime
        
        Args:
            model: Trained occupancy prediction model
            scaler: Fitted StandardScaler for time features
            time_features: List of time feature names
            datetime_input: datetime object or string in format 'YYYY-MM-DD HH:MM:SS'
            
        Returns:
            dict: Dictionary with occupancy prediction and probability
        """
        # Convert string to datetime if needed
        if isinstance(datetime_input, str):
            dt = pd.to_datetime(datetime_input)
        else:
            dt = datetime_input
        
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
        
        # Build feature vector
        features = [hour_sin, hour_cos, weekday_sin, weekday_cos, is_weekend, time_decimal]
        
        # Scale features
        features_scaled = scaler.transform([features])
        
        # Predict
        prediction_prob = model.predict(features_scaled)[0][0]
        prediction = 1 if prediction_prob > 0.5 else 0
        
        return {
            'datetime': dt,
            'occupancy': prediction,
            'probability': float(prediction_prob),
            'hour': hour,
            'minute': minute,
            'day_of_week': day_of_week,
            'is_weekend': is_weekend
        }
    
    # Save the function as a string to be included in the controller
    function_code = inspect.getsource(predict_occupancy_for_time)
    with open('occupancy_model/time_simulator_function.py', 'w') as f:
        f.write(function_code)
    
    print("Time simulation utility created and saved.")

def test_model_with_sample_times(model, scaler, time_features):
    """
    Test the trained model with sample times to verify predictions
    """
    print("\nTesting occupancy prediction with sample times:")
    
    # Create sample times for testing
    sample_times = [
        "2023-11-01 09:00:00",  # Wednesday morning
        "2023-11-01 12:30:00",  # Wednesday midday
        "2023-11-01 18:00:00",  # Wednesday evening
        "2023-11-01 03:00:00",  # Wednesday night
        "2023-11-04 10:00:00",  # Saturday morning
        "2023-11-05 15:00:00",  # Sunday afternoon
    ]
    
    # Define the prediction function
    def predict_for_time(dt_str):
        dt = pd.to_datetime(dt_str)
        
        # Extract features
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
        
        # Build feature vector
        features = [hour_sin, hour_cos, weekday_sin, weekday_cos, is_weekend, time_decimal]
        
        # Scale features
        features_scaled = scaler.transform([features])
        
        # Predict
        prediction_prob = model.predict(features_scaled)[0][0]
        prediction = 1 if prediction_prob > 0.5 else 0
        
        return {
            'datetime': dt_str,
            'day': ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'][day_of_week],
            'time': f"{hour:02d}:{minute:02d}",
            'occupancy': prediction,
            'probability': float(prediction_prob)
        }
    
    # Test predictions
    results = []
    for time_str in sample_times:
        prediction = predict_for_time(time_str)
        results.append(prediction)
        print(f"Time: {prediction['day']} at {prediction['time']} - " +
              f"Predicted: {'Occupied' if prediction['occupancy'] == 1 else 'Unoccupied'} " +
              f"(Probability: {prediction['probability']:.4f})")
    
    # Save test results
    pd.DataFrame(results).to_csv('occupancy_model/sample_predictions.csv', index=False)
    
    print("Sample predictions completed and saved.")

def main():
    """
    Main function to execute the occupancy model training process
    """
    print("=== Training Occupancy Prediction Model ===")
    
    # 1. Load and prepare data
    data = load_and_prepare_data('Modified_Occupancy_with_appliances.csv')
    
    # 2. Analyze occupancy patterns
    analyze_occupancy_patterns(data)
    
    # 3. Train the occupancy prediction model
    model, scaler, time_features = train_occupancy_model(data)
    
    # 4. Create time simulation utility
    create_time_simulation_utility()
    
    # 5. Test the model with sample times
    test_model_with_sample_times(model, scaler, time_features)
    
    print("\nOccupancy prediction model training complete!")
    print("Model and related files saved in the 'occupancy_model' directory")

if __name__ == "__main__":
    main()