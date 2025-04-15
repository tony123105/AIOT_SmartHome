import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Create output directory
MODEL_OUTPUT_DIR = 'model_output'
os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)

def preprocess_data(file_path):
    """
    Load and preprocess the occupancy dataset
    """
    print(f"Loading data from {file_path}...")
    try:
        data = pd.read_csv(file_path)
        print(f"Successfully loaded data with {len(data)} rows and {len(data.columns)} columns")
        
        # Check if the data has the expected columns
        required_columns = ['date', 'Temperature', 'Humidity', 'Light', 'Occupancy']
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            print(f"WARNING: Missing required columns: {missing_columns}")
            return None
            
        # Convert date string to datetime
        data['date'] = pd.to_datetime(data['date'])
        
        # Extract basic time features
        data['hour'] = data['date'].dt.hour
        data['minute'] = data['date'].dt.minute
        
        print("Data preprocessing complete")
        return data
        
    except Exception as e:
        print(f"Error loading or preprocessing data: {e}")
        return None

def create_appliance_targets(data):
    """
    Create target variables for each appliance based on sensor data
    """
    print("Creating appliance target variables...")
    
    # Make a copy to avoid modifying the original dataframe
    df = data.copy()
    
    # 1. AC Control Rules
    df['AC_On'] = 0
    # AC on when temperature is high and room is occupied
    df.loc[(df['Temperature'] > 23.0) & (df['Occupancy'] == 1), 'AC_On'] = 1
    # AC stays off when room is not occupied
    df.loc[df['Occupancy'] == 0, 'AC_On'] = 0
    
    # 2. Humidity Machine Control Rules
    df['Humidity_Machine_On'] = 0
    # Humidity machine on when humidity is low and room is occupied
    df.loc[(df['Humidity'] < 30.0) & (df['Occupancy'] == 1), 'Humidity_Machine_On'] = 1
    # Humidity machine off when humidity is high
    df.loc[df['Humidity'] > 60.0, 'Humidity_Machine_On'] = 0
    # Always off when room is unoccupied
    df.loc[df['Occupancy'] == 0, 'Humidity_Machine_On'] = 0
    
    # 3. Lights Control Rules
    df['Lights_On'] = 0
    # Lights on when light level is low and room is occupied
    df.loc[(df['Light'] < 480.0) & (df['Occupancy'] == 1), 'Lights_On'] = 1
    # Lights off when light level is high or room is unoccupied
    df.loc[df['Light'] > 800.0, 'Lights_On'] = 0
    df.loc[df['Occupancy'] == 0, 'Lights_On'] = 0
    
    # Check the distribution of each target
    for target in ['AC_On', 'Humidity_Machine_On', 'Lights_On']:
        counts = df[target].value_counts()
        print(f"{target} distribution:")
        for value, count in counts.items():
            percentage = count / len(df) * 100
            print(f"  {value}: {count} ({percentage:.1f}%)")
    
    print("Target variables created successfully")
    return df

def prepare_training_data(data):
    """
    Prepare features and targets for model training
    """
    print("Preparing training data...")
    
    # Define the feature set (inputs for the model)
    features = ['Temperature', 'Humidity', 'Light', 'Occupancy']
    
    # Define target variables (outputs for the model)
    targets = ['AC_On', 'Humidity_Machine_On', 'Lights_On']
    
    # Split features and targets
    X = data[features]
    y = data[targets]
    
    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Testing set: {X_test.shape[0]} samples")
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save the scaler and feature lists
    joblib.dump(scaler, os.path.join(MODEL_OUTPUT_DIR, 'appliance_scaler.pkl'))
    joblib.dump(features, os.path.join(MODEL_OUTPUT_DIR, 'appliance_features.pkl'))
    joblib.dump(targets, os.path.join(MODEL_OUTPUT_DIR, 'appliance_targets.pkl'))
    
    print("Training data prepared and files saved:")
    print(f"  - {os.path.join(MODEL_OUTPUT_DIR, 'appliance_scaler.pkl')}")
    print(f"  - {os.path.join(MODEL_OUTPUT_DIR, 'appliance_features.pkl')}")
    print(f"  - {os.path.join(MODEL_OUTPUT_DIR, 'appliance_targets.pkl')}")
    
    return {
        'X_train': X_train_scaled,
        'X_test': X_test_scaled,
        'y_train': y_train,
        'y_test': y_test,
        'features': features,
        'targets': targets,
        'scaler': scaler
    }

def build_model(input_dim, output_dim):
    """
    Build the neural network model for appliance control
    """
    print(f"Building model with {input_dim} inputs and {output_dim} outputs...")
    
    model = Sequential([
        Dense(64, activation='relu', input_shape=(input_dim,)),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(output_dim, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    print("Model architecture:")
    model.summary()
    
    return model

def train_model(model, training_data):
    """
    Train the appliance control model
    """
    print("Starting model training...")
    
    # Define callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            os.path.join(MODEL_OUTPUT_DIR, 'best_appliance_model.h5'),
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )
    ]
    
    # Train the model
    history = model.fit(
        training_data['X_train'], 
        training_data['y_train'],
        epochs=50,
        batch_size=32,
        validation_split=0.2,
        callbacks=callbacks,
        verbose=1
    )
    
    # Save the final model
    model_path = os.path.join(MODEL_OUTPUT_DIR, 'appliance_control_model.h5')
    model.save(model_path)
    print(f"Model saved to {model_path}")
    
    # Save training history
    history_df = pd.DataFrame(history.history)
    history_path = os.path.join(MODEL_OUTPUT_DIR, 'training_history.csv')
    history_df.to_csv(history_path, index=False)
    print(f"Training history saved to {history_path}")
    
    return history

def evaluate_model(model, training_data):
    """
    Evaluate the model on test data
    """
    print("Evaluating model on test data...")
    
    # Make predictions
    y_pred_probs = model.predict(training_data['X_test'])
    y_pred = (y_pred_probs > 0.5).astype(int)
    
    # Evaluate accuracy for each appliance
    results = {}
    for i, target in enumerate(training_data['targets']):
        true_values = training_data['y_test'][target].values
        pred_values = y_pred[:, i]
        
        accuracy = accuracy_score(true_values, pred_values)
        report = classification_report(true_values, pred_values, output_dict=True)
        cm = confusion_matrix(true_values, pred_values)
        
        results[target] = {
            'accuracy': accuracy,
            'report': report,
            'confusion_matrix': cm
        }
        
        print(f"\n--- {target} Evaluation ---")
        print(f"Accuracy: {accuracy:.4f}")
        print(classification_report(true_values, pred_values))
        
        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                  xticklabels=['Predicted OFF', 'Predicted ON'],
                  yticklabels=['Actual OFF', 'Actual ON'])
        plt.title(f'Confusion Matrix - {target}')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.tight_layout()
        
        cm_path = os.path.join(MODEL_OUTPUT_DIR, f'{target}_confusion_matrix.png')
        plt.savefig(cm_path)
        print(f"Confusion matrix saved to {cm_path}")
    
    # Save evaluation results
    eval_path = os.path.join(MODEL_OUTPUT_DIR, 'evaluation_results.pkl')
    joblib.dump(results, eval_path)
    print(f"Evaluation results saved to {eval_path}")
    
    # Plot training history
    history_df = pd.read_csv(os.path.join(MODEL_OUTPUT_DIR, 'training_history.csv'))
    
    plt.figure(figsize=(12, 5))
    
    # Plot loss
    plt.subplot(1, 2, 1)
    for col in history_df.columns:
        if 'loss' in col:
            plt.plot(history_df[col], label=col)
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot accuracy
    plt.subplot(1, 2, 2)
    for col in history_df.columns:
        if 'accuracy' in col:
            plt.plot(history_df[col], label=col)
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    history_plot_path = os.path.join(MODEL_OUTPUT_DIR, 'training_history_plot.png')
    plt.savefig(history_plot_path)
    print(f"Training history plot saved to {history_plot_path}")
    
    return results

def test_model_functionality():
    """
    Test the model functionality by loading it and making a sample prediction
    """
    print("\n=== Testing Model Functionality ===")
    
    try:
        # Check if all necessary files exist
        required_files = [
            'appliance_control_model.h5',
            'appliance_scaler.pkl',
            'appliance_features.pkl',
            'appliance_targets.pkl'
        ]
        
        missing_files = []
        for file in required_files:
            path = os.path.join(MODEL_OUTPUT_DIR, file)
            if not os.path.exists(path):
                missing_files.append(file)
        
        if missing_files:
            print(f"ERROR: Missing required files: {missing_files}")
            return False
        
        # Load the model and related components
        model = load_model(os.path.join(MODEL_OUTPUT_DIR, 'appliance_control_model.h5'))
        scaler = joblib.load(os.path.join(MODEL_OUTPUT_DIR, 'appliance_scaler.pkl'))
        features = joblib.load(os.path.join(MODEL_OUTPUT_DIR, 'appliance_features.pkl'))
        targets = joblib.load(os.path.join(MODEL_OUTPUT_DIR, 'appliance_targets.pkl'))
        
        print("Successfully loaded all components:")
        print(f"  - Model input features: {features}")
        print(f"  - Model target outputs: {targets}")
        
        # Create sample inputs for testing
        test_cases = [
            # {Temperature, Humidity, Light, Occupancy}
            {'name': 'Occupied, hot, bright', 'values': [25.0, 50.0, 600.0, 1]},
            {'name': 'Occupied, hot, dark', 'values': [25.0, 50.0, 200.0, 1]},
            {'name': 'Occupied, cool, dark', 'values': [21.0, 50.0, 200.0, 1]},
            {'name': 'Unoccupied', 'values': [25.0, 50.0, 200.0, 0]}
        ]
        
        # Test each case
        print("\nTest Case Predictions:")
        for test_case in test_cases:
            # Prepare input
            input_data = np.array([test_case['values']])
            input_scaled = scaler.transform(input_data)
            
            # Make prediction
            prediction_probs = model.predict(input_scaled)[0]
            prediction = (prediction_probs > 0.5).astype(int)
            
            # Print results
            print(f"\nTest Case: {test_case['name']}")
            print(f"Input: {dict(zip(features, test_case['values']))}")
            print("Predictions:")
            for i, target in enumerate(targets):
                status = "ON" if prediction[i] == 1 else "OFF"
                print(f"  - {target}: {status} ({prediction_probs[i]:.4f})")
        
        print("\nModel functionality test completed successfully!")
        return True
        
    except Exception as e:
        print(f"ERROR testing model functionality: {e}")
        return False

def main():
    """
    Main function to run the training process
    """
    print("=== Appliance Control Model Training ===")
    
    # Check if output directory exists
    if not os.path.exists(MODEL_OUTPUT_DIR):
        os.makedirs(MODEL_OUTPUT_DIR)
        print(f"Created output directory: {MODEL_OUTPUT_DIR}")
    
    # 1. Load and preprocess data
    data = preprocess_data('Modified_Occupancy_with_appliances.csv')
    if data is None:
        print("ERROR: Could not load or preprocess data. Exiting.")
        return
    
    # 2. Create appliance targets
    data_with_targets = create_appliance_targets(data)
    
    # 3. Prepare training data
    training_data = prepare_training_data(data_with_targets)
    
    # 4. Build model
    model = build_model(
        input_dim=len(training_data['features']),
        output_dim=len(training_data['targets'])
    )
    
    # 5. Train model
    history = train_model(model, training_data)
    
    # 6. Evaluate model
    evaluation = evaluate_model(model, training_data)
    
    # 7. Test model functionality
    test_successful = test_model_functionality()
    
    if test_successful:
        print("\n=== Training completed successfully! ===")
        print(f"All necessary files have been saved to {MODEL_OUTPUT_DIR}")
        print("\nYou can now proceed to the next step: training the occupancy predictor")
        print("Run: python train_occupancy_predictor.py")
    else:
        print("\n=== WARNING: Training completed but functionality test failed ===")
        print("Please check the error messages above and try to resolve the issues before proceeding")

if __name__ == "__main__":
    main()