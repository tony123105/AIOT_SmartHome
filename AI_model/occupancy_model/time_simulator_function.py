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
