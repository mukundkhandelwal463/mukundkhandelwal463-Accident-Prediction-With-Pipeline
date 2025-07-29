import streamlit as st
import pandas as pd
import numpy as np
from joblib import load

# Load saved preprocessor and model
preprocessor = load("preprocessor.joblib")
model = load("model.joblib")

# Title
st.title("🚧 Road Accident Severity Predictor")

st.markdown("""
This app predicts the **severity of a road accident** based on user inputs.  
Please fill in the information below:
""")

# Example input fields (customize based on your actual features)
def user_input_features():
    # Numerical input
    temperature = st.number_input("Temperature (°C)", value=20.0)
    humidity = st.number_input("Humidity (%)", value=50.0)
    
    # Categorical input
    weather = st.selectbox("Weather Condition", ["Clear", "Rain", "Snow", "Fog", "Cloudy"])
    visibility = st.selectbox("Visibility", ["High", "Medium", "Low"])
    road_surface = st.selectbox("Road Surface", ["Dry", "Wet", "Icy", "Snowy"])
    
    data = {
        'Temperature': temperature,
        'Humidity': humidity,
        'Weather_Condition': weather,
        'Visibility': visibility,
        'Road_Surface': road_surface
    }
    
    return pd.DataFrame([data])

input_df = user_input_features()

if st.button("Predict Severity"):
    try:
        # Preprocess input
        processed_input = preprocessor.transform(input_df)
        # Predict
        prediction = model.predict(processed_input)
        # Display result
        st.success(f"🚨 Predicted Accident Severity: **{prediction[0]}**")
    except Exception as e:
        st.error(f"Error: {e}")
