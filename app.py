import streamlit as st
import pandas as pd
import joblib

# Load your model
model = joblib.load('model.pkl')

st.title('Student Performance Prediction')

# Input features (adjust based on your model)
feature1 = st.number_input('Feature 1')
feature2 = st.number_input('Feature 2')
# Add more inputs as needed

if st.button('Predict'):
    input_features = [feature1, feature2]  # Adjust accordingly
    prediction = model.predict([input_features])
    st.success(f'Prediction: {prediction[0]}')
