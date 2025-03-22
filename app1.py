import streamlit as st
import numpy as np
import joblib

# Load trained model
model = joblib.load("insurance_model.pkl")

# Streamlit App
st.title("Medical Insurance Cost Prediction")

# User Inputs
age = st.number_input("Age", min_value=0, max_value=100, step=1)
diabetes = st.radio("Diabetes", [0, 1])
blood_pressure = st.radio("Blood Pressure Problems", [0, 1])
chronic_disease = st.radio("Any Chronic Disease", [0, 1])
height = st.number_input("Height (cm)", min_value=50, max_value=250, step=1)
weight = st.number_input("Weight (kg)", min_value=20, max_value=200, step=1)
transplants = st.radio("Any Transplants", [0, 1])
cancer_history = st.radio("History of Cancer in Family", [0, 1])
allergies = st.radio("Known Allergies", [0, 1])
surgeries = st.number_input("Number of Major Surgeries", min_value=0, max_value=20, step=1)

# Predict Button
if st.button("Predict Insurance Cost"):
    input_data = np.array([[age, diabetes, blood_pressure, chronic_disease, height, weight,
                            transplants, cancer_history, allergies, surgeries]])
    
    prediction = model.predict(input_data)[0]
    
    st.success(f"Estimated Insurance Cost: ${prediction:.2f}")
