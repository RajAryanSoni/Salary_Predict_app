import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the model and scaler
model = joblib.load('predict_salary.pkl')
scaler = joblib.load('scaler.pkl')

# Set page config
st.set_page_config(page_title="Salary Prediction App", page_icon=":money_with_wings:", layout="centered")

# Title and subtitle
st.title("Salary Prediction App")
st.subheader("Predict Salary based on Years of Experience")

# Input from user
st.write("Select the number of years of experience:")

# Create a dropdown for years of experience
years = [x for x in range(0, 20)]
years_of_experience = st.selectbox("Years of Experience", years)

# Predict the salary
if st.button("Predict Salary"):
    # Scale the input
    input_data = np.array([[years_of_experience]])
    input_scaled = scaler.transform(input_data)
    
    # Make prediction
    predicted_salary = model.predict(input_scaled)
    
    # Display the result
    st.success(f"The predicted salary is: Rs. predicted_salary[0][0]:,.2f")
