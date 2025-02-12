import streamlit as st
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load the trained model
model = joblib.load("model/randomforest.pkl")

# Title of the app
st.title("Sleep Disorder Prediction App")

# Sidebar for user input
st.sidebar.header("Enter Your Details")

# Function to get user input
def get_user_input():
    age = st.sidebar.slider("Age", 18, 80, 25)
    sleep_duration = st.sidebar.slider("Sleep Duration (hrs)", 3.0, 12.0, 6.5)
    quality_of_sleep = st.sidebar.slider("Quality of Sleep (1-10)", 1, 10, 5)
    physical_activity = st.sidebar.slider("Physical Activity Level", 0, 100, 3)
    stress_level = st.sidebar.slider("Stress Level (1-10)", 1, 10, 5)
    heart_rate = st.sidebar.slider("Heart Rate (bpm)", 50, 120, 70)
    daily_steps = st.sidebar.number_input("Daily Steps", min_value=1000, max_value=20000, value=5000)

    # One-hot encoding for categorical variables
    gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
    bp_category = st.sidebar.selectbox("BP Category", ["Normal", "Hypertension", "High"])

    BMI_category = st.sidebar.selectbox("BMI Category",["Normal","Obese","Overweight"])
    # One-hot encode categorical features (Must match training time encoding)
    
    overweight = 1 if BMI_category == "Overweight" else 0
    obese = 1 if BMI_category == "Obese" else 0
    normal_bmi = 1 if BMI_category == "Normal" else 0
    gender_male = 1 if gender == "Male" else 0
    bp_category_elevated = 1 if bp_category == 'Elevated' else 0 # Assuming there is no "Elevated" category in the provided options
    bp_category_normal = 1 if bp_category == "Normal" else 0
    bp_category_hypertension = 1 if bp_category == "Hypertension" else 0
    bp_category_high = 1 if bp_category == "High" else 0
    
    le = joblib.load("model/occupation_encoded.pkl")

    occupation = st.sidebar.selectbox("Occupation", le.classes_)
    occupation_encoded = le.transform([occupation])[0]

    # Create DataFrame (Match Training Features)
    user_data = pd.DataFrame(
        [[age, sleep_duration, quality_of_sleep, physical_activity, stress_level, heart_rate, 
          daily_steps, gender_male,normal_bmi,obese,overweight, occupation_encoded,bp_category_elevated, bp_category_high, bp_category_hypertension, bp_category_normal]],
        columns=['Age', 'Sleep Duration', 'Quality of Sleep', 'Physical Activity Level',
       'Stress Level', 'Heart Rate', 'Daily Steps', 'Gender_Male',
       'BMI Category_Normal', 'BMI Category_Obese', 'BMI Category_Overweight', 'Occupation_cod', 'BP Category_Elevated',
       'BP Category_High', 'BP Category_Hypertension', 'BP Category_Normal']
    )

    return user_data

# Get input from user
input_data = get_user_input()

# Display user input
st.subheader("User Input:")
st.write(input_data)

# Make predictions
if st.button("Predict Sleep Disorder"):
    prediction = model.predict(input_data)
    
    # Mapping back to actual labels
    sleep_disorder_mapping = {1: "Normal", 0: "Insomnia", 2: "Sleep Apnea"}
    prediction_label = sleep_disorder_mapping[prediction[0]]

    st.subheader("Prediction Result:")
    st.write(f"**{prediction_label}**")

