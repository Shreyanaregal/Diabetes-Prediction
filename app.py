
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import streamlit as st
import pickle

import pickle
log_reg=pickle.load(open('log25.pkl','rb'))

scaler=pickle.load(open('std25.pkl','rb'))

st.title('Diabetes Prediction App')
st.write("Enter the patient details below to predict the diabetes outcome.")

Pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1)

Glucose = st.number_input("Glucose", min_value=0, max_value=250, value=120)

BloodPressure = st.number_input("Blood Pressure", min_value=0, max_value=150, value=70)

SkinThickness = st.number_input("Skin Thickness", min_value=0, max_value=100, value=20)

Insulin = st.number_input("Insulin", min_value=0, max_value=900, value=80)

BMI = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0)

DiabetesPedigreeFunction = st.number_input(
    "Diabetes Pedigree Function",
    min_value=0.0,
    max_value=3.0,
    value=0.5
)

Age = st.number_input("Age", min_value=1, max_value=100, value=30)


# Prediction button
if st.button("Predict"):

    # Create input array
    input_data = np.array([[
        Pregnancies,
        Glucose,
        BloodPressure,
        SkinThickness,
        Insulin,
        BMI,
        DiabetesPedigreeFunction,
        Age
    ]])

    # Scale the input
    input_scaled = scaler.transform(input_data)

    # Prediction
    prediction = log_reg.predict(input_scaled)

    # Probability
    probability = log_reg.predict_proba(input_scaled)

    if prediction[0] == 1:
        st.error("⚠️ The model predicts: Diabetic")
    else:
        st.success("✅ The model predicts: Not Diabetic")

    st.write(
        "Diabetes Probability:",
        round(probability[0][1] * 100, 2),
        "%"
    )

