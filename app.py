
import streamlit as st
import numpy as np
from src.predict import load_model, predict
import joblib

st.title("Heart Disease Prediction")

model = load_model("models/heart_model.pkl")
scaler = joblib.load("models/scaler.pkl")

age = st.number_input("Age", 29, 77)
sex = st.selectbox("Sex", [0, 1])
cp = st.selectbox("Chest Pain Type", [0, 1, 2, 3])
trestbps = st.number_input("Resting BP", 80, 200)
chol = st.number_input("Cholesterol", 100, 600)
fbs = st.selectbox("Fasting Blood Sugar > 120", [0, 1])
restecg = st.selectbox("Rest ECG", [0, 1, 2])
thalach = st.number_input("Max Heart Rate", 60, 220)
exang = st.selectbox("Exercise Induced Angina", [0, 1])
oldpeak = st.number_input("ST depression", 0.0, 6.0)
slope = st.selectbox("Slope of ST", [0, 1, 2])
ca = st.selectbox("Number of major vessels", [0, 1, 2, 3])
thal = st.selectbox("Thal", [0, 1, 2, 3])

if st.button("Predict"):
    data = [age, sex, cp, trestbps, chol, fbs, restecg,
            thalach, exang, oldpeak, slope, ca, thal]
    result = predict(model, scaler, data)
    st.success("Heart Disease Detected" if result == 1 else "No Heart Disease")
