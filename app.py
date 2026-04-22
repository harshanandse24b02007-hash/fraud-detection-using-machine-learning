import streamlit as st
import numpy as np
import joblib

model = joblib.load("models/fraud_model.pkl")
scaler = joblib.load("models/scaler.pkl")

st.title("💳 Credit Card Fraud Detection")

features = []
for i in range(28):
    val = st.number_input(f"V{i+1}", value=0.0)
    features.append(val)

if st.button("Predict"):
    data = np.array([features])
    data = scaler.transform(data)
    pred = model.predict(data)

    if pred[0] == 1:
        st.error("Fraud Detected!")
    else:
        st.success("Legitimate Transaction")
