import streamlit as st
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
import joblib

st.set_page_config(page_title="Heart Failure Mortality Risk")

@st.cache_resource
def load_assets():
    model = joblib.load("best_xgb_model.pkl") 
    scaler = joblib.load("scaler.pkl")
    return model, scaler

model, scaler = load_assets()

st.title("Heart Failure Mortality Predictor")
st.markdown("This tool uses an **XGBoost model** to predict mortality risk in heart failure patients.")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", 40, 95, 60)
    ef = st.slider("Ejection Fraction (%)", 10, 80, 35)
    creatinine = st.number_input("Serum Creatinine", 0.5, 9.5, 1.1)
    time = st.number_input("Follow-up Period (Days)", 4, 285, 100)

with col2:
    st.info("The model was trained on 299 clinical records and optimized using SMOTE for better risk detection.")

if st.button("Predict Patient Risk"):
    features = np.array([[age, 0, 0, 0, ef, 0, 0, creatinine, 135, 1, 0, time]])
    features_scaled = scaler.transform(features)
    
    prediction = model.predict(features_scaled)
    prob = model.predict_proba(features_scaled)[0][1]

    if prediction[0] == 1:
        st.error(f"High Risk Detected (Probability: {prob:.2%})")
    else:
        st.success(f"Low Risk Detected (Probability: {prob:.2%})")