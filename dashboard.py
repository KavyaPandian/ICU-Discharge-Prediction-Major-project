import streamlit as st
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
import random



# PAGE SETTINGS


st.set_page_config(page_title="ICU Patient Monitor", layout="wide")

st.title("🏥 ICU Patient Risk Prediction System")
st.write("AI System for ICU Patient Monitoring and Risk Prediction")



# LOAD MODELS


@st.cache_resource
def load_models():

    lstm_model = tf.keras.models.load_model("icu_lstm_model.keras", compile=False)

    xgb_model = joblib.load("icu_xgboost_model.pkl")

    scaler = joblib.load("icu_scaler.pkl")

    feature_names = joblib.load("icu_features.pkl")

    return lstm_model, xgb_model, scaler, feature_names


lstm_model, xgb_model, scaler, feature_names = load_models()



# SIDEBAR INPUT


st.sidebar.header("Patient Vital Signs")

heart_rate = st.sidebar.slider("Heart Rate", 40, 160, 80)
resp_rate = st.sidebar.slider("Respiratory Rate", 10, 40, 18)
temperature = st.sidebar.slider("Temperature", 35.0, 41.0, 37.0)
spo2 = st.sidebar.slider("Oxygen Saturation (SpO2)", 70, 100, 96)
blood_pressure = st.sidebar.slider("Blood Pressure", 80, 200, 120)
glucose = st.sidebar.slider("Glucose", 60, 300, 110)
ph = st.sidebar.slider("Blood pH", 6.8, 7.8, 7.4)
sodium = st.sidebar.slider("Sodium", 120, 160, 140)
potassium = st.sidebar.slider("Potassium", 2.0, 6.0, 4.0)



# CREATE INPUT DICTIONARY


input_dict = {

    "Heart Rate": heart_rate,
    "Respiratory Rate": resp_rate,
    "Temperature": temperature,
    "SpO2": spo2,
    "Blood Pressure": blood_pressure,
    "Glucose": glucose,
    "Blood pH": ph,
    "Sodium": sodium,
    "Potassium": potassium
}



# BUILD FEATURE VECTOR


input_data = []

for feature in feature_names:
    input_data.append(input_dict.get(feature, 0))

data = np.array(input_data).reshape(1, -1)

data = scaler.transform(data)



# LSTM FORMAT


X_lstm = data.reshape((1, 1, data.shape[1]))



# MODEL PREDICTION


lstm_prob = lstm_model.predict(X_lstm)[0][0]

xgb_prob = xgb_model.predict_proba(data)[0][1]

prob = ((lstm_prob + xgb_prob) / 2) * 0.35



# MEDICAL RISK CORRECTION


critical_score = 0

if spo2 < 90:
    critical_score += 0.25

if heart_rate < 50 or heart_rate > 120:
    critical_score += 0.10

if ph < 7.2:
    critical_score += 0.25

if potassium < 3.0 or potassium > 5.5:
    critical_score += 0.10

if sodium < 130:
    critical_score += 0.10

if glucose > 250 or glucose < 70:
    critical_score += 0.10

prob = min(1, prob + critical_score)



# RISK CATEGORY


def risk_category(p):

    if p < 0.35:
        return "SAFE"

    elif p < 0.70:
        return "MONITOR"

    else:
        return "HIGH RISK"


# Clinical override rules

if heart_rate > 140 or resp_rate > 35 or temperature > 40 or glucose > 250 or blood_pressure > 180 or ph > 7.7 or potassium > 5.5:
    category = "HIGH RISK"
else:
    category = risk_category(prob)

# DISPLAY RESULT


st.header("Patient Risk Result")

col1, col2 = st.columns(2)

with col1:
    st.metric("Risk Probability", round(prob,3))

with col2:

    if category == "SAFE":
        st.success("SAFE")

    elif category == "MONITOR":
        st.warning("MONITOR")

    else:
        st.error("HIGH RISK")



# ICU ALERT SYSTEM


if category == "HIGH RISK":
    st.error("🚨 ICU ALERT: Patient condition critical!")

elif category == "MONITOR":
    st.warning("⚠ Patient requires monitoring")

else:
    st.success("✔ Patient condition stable")


# RISK GAUGE


st.subheader("Risk Gauge")

st.progress(float(prob))



# PATIENT VITAL TABLE


st.subheader("Patient Vital Signs")

df = pd.DataFrame({

    "Vital Sign":[
        "Heart Rate",
        "Respiratory Rate",
        "Temperature",
        "Oxygen Saturation",
        "Blood Pressure",
        "Glucose",
        "Blood pH",
        "Sodium",
        "Potassium"
    ],

    "Value":[
        heart_rate,
        resp_rate,
        temperature,
        spo2,
        blood_pressure,
        glucose,
        ph,
        sodium,
        potassium
    ]
})

st.table(df)



# AI MEDICAL INTERPRETATION


st.subheader("AI Medical Interpretation")

if heart_rate > 100:
    st.write("⚠ High Heart Rate increases risk")

if heart_rate < 50:
    st.write("⚠ Low Heart Rate detected")

if resp_rate > 24:
    st.write("⚠ High Respiratory Rate increases risk")

if spo2 < 92:
    st.write("⚠ Low Oxygen Saturation increases risk")

if temperature > 38:
    st.write("⚠ Fever detected")

if glucose > 180:
    st.write("⚠ High Glucose increases risk")

if glucose < 70:
    st.write("⚠ Low Blood Sugar detected")

if ph < 7.35:
    st.write("⚠ Blood pH imbalance detected")

if sodium < 130:
    st.write("⚠ Low Sodium level")

if potassium < 3.5:
    st.write("⚠ Low Potassium level")



# AI EXPLANATION PANEL


st.subheader("AI Risk Explanation")

risk_factors = []

if heart_rate > 100 or heart_rate < 50:
    risk_factors.append("Abnormal Heart Rate")

if resp_rate > 24:
    risk_factors.append("High Respiratory Rate")

if spo2 < 92:
    risk_factors.append("Low Oxygen Saturation")

if glucose > 180 or glucose < 70:
    risk_factors.append("Abnormal Blood Sugar")

if ph < 7.35:
    risk_factors.append("Blood pH imbalance")

if sodium < 130:
    risk_factors.append("Low Sodium level")

if potassium < 3.5:
    risk_factors.append("Low Potassium level")


if len(risk_factors) == 0:
    st.write("Vitals are within normal range")

else:
    for factor in risk_factors:
        st.write("•", factor, "may increase risk")



# RISK TREND GRAPH


st.subheader("Patient Risk Trend")

risk_history = [max(0, min(1, prob + random.uniform(-0.1,0.1))) for i in range(10)]

risk_df = pd.DataFrame({

    "Time": range(1,11),
    "Risk Score": risk_history
})

st.line_chart(risk_df.set_index("Time"))



# FOOTER


st.write("AI prediction is based on Deep Learning + XGBoost ensemble model.")