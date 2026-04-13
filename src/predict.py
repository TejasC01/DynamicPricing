import os
import joblib
import pandas as pd
import streamlit as st

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_PATH = os.path.join(BASE_DIR, "models")


# ------------------------------
# LOAD ONCE (CACHED)
# ------------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load(os.path.join(MODELS_PATH, "final_model.pkl"))
    scaler = joblib.load(os.path.join(MODELS_PATH, "scaler.pkl"))
    feature_names = joblib.load(os.path.join(MODELS_PATH, "feature_names.pkl"))
    return model, scaler, feature_names


model, scaler, feature_names = load_artifacts()


# ------------------------------
# PREDICT FUNCTION
# ------------------------------
def predict_demand(price, stock, expiry, day):
    input_data = pd.DataFrame([{
        "price": price,
        "stock_level": stock,
        "days_to_expiry": expiry,
        "day_of_week": day
    }])

    input_data = input_data[feature_names]
    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)[0]

    return max(0, int(prediction))