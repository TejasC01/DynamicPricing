import joblib
import pandas as pd
import os

# ------------------------------
# PATH SETUP
# ------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

model_path = os.path.join(BASE_DIR, "models", "final_model.pkl")
scaler_path = os.path.join(BASE_DIR, "models", "scaler.pkl")
features_path = os.path.join(BASE_DIR, "models", "feature_names.pkl")

print("Loading model from:", model_path)

# ------------------------------
# LOAD ARTIFACTS (FIXED)
# ------------------------------
model = joblib.load(model_path)
scaler = joblib.load(scaler_path)
feature_names = joblib.load(features_path)


# ------------------------------
# PREDICTION FUNCTION
# ------------------------------
def predict_demand(price, stock, expiry, day):
    day_mapping = {
        "Monday": 0, "Tuesday": 1, "Wednesday": 2,
        "Thursday": 3, "Friday": 4, "Saturday": 5, "Sunday": 6
    }

    day_encoded = day_mapping[day]

    input_data = pd.DataFrame([{
        "price": price,
        "stock_level": stock,
        "days_to_expiry": expiry,
        "day_of_week": day_encoded
    }])

    # Ensure correct feature order
    input_data = input_data[feature_names]

    # Scale
    input_scaled = scaler.transform(input_data)

    # Predict
    prediction = model.predict(input_scaled)[0]

    return prediction