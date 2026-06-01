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
# LOAD ARTIFACTS
# ------------------------------
model = joblib.load(model_path)
scaler = joblib.load(scaler_path)
feature_names = joblib.load(features_path)


def validate_input_df(input_data: pd.DataFrame) -> pd.DataFrame:
    """Validate that production input is a DataFrame matching training columns."""
    if not isinstance(input_data, pd.DataFrame):
        raise TypeError("Production input must be a pandas DataFrame.")

    missing_columns = [c for c in feature_names if c not in input_data.columns]
    extra_columns = [c for c in input_data.columns if c not in feature_names]

    if missing_columns or extra_columns:
        raise ValueError(
            f"Input DataFrame column mismatch. "
            f"Missing: {missing_columns}. Extra: {extra_columns}."
        )

    return input_data[feature_names].copy()


def prepare_production_input(price, stock, expiry, day) -> pd.DataFrame:
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

    return validate_input_df(input_data)


def predict_demand_from_df(input_data: pd.DataFrame) -> float:
    """Predict demand from a validated production DataFrame."""
    validated_data = validate_input_df(input_data)
    input_scaled = scaler.transform(validated_data)
    prediction = model.predict(input_scaled)[0]
    return prediction


def predict_demand(price, stock, expiry, day):
    input_data = prepare_production_input(price, stock, expiry, day)
    return predict_demand_from_df(input_data)
