import pickle
import pandas as pd

# Load once (not inside function repeatedly)
with open("models/final_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("models/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("models/feature_names.pkl", "rb") as f:
    feature_names = pickle.load(f)


def predict_demand(price, stock, expiry, day):
    # Convert day to numeric (example)
    day_mapping = {
        "Monday": 0, "Tuesday": 1, "Wednesday": 2,
        "Thursday": 3, "Friday": 4, "Saturday": 5, "Sunday": 6
    }

    day_encoded = day_mapping[day]

    # Create input dataframe EXACTLY like training
    input_data = pd.DataFrame([{
        "price": price,
        "stock_level": stock,
        "days_to_expiry": expiry,
        "day_of_week": day_encoded
    }])

    # Ensure column order matches training
    input_data = input_data[feature_names]

    # Scale if needed
    input_scaled = scaler.transform(input_data)

    # Predict
    prediction = model.predict(input_scaled)[0]

    return prediction