import os
import joblib
import pandas as pd

# ------------------------------
# PATH SETUP
# ------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

model_path = os.path.join(BASE_DIR, "models", "final_model.pkl")
scaler_path = os.path.join(BASE_DIR, "models", "scaler.pkl")
features_path = os.path.join(BASE_DIR, "models", "feature_names.pkl")  # IMPORTANT

# ------------------------------
# LOAD
# ------------------------------
model = joblib.load(model_path)
print("Model loaded")

scaler = None
if os.path.exists(scaler_path):
    scaler = joblib.load(scaler_path)
    print("Scaler loaded")

feature_names = None
if os.path.exists(features_path):
    feature_names = joblib.load(features_path)
    print("Feature names loaded:", feature_names)
else:
    raise Exception("feature_names.pkl NOT FOUND → fix preprocessing save")

# ------------------------------
# TEST FUNCTION (CORRECT)
# ------------------------------
def test_case(price, stock, expiry, day):
    # Map values to correct feature names
    input_dict = {
        "price": price,
        "stock_level": stock,
        "days_to_expiry": expiry,
        "day_of_week": day
    }

    # Create row in EXACT training order
    sample = pd.DataFrame([[input_dict[col] for col in feature_names]],
                          columns=feature_names)

    if scaler:
        sample = scaler.transform(sample)
        sample = pd.DataFrame(sample, columns=feature_names)  # 🔥 FIX

    pred = model.predict(sample)[0]

    # Apply business constraint
    pred = max(0, min(pred, stock))

    print(f"Input → {input_dict} | Predicted = {pred:.2f}")


# ------------------------------
# RUN TESTS
# ------------------------------
print("\nManual Tests:")

test_case(100, 50, 2, 6)
test_case(300, 20, 10, 2)
test_case(50, 100, 1, 5)
test_case(200, 0, 5, 3)

# ------------------------------
# LOAD TEST DATA (NOW EXISTS)
# ------------------------------
X_test_path = os.path.join(BASE_DIR, "models", "X_test.pkl")
y_test_path = os.path.join(BASE_DIR, "models", "y_test.pkl")

if os.path.exists(X_test_path) and os.path.exists(y_test_path):
    print("\nRunning full evaluation...")

    X_test = joblib.load(X_test_path)
    y_test = joblib.load(y_test_path)

    # IMPORTANT: maintain feature names
    X_test = pd.DataFrame(X_test, columns=feature_names)


    y_pred = model.predict(X_test)
    print("\nPrediction Range Check:")
    print("Pred min:", y_pred.min())
    print("Pred max:", y_pred.max())

    from sklearn.metrics import mean_absolute_error, mean_squared_error
    import numpy as np
    import matplotlib.pyplot as plt

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print("\nEvaluation Results:")
    print("MAE:", mae)
    print("RMSE:", rmse)

    # ---- Plot ----
    plt.scatter(y_test, y_pred)
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title("Actual vs Predicted")
    plt.show()

else:
    print("\nTest data not found")