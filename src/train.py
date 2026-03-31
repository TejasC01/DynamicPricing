import os
import joblib
import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from data_generation import generate_data
from preprocessing import preprocess


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "demand_data.csv")
MODELS_PATH = os.path.join(BASE_DIR, "models")


def evaluate(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return mae, rmse


def load_or_generate_data():
    if os.path.exists(DATA_PATH):
        print("Loading existing data...")
        return pd.read_csv(DATA_PATH)
    else:
        print("Generating new data...")
        return generate_data(save=True)


def main():
    df = load_or_generate_data()

    print("Preprocessing data...")
    X_train, X_test, y_train, y_test, scaler, feature_names = preprocess(df, save=True)

    # ------------------------------
    # LINEAR REGRESSION
    # ------------------------------
    print("\nTraining Linear Regression...")
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)

    mae_lr, rmse_lr = evaluate(y_test, y_pred_lr)

    print("Linear Regression")
    print("MAE:", mae_lr)
    print("RMSE:", rmse_lr)

    # ------------------------------
    # RANDOM FOREST
    # ------------------------------
    print("\nTraining Random Forest...")
    rf = RandomForestRegressor(
        n_estimators=300,
        max_depth=10,
        min_samples_split=5,
        random_state=42
    )
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)

    mae_rf, rmse_rf = evaluate(y_test, y_pred_rf)

    print("Random Forest")
    print("MAE:", mae_rf)
    print("RMSE:", rmse_rf)

    # ------------------------------
    # SELECT BEST MODEL
    # ------------------------------
    if rmse_rf < rmse_lr:
        best_model = rf
        model_name = "Random Forest"
    else:
        best_model = lr
        model_name = "Linear Regression"

    print(f"\n✅ Best Model: {model_name}")

    # ------------------------------
    # FEATURE IMPORTANCE
    # ------------------------------
    if model_name == "Random Forest":
        print("\nFeature Importance:")
        importance = rf.feature_importances_

        for name, val in sorted(zip(feature_names, importance), key=lambda x: x[1], reverse=True):
            print(f"{name}: {val:.4f}")

    # ------------------------------
    # SAVE
    # ------------------------------
    os.makedirs(MODELS_PATH, exist_ok=True)

    joblib.dump(best_model, os.path.join(MODELS_PATH, "final_model.pkl"))
    joblib.dump(X_test, os.path.join(MODELS_PATH, "X_test.pkl"))
    joblib.dump(y_test, os.path.join(MODELS_PATH, "y_test.pkl"))

    print("\n✅ Model saved!")


if __name__ == "__main__":
    main()