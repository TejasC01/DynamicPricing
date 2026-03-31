import os
import joblib
import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ------------------------------
# PATH SETUP
# ------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_PATH = os.path.join(BASE_DIR, "models")


# ------------------------------
# LOAD DATA
# ------------------------------
def load_data():
    X_train = joblib.load(os.path.join(MODELS_PATH, "X_train.pkl"))
    X_test = joblib.load(os.path.join(MODELS_PATH, "X_test.pkl"))
    y_train = joblib.load(os.path.join(MODELS_PATH, "y_train.pkl"))
    y_test = joblib.load(os.path.join(MODELS_PATH, "y_test.pkl"))

    feature_names = joblib.load(os.path.join(MODELS_PATH, "feature_names.pkl"))

    return X_train, X_test, y_train, y_test, feature_names


# ------------------------------
# EVALUATION
# ------------------------------
def evaluate(y_true, y_pred, name):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    print(f"\n{name}")
    print("MAE:", mae)
    print("RMSE:", rmse)


# ------------------------------
# MAIN
# ------------------------------
def main():
    print("Loading data...")
    X_train, X_test, y_train, y_test, feature_names = load_data()
    print("Data loaded!")

    # ---- Linear Regression ----
    print("\nTraining Linear Regression...")
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    evaluate(y_test, y_pred_lr, "Linear Regression")

    # ---- Random Forest ----
    print("\nTraining Random Forest...")
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    evaluate(y_test, y_pred_rf, "Random Forest")

    # ------------------------------
    # FEATURE IMPORTANCE (FIXED)
    # ------------------------------
    print("\nFeature Importance:")

    importance = rf.feature_importances_

    df_imp = pd.DataFrame({
        "feature": feature_names,
        "importance": importance
    }).sort_values(by="importance", ascending=False)

    print(df_imp)

    # ------------------------------
    # SAVE MODELS
    # ------------------------------
    joblib.dump(rf, os.path.join(MODELS_PATH, "final_model.pkl"))

    print("\n✅ Final model saved!")


if __name__ == "__main__":
    main()