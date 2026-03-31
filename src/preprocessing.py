import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def preprocess(df, save=False, models_path=None):
    if models_path is None:
        models_path = os.path.join(BASE_DIR, "models")
    df = df.copy()

    # ------------------------------
    # DROP USELESS COLUMN
    # ------------------------------
    df = df.drop("product_id", axis=1)

    # ------------------------------
    # FEATURES / TARGET
    # ------------------------------
    X = df.drop("units_sold", axis=1)
    y = df["units_sold"]

    feature_names = X.columns.tolist()

    # ------------------------------
    # TRAIN-TEST SPLIT
    # ------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ------------------------------
    # SCALING (for LR only, but kept for consistency)
    # ------------------------------
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)

    # ------------------------------
    # SAVE
    # ------------------------------
    if save:
        os.makedirs(models_path, exist_ok=True)

        joblib.dump(scaler, os.path.join(models_path, "scaler.pkl"))
        joblib.dump(feature_names, os.path.join(models_path, "feature_names.pkl"))

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_names