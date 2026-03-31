import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib


def preprocess(df):
    # Create models folder if it doesn't exist
    os.makedirs("models", exist_ok=True)

    # Feature engineering
    df["is_expiring_soon"] = (df["days_to_expiry"] < 3).astype(int)

    # Drop useless feature
    df = df.drop("product_id", axis=1)

    # One-hot encoding
    df = pd.get_dummies(df, columns=["day_of_week"], drop_first=True)

    # Split
    X = df.drop("units_sold", axis=1)
    y = df["units_sold"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Save scaler
    joblib.dump(scaler, "models/scaler.pkl")

    feature_names = X.columns.tolist()

    return X_train_scaled, X_test_scaled, y_train, y_test, feature_names