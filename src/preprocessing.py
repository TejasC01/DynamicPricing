import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ------------------------------
# PATH SETUP
# ------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(BASE_DIR, "data", "demand_data.csv")
MODELS_PATH = os.path.join(BASE_DIR, "models")

os.makedirs(MODELS_PATH, exist_ok=True)


# ------------------------------
# LOAD DATA
# ------------------------------
df = pd.read_csv(DATA_PATH)

print("Sample Data:\n", df.head())


# ------------------------------
# FEATURE ENGINEERING (IMPORTANT)
# ------------------------------
df["is_expiring_soon"] = (df["days_to_expiry"] < 3).astype(int)

# Drop useless feature
df = df.drop("product_id", axis=1)


# ------------------------------
# ENCODE CATEGORICAL
# ------------------------------
df = pd.get_dummies(df, columns=["day_of_week"], drop_first=True)


# ------------------------------
# SPLIT
# ------------------------------
X = df.drop("units_sold", axis=1)
y = df["units_sold"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# ------------------------------
# SCALING
# ------------------------------
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# ------------------------------
# SAVE EVERYTHING
# ------------------------------
joblib.dump(scaler, os.path.join(MODELS_PATH, "scaler.pkl"))

joblib.dump(X_train_scaled, os.path.join(MODELS_PATH, "X_train.pkl"))
joblib.dump(X_test_scaled, os.path.join(MODELS_PATH, "X_test.pkl"))

joblib.dump(y_train, os.path.join(MODELS_PATH, "y_train.pkl"))
joblib.dump(y_test, os.path.join(MODELS_PATH, "y_test.pkl"))

# ✅ Save feature names (CRITICAL FIX)
joblib.dump(list(X.columns), os.path.join(MODELS_PATH, "feature_names.pkl"))

print("\n✅ Preprocessing complete.")