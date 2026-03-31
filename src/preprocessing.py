import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os


# ------------------------------
# 1. LOAD DATA
# ------------------------------
# Read the dataset from the data folder
df = pd.read_csv("../data/demand_data.csv")

# Preview first few rows → sanity check
print("Sample Data:\n", df.head())

# Check structure → data types + null values
print("\nData Info:")
print(df.info())

# ------------------------------
# 2. HANDLE MISSING VALUES
# ------------------------------
# Count missing values in each column
print("\nMissing Values:\n", df.isnull().sum())

# NOTE:
# Your dataset currently has no missing values.
# But in real-world data, you would fill or drop them here.


# ------------------------------
# 3. ENCODE CATEGORICAL VARIABLES
# ------------------------------
# 'day_of_week' is categorical (even if numeric)
# Convert it into one-hot encoded columns

df = pd.get_dummies(df, columns=["day_of_week"], drop_first=True)

# WHY drop_first=True?
# Avoids redundancy → prevents multicollinearity in linear models


# ------------------------------
# 4. FEATURE-TARGET SPLIT
# ------------------------------
# X = input features
# y = target variable (what we want to predict)

X = df.drop("units_sold", axis=1)
y = df["units_sold"]


# ------------------------------
# 5. TRAIN-TEST SPLIT
# ------------------------------
# Split data into:
# 80% training → model learns from this
# 20% testing → model is evaluated on this

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# random_state ensures same split every time (reproducibility)


# ------------------------------
# 6. FEATURE SCALING
# ------------------------------
# Standardize features → mean=0, std=1
# Important for Linear Regression

scaler = StandardScaler()

# Fit ONLY on training data → learn scaling parameters
X_train_scaled = scaler.fit_transform(X_train)

# Apply SAME transformation to test data
X_test_scaled = scaler.transform(X_test)

# IMPORTANT:
# Never fit on test data → causes data leakage


# ------------------------------
# 7. DEBUG / SANITY CHECKS
# ------------------------------

# Check dataset sizes
print("\nTrain shape:", X_train.shape)
print("Test shape:", X_test.shape)

# Check final feature columns
print("\nFeature Columns:\n", X.columns)

# Check number of features
print("\nNumber of features:", X.shape[1])

# Check scaled data sample
print("\nScaled Train Sample:\n", X_train_scaled[:2])


# ------------------------------
# 8. SAVE ARTIFACTS (CRITICAL)
# ------------------------------
# Create models folder if it doesn't exist
os.makedirs("../models", exist_ok=True)

# Save scaler → needed during inference
joblib.dump(scaler, "../models/scaler.pkl")

# Save processed datasets → avoids recomputation later
joblib.dump(X_train, "../models/X_train.pkl")
joblib.dump(X_test, "../models/X_test.pkl")
joblib.dump(y_train, "../models/y_train.pkl")
joblib.dump(y_test, "../models/y_test.pkl")


# ------------------------------
# FINAL MESSAGE
# ------------------------------
print("\nPreprocessing completed successfully.")