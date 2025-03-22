import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib

# Load dataset
df = pd.read_csv("medical_insurance_dataset.csv")

# Rename columns to match dataset exactly
df.columns = df.columns.str.strip()  # Remove extra spaces if any

# Print dataset columns to check
print("Dataset Columns:", df.columns)

# Define features (X) and target variable (y)
X = df[['Age', 'Diabetes', 'BloodPressureProblems', 'AnyChronicDiseases', 
        'Height', 'Weight', 'AnyTransplants', 'HistoryOfCancerInFamily', 
        'KnownAllergies', 'NumberOfMajorSurgeries']]

y = df['PremiumPrice']  # Target column

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "insurance_model.pkl")

print("Model trained and saved as insurance_model.pkl")
