import joblib
import numpy as np

# Load
model = joblib.load("models/fraud_model.pkl")
scaler = joblib.load("models/scaler.pkl")

# Example input (28 features)
sample = np.zeros((1, 28))

# Scale
sample = scaler.transform(sample)

# Predict
pred = model.predict(sample)

print("Fraud" if pred[0] == 1 else "Not Fraud")
