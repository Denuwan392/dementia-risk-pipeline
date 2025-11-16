# test.py
"""
Test script for dementia risk prediction model.
Validates model loading and prediction on a realistic patient.
"""

import pandas as pd
import joblib
import os

# ----------------------------
# 1. Load the model
# ----------------------------
model_path = "models/dementia_risk_model_20251115_1748_auc09704.pkl"

if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model not found at {model_path}. Did you run the full notebook?")

print("🔍 Loading model...")
model = joblib.load(model_path)
print("✅ Model loaded successfully.\n")

# ----------------------------
# 2. Define a realistic test patient (RAW features only)
# ----------------------------
raw_patient = pd.DataFrame([{
    # Demographics
    'AGE': 82,
    'EDUC': 8,
    'SEX': 2,
    'HISPANIC': 0,
    'RACE': 1,
    'PRIMLANG': 1,
    'MARISTAT': 2,

    # Lifestyle
    'SMOKYRS': 40,
    'PACKSPER': 1,
    'QUITSMOK': 2010,
    'ALCOCCAS': 1,
    'ALCFREQ': 2,

    # Living / Social
    'NACCLIVS': 2,
    'INLIVWTH': 1,
    'INVISITS': 3,
    'INCALLS': 4,
    'NACCFAM': 1,
    'NACCOM': -4,

    # Functional status
    'HOMEHOBB': 3,
    'PERSCARE': 2,
}])

print("🧑 Raw test patient (before engineering):")
print(raw_patient.T.to_string(header=False))
print()

# ----------------------------
# 3. Apply Feature Engineering (MUST match notebook logic)
# ----------------------------
def engineer_features(df):
    """Apply the same engineering as in the notebook."""
    df = df.copy()
    # AGE_GROUP
    df['AGE_GROUP'] = pd.cut(
        df['AGE'],
        bins=[0, 64, 74, 84, 130],
        labels=['65-', '65-74', '75-84', '85+'],
        right=False
    )
    # EDUC_GROUP
    df['EDUC_GROUP'] = pd.cut(
        df['EDUC'],
        bins=[0, 8, 12, 16, 100],
        labels=['<HS', 'HS', 'Some College', 'College+'],
        right=False
    )
    return df

test_patient = engineer_features(raw_patient)

print("🛠️ After feature engineering:")
print("Added:", [col for col in test_patient.columns if col not in raw_patient.columns])
print()

# ----------------------------
# 4. Predict dementia risk
# ----------------------------
try:
    risk_prob = model.predict_proba(test_patient)[0, 1]
    risk_percent = risk_prob * 100

    print("🧠 Model Prediction:")
    print(f"   Dementia Risk: {risk_percent:.1f}%")
    print(f"   Raw Probability: {risk_prob:.4f}")

    if risk_percent >= 80:
        print("   🚨 High Risk")
    elif risk_percent >= 50:
        print("   ⚠️  Moderate Risk")
    else:
        print("   ✅ Low Risk")

except Exception as e:
    print(f"❌ Prediction failed: {e}")
    print("\n💡 Check that engineered features match training exactly.")