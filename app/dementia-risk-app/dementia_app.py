# dementia_app.py
import streamlit as st
import pandas as pd
import joblib
import os
import numpy as np

# ----------------------------
# Setup
# ----------------------------
st.set_page_config(page_title="🧠 Dementia Risk Predictor", page_icon="🧠")
st.title("🧠 Dementia Risk Prediction")
st.markdown("Enter key patient details to estimate 5-year dementia risk.")

MODEL_PATH = "models/dementia_risk_model_20251116_0335_auc09704.pkl"
if not os.path.exists(MODEL_PATH):
    st.error(f"Model not found at `{MODEL_PATH}`. Please ensure it exists.")
    st.stop()

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

# ----------------------------
# Feature Engineering Function (IN-LINE for simplicity)
# ----------------------------
def engineer_features(df):
    """Apply the same engineering as in the training pipeline."""
    df = df.copy()

    # Convert NACC sentinel codes to NaN
    df = df.replace(-4, np.nan)  # -4 = "Not assessed"

    # 1. Age and Education Binning
    if 'AGE' in df.columns:
        df['AGE_GROUP'] = pd.cut(
            df['AGE'],
            bins=[0, 64, 74, 84, 130],
            labels=['65-', '65-74', '75-84', '85+'],
            right=False
        )
    if 'EDUC' in df.columns:
        df['EDUC_GROUP'] = pd.cut(
            df['EDUC'],
            bins=[0, 8, 12, 16, 100],
            labels=['<HS', 'HS', 'Some College', 'College+'],
            right=False
        )

    # 2. Missingness Indicators (MUST match training)
    HIGH_MISSING_COLS = ['NACCOM', 'RACE']
    for col in HIGH_MISSING_COLS:
        if col in df.columns:
            df[f'{col}_MISSING'] = df[col].isna().astype(int)
        else:
            # If column is missing, assume it's missing
            df[f'{col}_MISSING'] = 1

    # 3. Handle NaN in engineered categorical columns
    for col in ['AGE_GROUP', 'EDUC_GROUP']:
        if col in df.columns:
            df[col] = df[col].cat.add_categories(['Missing']).fillna('Missing')

    return df

# ----------------------------
# Sidebar: Load Example Patients
# ----------------------------
st.sidebar.header("🧪 Quick Examples")
example = st.sidebar.selectbox(
    "Load example patient:",
    ["Custom", "Low Risk (Healthy 82F)", "Moderate Risk (76M)", "High Risk (88F)"]
)

patient_data = {}

if example == "Low Risk (Healthy 82F)":
    patient_data = {
        'AGE': 82, 'EDUC': 16, 'SEX': 2, 'HISPANIC': 0, 'RACE': 1, 'PRIMLANG': 1, 'MARISTAT': 1,
        'SMOKYRS': 0, 'PACKSPER': 0, 'QUITSMOK': 0, 'ALCOCCAS': 0, 'ALCFREQ': 0,
        'NACCLIVS': 1, 'INLIVWTH': 1, 'INVISITS': 5, 'INCALLS': 7, 'NACCFAM': 2, 'NACCOM': 3,
        'HOMEHOBB': 1, 'PERSCARE': 1
    }
elif example == "Moderate Risk (76M)":
    patient_data = {
        'AGE': 76, 'EDUC': 10, 'SEX': 1, 'HISPANIC': 0, 'RACE': 1, 'PRIMLANG': 1, 'MARISTAT': 2,
        'SMOKYRS': 20, 'PACKSPER': 1, 'QUITSMOK': 2015, 'ALCOCCAS': 1, 'ALCFREQ': 3,
        'NACCLIVS': 2, 'INLIVWTH': 2, 'INVISITS': 2, 'INCALLS': 2, 'NACCFAM': 1, 'NACCOM': 1,
        'HOMEHOBB': 3, 'PERSCARE': 2
    }
elif example == "High Risk (88F)":
    patient_data = {
        'AGE': 88, 'EDUC': 6, 'SEX': 2, 'HISPANIC': 1, 'RACE': 2, 'PRIMLANG': 2, 'MARISTAT': 3,
        'SMOKYRS': 50, 'PACKSPER': 2, 'QUITSMOK': 2005, 'ALCOCCAS': 1, 'ALCFREQ': 4,
        'NACCLIVS': 3, 'INLIVWTH': 3, 'INVISITS': 1, 'INCALLS': 1, 'NACCFAM': 0, 'NACCOM': -4,  # -4 = missing
        'HOMEHOBB': 4, 'PERSCARE': 3
    }
else:
    patient_data = {
        'AGE': 75, 'EDUC': 12, 'SEX': 2, 'HISPANIC': 0, 'RACE': 1, 'PRIMLANG': 1, 'MARISTAT': 1,
        'SMOKYRS': 0, 'PACKSPER': 0, 'QUITSMOK': 0, 'ALCOCCAS': 0, 'ALCFREQ': 0,
        'NACCLIVS': 1, 'INLIVWTH': 1, 'INVISITS': 3, 'INCALLS': 4, 'NACCFAM': 1, 'NACCOM': 0,
        'HOMEHOBB': 2, 'PERSCARE': 1
    }

# ----------------------------
# Main Form
# ----------------------------
st.subheader("1️⃣ Demographics")
col1, col2, col3 = st.columns(3)
AGE = col1.number_input("Age", 60, 110, patient_data.get('AGE', 75))
EDUC = col2.slider("Education (years)", 0, 20, patient_data.get('EDUC', 12))
SEX = col3.selectbox("Sex", [(1, "Male"), (2, "Female")], 
                     format_func=lambda x: x[1],
                     index=1 if patient_data.get('SEX') == 2 else 0)[0]

col1, col2, col3 = st.columns(3)
HISPANIC = col1.selectbox("Hispanic?", [(0, "No"), (1, "Yes")], 
                          format_func=lambda x: x[1],
                          index=patient_data.get('HISPANIC', 0))[0]
RACE = col2.selectbox("Race", 
                      [(1, "White"), (2, "Black"), (3, "Other")],
                      format_func=lambda x: x[1],
                      index=(patient_data.get('RACE', 1) - 1))[0]
PRIMLANG = col3.selectbox("Primary Language", 
                          [(1, "English"), (2, "Non-English")],
                          format_func=lambda x: x[1],
                          index=patient_data.get('PRIMLANG', 1) - 1)[0]

MARISTAT = st.selectbox("Marital Status",
                        [(1, "Married"), (2, "Widowed"), (3, "Divorced/Separated"), (4, "Never married")],
                        format_func=lambda x: x[1],
                        index=patient_data.get('MARISTAT', 1) - 1)[0]

st.subheader("2️⃣ Lifestyle")
SMOKYRS = st.number_input("Smoking: Years smoked", 0, 80, patient_data.get('SMOKYRS', 0))
if SMOKYRS > 0:
    PACKSPER = st.slider("Packs per day", 0.0, 5.0, float(patient_data.get('PACKSPER', 1.0)))
    QUITSMOK = st.number_input("Year quit smoking (e.g., 2010)", 1950, 2025, patient_data.get('QUITSMOK', 2020))
else:
    PACKSPER = 0
    QUITSMOK = 0

ALCOCCAS = st.selectbox("Alcohol use?",
                        [(0, "Never"), (1, "Yes")],
                        format_func=lambda x: x[1],
                        index=patient_data.get('ALCOCCAS', 0))[0]
if ALCOCCAS == 1:
    ALCFREQ = st.slider("Alcohol frequency (1=Rare, 5=Daily)", 1, 5, patient_data.get('ALCFREQ', 2))
else:
    ALCFREQ = 0

st.subheader("3️⃣ Social & Functional")
NACCLIVS = st.selectbox("Living situation",
                        [(1, "Alone"), (2, "With family"), (3, "Facility")],
                        format_func=lambda x: x[1],
                        index=patient_data.get('NACCLIVS', 1) - 1)[0]

INLIVWTH = st.slider("People living with patient", 0, 10, patient_data.get('INLIVWTH', 1))
INVISITS = st.slider("Weekly visits from others (0–7)", 0, 7, patient_data.get('INVISITS', 3))
INCALLS = st.slider("Weekly calls from others (0–7)", 0, 7, patient_data.get('INCALLS', 4))

NACCFAM = st.slider("Family support (0=Low, 2=High)", 0, 2, patient_data.get('NACCFAM', 1))
NACCOM = st.slider("Communication ability (-4=Not assessed, -3=Severe impairment, ..., 3=Excellent)", 
                   -4, 3, patient_data.get('NACCOM', 0))

HOMEHOBB = st.slider("Hobbies at home (1=Active, 4=Inactive)", 1, 4, patient_data.get('HOMEHOBB', 2))
PERSCARE = st.slider("Personal care independence (1=Independent, 3=Dependent)", 1, 3, patient_data.get('PERSCARE', 1))

# ----------------------------
# Predict
# ----------------------------
if st.button("🧠 Predict Dementia Risk"):
    raw_df = pd.DataFrame([{
        'AGE': AGE, 'EDUC': EDUC, 'SEX': SEX, 'HISPANIC': HISPANIC, 'RACE': RACE,
        'PRIMLANG': PRIMLANG, 'MARISTAT': MARISTAT,
        'SMOKYRS': SMOKYRS, 'PACKSPER': PACKSPER, 'QUITSMOK': QUITSMOK,
        'ALCOCCAS': ALCOCCAS, 'ALCFREQ': ALCFREQ,
        'NACCLIVS': NACCLIVS, 'INLIVWTH': INLIVWTH, 'INVISITS': INVISITS,
        'INCALLS': INCALLS, 'NACCFAM': NACCFAM, 'NACCOM': NACCOM,
        'HOMEHOBB': HOMEHOBB, 'PERSCARE': PERSCARE
    }])

    try:
        engineered = engineer_features(raw_df)
        
        # Final safety: ensure no unexpected NaN in categorical engineered cols
        for col in ['AGE_GROUP', 'EDUC_GROUP']:
            if col in engineered.columns:
                engineered[col] = engineered[col].fillna('Missing')

        prob = model.predict_proba(engineered)[0, 1]
        risk = prob * 100

        st.subheader("🎯 Prediction Result")
        st.metric("Dementia Risk", f"{risk:.1f}%")

        if risk >= 80:
            st.error("🚨 High Risk")
        elif risk >= 50:
            st.warning("⚠️ Moderate Risk")
        else:
            st.success("✅ Low Risk")

        st.caption(f"Raw probability: {prob:.4f}")
        st.info("Note: This is not a medical diagnosis. Consult a healthcare professional.")

    except Exception as e:
        st.exception(f"Prediction error: {e}")