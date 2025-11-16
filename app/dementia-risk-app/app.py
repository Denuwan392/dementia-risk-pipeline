# app.py
from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib
import os
from utils import engineer_features

app = Flask(__name__)

# Load model once at startup
MODEL_PATH = "models/dementia_risk_model_20251115_1748_auc09704.pkl"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}")

model = joblib.load(MODEL_PATH)

# Define all input fields (same as in test.py)
INPUT_FIELDS = [
    'AGE', 'EDUC', 'SEX', 'HISPANIC', 'RACE', 'PRIMLANG', 'MARISTAT',
    'SMOKYRS', 'PACKSPER', 'QUITSMOK', 'ALCOCCAS', 'ALCFREQ',
    'NACCLIVS', 'INLIVWTH', 'INVISITS', 'INCALLS', 'NACCFAM', 'NACCOM',
    'HOMEHOBB', 'PERSCARE'
]

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    error = None

    if request.method == 'POST':
        try:
            # Collect form data
            data = {}
            for field in INPUT_FIELDS:
                val = request.form.get(field, '').strip()
                if val == '':
                    raise ValueError(f"Missing value for {field}")
                # Convert to appropriate type
                if field in ['AGE', 'EDUC', 'SEX', 'HISPANIC', 'RACE', 'PRIMLANG',
                             'MARISTAT', 'SMOKYRS', 'PACKSPER', 'QUITSMOK',
                             'ALCOCCAS', 'ALCFREQ', 'NACCLIVS', 'INLIVWTH',
                             'INVISITS', 'INCALLS', 'NACCFAM', 'NACCOM',
                             'HOMEHOBB', 'PERSCARE']:
                    data[field] = int(val)
                else:
                    data[field] = float(val)

            # Create DataFrame
            raw_df = pd.DataFrame([data])

            # Engineer features
            engineered_df = engineer_features(raw_df)

            # Predict
            prob = model.predict_proba(engineered_df)[0, 1]
            risk_percent = prob * 100

            # Risk category
            if risk_percent >= 80:
                category = "🚨 High Risk"
            elif risk_percent >= 50:
                category = "⚠️ Moderate Risk"
            else:
                category = "✅ Low Risk"

            result = {
                'risk_percent': f"{risk_percent:.1f}",
                'raw_prob': f"{prob:.4f}",
                'category': category
            }

        except Exception as e:
            error = f"Prediction failed: {str(e)}"

    return render_template('index.html', result=result, error=error, fields=INPUT_FIELDS)

if __name__ == '__main__':
    app.run(debug=True)