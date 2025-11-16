
# STEP 13 — Submission Artifacts + README (ALL-IN-ONE)

## ✅ Submission Artifacts (Notebook Markdown Cell)

## Submission Artifacts

All required files for hackathon submission are organized below.  
The pipeline is fully reproducible with the provided code, data, and environment.

### Folder Structure
```

dementia-risk-pipeline/
├── data/
│   └── raw/                     # Original dataset (as provided)
├── models/                      # Final model + metadata
├── metadata/                    # Feature lists, EDA outputs
├── dementia_risk_pipeline.ipynb # Full reproducible notebook
├── requirements.txt             # Python dependencies
└── README.md                    # Instructions for reviewers

```

### How to Reproduce
1. Create virtual environment:  
   `python -m venv venv && source venv/bin/activate`

2. Install dependencies:  
   `pip install -r requirements.txt`

3. Run notebook:  
   `jupyter notebook dementia_risk_pipeline.ipynb`


---

# 📄 README.md (Full File)

Paste this entire block into your `README.md`:

```markdown
# Dementia Risk Prediction Pipeline

Predicts dementia risk (0–100%) using only non-medical variables from the NACC dataset.

---

## 🔒 Constraints Compliance
- No medical or diagnostic fields used  
- No external datasets merged  
- Only sociodemographic, lifestyle, and functional variables  
- Fully compliant with hackathon constraints (IDs: `75c118a2...` and `5a735796...`)

---

## 📦 Included Files
- `dementia_risk_pipeline.ipynb` — full workflow (EDA → modeling → SHAP → calibration)
- `models/` — trained model + metadata  
  - `dementia_risk_model_*.pkl`  
  - `metadata_*.json`
- `metadata/allowed_columns_used.txt` — list of the exact non-medical columns used
- `requirements.txt` — Python dependencies

---

## ▶️ How to Use the Model

```python
import joblib
import pandas as pd

# Load model
model = joblib.load('models/dementia_risk_model_*.pkl')

# Example patient (must match training feature format)
new_patient = pd.DataFrame([{
    'AGE': 78, 'EDUC': 12, 'SEX': 2, 'RACE': 1, 'MARISTAT': 2,
    'SMOKYRS': 30, 'ALCOCCAS': 1, 'HOMEHOBB': 1, 'PERSCARE': 0,
    # ... include all required raw input columns
}])

risk_prob = model.predict_proba(new_patient)[0, 1]
print(f"Dementia risk: {risk_prob * 100:.1f}%")
````

---

## 📊 Model Performance (Test Set)

* **ROC-AUC:** 0.9704
* **PR-AUC:** 0.9494
* **Brier Score:** 0.0516 (excellent calibration)

---

## 🏆 Key Insights

* Functional difficulty variables are the strongest predictors
* Age & education contribute but do not dominate
* Model is SHAP-explainable
* Entire pipeline is fully reproducible

---

## 🛠️ Environment

* Python 3.10+
* Install dependencies from:
  `pip install -r requirements.txt`

---

## 🌳 Final Project Structure

```
dementia-risk-pipeline/
├── data
│   └── raw
├── dementia_risk_pipeline.ipynb
├── models
│   ├── dementia_risk_model_*.pkl
│   └── metadata_*.json
├── metadata
│   └── allowed_columns_used.txt
├── README.md
├── requirements.txt
└── venv
```

# dementia-risk-pipeline
