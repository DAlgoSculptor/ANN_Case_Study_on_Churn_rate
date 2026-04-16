from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from tensorflow import keras


ROOT = Path(__file__).resolve().parent
ARTIFACTS_DIR = ROOT / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "churn_ann_model.keras"
PREPROCESSOR_PATH = ARTIFACTS_DIR / "churn_preprocessor.joblib"


FEATURE_COLUMNS = [
    "CreditScore",
    "Geography",
    "Gender",
    "Age",
    "Tenure",
    "Balance",
    "NumOfProducts",
    "HasCrCard",
    "IsActiveMember",
    "EstimatedSalary",
]


@st.cache_resource
def load_artifacts() -> Dict[str, Any]:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    if not PREPROCESSOR_PATH.exists():
        raise FileNotFoundError(f"Preprocessor not found: {PREPROCESSOR_PATH}")

    model = keras.models.load_model(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    return {"model": model, "preprocessor": preprocessor}


def predict_proba(model: keras.Model, preprocessor: Any, row: Dict[str, Any]) -> float:
    df = pd.DataFrame([row])[FEATURE_COLUMNS]
    x = preprocessor.transform(df)
    if hasattr(x, "toarray"):
        x = x.toarray()
    proba = float(model.predict(x, verbose=0).ravel()[0])
    return proba


st.set_page_config(page_title="Churn Predictor (ANN)", page_icon="📈", layout="centered")

st.title("Customer Churn Predictor (ANN)")
st.write(
    "Streamlit deployment of the trained ANN churn model. "
    "Enter customer attributes to compute churn probability."
)

with st.sidebar:
    st.header("Settings")
    threshold = st.slider("Decision threshold", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
    st.caption("Prediction = 1 if probability ≥ threshold.")

st.subheader("Customer input")

col1, col2 = st.columns(2)

with col1:
    credit_score = st.number_input("CreditScore", min_value=0, value=619, step=1)
    geography = st.selectbox("Geography", ["France", "Germany", "Spain"], index=0)
    gender = st.selectbox("Gender", ["Female", "Male"], index=0)
    age = st.number_input("Age", min_value=0, value=42, step=1)
    tenure = st.number_input("Tenure", min_value=0, value=2, step=1)

with col2:
    balance = st.number_input("Balance", min_value=0.0, value=0.0, step=100.0, format="%.2f")
    num_products = st.number_input("NumOfProducts", min_value=0, value=1, step=1)
    has_card = st.selectbox("HasCrCard", [0, 1], index=1)
    is_active = st.selectbox("IsActiveMember", [0, 1], index=1)
    salary = st.number_input(
        "EstimatedSalary", min_value=0.0, value=101348.88, step=100.0, format="%.2f"
    )

row = {
    "CreditScore": int(credit_score),
    "Geography": geography,
    "Gender": gender,
    "Age": int(age),
    "Tenure": int(tenure),
    "Balance": float(balance),
    "NumOfProducts": int(num_products),
    "HasCrCard": int(has_card),
    "IsActiveMember": int(is_active),
    "EstimatedSalary": float(salary),
}

st.divider()

left, right = st.columns([1, 1])
with left:
    predict_clicked = st.button("Predict churn", type="primary", use_container_width=True)
with right:
    example_clicked = st.button("Load higher-risk example", use_container_width=True)

if example_clicked:
    st.session_state["example_loaded"] = True
    st.rerun()

if st.session_state.get("example_loaded"):
    row.update(
        {
            "CreditScore": 450,
            "Geography": "Germany",
            "Gender": "Female",
            "Age": 50,
            "Tenure": 1,
            "Balance": 120000.0,
            "NumOfProducts": 1,
            "HasCrCard": 0,
            "IsActiveMember": 0,
            "EstimatedSalary": 60000.0,
        }
    )
    st.session_state["example_loaded"] = False
    st.rerun()

if predict_clicked:
    try:
        artifacts = load_artifacts()
        model = artifacts["model"]
        preprocessor = artifacts["preprocessor"]
        proba = predict_proba(model, preprocessor, row)
        pred = int(proba >= threshold)

        st.success("Prediction complete.")
        st.metric("Churn probability", f"{proba:.4f}")
        st.metric("Churn prediction", str(pred))

        with st.expander("Details"):
            st.json({"input": row, "threshold": threshold, "probability": proba, "prediction": pred})
    except Exception as e:
        st.error(f"Failed to predict: {e}")

