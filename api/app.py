from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from tensorflow import keras


ARTIFACTS_DIR = Path(__file__).resolve().parents[1] / "artifacts"
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


class CustomerFeatures(BaseModel):
    CreditScore: int = Field(..., ge=0)
    Geography: str
    Gender: str
    Age: int = Field(..., ge=0)
    Tenure: int = Field(..., ge=0)
    Balance: float = Field(..., ge=0)
    NumOfProducts: int = Field(..., ge=0)
    HasCrCard: int = Field(..., ge=0, le=1)
    IsActiveMember: int = Field(..., ge=0, le=1)
    EstimatedSalary: float = Field(..., ge=0)


class PredictRequest(BaseModel):
    customer: CustomerFeatures
    threshold: float = Field(0.5, ge=0.0, le=1.0)


class PredictBatchRequest(BaseModel):
    customers: List[CustomerFeatures]
    threshold: float = Field(0.5, ge=0.0, le=1.0)


class PredictResponse(BaseModel):
    churn_probability: float
    churn_prediction: int
    threshold: float


class PredictBatchResponse(BaseModel):
    results: List[PredictResponse]
    threshold: float


app = FastAPI(
    title="Customer Churn ANN API",
    version="1.0.0",
    description="Predict customer churn probability using a trained ANN (TensorFlow/Keras).",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


_model: Optional[keras.Model] = None
_preprocessor: Any = None


def _ensure_loaded() -> None:
    global _model, _preprocessor
    if _model is not None and _preprocessor is not None:
        return

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    if not PREPROCESSOR_PATH.exists():
        raise FileNotFoundError(f"Preprocessor not found: {PREPROCESSOR_PATH}")

    _model = keras.models.load_model(MODEL_PATH)
    _preprocessor = joblib.load(PREPROCESSOR_PATH)


def _to_dataframe(customers: List[CustomerFeatures]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for c in customers:
        row = c.model_dump()
        rows.append(row)
    df = pd.DataFrame(rows)
    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required feature columns: {missing}")
    return df[FEATURE_COLUMNS]


def _predict_proba(df: pd.DataFrame) -> np.ndarray:
    _ensure_loaded()
    assert _model is not None
    assert _preprocessor is not None

    x = _preprocessor.transform(df)
    if hasattr(x, "toarray"):
        x = x.toarray()
    proba = _model.predict(x, verbose=0).ravel()
    return proba


@app.get("/health")
def health() -> Dict[str, str]:
    try:
        _ensure_loaded()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    try:
        df = _to_dataframe([req.customer])
        proba = float(_predict_proba(df)[0])
        pred = int(proba >= req.threshold)
        return PredictResponse(
            churn_probability=proba,
            churn_prediction=pred,
            threshold=req.threshold,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/predict/batch", response_model=PredictBatchResponse)
def predict_batch(req: PredictBatchRequest) -> PredictBatchResponse:
    try:
        if len(req.customers) == 0:
            raise ValueError("customers must be a non-empty list")
        df = _to_dataframe(req.customers)
        probas = _predict_proba(df)
        results = [
            PredictResponse(
                churn_probability=float(p),
                churn_prediction=int(p >= req.threshold),
                threshold=req.threshold,
            )
            for p in probas
        ]
        return PredictBatchResponse(results=results, threshold=req.threshold)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

