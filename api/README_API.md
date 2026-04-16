# API (FastAPI) — Customer Churn Prediction

This API serves churn predictions using the trained ANN model and preprocessing pipeline in `artifacts/`.

## Requirements

- Python 3.10+
- Artifacts exist (run `churn_ann_case_study.py` first):
  - `artifacts/churn_ann_model.keras`
  - `artifacts/churn_preprocessor.joblib`

## Install

```bash
pip install fastapi uvicorn joblib pandas numpy tensorflow-intel
```

## Run

From the project root (`d:\casestudy`):

```bash
python -m uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload
```

## Endpoints

- `GET /health`
- `POST /predict`
- `POST /predict/batch`

## Example Request

```bash
curl -X POST "http://127.0.0.1:8000/predict" ^
  -H "Content-Type: application/json" ^
  -d "{\n    \"customer\": {\n      \"CreditScore\": 619,\n      \"Geography\": \"France\",\n      \"Gender\": \"Female\",\n      \"Age\": 42,\n      \"Tenure\": 2,\n      \"Balance\": 0,\n      \"NumOfProducts\": 1,\n      \"HasCrCard\": 1,\n      \"IsActiveMember\": 1,\n      \"EstimatedSalary\": 101348.88\n    },\n    \"threshold\": 0.5\n  }"
```

Response includes:
- `churn_probability` (0–1)
- `churn_prediction` (0 or 1)

