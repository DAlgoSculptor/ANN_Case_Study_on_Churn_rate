# ANN Case Study on Customer Churn Rate

End-to-end business case study for predicting bank customer churn with an Artificial Neural Network (ANN) built using TensorFlow and Keras.

## Executive Summary

This project develops a supervised binary-classification model to identify customers likely to churn (`Exited = 1`).  
The solution includes data preparation, ANN training, model evaluation, reproducible artifacts, and a professional PDF report with supporting visuals.

## Objectives

- Predict customer attrition risk at the individual level.
- Quantify model quality with classification metrics and ROC-AUC.
- Provide a reproducible baseline for retention strategy optimization.

## Dataset

- File: `Artificial_Neural_Network_Case_Study_data.csv`
- Size: `10,000` records
- Target variable: `Exited` (`0` = retained, `1` = churned)
- Feature groups:
  - Credit and account profile (`CreditScore`, `Balance`, `NumOfProducts`)
  - Demographics (`Geography`, `Gender`, `Age`)
  - Engagement (`HasCrCard`, `IsActiveMember`, `Tenure`)
  - Financial indicator (`EstimatedSalary`)

Identifier fields excluded from modeling:
- `RowNumber`
- `CustomerId`
- `Surname`

## Methodology

### Preprocessing
- Train/test split: `80/20` with stratification.
- Numeric features: standardized using `StandardScaler`.
- Categorical features: encoded with `OneHotEncoder(handle_unknown="ignore")`.

### ANN Architecture
- Dense(32, ReLU)
- Dropout(0.20)
- Dense(16, ReLU)
- Dropout(0.10)
- Dense(1, Sigmoid)

### Training Configuration
- Loss function: `binary_crossentropy`
- Optimizer: `Adam` (`learning_rate=0.001`)
- Batch size: `32`
- Epochs: up to `100`
- Regularization: early stopping on validation loss

## Performance (Latest Run)

- Accuracy: `0.8650`
- Precision: `0.7915`
- Recall: `0.4570`
- F1-score: `0.5794`
- ROC-AUC: `0.8664`

Interpretation:
- The model shows strong ranking performance (ROC-AUC).
- Recall can be improved further through threshold tuning or class-weighting, depending on retention campaign priorities.

## Repository Structure

### Core Scripts
- `churn_ann_case_study.py` - Model training and evaluation pipeline
- `generate_case_study_pdf.py` - Professional PDF report generator

### Case Study Documents
- `CASE_STUDY_ANN_CHURN.md` - Narrative business case study
- `ANN_Customer_Churn_Case_Study.pdf` - Final report with visuals

### Model and Data Artifacts
- `artifacts/churn_ann_model.keras` - Trained ANN model
- `artifacts/churn_preprocessor.joblib` - Fitted preprocessing pipeline
- `artifacts/training_history.joblib` - Epoch-level training history

### Visual Outputs
- `ann_training_curves.png`
- `ann_test_metrics.png`

## Quick Start

1) Install dependencies
```bash
pip install tensorflow-intel pandas scikit-learn numpy joblib matplotlib
```

2) Train and evaluate the ANN
```bash
python churn_ann_case_study.py
```

3) Generate the professional PDF report
```bash
python generate_case_study_pdf.py
```

## API (FastAPI)

This repo includes a small REST API to serve churn predictions from the saved artifacts.

Run (from project root):

```bash
pip install fastapi uvicorn
python -m uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload
```

Docs:
- Interactive Swagger UI: `http://127.0.0.1:8000/docs`
- API notes: `api/README_API.md`

## Frontend (Browser UI)

A lightweight frontend is included in `frontend/` and calls the API.

1) Start the API (see section above), then start a static server for the frontend:

```bash
python -m http.server 5173 --directory frontend
```

2) Open:
- `http://127.0.0.1:5173`

Note: CORS is enabled in the API to allow browser requests from the frontend.

## Streamlit Deployment

This repo includes a Streamlit app (`streamlit_app.py`) that loads the saved model artifacts and runs predictions in the browser.

Run locally:

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Deploy on Streamlit Community Cloud:
- Push the repo to GitHub (already done)
- In Streamlit Cloud, set:
  - **Repository**: your GitHub repo
  - **Main file path**: `streamlit_app.py`
  - **Python dependencies**: `requirements.txt`
  - **Runtime**: uses `runtime.txt` (`python-3.11`) for TensorFlow wheel compatibility

If you see `ModuleNotFoundError: No module named 'tensorflow'` on Streamlit Cloud:
- Ensure `requirements.txt` uses `tensorflow` (not `tensorflow-intel`)
- Reboot/redeploy the app from Streamlit Cloud after pulling latest commit

## Business Impact

- Enables early identification of high-risk customers.
- Supports targeted retention interventions and budget allocation.
- Establishes a reusable ML baseline for future model comparisons and improvements.

## Git workflow

Stage, commit, and push using `-m` (with a space). Using `git commit m-"message"` is invalid and triggers a pathspec error.

```bash
git add .
git commit -m "Your clear commit message"
git push
```

Full narrative case study (abstract, methodology, results, appendices): see `CASE_STUDY_ANN_CHURN.md`.

## Repository

[DAlgoSculptor/ANN_Case_Study_on_Churn_rate](https://github.com/DAlgoSculptor/ANN_Case_Study_on_Churn_rate)