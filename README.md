# ANN Customer Churn Prediction

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-REST_API-009688?logo=fastapi&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Deployed-FF4B4B?logo=streamlit&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

A production-ready binary classification pipeline for identifying bank customers at risk of churn, built with TensorFlow/Keras. The solution covers the full ML lifecycle — data preprocessing, model training, evaluation, REST API serving, and a browser-based prediction interface.

---

## Table of Contents

- [Executive Summary](#executive-summary)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Model Performance](#model-performance)
- [Repository Structure](#repository-structure)
- [Getting Started](#getting-started)
- [REST API](#rest-api)
- [Frontend](#frontend)
- [Streamlit Deployment](#streamlit-deployment)
- [Business Impact](#business-impact)

---

## Executive Summary

This project develops a supervised binary-classification model to identify customers likely to churn (`Exited = 1`). The solution provides a reproducible ML baseline for retention strategy optimization, supported by a professional PDF report with diagnostic visuals.

---

## Dataset

![Dataset](https://img.shields.io/badge/Records-10%2C000-lightgrey)
![Target](https://img.shields.io/badge/Target-Exited%20%280%2F1%29-lightgrey)

| Property | Value |
|---|---|
| File | `Artificial_Neural_Network_Case_Study_data.csv` |
| Records | 10,000 |
| Target Variable | `Exited` — `0` Retained · `1` Churned |

**Feature Groups**

| Group | Features |
|---|---|
| Credit & Account | `CreditScore`, `Balance`, `NumOfProducts` |
| Demographics | `Geography`, `Gender`, `Age` |
| Engagement | `HasCrCard`, `IsActiveMember`, `Tenure` |
| Financial | `EstimatedSalary` |

> Identifier columns excluded from modeling: `RowNumber`, `CustomerId`, `Surname`

---

## Methodology

### Preprocessing

- Train / test split: **80 / 20** with stratification
- Numeric features: standardized via `StandardScaler`
- Categorical features: encoded via `OneHotEncoder(handle_unknown="ignore")`

### Model Architecture

```
Layer               Output Shape    
─────────────────────────────────────
Dense (ReLU)        (None, 32)      
Dropout (0.20)      (None, 32)      
Dense (ReLU)        (None, 16)      
Dropout (0.10)      (None, 16)      
Dense (Sigmoid)     (None, 1)       ← Churn probability
```

### Training Configuration

| Parameter | Value |
|---|---|
| Loss | `binary_crossentropy` |
| Optimizer | `Adam` · `lr = 0.001` |
| Batch Size | `32` |
| Max Epochs | `100` |
| Regularization | Early stopping on `val_loss` |

---

## Model Performance

![ROC-AUC](https://img.shields.io/badge/ROC--AUC-0.8664-brightgreen)
![Accuracy](https://img.shields.io/badge/Accuracy-86.50%25-brightgreen)

| Metric | Score |
|---|---|
| Accuracy | 0.8650 |
| Precision | 0.7915 |
| Recall | 0.4570 |
| F1-Score | 0.5794 |
| ROC-AUC | 0.8664 |

The model demonstrates strong ranking performance (ROC-AUC ≈ 0.87). Recall can be improved further through threshold tuning or class-weighting, depending on the cost tolerance of the retention campaign.

**Generated Visuals**

| Artifact | Description |
|---|---|
| `ann_training_curves.png` | Loss and accuracy over training epochs |
| `ann_test_metrics.png` | Confusion matrix and classification report |

---

## Repository Structure

```
ANN_Case_Study_on_Churn_rate/
│
├── churn_ann_case_study.py          # Model training and evaluation pipeline
├── generate_case_study_pdf.py       # PDF report generator
├── streamlit_app.py                 # Streamlit browser application
│
├── api/
│   ├── app.py                       # FastAPI prediction service
│   └── README_API.md
│
├── frontend/                        # Static browser UI
│
├── artifacts/
│   ├── churn_ann_model.keras        # Trained ANN model
│   ├── churn_preprocessor.joblib    # Fitted preprocessing pipeline
│   └── training_history.joblib      # Epoch-level training history
│
├── CASE_STUDY_ANN_CHURN.md          # Full narrative case study
├── ANN_Customer_Churn_Case_Study.pdf
├── requirements.txt
└── requirements_api.txt
```

---

## Getting Started

**1. Clone the repository**

```bash
git clone https://github.com/DAlgoSculptor/ANN_Case_Study_on_Churn_rate.git
cd ANN_Case_Study_on_Churn_rate
```

**2. Install dependencies**

```bash
pip install tensorflow-intel pandas scikit-learn numpy joblib matplotlib
```

**3. Train and evaluate the model**

```bash
python churn_ann_case_study.py
```

**4. Generate the PDF report**

```bash
python generate_case_study_pdf.py
```

---

## REST API

A FastAPI service for serving churn predictions from the saved model artifacts.

**Start the server**

```bash
pip install -r requirements_api.txt
python -m uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload
```

| Endpoint | Description |
|---|---|
| `http://127.0.0.1:8000/docs` | Interactive Swagger UI |
| `http://127.0.0.1:8000/redoc` | ReDoc documentation |

For full endpoint details, see [`api/README_API.md`](./api/README_API.md).

---

## Frontend

A lightweight static UI in `frontend/` that calls the prediction API from the browser.

```bash
# Start the API first, then:
python -m http.server 5173 --directory frontend
```

Open `http://127.0.0.1:5173` in your browser. CORS is enabled in the API to allow browser requests from the frontend.

---

## Streamlit Deployment

**Run locally**

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

**Deploy to Streamlit Community Cloud**

1. Push this repository to GitHub.
2. In Streamlit Cloud, set the following:

| Setting | Value |
|---|---|
| Repository | Your GitHub repo |
| Main file path | `streamlit_app.py` |
| Python dependencies | `requirements.txt` |
| Runtime | `python-3.11` (via `runtime.txt`) |

**Troubleshooting**

- `ModuleNotFoundError: No module named 'tensorflow'` — Ensure `requirements.txt` specifies `tensorflow`, not `tensorflow-intel`. Redeploy after pulling the latest commit.
- `Error installing requirements` — Use the pinned `requirements.txt` included in this repo (API dependencies are intentionally excluded). Go to **Manage app → Settings → Advanced → Clear cache**, then reboot.

---

## Business Impact

- Enables early identification of high-risk customers before attrition occurs.
- Supports targeted retention interventions and efficient campaign budget allocation.
- Provides a reproducible ML baseline for future model comparisons and iterative improvement.

---

## Documentation

| Document | Description |
|---|---|
| [`CASE_STUDY_ANN_CHURN.md`](./CASE_STUDY_ANN_CHURN.md) | Full narrative — abstract, methodology, results, appendices |
| [`ANN_Customer_Churn_Case_Study.pdf`](./ANN_Customer_Churn_Case_Study.pdf) | Professional PDF report with visuals |
| [`api/README_API.md`](./api/README_API.md) | REST API reference |

---

## Git Workflow

```bash
git add .
git commit -m "Your commit message"
git push
```

> Use `git commit -m "message"` with a space. The form `git commit m-"message"` is invalid and triggers a pathspec error.

---

*Built with TensorFlow · FastAPI · Streamlit*
