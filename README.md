# ANN Case Study on Customer Churn Rate

Business case study for predicting customer churn using an Artificial Neural Network (ANN) with TensorFlow and Keras in Python.

## Project Overview

This project builds and evaluates an ANN model to predict whether a bank customer is likely to churn (`Exited = 1`).

The workflow includes:
- Data preprocessing (scaling + encoding)
- Train/test split
- ANN model training
- Model evaluation using classification metrics
- Export of model artifacts
- PDF case study report with visuals

## Dataset

- Source file: `Artificial_Neural_Network_Case_Study_data.csv`
- Records: `10,000`
- Target variable: `Exited`
- Input features include:
  - Credit profile (`CreditScore`, `Balance`, `NumOfProducts`)
  - Demographics (`Geography`, `Gender`, `Age`)
  - Engagement (`HasCrCard`, `IsActiveMember`, `Tenure`)
  - Financial indicator (`EstimatedSalary`)

Dropped identifier columns:
- `RowNumber`
- `CustomerId`
- `Surname`

## Model Architecture (TensorFlow/Keras)

- Dense(32, ReLU)
- Dropout(0.2)
- Dense(16, ReLU)
- Dropout(0.1)
- Dense(1, Sigmoid)

Training setup:
- Loss: `binary_crossentropy`
- Optimizer: `Adam`
- Validation split: `0.2`
- Early stopping enabled

## Evaluation Results

From the latest run:
- Accuracy: `0.8650`
- Precision: `0.7915`
- Recall: `0.4570`
- F1-score: `0.5794`
- ROC-AUC: `0.8664`

## Generated Outputs

### Code and Documentation
- `churn_ann_case_study.py` - ANN training and evaluation pipeline
- `CASE_STUDY_ANN_CHURN.md` - Written business case study
- `generate_case_study_pdf.py` - PDF report generator with images

### Artifacts
- `artifacts/churn_ann_model.keras` - Trained ANN model
- `artifacts/churn_preprocessor.joblib` - Fitted preprocessing pipeline
- `artifacts/training_history.joblib` - Training history

### Visuals and Report
- `ann_training_curves.png`
- `ann_test_metrics.png`
- `ANN_Customer_Churn_Case_Study.pdf`

## How to Run

1. Install dependencies:

```bash
pip install tensorflow-intel pandas scikit-learn numpy joblib matplotlib
```

2. Train and evaluate model:

```bash
python churn_ann_case_study.py
```

3. Generate PDF report with images:

```bash
python generate_case_study_pdf.py
```

## Business Value

- Supports proactive retention strategy by identifying likely churners.
- Enables campaign targeting using churn probability scores.
- Provides a reproducible baseline ANN pipeline for further optimization.

## Author

Repository: [DAlgoSculptor/ANN_Case_Study_on_Churn_rate](https://github.com/DAlgoSculptor/ANN_Case_Study_on_Churn_rate)