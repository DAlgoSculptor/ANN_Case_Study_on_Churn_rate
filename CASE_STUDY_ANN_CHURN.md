# Business Case Study: Customer Churn Prediction with ANN

## 1) Problem Statement
A retail bank wants to reduce customer attrition. The objective is to predict which customers are likely to churn (`Exited = 1`) so the business can proactively retain them.

## 2) Dataset
- File: `Artificial_Neural_Network_Case_Study_data.csv`
- Rows: 10,000
- Target variable: `Exited` (0 = stayed, 1 = churned)
- Features include:
  - Demographics: `Geography`, `Gender`, `Age`
  - Account profile: `CreditScore`, `Balance`, `NumOfProducts`, `Tenure`
  - Engagement signals: `HasCrCard`, `IsActiveMember`
  - Economic profile: `EstimatedSalary`

Identifier columns (`RowNumber`, `CustomerId`, `Surname`) are excluded from training because they do not represent behavioral signal.

## 3) ANN Methodology
The model uses TensorFlow/Keras with the following pipeline:

1. Train-test split (`80/20`) with stratified target distribution  
2. Preprocessing:
   - Standard scaling for numeric features
   - One-hot encoding for categorical features (`Geography`, `Gender`)
3. ANN architecture:
   - Dense(32, ReLU) + Dropout(0.2)
   - Dense(16, ReLU) + Dropout(0.1)
   - Dense(1, Sigmoid)
4. Loss/optimizer:
   - Binary cross-entropy
   - Adam optimizer
5. Overfitting control:
   - Early stopping on validation loss

## 4) Evaluation Framework
Primary test metrics:
- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion matrix and full classification report

These metrics give a balanced view of churn prediction quality, especially for identifying churners (minority class).

## 5) Business Interpretation
- **Higher recall** on churners helps catch more at-risk customers for retention campaigns.
- **Higher precision** avoids spending retention budget on customers who would not churn.
- **ROC-AUC** measures ranking quality for campaign targeting thresholds.

In production, decision thresholds can be tuned by campaign capacity and retention costs.

## 6) Deliverables
- Training and evaluation code: `churn_ann_case_study.py`
- Saved artifacts after execution:
  - `artifacts/churn_ann_model.keras`
  - `artifacts/churn_preprocessor.joblib`
  - `artifacts/training_history.joblib`

## 7) How to Run
From `d:\casestudy`:

```bash
pip install tensorflow pandas scikit-learn numpy joblib
python churn_ann_case_study.py
```

## 8) Recommended Next Enhancements
- Perform threshold optimization based on retention ROI.
- Add class weighting or focal loss to improve churn recall.
- Compare ANN performance with XGBoost and logistic regression baselines.
- Add model explainability (SHAP) for actionable churn drivers.
