# Customer Churn Prediction Using an Artificial Neural Network  
## A Business Case Study (TensorFlow / Keras)

**Course / project:** ANN Case Study on Churn Rate  
**Stack:** Python, TensorFlow, Keras, scikit-learn  

---

## Abstract

This case study addresses customer churn in a retail banking context. A supervised binary classifier predicts whether a customer will exit (`Exited = 1`) based on demographic, account, and engagement attributes. An artificial neural network (ANN) is trained and evaluated on a dataset of 10,000 records. Results on a held-out test set are reported using accuracy, precision, recall, F1-score, and ROC-AUC, together with learning-curve diagnostics. The report concludes with business interpretation and recommended next steps for deployment-oriented use.

---

## 1. Introduction

Customer churn directly affects revenue and lifetime value. Machine learning can prioritize retention efforts by estimating churn probability at the customer level. This study implements a feedforward ANN as a baseline model and documents data preparation, model design, evaluation, and reproducibility.

---

## 2. Problem Statement

**Objective:** Predict churn (`Exited = 1`) so the bank can target high-risk customers for retention.  
**Outcome:** A probability score per customer (sigmoid output), thresholded for operational decisions.

---

## 3. Dataset

| Item | Description |
|------|-------------|
| **File** | `Artificial_Neural_Network_Case_Study_data.csv` |
| **Size** | 10,000 rows |
| **Target** | `Exited` — 0 = retained, 1 = churned |
| **Features (examples)** | `CreditScore`, `Geography`, `Gender`, `Age`, `Tenure`, `Balance`, `NumOfProducts`, `HasCrCard`, `IsActiveMember`, `EstimatedSalary` |

**Excluded from training:** `RowNumber`, `CustomerId`, `Surname` (identifiers, not predictive signal).

---

## 4. Methodology

### 4.1 Data splitting

- Train / test split: **80% / 20%**, **stratified** on `Exited` to preserve class proportions.

### 4.2 Preprocessing

- **Numeric** features: `StandardScaler` (zero mean, unit variance).  
- **Categorical** features (`Geography`, `Gender`): `OneHotEncoder` with `handle_unknown="ignore"`.

### 4.3 Model architecture (ANN)

| Layer | Configuration |
|-------|----------------|
| Hidden 1 | Dense(32), ReLU |
| Regularization | Dropout(0.20) |
| Hidden 2 | Dense(16), ReLU |
| Regularization | Dropout(0.10) |
| Output | Dense(1), Sigmoid |

### 4.4 Training

- **Loss:** binary cross-entropy.  
- **Optimizer:** Adam (learning rate 0.001).  
- **Batch size:** 32.  
- **Epochs:** up to 100, with **early stopping** on validation loss to limit overfitting.

---

## 5. Results (latest documented run)

Test-set metrics:

| Metric | Value |
|--------|--------|
| Accuracy | 0.8650 |
| Precision | 0.7915 |
| Recall | 0.4570 |
| F1-score | 0.5794 |
| ROC-AUC | 0.8664 |

**Confusion matrix (test):** reported in the training script output (see console log when running `churn_ann_case_study.py`).

---

## 6. Discussion

- **ROC-AUC** indicates reasonable **ranking** of customers by churn risk, useful for prioritization.  
- **Precision** suggests that when the model flags churn, a large fraction of those flags are true positives—relevant when retention budget is limited.  
- **Recall** is moderate; more churners could be captured by **threshold tuning**, **class weights**, or alternative algorithms, traded off against campaign cost.

---

## 7. Conclusion

The ANN provides a reproducible baseline for churn prediction with strong discrimination (ROC-AUC) and actionable outputs for retention planning. Further work should align the decision threshold with business costs and compare against strong tabular baselines (e.g., gradient boosting) and explainability tools (e.g., SHAP).

---

## 8. References (placeholders)

1. Goodfellow, Bengio, Courville — *Deep Learning* (foundations).  
2. TensorFlow — https://www.tensorflow.org/  
3. scikit-learn — https://scikit-learn.org/  

*(Replace with your course or institution citation style if required.)*

---

## Appendix A — Reproducibility (commands)

Run from the project folder (e.g. `D:\casestudy`).

**Environment**

```bash
pip install tensorflow-intel pandas scikit-learn numpy joblib matplotlib
```

**Train and evaluate**

```bash
python churn_ann_case_study.py
```

**Generate the PDF report and figures**

```bash
python generate_case_study_pdf.py
```

---

## Appendix B — Version control (Git)

Use a **space** between `commit` and `-m`. The following is **incorrect** and will fail:

```text
git commit m-"edit readme.md file"
```

**Correct sequence:**

```bash
git add .
git commit -m "Describe your change clearly"
git push
```

Example message: `docs: refine case study and README`.
