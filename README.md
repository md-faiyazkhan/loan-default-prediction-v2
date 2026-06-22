# Loan Approval Prediction (V2)

A machine learning project that predicts whether a loan application will be approved or
rejected based on applicant profile — built with a strong focus on model validation,
transparent reporting, and explainability through feature importance analysis.

---

## Business Impact

Non-performing assets (NPAs) are one of the biggest financial challenges for banks and
NBFCs in India. A wrong approval decision leads to direct financial loss, while a wrong
rejection means losing a genuine customer to a competitor.

| Business Metric | Impact |
|---|---|
| **Faster Decisions** | Manual review takes days — this model predicts in seconds |
| **Reduced NPA Risk** | CIBIL score and income-based filtering reduces bad loan approvals |
| **Cost Saving** | Automating screening reduces dependency on manual underwriters |
| **Consistency** | Model applies the same logic to every application — no human bias |
| **Scalability** | Can process thousands of applications simultaneously |

> Even a 1% reduction in loan defaults can save a mid-sized bank crores of rupees annually.

---

## Problem Statement

India's lending industry processes millions of loan applications annually. Manual review
is time-consuming, inconsistent, and prone to human bias. This project builds a binary
classification model that predicts loan approval outcome based on applicant profile —
helping financial institutions make data-driven, faster, and more consistent lending
decisions.

---

## Project Structure

```
loan-default-prediction-v2/
│
├── data/
│   ├── raw/
│   │   └── loan_approval_dataset.csv
│   └── processed/
│       ├── X_train.csv
│       ├── X_test.csv
│       ├── y_train.csv
│       └── y_test.csv
│
├── notebooks/
│   └── loan_approval_prediction.ipynb
│
├── models/
│   ├── loan_approval_model.pkl
│   └── scaler.pkl
│
├── app/
│   └── app.py
│
├── .gitignore
├── README.md
└── requirements.txt
```

---

## Dataset

- **Source:** Kaggle — Loan Approval Prediction Dataset
- **Size:** 4,269 rows, 13 columns
- **Target column:** `loan_status` (Approved / Rejected)

**Features used:**

| Feature | Description |
|---|---|
| no_of_dependents | Number of dependents of the applicant |
| education | Graduate or Not Graduate |
| self_employed | Self employed or salaried |
| income_annum | Annual income of the applicant (₹) |
| loan_amount | Loan amount requested (₹) |
| loan_term | Loan repayment term in years |
| cibil_score | Credit score of the applicant (300–900) |
| residential_assets_value | Value of residential assets (₹) |
| commercial_assets_value | Value of commercial assets (₹) |
| luxury_assets_value | Value of luxury assets (₹) |
| bank_asset_value | Value of bank assets (₹) |

---

## ML Workflow

1. Data Loading
2. Data Exploration (shape, dtypes, missing values, duplicates)
3. Visualization (distribution, boxplot, correlation heatmap)
4. Data Preprocessing (encoding, feature & target split, train-test split, scaling)
5. Model Training (Logistic Regression, Decision Tree, Random Forest, XGBoost)
6. Model Evaluation (Accuracy, ROC-AUC, Classification Report, Confusion Matrix)
7. Feature Importance Analysis (XGBoost + SHAP)
8. Cross-Validation (5-fold, all models)
9. Model Saving (joblib)
10. Streamlit Deployment

---

## Models Trained

| Model | Accuracy | ROC-AUC |
|---|---|---|
| Logistic Regression | 93.44% | 0.9812 |
| Decision Tree | 97.54% | 0.9772 |
| Random Forest | 98.13% | 0.9992 |
| **XGBoost** | **99.06%** | **0.9995** |

XGBoost performed best and was selected as the final model.

---

## Model Validation

5-fold cross-validation was performed on every model to ensure a fair comparison and
to confirm that results were consistent across different subsets of the data — not
dependent on a single train-test split.

| Model | Mean CV Accuracy | Std Dev |
|---|---|---|
| Logistic Regression | 91.01% | 1.06% |
| Decision Tree | 97.19% | 0.74% |
| Random Forest | 97.80% | 0.53% |
| **XGBoost** | **97.95%** | **0.49%** |

XGBoost achieved both the highest mean CV accuracy and the lowest standard deviation —
confirming it as the most accurate and most stable model across all validation folds.

---

## Model Analysis & Feature Importance

A feature importance analysis was conducted on the final XGBoost model to understand
which factors drive the predictions.

| Feature | Importance |
|---|---|
| cibil_score | 0.619 |
| loan_term | 0.270 |
| income_annum | 0.030 |
| loan_amount | 0.023 |
| all other features combined | ~0.06 |

**Key insight:** CIBIL Score and Loan Term together account for approximately 89% of
the model's decision-making weight. This aligns closely with how real-world Indian
lending actually works — CIBIL score is the primary filter used by banks and NBFCs
when evaluating loan eligibility, and loan term directly impacts repayment risk
assessment.

SHAP (SHapley Additive exPlanations) analysis was also performed in the notebook to
validate feature contributions at the individual prediction level, further confirming
that the model's reasoning is consistent with domain knowledge.

The high accuracy (99.06%) was investigated and validated through cross-validation
(mean CV: 97.95%, std: 0.49%), confirming the model generalizes well and is not
overfitting to the training data.

---

## Key Findings

- **CIBIL Score** is the single most influential feature at 62% importance — consistent
  with how Indian banks prioritize credit history in lending decisions
- **Loan Term** is the second most important factor at 27% — together with CIBIL Score,
  these two features capture the core repayment risk signal
- XGBoost outperformed all other models across accuracy, ROC-AUC, and cross-validation
  stability
- 5-fold cross-validation confirmed the model is stable and generalizes well across
  different data subsets

---

## EMI Calculator

The deployed Streamlit app includes a built-in EMI calculator. Users can enter the
loan amount, loan term, and interest rate to instantly calculate the monthly EMI using
the standard reducing-balance formula:

```
EMI = P × r × (1 + r)^n / ((1 + r)^n − 1)
```

Where P is the principal loan amount, r is the monthly interest rate, and n is the
total number of monthly installments.

---

## How to Run

**1. Clone the repository**
```
git clone https://github.com/md-faiyazkhan/loan-default-prediction-v2.git
cd loan-default-prediction-v2
```

**2. Install dependencies**
```
pip install -r requirements.txt
```

**3. Run the Streamlit app**
```
streamlit run app/app.py
```

---

## Dataset Source

Download from Kaggle: [Loan Approval Prediction Dataset](https://www.kaggle.com/datasets/architsharma01/loan-approval-prediction-dataset)

After downloading, place the file here:
```
data/raw/loan_approval_dataset.csv
```

---

## Requirements

```
pandas
numpy
scikit-learn
matplotlib
seaborn
xgboost==2.1.4
shap
joblib
streamlit
```

---

## Skills Demonstrated

- Data Cleaning & Preprocessing
- Exploratory Data Analysis
- Feature Engineering
- Binary Classification
- Model Evaluation & Cross-Validation
- Feature Importance Analysis (XGBoost + SHAP)
- Model Interpretability
- Web App Deployment using Streamlit

---

## ⚠️ Disclaimer

This project is built for educational and portfolio purposes only. The model is trained
on a publicly available Kaggle dataset and should not be used to make real financial or
lending decisions without further validation on production-grade data.

---

## 👤 Author

**Md Faiyaz Khan**
- GitHub: [@md-faiyazkhan](https://github.com/md-faiyazkhan)
- LinkedIn: [@mdfaiyazkhan](https://www.linkedin.com/in/mdfaiyazkhan)
- Email: faiyazkhan.work@gmail.com