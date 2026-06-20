# Loan Approval Prediction (V2)

A machine learning project that predicts whether a loan application will be approved or
rejected based on applicant details such as annual income, CIBIL score, assets value,
and employment status — built with a strong focus on model validation, explainability,
and honest reporting of limitations.

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
helping financial institutions make data-driven, faster lending decisions, while being
transparent about how and why each decision is made.

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

## A Note on Dataset Quality — Read Before Trusting the Accuracy Numbers

This is an important section, and it is included deliberately.

Every model trained on this dataset — including a simple Logistic Regression — crossed
**90%+ accuracy**, and the final XGBoost model reached **99.06% accuracy**. Numbers this
high should always raise a question before they raise confidence.

After investigating, here is the honest explanation:

**1. The dataset appears synthetically generated or heavily cleaned.**
There are zero missing values, no noisy entries, no inconsistent formatting beyond minor
whitespace issues, and very clean class separation between Approved and Rejected cases.
Real-world banking data is rarely this clean.

**2. One feature dominates the decision almost entirely.**
A feature importance analysis on the final XGBoost model showed:

| Feature | Importance |
|---|---|
| cibil_score | 0.619 |
| loan_term | 0.270 |
| income_annum | 0.030 |
| loan_amount | 0.023 |
| all other features combined | ~0.06 |

Nearly **89% of the model's decision-making weight comes from just two features** —
`cibil_score` and `loan_term`. The remaining nine features have a marginal combined
contribution. This is not necessarily wrong — CIBIL score genuinely is the single most
influential factor in real-world Indian lending decisions — but a model this dependent
on one or two signals is **less robust** than one that reasons across a balanced set of
features. If `cibil_score` were missing, delayed, or incorrect for an applicant in a
production setting, this model's reliability would drop sharply.

**3. The high accuracy was validated, not just assumed.**
To rule out overfitting, 5-fold cross-validation was performed on the final model:

| Metric | Value |
|---|---|
| Mean CV Accuracy | 97.95% |
| Standard Deviation | 0.49% |

The low standard deviation confirms the model is stable and consistent across different
subsets of the training data — so the 99.06% test accuracy is not a fluke caused by a
lucky train-test split. It genuinely reflects how easily separable this particular
dataset is, not necessarily how well the model would perform on messier, real-world data.

**Conclusion:** This model is technically sound and well-validated for the dataset it was
trained on. However, it should be understood as a **learning and demonstration project**
that showcases the complete ML workflow — not as a production-ready underwriting system.
A production model would need to be retrained on more diverse, real-world Indian banking
data with less feature dominance before being deployed for actual lending decisions.

---

## ML Workflow

1. Data Loading
2. Data Exploration (shape, dtypes, missing values, duplicates)
3. Visualization (distribution, boxplot, correlation heatmap)
4. Data Preprocessing (encoding, feature & target split, train-test split, scaling)
5. Model Training (Logistic Regression, Decision Tree, Random Forest, XGBoost)
6. Model Evaluation (Accuracy, ROC-AUC, Classification Report, Confusion Matrix)
7. Feature Importance Analysis
8. Model Saving (joblib)
9. Streamlit Deployment

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

5-fold cross-validation was performed on every model — not just the final one — to make
the comparison fair and to confirm that high accuracy was not isolated to a single
train-test split.

| Model | Mean CV Accuracy | Std Dev |
|---|---|---|
| Logistic Regression | 91.01% | 1.06% |
| Decision Tree | 97.19% | 0.74% |
| Random Forest | 97.80% | 0.53% |
| **XGBoost** | **97.95%** | **0.49%** |

XGBoost had both the highest mean accuracy and the lowest variance across folds, making
it the most consistent model in addition to the most accurate one.

---

## Key Findings

- **CIBIL Score** is the single most influential feature, contributing roughly 62% of
  the model's decision-making weight — consistent with how Indian banks actually
  prioritize credit history in lending decisions
- **Loan Term** is the second most influential factor at 27% — together with CIBIL
  Score, these two features account for nearly 89% of the model's reasoning
- The remaining nine features — including income, loan amount, and all asset values —
  have a comparatively minor combined influence on the outcome
- This concentration of importance is a notable limitation: a more robust model would
  reason across a broader, more balanced set of features
- XGBoost outperformed all other models across accuracy, ROC-AUC, and cross-validation
  stability

---

## EMI Calculator

In addition to the prediction model, the deployed app includes a built-in EMI
calculator. Once a user enters the loan amount, loan term, and interest rate, the
monthly EMI is calculated in real time using the standard reducing-balance EMI formula:

```
EMI = P × r × (1 + r)^n / ((1 + r)^n − 1)
```

Where P is the principal loan amount, r is the monthly interest rate, and n is the
total number of monthly installments. This gives applicants immediate insight into
loan affordability alongside the approval prediction.

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
xgboost
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
- Feature Importance Analysis
- Critical Evaluation of Model Reliability and Dataset Quality
- Web App Deployment using Streamlit

---

## Author

**Md Faiyaz Khan**
Self-taught ML Engineer | AI & ML Certified — Intellipaat x IIT Patna
[LinkedIn](https://linkedin.com/in/mdfaiyazkhan) | [GitHub](https://github.com/md-faiyazkhan)