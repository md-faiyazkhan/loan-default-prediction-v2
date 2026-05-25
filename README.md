# Loan Approval Prediction

A machine learning project that predicts whether a loan application will be approved or 
rejected based on applicant details such as annual income, CIBIL score, assets value, 
and employment status.

---

## Business Impact

Non-performing assets (NPAs) are one of the biggest financial challenges for banks and 
NBFCs in India. A wrong approval decision leads to direct financial loss, while a wrong 
rejection means losing a genuine customer to a competitor.

| Business Metric | Impact |
|---|---|
| **Faster Decisions** | Manual review takes days — this model predicts in seconds |
| **Reduced NPA Risk** | CIBIL score + income based filtering reduces bad loan approvals |
| **Cost Saving** | Automating screening reduces dependency on manual underwriters |
| **Consistency** | Model applies same logic to every application — no human bias |
| **Scalability** | Can process thousands of applications simultaneously |

> Even a 1% reduction in loan defaults can save a mid-sized bank crores of rupees annually.

---

## Problem Statement

India's lending industry processes millions of loan applications annually. Manual review 
is time-consuming, inconsistent, and prone to human bias. This project builds a binary 
classification model that predicts loan approval outcome based on applicant profile — 
helping financial institutions make data-driven, unbiased, and faster lending decisions.

---

## Project Structure

```bash
loan-default-prediction-v2/
│
├── data/
│   └── raw/
│       └── loan_approval_dataset.csv
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
7. Cross Validation (5-fold)
8. Model Saving (joblib)
9. Streamlit Deployment

---

## Models Trained

| Model | Accuracy | ROC-AUC |
|---|---|---|
| Logistic Regression | 93.44% | 0.9812 |
| Decision Tree | 97.54% | 0.9772 |
| Random Forest | 98.13% | 0.9992 |
| **XGBoost** | **98.95%** | **0.9996** |

XGBoost performed best and was selected as the final model.

---

## Model Validation

5-fold cross-validation was performed to confirm that the high accuracy is genuine and 
not a result of overfitting.

| Metric | Value |
|---|---|
| Mean CV Accuracy | 97.95% |
| Standard Deviation | 0.48% |

Low standard deviation confirms that the model is stable and consistent across 
different subsets of data.

---

## Key Findings

- **CIBIL Score** is the most influential feature — applicants with high CIBIL score 
  have significantly higher approval chances
- Applicants with higher annual income and strong asset base have better approval rates
- Graduates and salaried applicants show slightly higher approval rates
- XGBoost outperformed all other models across both accuracy and ROC-AUC

---

## Improvements Over V1

| | V1 | V2 |
|---|---|---|
| Dataset size | 614 rows | 4,269 rows |
| Features | 12 | 11 (more meaningful) |
| CIBIL Score | Binary (0/1) | Continuous (300–900) |
| Best Model | Logistic Regression | XGBoost |
| Cross Validation | No | Yes (5-fold) |
| ROC-AUC | No | Yes |

---

## How to Run

**1. Clone the repository**
git clone https://github.com/md-faiyazkhan/loan-default-prediction-v2.git
cd loan-default-prediction-v2

**2. Install dependencies**
pip install -r requirements.txt

**3. Run the Streamlit app**
streamlit run app/app.py

---

## Dataset

Download from Kaggle: [Loan Approval Prediction Dataset](https://www.kaggle.com/datasets/architsharma01/loan-approval-prediction-dataset)

After downloading, place the file here:
data/raw/loan_approval_dataset.csv

---

## Requirements

pandas
numpy
scikit-learn
matplotlib
seaborn
xgboost
joblib
streamlit

---

## Skills Demonstrated

- Data Cleaning & Preprocessing
- Exploratory Data Analysis
- Feature Engineering
- Binary Classification
- Model Evaluation & Validation
- Cross Validation
- Web App Deployment using Streamlit

---

## Author

**Md Faiyaz Khan**
Self-taught ML Engineer | IIT Patna Certified (AI & ML — Intellipaat)
[LinkedIn](https://linkedin.com/in/mdfaiyazkhan) | [GitHub](https://github.com/md-faiyazkhan)