import streamlit as st
import numpy as np
import joblib
import os

# Loading Model + Scaler Safely 
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

model = joblib.load(os.path.join(BASE_DIR, "models", "loan_default_model.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "models", "scaler.pkl"))

# UI
st.title("Loan Approval Prediction")

no_of_dependents = st.selectbox("Number of Dependents", [0, 1, 2, 3, 4, 5])
education = st.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.selectbox("Self Employed", ["No", "Yes"])
income_annum = st.number_input("Annual Income (₹)", min_value=0)
loan_amount = st.number_input("Loan Amount (₹)", min_value=0)
loan_term = st.number_input("Loan Term (in years)", min_value=0, max_value=20)
cibil_score = st.number_input("CIBIL Score", min_value=300, max_value=900)   
residential_assets_value = st.number_input("Residential Assets Value (₹)", min_value=0)
commercial_assets_value = st.number_input("Commercial Assets Value (₹)", min_value=0)
luxury_assets_value = st.number_input("Luxury Assets Value (₹)", min_value=0)
bank_asset_value = st.number_input("Bank Asset Value (₹)", min_value=0)

# Encoding
education = 0 if education == "Graduate" else 1
self_employed = 0 if self_employed == "No" else 1

input_data = np.array([[
    no_of_dependents, education, self_employed,
    income_annum, loan_amount, loan_term, cibil_score,
    residential_assets_value, commercial_assets_value,
    luxury_assets_value, bank_asset_value
]])

input_scaled = scaler.transform(input_data)

if st.button("Predict"):
    prediction = model.predict(input_scaled)[0]
    confidence = round(max(model.predict_proba(input_scaled)[0] * 100, 2))

    if prediction == 0:
        st.success(f"Loan Approved - {confidence}% confidence")
    else:
        st.error(f"Loan Rejected - {confidence}% confidence")

    # CIBIL Guidance
    st.divider()
    if cibil_score < 500: 
        st.warning("Poor CIBIL Score - Low approval chances")
    elif cibil_score < 700:
        st.info("Average CIBIL Score - Moderate approval chances")
    else: 
        st.success("Good CIBIL Score - High approval chances")