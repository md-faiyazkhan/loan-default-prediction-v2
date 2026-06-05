import streamlit as st
import numpy as np
import joblib
import os

# Page Configuration 
st.set_page_config(page_title="Loan Approval Prediction", page_icon="🏦")

# Loading Model + Scaler Safely 
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

model = joblib.load(os.path.join(BASE_DIR, "models", "loan_default_model.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "models", "scaler.pkl"))

# UI
st.title("Loan Approval Prediction")

no_of_dependents = st.selectbox("Number of Dependents", [0, 1, 2, 3, 4, 5])
education = st.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.selectbox("Self Employed", ["No", "Yes"])
income_annum = st.number_input("Annual Income (₹)", min_value=0, value=None, placeholder="Enter annual income")
loan_amount = st.number_input("Loan Amount (₹)", min_value=0, value=None, placeholder="Enter loan amount")
loan_term = st.number_input("Loan Term (in years)", min_value=0, max_value=20, value=None, placeholder="Enter loan term")
cibil_score = st.number_input("CIBIL Score", min_value=300, max_value=900, value=None, placeholder="Enter CIBIL score (300-900)")   
residential_assets_value = st.number_input("Residential Assets Value (₹)", min_value=0, value=None, placeholder="Enter residential assets value")
commercial_assets_value = st.number_input("Commercial Assets Value (₹)", min_value=0, value=None, placeholder="Enter commercial assets value")
luxury_assets_value = st.number_input("Luxury Assets Value (₹)", min_value=0, value=None, placeholder="Enter luxury assets value")
bank_asset_value = st.number_input("Bank Asset Value (₹)", min_value=0, value=None, placeholder="Enter bank assets value")

# Encoding
education = 0 if education == "Graduate" else 1
self_employed = 0 if self_employed == "No" else 1

input_data = np.array([[
    no_of_dependents, education, self_employed,
    income_annum, loan_amount, loan_term, cibil_score,
    residential_assets_value, commercial_assets_value,
    luxury_assets_value, bank_asset_value
]])


# Prediction
if st.button("Predict"):
    if None in input_data[0] or cibil_score is None or "Select" in [no_of_dependents, education, self_employed]:
        st.warning("Please fill all fields before predicting.")
    else:
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)[0]
        confidence = round(float(max(model.predict_proba(input_scaled)[0]) * 100), 2)

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

        st.divider()
        st.caption("Note: This prediction is based on a machine learning model trained on historical loan data and should be used as a decision-support guide only, not as a final financial assessment.")


# EMI Calculator
st.divider()
st.subheader("EMI Calculator")

emi_rate = st.number_input("Annual Interest Rate (%)", min_value=0.0, max_value=30.0, value=8.5)

if loan_amount and loan_term and emi_rate and loan_amount > 0 and loan_term > 0 and emi_rate > 0:
    r = (emi_rate / 100) / 12          # monthly interest rate
    n = loan_term * 12                  # total months
    emi = loan_amount * r * (1 + r)**n / ((1 + r)**n - 1)
    st.success(f"Monthly EMI: ₹{emi:,.2f}")