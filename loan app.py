
import streamlit as st
import joblib
import numpy as np

# Load the trained Random Forest model
model = joblib.load('final_random_forest_model.pkl')

st.title("Loan Default Prediction System")
st.write("Predict customer loan default risk using behavioral data from Lending Club dataset.")

# Borrower ID (optional)
borrower_id = st.text_input("Borrower ID ")

# Behavioral Inputs
transaction_freq_open = st.number_input("Number of Open Credit Lines", min_value=0, step=1)
total_acc = st.number_input("Total Number of Credit Accounts", min_value=0, step=1)
delinq_2yrs = st.number_input("Delinquencies in Last 2 Years", min_value=0, step=1)
mths_since_last_delinq = st.number_input("Months Since Last Delinquency (0 if none)", min_value=0, step=1)
revol_bal = st.number_input("Revolving Balance ($)", min_value=0.0, step=100.0)
revol_util = st.number_input("Revolving Utilization (%)", min_value=0.0, step=0.1)
inq_last_6mths = st.number_input("Credit Inquiries in Last 6 Months", min_value=0, step=1)

# Financial/Demographic Inputs
annual_inc = st.number_input("Annual Income ($)", min_value=0.0, step=1000.0)
emp_length = st.number_input("Employment Length (years)", min_value=0, step=1)
loan_amnt = st.number_input("Loan Amount ($)", min_value=100.0, step=100.0)
purpose = st.selectbox("Loan Purpose", (["debt_consolidation", "credit_card", "home_improvement", "major_purchase", "small_business", "car", "med" "vacation", "moving", "other"]))

# Convert categorical 'purpose' to numeric if your model expects encoding
# For simplicity, you can one-hot encode manually or use your trained encoder
purpose_mapping = {
    "debt_consolidation": 0, "credit_card": 1, "home_improvement": 2,
    "major_purchase": 3, "small_business": 4, "car": 5,
    "medical": 6, "vacation": 7, "moving": 8, "other": 9
}
purpose_encoded = purpose_mapping[purpose]

# Predict button
if st.button("Predict Default Risk"):
    input_features = np.array([
        transaction_freq_open,
        total_acc,
        delinq_2yrs,
        mths_since_last_delinq,
        revol_bal,
        revol_util,
        inq_last_6mths,
        annual_inc,
        emp_length,
        loan_amnt,
        purpose_encoded
    ]).reshape(1, -1)
    
    prediction = model.predict(input_features)[0]
    risk_mapping = {0: "Not Defaulted", 1: "Defaulted"}
    
    st.success(f"Prediction for Borrower {borrower_id if borrower_id else '(no ID)'}: {risk_mapping[prediction]}")