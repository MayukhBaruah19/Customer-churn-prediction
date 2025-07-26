import os
import streamlit as st
import pandas as pd
import numpy as np
import dill

with open('artifacts/model.pkl', 'rb') as f:
    model = dill.load(f)

with open('artifacts/preprocessor.pkl', 'rb') as f: 
    preprocessor = dill.load(f)    

with open('artifacts/label_encoder.pkl', 'rb') as f:
    label_encoder = dill.load(f)

def predict_single_input(input_df, model, preprocessor, label_encoder):
    processed = preprocessor.transform(input_df)
    pred = model.predict(processed).astype(int)  # Ensure integer
    decoded = label_encoder.inverse_transform(pred)
    return decoded[0]    

st.set_page_config(page_title="Customer Churn Predictor", layout="centered")
st.title("Customer Churn Prediction")

st.subheader("📋 Enter Customer Details")
with st.form("customer_details"):
    gender = st.selectbox("Gender", ["Male", "Female"])

    senior_citizen_map={"No": 0, "Yes": 1}
    senior_citizen = st.selectbox("Senior Citizen", list(senior_citizen_map.keys()))
    senior=senior_citizen_map[senior_citizen]
   

    partner = st.selectbox("Partner", ['Yes', 'No'])
    dependents = st.selectbox("Dependents", ["Yes", "No"],help="This refers to whether the customer has people relying on them financially, such as children or elderly family members.?")
    tenure = st.number_input("Tenure",help="The number of months the customer has been with the company.",min_value=0,max_value=72)
    phone_service = st.selectbox("Phone Service",["Yes", "No"],help='Whether the customer has a phone service or not ')
    multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
    online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
    device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
    tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
    streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"],help='Whether the customer has streaming TV or not')
    streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"],help='Whether the customer has streaming movies or not')
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
    payment_method = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
    monthly_charges = st.number_input("Monthly Charges", min_value=20.0, max_value=120.0, value=40.0)
    total_charges = st.number_input("Total Charges", min_value=20.0,max_value=8590.00,value=400.0) 

    submit_button = st.form_submit_button("Predict")

if submit_button:
    input_data = {
        'gender': [gender],
        'SeniorCitizen': [senior],  # uses mapped value
        'Partner': [partner],
        'Dependents': [dependents],
        'tenure': [tenure],
        'PhoneService': [phone_service],
        'MultipleLines': [multiple_lines],
        'InternetService': [internet_service],
        'OnlineSecurity': [online_security],
        'OnlineBackup': [online_backup],
        'DeviceProtection': [device_protection],
        'TechSupport': [tech_support],
        'StreamingTV': [streaming_tv],
        'StreamingMovies': [streaming_movies],
        'Contract': [contract],
        'PaperlessBilling': [paperless_billing],
        'PaymentMethod': [payment_method],
        'MonthlyCharges': [monthly_charges],
        'TotalCharges': [total_charges],
    }
    input_df = pd.DataFrame(input_data)
    prediction = predict_single_input(input_df, model, preprocessor, label_encoder)
    st.success(f"🎯 Prediction: **{prediction}**")