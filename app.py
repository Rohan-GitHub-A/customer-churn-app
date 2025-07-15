import streamlit as st
import pandas as pd
import pickle

# Load model and encoders
with open("Customer_churn_model.pkl", "rb") as f:
    model_data = pickle.load(f)
model = model_data["model"]
features = model_data["features_names"]

with open("encoders.pkl", "rb") as f:
    encoders = pickle.load(f)

st.title("📉 Customer Churn Prediction")

# User input fields
input_data = {
    'gender': st.selectbox("Gender", ["Male", "Female"]),
    'SeniorCitizen': st.selectbox("Senior Citizen", [0, 1]),
    'Partner': st.selectbox("Partner", ["Yes", "No"]),
    'Dependents': st.selectbox("Dependents", ["Yes", "No"]),
    'tenure': st.slider("Tenure (months)", 0, 72, 1),
    'PhoneService': st.selectbox("Phone Service", ["Yes", "No"]),
    'MultipleLines': st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"]),
    'InternetService': st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"]),
    'OnlineSecurity': st.selectbox("Online Security", ["Yes", "No", "No internet service"]),
    'OnlineBackup': st.selectbox("Online Backup", ["Yes", "No", "No internet service"]),
    'DeviceProtection': st.selectbox("Device Protection", ["Yes", "No", "No internet service"]),
    'TechSupport': st.selectbox("Tech Support", ["Yes", "No", "No internet service"]),
    'StreamingTV': st.selectbox("Streaming TV", ["Yes", "No", "No internet service"]),
    'StreamingMovies': st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"]),
    'Contract': st.selectbox("Contract", ["Month-to-month", "One year", "Two year"]),
    'PaperlessBilling': st.selectbox("Paperless Billing", ["Yes", "No"]),
    'PaymentMethod': st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]),
    'MonthlyCharges': st.number_input("Monthly Charges", value=29.85),
    'TotalCharges': st.number_input("Total Charges", value=29.85),
}

if st.button("Predict"):
    df = pd.DataFrame([input_data])
    for col, encoder in encoders.items():
        df[col] = encoder.transform(df[col])
    pred = model.predict(df)[0]
    prob = model.predict_proba(df)[0][1]

    st.subheader("📊 Result:")
    st.write(f"Prediction: **{'Churn' if pred == 1 else 'No Churn'}**")
    st.write(f"Confidence: **{prob*100:.2f}%**")
