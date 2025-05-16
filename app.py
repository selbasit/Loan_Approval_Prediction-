
import streamlit as st
import pandas as pd
import joblib

st.title("Loan Approval Prediction")

model = joblib.load("loan_model.joblib")

st.write("Upload a CSV file with the same structure as the training data (without 'id' and 'loan_status'):")

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
if uploaded_file is not None:
    input_data = pd.read_csv(uploaded_file)
    prediction = model.predict(input_data)
    prediction_prob = model.predict_proba(input_data)[:, 1]
    input_data["Prediction"] = prediction
    input_data["Approval Probability"] = prediction_prob
    st.write(input_data)
    csv = input_data.to_csv(index=False).encode("utf-8")
    st.download_button("Download Predictions", csv, "predictions.csv", "text/csv")
