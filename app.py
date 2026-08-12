"""
Customer Churn Prediction — Streamlit App
------------------------------------------
Ye app trained Random Forest model (churn_model.pkl) load karke
real-time mein kisi bhi customer ka churn risk predict karti hai.

Run:  streamlit run app.py

Required files in same folder:
  - churn_model.pkl
  - scaler.pkl
  - model_columns.pkl
(Ye teeno files ChurnAnalysis notebook run karne ke baad generate honge)
"""

import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Churn Prediction", page_icon="📉", layout="centered")

# ---------- Load model artifacts ----------
@st.cache_resource
def load_artifacts():
    model = joblib.load("churn_model.pkl")
    scaler = joblib.load("scaler.pkl")
    columns = joblib.load("model_columns.pkl")
    return model, scaler, columns

try:
    model, scaler, model_columns = load_artifacts()
except FileNotFoundError:
    st.error(
        "Model files nahi mile. Pehle `ChurnAnalysis_Enhanced.ipynb` run karo "
        "(Colab/Jupyter mein) — usse `churn_model.pkl`, `scaler.pkl`, "
        "aur `model_columns.pkl` generate honge. Unhe is app.py ke saath "
        "same folder mein rakho."
    )
    st.stop()

st.title("📉 Customer Churn Predictor")
st.write("Customer ki details daalo aur turant unka churn risk dekho — sath mein recommended retention action bhi milega.")

st.divider()

# ---------- Input form ----------
col1, col2 = st.columns(2)

with col1:
    tenure = st.slider("Tenure (months with company)", 0, 72, 12)
    monthly_charges = st.slider("Monthly Charges ($)", 18.0, 120.0, 65.0)
    total_charges = st.number_input("Total Charges ($)", 0.0, 9000.0, float(tenure * monthly_charges))
    contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    payment_method = st.selectbox(
        "Payment Method",
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
    )

with col2:
    gender = st.selectbox("Gender", ["Male", "Female"])
    senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
    partner = st.selectbox("Has Partner", ["No", "Yes"])
    dependents = st.selectbox("Has Dependents", ["No", "Yes"])
    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])

paperless = st.selectbox("Paperless Billing", ["No", "Yes"])
phone_service = st.selectbox("Phone Service", ["No", "Yes"])

# ---------- Build a single-row dataframe matching training format ----------
raw_input = {
    "gender": gender,
    "SeniorCitizen": 1 if senior_citizen == "Yes" else 0,
    "Partner": partner,
    "Dependents": dependents,
    "tenure": tenure,
    "PhoneService": phone_service,
    "MultipleLines": "No",
    "InternetService": internet_service,
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": contract,
    "PaperlessBilling": paperless,
    "PaymentMethod": payment_method,
    "MonthlyCharges": monthly_charges,
    "TotalCharges": total_charges,
}

input_df = pd.DataFrame([raw_input])
input_encoded = pd.get_dummies(input_df)

# Align columns with training data (fill missing dummy columns with 0)
input_final = input_encoded.reindex(columns=model_columns, fill_value=0)

# Scale numeric columns the same way training data was scaled
num_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
input_final[num_cols] = scaler.transform(input_final[num_cols])

st.divider()

if st.button("🔍 Predict Churn Risk", type="primary", use_container_width=True):
    proba = model.predict_proba(input_final)[0][1]
    prediction = "Likely to Churn" if proba >= 0.5 else "Likely to Stay"

    risk_color = "red" if proba >= 0.5 else "green"
    st.markdown(f"### Prediction: :{risk_color}[{prediction}]")
    st.progress(min(int(proba * 100), 100))
    st.write(f"**Churn Probability: {proba*100:.1f}%**")

    st.divider()
    st.subheader("📋 Recommended Action")
    if tenure <= 3 and contract == "Month-to-month":
        st.warning("⚠️ High-risk profile: new customer + month-to-month contract. "
                    "→ Improve onboarding, offer discount for switching to a long-term plan.")
    elif proba >= 0.5:
        st.warning("→ Send retention offer / personalized discount before customer churns.")
    else:
        st.success("→ Low risk. Continue standard engagement, consider upsell for high-value tier.")

st.divider()
st.caption("Model: Random Forest (class-balanced) | Trained on Telco Customer Churn dataset")
