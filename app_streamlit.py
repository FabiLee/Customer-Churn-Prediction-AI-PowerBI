import streamlit as st
import requests

# Page Configuration
st.set_page_config(page_title="Churn Predictor", page_icon="📉")

st.title("🛡️ AI Decision Support System")
st.subheader("Customer Churn Prediction")
st.write("Fill in the customer details below to calculate the risk score via FastAPI.")

# 1. Input Form
with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📊 Billing Info")
        tenure = st.number_input("Tenure (months)", min_value=0, max_value=120, value=12)
        monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, value=50.0)
        total_charges = st.number_input("Total Charges ($)", min_value=0.0, value=600.0)
        
    with col2:
        st.markdown("### 📝 Contract Info")
        contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
        internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])

    submit = st.form_submit_button("Run AI Prediction")

# 2. API Communication
if submit:
    payload = {
        "tenure": int(tenure),
        "MonthlyCharges": float(monthly_charges),
        "TotalCharges": float(total_charges),
        "Contract": contract,
        "InternetService": internet,
        "TechSupport": tech_support
    }
    
    try:
        # Requesting prediction from FastAPI
        response = requests.post("http://localhost:8000/predict", json=payload)
        result = response.json()
        prob = result["churn_probability"]
        
        # 3. Displaying Results
        st.divider()
        st.markdown(f"### Result for Customer:")
        
        if prob > 0.5:
            st.error(f"🚨 **HIGH RISK**: {prob:.2%}")
        else:
            st.success(f"✅ **LOW RISK**: {prob:.2%}")
            
        st.progress(prob)
        st.caption("Probability calculated by XGBoost Model via REST API.")
        
    except Exception as e:
        st.error("Error: Could not connect to FastAPI. Please ensure the server is running on port 8000.")