import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

# 1. Load the Model and Metadata
@st.cache_resource
def load_model_assets():
    # Load the dictionary containing model and feature names
    artifacts = joblib.load('telco_churn_model.pkl')
    return artifacts['model'], artifacts['features']

model, model_features = load_model_assets()

# 2. UI Configuration
st.set_page_config(page_title="Churn Analytics Pro", layout="wide")
st.title("📊 Customer Churn Prediction Dashboard")
st.markdown("---")

# 3. Sidebar: User Input (The "What-If" Simulator)
st.sidebar.header("Customer Profile Input")

def get_user_inputs():
    tenure = st.sidebar.slider("Tenure (Months)", 0, 72, 12)
    monthly_charges = st.sidebar.slider("Monthly Charges ($)", 18, 120, 60)
    
    contract = st.sidebar.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    internet = st.sidebar.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
    tech_support = st.sidebar.selectbox("Tech Support", ["Yes", "No", "No internet service"])
    
    # Initialize a DataFrame with zeros for all trained features
    input_data = pd.DataFrame(0, index=[0], columns=model_features)
    
    # Map numerical inputs
    input_data['tenure'] = tenure
    input_data['MonthlyCharges'] = monthly_charges
    input_data['TotalCharges'] = tenure * monthly_charges # Proxy
    
    # Map Categorical inputs (One-Hot Encoding handling)
    if f"Contract_{contract}" in model_features:
        input_data[f"Contract_{contract}"] = 1
    if f"InternetService_{internet}" in model_features:
        input_data[f"InternetService_{internet}"] = 1
    if f"TechSupport_{tech_support}" in model_features:
        input_data[f"TechSupport_{tech_support}"] = 1
        
    return input_data

user_data = get_user_inputs()

# 4. Main Dashboard Area
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Churn Risk Analysis")
    # Get probability from model
    prob = model.predict_proba(user_data)[0][1]
    
    # Visual Gauge using Plotly
    fig = px.pie(values=[prob, 1-prob], 
                 names=["Churn Risk", "Loyalty"],
                 hole=0.6,
                 color_discrete_sequence=['#E74C3C', '#2ECC71'])
    st.plotly_chart(fig, use_container_width=True)
    
    st.metric(label="Probability of Leaving", value=f"{prob*100:.2f}%")

with col2:
    st.subheader("Recommended Business Actions")
    if prob > 0.5:
        st.error("⚠️ HIGH RISK CUSTOMER")
        st.write("- **Immediate Action:** Offer a retention discount.")
        st.write("- **Strategy:** Propose a 1-year or 2-year contract upgrade.")
        st.write("- **Support:** Assign a priority technical support agent.")
    else:
        st.success("✅ LOW RISK CUSTOMER")
        st.write("- **Strategy:** Target for upselling premium services.")
        st.write("- **Engagement:** Send a 'Thank You' loyalty email.")

st.markdown("---")
st.caption("Advanced AI Analytics Engine - Powered by XGBoost & SHAP")