from fastapi import FastAPI
import joblib
import pandas as pd
from pydantic import BaseModel

# 1. Initialize API and Load ML Model
app = FastAPI(title="Customer Churn Prediction API")

# Load your saved artifacts (Model + Feature list)
artifacts = joblib.load('telco_churn_model.pkl')
model = artifacts['model']
features = artifacts['features']

# 2. Define the Input Schema (Data Structure)
class CustomerData(BaseModel):
    tenure: int
    MonthlyCharges: float
    TotalCharges: float
    Contract: str  # Expected: 'Month-to-month', 'One year', 'Two year'
    InternetService: str
    TechSupport: str

@app.post("/predict")
def predict_churn(data: CustomerData):
    # Create a DataFrame with zeros for all features
    df_input = pd.DataFrame(0, index=[0], columns=features)
    
    # Fill Numerical features
    df_input['tenure'] = data.tenure
    df_input['MonthlyCharges'] = data.MonthlyCharges
    df_input['TotalCharges'] = data.TotalCharges
    
    # Fill Categorical features (One-Hot Encoding mapping)
    if f"Contract_{data.Contract}" in features:
        df_input[f"Contract_{data.Contract}"] = 1
        
    # Calculate Prediction Probability
    # [0][1] gets the probability of Churn (Class 1)
    probability = model.predict_proba(df_input)[0][1]
    
    return {"churn_probability": float(probability)}