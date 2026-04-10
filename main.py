from fastapi import FastAPI
import joblib
import pandas as pd
from pydantic import BaseModel

# 1. Initialize API and Load Model
app = FastAPI(title="Churn Prediction API")
artifacts = joblib.load('telco_churn_model.pkl')
model = artifacts['model']
features = artifacts['features']

# 2. Define the data structure (Input Schema)
class CustomerData(BaseModel):
    tenure: int
    MonthlyCharges: float
    TotalCharges: float
    Contract: str # Options: 'Month-to-month', 'One year', 'Two year'
    InternetService: str
    TechSupport: str

@app.post("/predict")
def predict_churn(data: CustomerData):
    # Convert input to DataFrame
    df_input = pd.DataFrame(0, index=[0], columns=features)
    df_input['tenure'] = data.tenure
    df_input['MonthlyCharges'] = data.MonthlyCharges
    df_input['TotalCharges'] = data.TotalCharges
    
    # Simple mapping for Categorical
    if f"Contract_{data.Contract}" in features:
        df_input[f"Contract_{data.Contract}"] = 1
        
    # Get Probability
    probability = model.predict_proba(df_input)[0][1]
    return {"churn_probability": float(probability)}