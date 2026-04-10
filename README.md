# 📊 End-to-End Decision Intelligence System

This project demonstrates a complete machine learning pipeline, from model deployment to business visualization. It predicts customer churn using an **XGBoost** model served via a **FastAPI** REST API.

## 🏗️ System Architecture
* **Backend:** FastAPI (Python) serving real-time predictions.
* **ML Model:** XGBoost Classifier.
* **Web Frontend:** Streamlit application for individual testing.
* **BI Dashboard:** Power BI connected to the API for executive insights.

## 🚀 How to Run

### 1. Start the API (Backend)
```bash
cd api
uvicorn main:app --reload