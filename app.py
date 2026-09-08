"""
app.py

FastAPI application for Customer Churn Prediction.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import pandas as pd

from src.serving.predict import PredictionPipeline
from src.utils.logger import logger

from src.monitoring.monitor import ModelMonitor


# --------------------------------------------------
# Initialize FastAPI
# --------------------------------------------------

app = FastAPI(
    title="Customer Churn Prediction API",
    description="Predict whether a customer will churn.",
    version="1.0.0"
)


# --------------------------------------------------
# Load prediction pipeline
# --------------------------------------------------

pipeline = PredictionPipeline()


# --------------------------------------------------
# Request Schema
# --------------------------------------------------

class CustomerData(BaseModel):

    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float


# --------------------------------------------------
# Health Check
# --------------------------------------------------

@app.get("/")
def home():

    return {
        "message": "Customer Churn Prediction API is running."
    }


# --------------------------------------------------
# Prediction Endpoint
# --------------------------------------------------

@app.post("/predict")
def predict(customer: CustomerData):

    try:

        input_df = pd.DataFrame([customer.model_dump()])

        result = pipeline.predict(input_df)

        monitor = ModelMonitor()

        monitor.update(result["prediction"])

        logger.info("Prediction request completed.")

        result["model_version"] = "1.0.0"

        return result

    except Exception as e:

        logger.error(str(e))

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )