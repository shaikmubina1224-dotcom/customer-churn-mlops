import pandas as pd

from src.serving.predict import PredictionPipeline


def test_prediction():

    sample = pd.DataFrame({
        "gender": ["Female"],
        "SeniorCitizen": [0],
        "Partner": ["Yes"],
        "Dependents": ["No"],
        "tenure": [24],
        "PhoneService": ["Yes"],
        "MultipleLines": ["No"],
        "InternetService": ["Fiber optic"],
        "OnlineSecurity": ["No"],
        "OnlineBackup": ["Yes"],
        "DeviceProtection": ["No"],
        "TechSupport": ["No"],
        "StreamingTV": ["Yes"],
        "StreamingMovies": ["Yes"],
        "Contract": ["Month-to-month"],
        "PaperlessBilling": ["Yes"],
        "PaymentMethod": ["Electronic check"],
        "MonthlyCharges": [89.5],
        "TotalCharges": [2148.0]
    })

    pipeline = PredictionPipeline()

    result = pipeline.predict(sample)

    print(result)

    assert result is not None
    assert "prediction" in result
    assert "churn_probability" in result
    assert "no_churn_probability" in result