from fastapi.testclient import TestClient

from app import app


client = TestClient(app)


def test_home():

    response = client.get("/")

    assert response.status_code == 200

    assert response.json()["message"] == (
        "Customer Churn Prediction API is running."
    )


def test_prediction_api():

    sample = {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 24,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 89.5,
        "TotalCharges": 2148.0
    }

    response = client.post(
        "/predict",
        json=sample
    )

    assert response.status_code == 200

    result = response.json()

    assert "prediction" in result
    assert "churn_probability" in result
    assert "no_churn_probability" in result
    assert "model_version" in result