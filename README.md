# Customer Churn Prediction — MLOps

An end-to-end machine learning project that predicts whether a telecom customer is likely to churn. The project follows an MLOps-oriented structure with data processing, feature engineering, model training, evaluation, prediction API, monitoring, drift detection, benchmarking, retraining, and automated testing.

## Project Overview

Customer churn is an important business problem for telecom companies. Identifying customers who are likely to leave can help businesses take proactive retention actions.

This project uses the Telco Customer Churn dataset to:

* Prepare and preprocess customer data
* Create additional customer-level features
* Train multiple machine learning models
* Select the best-performing model
* Save the trained model and preprocessing pipeline
* Provide predictions through an API
* Monitor model/data behavior
* Detect data drift
* Support model retraining
* Validate the complete project using automated tests

## MLOps Workflow

```text
Raw Data
   │
   ▼
Data Ingestion
   │
   ▼
Data Preprocessing
   │
   ▼
Feature Engineering
   │
   ▼
Train / Test Split
   │
   ▼
Preprocessing Pipeline
   │
   ├───────────────┐
   ▼               ▼
Logistic        Random Forest
Regression      + Grid Search
   │               │
   └───────┬───────┘
           ▼
      Model Selection
           │
           ▼
      Model Evaluation
           │
           ▼
     Saved Model + Preprocessor
           │
           ▼
       Prediction API
           │
           ▼
 Monitoring / Drift Detection
           │
           ▼
        Retraining
```

## Project Structure

```text
customer-churn-mlops/
│
├── artifacts/
│   ├── logs/
│   ├── models/
│   │   ├── preprocessor.pkl
│   │   └── telco_churn_model.pkl
│   └── reports/
│       ├── benchmark_report.json
│       ├── evaluation_report.json
│       └── monitoring_report.json
│
├── configs/
│   └── config.yaml
│
├── data/
│   ├── raw/
│   │   └── Telco-Customer-Churn.csv
│   ├── processed/
│   │   └── processed_telco.csv
│   └── recent_batch/
│       └── recent_batch.csv
│
├── docs/
│
├── notebooks/
│   └── 01_data_exploration.ipynb
│
├── src/
│   ├── feature_engineering/
│   ├── ingestion/
│   ├── monitoring/
│   ├── pipeline/
│   ├── preprocessing/
│   ├── serving/
│   ├── training/
│   └── utils/
│
├── tests/
│   ├── test_api.py
│   ├── test_benchmark.py
│   ├── test_drift.py
│   ├── test_monitor.py
│   ├── test_prediction.py
│   ├── test_preprocessing.py
│   ├── test_retrain.py
│   └── test_training.py
│
├── app.py
├── main.py
├── Dockerfile
├── pytest.ini
├── requirements.txt
├── .gitignore
└── README.md
```

## Dataset

The project uses the **Telco Customer Churn** dataset.

The target variable is:

```text
Churn
```

Target mapping:

```text
No  → 0
Yes → 1
```

The project removes `customerID` from the model features.

## Feature Engineering

Two additional features are created:

### MonthlyChargesPerTenure

```text
MonthlyCharges / (tenure + 1)
```

This provides an additional relationship between monthly charges and customer tenure.

### IsLongTermCustomer

```text
1 if tenure >= 24
0 otherwise
```

This identifies customers who have been with the company for at least 24 months.

## Preprocessing

The preprocessing pipeline automatically identifies numerical and categorical columns.

### Numerical Features

The numerical pipeline uses:

* Median imputation
* Standard scaling

### Categorical Features

The categorical pipeline uses:

* Most-frequent imputation
* One-hot encoding
* `handle_unknown="ignore"`

The fitted preprocessing pipeline is saved as:

```text
artifacts/models/preprocessor.pkl
```

This same saved preprocessing logic is used during prediction.

## Model Training

The project trains two models:

### Logistic Regression

Used as the baseline model.

```text
max_iter = 1000
```

### Random Forest

A Random Forest model is tuned using `GridSearchCV`.

The search includes:

```text
n_estimators:
    100
    200

max_depth:
    5
    10
    None

min_samples_split:
    2
    5
```

The project uses:

```text
Cross-validation: 5 folds
Scoring: f1_macro
```

The model with the better test score is selected as the final model.

## Model Evaluation

Current evaluation results:

| Metric    |  Score |
| --------- | -----: |
| Accuracy  | 0.8084 |
| Precision | 0.6818 |
| Recall    | 0.5214 |
| F1 Score  | 0.5909 |
| ROC AUC   | 0.7167 |

The current training run selected **Logistic Regression** as the final model.

The trained model is saved to:

```text
artifacts/models/telco_churn_model.pkl
```

The evaluation report is saved to:

```text
artifacts/reports/evaluation_report.json
```

## Prediction API

The project provides a prediction API using FastAPI.

Start the application with:

```powershell
python app.py
```

The API is configured to run on:

```text
127.0.0.1:8000
```

The prediction response contains:

```text
prediction
churn_probability
no_churn_probability
```

Example response:

```json
{
    "prediction": 1,
    "churn_probability": 0.72,
    "no_churn_probability": 0.28
}
```

## Monitoring

The project includes monitoring components for:

* Model performance
* Data quality
* Data drift
* Benchmarking
* Retraining

Monitoring reports are stored under:

```text
artifacts/reports/
```

## Data Drift Detection

The project includes a drift detection module.

The configured monitoring feature is:

```text
MonthlyCharges
```

The configured drift threshold is:

```text
10
```

The drift monitoring results can be used to determine whether incoming customer data has changed significantly from the expected data distribution.

## Retraining

The project includes a retraining module that can be used when monitoring indicates that the model should be updated.

The retraining workflow can regenerate:

```text
preprocessor.pkl
telco_churn_model.pkl
evaluation_report.json
```

## Configuration

Project settings are maintained in:

```text
configs/config.yaml
```

This includes:

* Project information
* Dataset paths
* Model paths
* Test split
* Random state
* Grid-search settings
* Monitoring thresholds
* API settings
* Drift settings

Keeping these values in configuration makes the project easier to maintain and modify.

## Installation

### 1. Clone or copy the project

Open the project directory:

```powershell
cd customer-churn-mlops
```

### 2. Create virtual environment

```powershell
python -m venv venv
```

### 3. Activate the environment

Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

### 4. Install dependencies

```powershell
pip install -r requirements.txt
```

## Run the Training Pipeline

Run:

```powershell
python main.py
```

The training pipeline will:

1. Load the processed dataset
2. Apply feature engineering
3. Split features and target
4. Create the train/test split
5. Build the preprocessing pipeline
6. Train Logistic Regression
7. Train and tune Random Forest
8. Select the best model
9. Save the model
10. Generate the evaluation report

## Run Tests

Run the complete test suite:

```powershell
python -m pytest
```

Current test status:

```text
9 passed
```

The test suite covers:

* API
* Prediction
* Benchmarking
* Drift detection
* Monitoring
* Preprocessing
* Retraining
* Training

## Docker

The project includes a `Dockerfile` for containerization.

Build the image:

```powershell
docker build -t customer-churn-mlops .
```

Run the container:

```powershell
docker run -p 8000:8000 customer-churn-mlops
```

## Technology Stack

* Python
* Pandas
* NumPy
* Scikit-learn
* PyYAML
* Joblib
* FastAPI
* Uvicorn
* Pytest
* Docker

## Key MLOps Practices

This project demonstrates several practical MLOps concepts:

* Configuration-driven pipelines
* Reusable preprocessing
* Feature engineering
* Model comparison
* Hyperparameter tuning
* Model artifact management
* API-based inference
* Automated testing
* Data drift monitoring
* Model monitoring
* Retraining workflow
* Containerization

## Future Improvements

Possible future improvements include:

* CI/CD integration
* Experiment tracking with MLflow
* Model registry
* Cloud deployment
* Automated scheduled retraining
* Advanced data validation
* Production database integration
* API authentication
* Model explainability
* Improved recall for churn detection
* Production monitoring dashboards

## Project Status

**Status: Working**

The current automated test suite passes successfully:

```text
9 passed
```

The current model baseline achieves:

```text
Accuracy: 80.84%
ROC AUC: 71.67%
```
