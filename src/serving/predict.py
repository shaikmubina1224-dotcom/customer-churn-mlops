"""
predict.py

Prediction pipeline for the Customer Churn project.
Loads the trained model and preprocessing pipeline
and generates churn predictions.
"""

import sys
import pandas as pd

from src.utils.logger import logger
from src.utils.exception import CustomException
from src.utils.common import read_yaml, load_object
from src.feature_engineering.feature_engineering import FeatureEngineering


class PredictionPipeline:

    def __init__(self):

        try:
            self.config = read_yaml("configs/config.yaml")

            self.model_path = self.config["paths"]["model_path"]
            self.preprocessor_path = self.config["paths"]["preprocessor_path"]

            # Load trained model
            self.model = load_object(
                self.model_path
            )

            # Load fitted preprocessor
            self.preprocessor = load_object(
                self.preprocessor_path
            )

            logger.info(
                "Model and Preprocessor loaded successfully."
            )

        except Exception as e:
            raise CustomException(e, sys)

    def predict(self, input_data):

        try:
            logger.info(
                "Starting prediction pipeline."
            )

            # Convert input data to DataFrame
            data = pd.DataFrame(input_data)

            # Ensure TotalCharges is consistently numeric
            if "TotalCharges" in data.columns:
                data["TotalCharges"] = pd.to_numeric(
                    data["TotalCharges"],
                    errors="coerce"
                )

            # Apply feature engineering
            data = FeatureEngineering().create_features(
                data
            )

            logger.info(
                f"Prediction input shape: {data.shape}"
            )

            # Apply the saved preprocessing pipeline
            processed_data = self.preprocessor.transform(
                data
            )

            logger.info(
                f"Processed prediction shape: "
                f"{processed_data.shape}"
            )

            # Generate prediction
            prediction = self.model.predict(
                processed_data
            )

            # Generate probabilities
            probability = self.model.predict_proba(
                processed_data
            )

            # Prediction value
            churn_prediction = int(
                prediction[0]
            )

            # Probability of churn
            churn_probability = float(
                probability[0][1]
            )

            # Probability of no churn
            no_churn_probability = float(
                probability[0][0]
            )

            logger.info(
                "Prediction completed successfully."
            )

            # Return prediction results
            return {
                "prediction": churn_prediction,
                "churn_probability": churn_probability,
                "no_churn_probability": no_churn_probability
            }

        except Exception as e:

            logger.error(
                "Prediction failed."
            )

            raise CustomException(e, sys)