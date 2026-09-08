"""
data_preprocessing.py

Preprocessing pipeline for the Customer Churn project.
Builds and saves a reusable preprocessing transformer.
"""

import sys
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from src.utils.logger import logger
from src.utils.exception import CustomException
from src.utils.common import read_yaml, save_object


class DataPreprocessing:

    def __init__(self):

        try:
            self.config = read_yaml("configs/config.yaml")

            self.target_column = self.config["data"]["target_column"]

            self.preprocessor_path = self.config["paths"]["preprocessor_path"]

        except Exception as e:
            raise CustomException(e, sys)

    def split_features_target(self, df: pd.DataFrame):
        """
        Split dataset into features and target.
        """

        try:
            logger.info("Splitting features and target.")

            df = df.copy()

            # Ensure TotalCharges is consistently treated as numeric.
            if "TotalCharges" in df.columns:
                df["TotalCharges"] = pd.to_numeric(
                    df["TotalCharges"],
                    errors="coerce"
                )

            if "customerID" in df.columns:
                df.drop(columns=["customerID"], inplace=True)

            X = df.drop(columns=[self.target_column])

            y = df[self.target_column].map({
                "No": 0,
                "Yes": 1
            })

            logger.info("Feature-target split completed.")

            return X, y

        except Exception as e:
            raise CustomException(e, sys)

    def build_preprocessor(self, X):

        try:
            logger.info("Building preprocessing pipeline.")

            # Detect numerical columns
            numerical_columns = X.select_dtypes(
                include=["number"]
            ).columns.tolist()

            # Detect categorical columns.
            # Including "str" avoids the Pandas warning
            # about future string dtype behavior.
            categorical_columns = X.select_dtypes(
                include=["object", "category", "str"]
            ).columns.tolist()

            logger.info(
                f"Numerical Columns : {numerical_columns}"
            )

            logger.info(
                f"Categorical Columns : {categorical_columns}"
            )

            # Numerical preprocessing
            numeric_pipeline = Pipeline(
                steps=[
                    (
                        "imputer",
                        SimpleImputer(strategy="median")
                    ),
                    (
                        "scaler",
                        StandardScaler()
                    )
                ]
            )

            # Categorical preprocessing
            categorical_pipeline = Pipeline(
                steps=[
                    (
                        "imputer",
                        SimpleImputer(strategy="most_frequent")
                    ),
                    (
                        "encoder",
                        OneHotEncoder(
                            handle_unknown="ignore",
                            sparse_output=False
                        )
                    )
                ]
            )

            # Combine numerical and categorical pipelines
            preprocessor = ColumnTransformer(
                transformers=[
                    (
                        "num",
                        numeric_pipeline,
                        numerical_columns
                    ),
                    (
                        "cat",
                        categorical_pipeline,
                        categorical_columns
                    )
                ]
            )

            logger.info(
                "Preprocessing pipeline created successfully."
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def fit_transform(self, X_train):

        try:
            preprocessor = self.build_preprocessor(X_train)

            logger.info(
                "Fitting preprocessing pipeline."
            )

            X_train_processed = preprocessor.fit_transform(
                X_train
            )

            save_object(
                self.preprocessor_path,
                preprocessor
            )

            logger.info(
                f"Preprocessor saved to "
                f"{self.preprocessor_path}"
            )

            return X_train_processed, preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def transform(self, X_test, preprocessor):

        try:
            logger.info(
                "Transforming test data."
            )

            X_test_processed = preprocessor.transform(
                X_test
            )

            logger.info(
                "Transformation completed."
            )

            return X_test_processed

        except Exception as e:
            raise CustomException(e, sys)