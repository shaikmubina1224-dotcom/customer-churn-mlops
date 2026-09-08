"""
feature_engineering.py

Feature engineering module for the Customer Churn project.
Creates additional features before preprocessing.
"""

import sys
import pandas as pd

from src.utils.logger import logger
from src.utils.exception import CustomException


class FeatureEngineering:
    """
    Performs feature engineering on the dataset.
    """

    def __init__(self):
        logger.info("Feature Engineering Module Initialized.")

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create engineered features.

        Parameters
        ----------
        df : pd.DataFrame

        Returns
        -------
        pd.DataFrame
        """

        try:

            logger.info("Starting feature engineering...")

            df = df.copy()

            # Monthly Charges per Tenure
            if {"MonthlyCharges", "tenure"}.issubset(df.columns):

                df["MonthlyChargesPerTenure"] = (
                    df["MonthlyCharges"] /
                    (df["tenure"] + 1)
                )

            # Long-term Customer Flag
            if "tenure" in df.columns:

                df["IsLongTermCustomer"] = (
                    df["tenure"] >= 24
                ).astype(int)

            logger.info("Feature engineering completed successfully.")

            return df

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":

    df = pd.read_csv(
        "data/processed/cleaned_telco.csv"
    )

    feature_engineering = FeatureEngineering()

    df = feature_engineering.create_features(df)

    df.to_csv(
        "data/processed/features_telco.csv",
        index=False
    )

    print("Feature engineering completed successfully!")
    print("Dataset shape:", df.shape)