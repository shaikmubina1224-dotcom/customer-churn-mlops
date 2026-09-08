"""
Lightweight Data Drift Monitoring

Compares the mean and standard deviation of a selected feature
between the processed training dataset and the most recent batch.

Logs a warning when either difference exceeds the configured threshold.
"""

import sys
import pandas as pd

from src.utils.logger import logger
from src.utils.exception import CustomException
from src.utils.common import read_yaml


class DataDriftMonitor:

    def __init__(self):

        try:

            self.config = read_yaml("configs/config.yaml")

            self.train_data_path = self.config["paths"]["processed_data"]

            self.recent_batch_path = self.config["paths"]["recent_batch"]

            self.feature = self.config["drift"]["feature"]

            self.threshold = self.config["drift"]["threshold"]

        except Exception as e:

            raise CustomException(e, sys)

    def check_drift(self):

        try:

            logger.info("Starting data drift monitoring...")

            # Load training data
            train_df = pd.read_csv(
                self.train_data_path
            )

            # Load recent production batch
            recent_df = pd.read_csv(
                self.recent_batch_path
            )

            logger.info(
                f"Training dataset shape : {train_df.shape}"
            )

            logger.info(
                f"Recent batch shape : {recent_df.shape}"
            )

            # Validate feature in training data
            if self.feature not in train_df.columns:

                raise ValueError(
                    f"Feature '{self.feature}' not found "
                    f"in training data."
                )

            # Validate feature in recent batch
            if self.feature not in recent_df.columns:

                raise ValueError(
                    f"Feature '{self.feature}' not found "
                    f"in recent batch."
                )

            # Convert selected feature to numeric
            train_feature = pd.to_numeric(
                train_df[self.feature],
                errors="coerce"
            )

            recent_feature = pd.to_numeric(
                recent_df[self.feature],
                errors="coerce"
            )

            # Remove invalid / missing values
            train_feature = train_feature.dropna()

            recent_feature = recent_feature.dropna()

            # Make sure usable data exists
            if train_feature.empty:

                raise ValueError(
                    f"No valid numeric values found for "
                    f"'{self.feature}' in training data."
                )

            if recent_feature.empty:

                raise ValueError(
                    f"No valid numeric values found for "
                    f"'{self.feature}' in recent batch."
                )

            # Calculate statistics
            train_mean = train_feature.mean()

            recent_mean = recent_feature.mean()

            train_std = train_feature.std()

            recent_std = recent_feature.std()

            # For a single-value batch, std can become NaN.
            # Treat it as zero instead of allowing NaN
            # to break the drift calculation.
            if pd.isna(train_std):

                train_std = 0.0

            if pd.isna(recent_std):

                recent_std = 0.0

            # Calculate differences
            mean_diff = abs(
                train_mean - recent_mean
            )

            std_diff = abs(
                train_std - recent_std
            )

            # Log statistics
            logger.info(
                f"Feature being monitored : {self.feature}"
            )

            logger.info(
                f"Training Mean : {train_mean:.2f}"
            )

            logger.info(
                f"Recent Mean : {recent_mean:.2f}"
            )

            logger.info(
                f"Training Std : {train_std:.2f}"
            )

            logger.info(
                f"Recent Std : {recent_std:.2f}"
            )

            logger.info(
                f"Mean Difference : {mean_diff:.2f}"
            )

            logger.info(
                f"Std Difference : {std_diff:.2f}"
            )

            logger.info(
                f"Configured Drift Threshold : "
                f"{self.threshold}"
            )

            # Check drift
            drift_detected = (
                mean_diff > self.threshold
                or std_diff > self.threshold
            )

            if drift_detected:

                logger.warning(
                    "Possible data drift detected!"
                )

                status = "Drift Detected"

            else:

                logger.info(
                    "No significant data drift detected."
                )

                status = "No Drift"

            # Return monitoring result
            return {
                "status": status,
                "feature": self.feature,
                "threshold": self.threshold,
                "training_mean": round(train_mean, 2),
                "recent_mean": round(recent_mean, 2),
                "training_std": round(train_std, 2),
                "recent_std": round(recent_std, 2),
                "mean_difference": round(mean_diff, 2),
                "std_difference": round(std_diff, 2),
                "training_rows": len(train_feature),
                "recent_rows": len(recent_feature)
            }

        except Exception as e:

            logger.error(
                "Data drift monitoring failed."
            )

            raise CustomException(e, sys)


if __name__ == "__main__":

    monitor = DataDriftMonitor()

    result = monitor.check_drift()

    print(result)