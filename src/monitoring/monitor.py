"""
monitor.py

Monitoring module for Customer Churn Prediction.
Keeps track of prediction statistics.
"""

import sys
from datetime import datetime

from src.utils.logger import logger
from src.utils.exception import CustomException
from src.utils.common import (
    read_yaml,
    load_json,
    save_json
)


class ModelMonitor:

    def __init__(self):

        try:

            self.config = read_yaml("configs/config.yaml")

            self.monitor_path = self.config["paths"]["monitoring_report"]

        except Exception as e:
            raise CustomException(e, sys)

    def update(self, prediction):

        """
        Update monitoring statistics.
        """

        try:

            logger.info("Updating monitoring report...")

            # Load existing monitoring report
            try:

                report = load_json(
                    self.monitor_path
                )

            except Exception:

                logger.info(
                    "Monitoring report not found. "
                    "Creating a new report."
                )

                report = {
                    "total_predictions": 0,
                    "churn_predictions": 0,
                    "no_churn_predictions": 0,
                    "churn_percentage": 0.0,
                    "no_churn_percentage": 0.0,
                    "last_prediction_time": None
                }

            # Update total predictions
            report["total_predictions"] += 1

            # Update prediction counters
            if prediction == 1:

                report["churn_predictions"] += 1

            else:

                report["no_churn_predictions"] += 1

            # Calculate percentages
            total_predictions = report["total_predictions"]

            report["churn_percentage"] = round(
                (
                    report["churn_predictions"]
                    / total_predictions
                ) * 100,
                2
            )

            report["no_churn_percentage"] = round(
                (
                    report["no_churn_predictions"]
                    / total_predictions
                ) * 100,
                2
            )

            # Record latest prediction time
            report["last_prediction_time"] = (
                datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
            )

            # Save monitoring report
            save_json(
                self.monitor_path,
                report
            )

            logger.info(
                "Monitoring report updated."
            )

            logger.info(
                f"Total Predictions : "
                f"{report['total_predictions']}"
            )

            logger.info(
                f"Churn Predictions : "
                f"{report['churn_predictions']}"
            )

            logger.info(
                f"No-Churn Predictions : "
                f"{report['no_churn_predictions']}"
            )

            logger.info(
                f"Churn Percentage : "
                f"{report['churn_percentage']}%"
            )

            return report

        except Exception as e:

            logger.error(
                "Failed to update monitoring report."
            )

            raise CustomException(e, sys)