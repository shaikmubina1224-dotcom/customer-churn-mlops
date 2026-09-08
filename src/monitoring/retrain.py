"""
retrain.py

Checks for new data and retrains the model.
"""

import os
import sys
import pandas as pd

from src.utils.logger import logger
from src.utils.exception import CustomException
from src.utils.common import read_yaml

from src.training.train import ModelTrainer


class ModelRetrainer:

    def __init__(self):

        try:

            self.config = read_yaml(
                "configs/config.yaml"
            )

            self.recent_batch_path = self.config["paths"][
                "recent_batch"
            ]

            self.processed_data_path = self.config["paths"][
                "processed_data"
            ]

        except Exception as e:

            raise CustomException(e, sys)

    def check_new_data(self):

        """
        Check whether new batch data exists.
        """

        try:

            logger.info(
                "Checking for new batch data..."
            )

            # Check whether batch file exists
            if not os.path.exists(
                self.recent_batch_path
            ):

                logger.info(
                    "No new batch found."
                )

                return False

            # Read recent batch
            df = pd.read_csv(
                self.recent_batch_path
            )

            # Check empty batch
            if df.empty:

                logger.info(
                    "Batch file is empty."
                )

                return False

            logger.info(
                f"New batch found with "
                f"{len(df)} records."
            )

            return True

        except Exception as e:

            raise CustomException(
                e,
                sys
            )

    def merge_new_data(self):

        """
        Merge the recent batch with the existing
        processed training dataset.
        """

        try:

            logger.info(
                "Merging new batch with training data..."
            )

            # Load existing training data
            existing_df = pd.read_csv(
                self.processed_data_path
            )

            # Load recent batch
            recent_df = pd.read_csv(
                self.recent_batch_path
            )

            logger.info(
                f"Existing training records: "
                f"{len(existing_df)}"
            )

            logger.info(
                f"Recent batch records: "
                f"{len(recent_df)}"
            )

            # Combine datasets
            combined_df = pd.concat(
                [
                    existing_df,
                    recent_df
                ],
                ignore_index=True
            )

            # Remove duplicate records
            combined_df = combined_df.drop_duplicates()

            # Save updated training dataset
            combined_df.to_csv(
                self.processed_data_path,
                index=False
            )

            logger.info(
                f"Updated training dataset saved. "
                f"Total records: {len(combined_df)}"
            )

            return len(combined_df)

        except Exception as e:

            raise CustomException(
                e,
                sys
            )

    def retrain(self):

        """
        Retrain model if new data exists.
        """

        try:

            # Check for new data
            if not self.check_new_data():

                logger.info(
                    "Retraining skipped."
                )

                return {
                    "status": "No new data"
                }

            # Merge new data into training data
            total_records = self.merge_new_data()

            logger.info(
                "Starting model retraining..."
            )

            # Train model using updated dataset
            trainer = ModelTrainer()

            trainer.train()

            logger.info(
                "Model retrained successfully."
            )

            return {
                "status": "Retrained",
                "total_training_records": total_records
            }

        except Exception as e:

            logger.error(
                "Model retraining failed."
            )

            raise CustomException(
                e,
                sys
            )