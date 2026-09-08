"""
train.py

Model training module for the Customer Churn project.
"""

import sys
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from src.utils.logger import logger
from src.utils.exception import CustomException
from src.utils.common import read_yaml, save_object

from src.feature_engineering.feature_engineering import FeatureEngineering
from src.preprocessing.data_preprocessing import DataPreprocessing


class ModelTrainer:

    def __init__(self):

        try:

            self.config = read_yaml("configs/config.yaml")

            self.processed_data_path = self.config["paths"]["processed_data"]

            self.model_path = self.config["paths"]["model_path"]

            self.test_size = self.config["data"]["test_size"]

            self.random_state = self.config["data"]["random_state"]

        except Exception as e:
            raise CustomException(e, sys)

    def load_data(self):

        try:

            logger.info("Loading processed dataset...")

            df = pd.read_csv(self.processed_data_path)

            logger.info(f"Dataset Shape : {df.shape}")

            return df

        except Exception as e:
            raise CustomException(e, sys)

    def train(self):

        try:

            # Load data
            df = self.load_data()

            # Feature Engineering
            feature_engineering = FeatureEngineering()

            df = feature_engineering.create_features(df)

            # Preprocessing
            preprocessing = DataPreprocessing()

            X, y = preprocessing.split_features_target(df)

            # Train-Test Split
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=self.test_size,
                random_state=self.random_state,
                stratify=y
            )

            logger.info("Train-Test Split Completed.")

            # Fit preprocessing on training data
            X_train_processed, preprocessor = preprocessing.fit_transform(
                X_train
            )

            # Transform test data
            X_test_processed = preprocessing.transform(
                X_test,
                preprocessor
            )

            # --------------------------------------------------
            # Logistic Regression
            # --------------------------------------------------

            logger.info("Training Logistic Regression...")

            logistic = LogisticRegression(
                max_iter=1000
            )

            logistic.fit(
                X_train_processed,
                y_train
            )

            logistic_score = logistic.score(
                X_test_processed,
                y_test
            )

            logger.info(
                f"Logistic Regression Accuracy : {logistic_score:.4f}"
            )

            # --------------------------------------------------
            # Random Forest + Grid Search
            # --------------------------------------------------

            logger.info("Training Random Forest...")

            rf = RandomForestClassifier(
                random_state=self.random_state
            )

            param_grid = {
                "n_estimators": [100, 200],
                "max_depth": [5, 10, None],
                "min_samples_split": [2, 5]
            }

            grid_search = GridSearchCV(
                estimator=rf,
                param_grid=param_grid,
                cv=self.config["grid_search"]["cv"],
                scoring=self.config["grid_search"]["scoring"],
                n_jobs=self.config["grid_search"]["n_jobs"]
            )

            grid_search.fit(
                X_train_processed,
                y_train
            )

            best_model = grid_search.best_estimator_

            best_score = best_model.score(
                X_test_processed,
                y_test
            )

            logger.info(
                f"Random Forest Accuracy : {best_score:.4f}"
            )

            # --------------------------------------------------
            # Select Best Model
            # --------------------------------------------------

            if best_score > logistic_score:

                logger.info("Random Forest Selected.")

                final_model = best_model

            else:

                logger.info("Logistic Regression Selected.")

                final_model = logistic

            # Save final model
            save_object(
                self.model_path,
                final_model
            )

            logger.info(
                f"Model saved to {self.model_path}"
            )

            return (
                final_model,
                X_test_processed,
                y_test,
                preprocessor
            )

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":

    trainer = ModelTrainer()

    trainer.train()