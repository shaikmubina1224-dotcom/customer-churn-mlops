"""
evaluation.py

Evaluates the trained model and saves evaluation metrics.
"""

import sys

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

from src.utils.logger import logger
from src.utils.exception import CustomException
from src.utils.common import (
    read_yaml,
    save_json
)


class ModelEvaluation:

    def __init__(self):

        try:

            self.config = read_yaml("configs/config.yaml")

            self.report_path = self.config["paths"]["report_path"]

        except Exception as e:
            raise CustomException(e, sys)

    def evaluate(
        self,
        model,
        X_test,
        y_test
    ):
        """
        Evaluate trained model.
        """

        try:

            logger.info("Evaluating model...")

            predictions = model.predict(X_test)

            accuracy = accuracy_score(
                y_test,
                predictions
            )

            precision = precision_score(
                y_test,
                predictions
            )

            recall = recall_score(
                y_test,
                predictions
            )

            f1 = f1_score(
                y_test,
                predictions
            )

            roc_auc = roc_auc_score(
                y_test,
                predictions
            )

            confusion = confusion_matrix(
                y_test,
                predictions
            )

            report = {

                "Accuracy": round(accuracy, 4),

                "Precision": round(precision, 4),

                "Recall": round(recall, 4),

                "F1 Score": round(f1, 4),

                "ROC AUC": round(roc_auc, 4),

                "Confusion Matrix": confusion.tolist()

            }

            save_json(
                self.report_path,
                report
            )

            logger.info(
                f"Evaluation report saved to {self.report_path}"
            )

            return report

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":

    logger.info(
        "Run train.py first before evaluating the model."
    )