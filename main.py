"""
main.py

Main MLOps pipeline for Customer Churn Prediction.

Workflow:
1. Train model
2. Evaluate model
3. Check data drift
4. Run prediction benchmark
"""

from src.training.train import ModelTrainer
from src.training.evaluation import ModelEvaluation

from src.monitoring.drift_monitor import DataDriftMonitor
from src.monitoring.benchmark import ModelBenchmark

from src.serving.predict import PredictionPipeline


def main():

    print("\n" + "=" * 60)
    print("CUSTOMER CHURN MLOPS PIPELINE")
    print("=" * 60)

    # ---------------------------------------------------------
    # 1. Model Training
    # ---------------------------------------------------------

    print("\n[1] Training Model...")

    trainer = ModelTrainer()

    model, X_test, y_test, _ = trainer.train()

    print("Training completed successfully.")

    # ---------------------------------------------------------
    # 2. Model Evaluation
    # ---------------------------------------------------------

    print("\n[2] Evaluating Model...")

    evaluator = ModelEvaluation()

    report = evaluator.evaluate(
        model,
        X_test,
        y_test
    )

    print("\nEvaluation Report:")

    for metric, value in report.items():

        print(
            f"{metric}: {value}"
        )

    # ---------------------------------------------------------
    # 3. Data Drift Monitoring
    # ---------------------------------------------------------

    print("\n[3] Checking Data Drift...")

    drift_monitor = DataDriftMonitor()

    drift_result = drift_monitor.check_drift()

    print("\nDrift Monitoring Result:")

    print(drift_result)

    # ---------------------------------------------------------
    # 4. Prediction Benchmark
    # ---------------------------------------------------------

    print("\n[4] Running Prediction Benchmark...")

    prediction_pipeline = PredictionPipeline()

    benchmark = ModelBenchmark()

    # Use one sample from the raw dataset
    import pandas as pd

    sample_data = pd.read_csv(
        "data/raw/Telco-Customer-Churn.csv"
    ).iloc[[0]].drop(
        columns=["Churn"]
    )

    benchmark_result = benchmark.measure(
        prediction_pipeline,
        sample_data
    )

    print("\nBenchmark Result:")

    print(benchmark_result)

    # ---------------------------------------------------------
    # 5. Pipeline Summary
    # ---------------------------------------------------------

    print("\n" + "=" * 60)
    print("MLOPS PIPELINE COMPLETED")
    print("=" * 60)

    print(
        f"\nModel Accuracy: "
        f"{report.get('Accuracy')}"
    )

    print(
        f"Data Drift: "
        f"{drift_result.get('status')}"
    )

    print(
        f"Average Prediction Latency: "
        f"{benchmark_result.get('average_prediction_time_milliseconds')} ms"
    )

    print("\nPipeline completed successfully.")


if __name__ == "__main__":

    main()