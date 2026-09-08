import pandas as pd

from src.serving.predict import PredictionPipeline
from src.monitoring.benchmark import ModelBenchmark


def test_benchmark():

    pipeline = PredictionPipeline()

    sample = pd.read_csv(
        "data/raw/Telco-Customer-Churn.csv"
    ).iloc[[0]].drop(columns=["Churn"])

    benchmark = ModelBenchmark()

    result = benchmark.measure(
        pipeline,
        sample
    )

    assert result is not None
    assert "prediction_time_seconds" in result
    assert "prediction_time_milliseconds" in result
    assert "status" in result

    assert result["prediction_time_seconds"] >= 0
    assert result["prediction_time_milliseconds"] >= 0
    assert result["status"] == "Success"