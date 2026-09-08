from src.monitoring.monitor import ModelMonitor


def test_monitor():

    monitor = ModelMonitor()

    result = monitor.update(1)

    assert result is not None
    assert "total_predictions" in result
    assert "churn_predictions" in result
    assert "no_churn_predictions" in result
    assert "last_prediction_time" in result

    assert result["total_predictions"] > 0