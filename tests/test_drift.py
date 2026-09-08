from src.monitoring.drift_monitor import DataDriftMonitor


def test_drift():

    monitor = DataDriftMonitor()

    result = monitor.check_drift()

    assert result is not None
    assert "status" in result
    assert "feature" in result
    assert "mean_difference" in result
    assert "std_difference" in result