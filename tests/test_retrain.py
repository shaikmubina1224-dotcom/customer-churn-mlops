from src.monitoring.retrain import ModelRetrainer


def test_retrain_module():

    retrainer = ModelRetrainer()

    assert retrainer is not None