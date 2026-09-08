from src.training.train import ModelTrainer


def test_training_module():

    trainer = ModelTrainer()

    assert trainer is not None