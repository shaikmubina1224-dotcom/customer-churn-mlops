import pandas as pd

from src.preprocessing.data_preprocessing import DataPreprocessing


def test_preprocessing():

    # Load processed dataset
    df = pd.read_csv(
        "data/processed/processed_telco.csv"
    )

    preprocessing = DataPreprocessing()

    # Split features and target
    X, y = preprocessing.split_features_target(df)

    # Build preprocessing pipeline for testing
    # Do NOT use fit_transform() here because it saves
    # a test preprocessor over the production preprocessor.
    preprocessor = preprocessing.build_preprocessor(X)

    # Fit the preprocessing pipeline
    X_processed = preprocessor.fit_transform(X)

    # Assertions
    assert X_processed is not None
    assert preprocessor is not None
    assert y is not None

    assert X_processed.shape[0] == len(y)