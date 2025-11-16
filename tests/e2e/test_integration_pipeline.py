"""
Integration tests for the end-to-end ML pipeline:
    raw data -> preprocessing -> feature transformation -> model training -> prediction -> metrics.

These tests validate that the main building blocks work together as expected.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from mlops.config import RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR
from src.data_processor import DataProcessor
from src.feature_transformer import FeatureTransformer
from src.model_trainer import ModelTrainer
from src.train_pipeline import evaluate_model


def _load_raw_sample(n_rows: int = 40) -> pd.DataFrame:
    """Load a small sample of the raw dataset to keep the test fast."""
    data_path = RAW_DATA_DIR / "turkis_music_emotion_original.csv"
    df = pd.read_csv(data_path)
    # Use a subset to keep tests lightweight
    return df.head(n_rows).copy()


def test_end_to_end_random_forest_pipeline(tmp_path):
    """
    End-to-end test:
        1. Load raw data
        2. Preprocess (remove outliers)
        3. Feature/target split
        4. Encode + transform features
        5. Train RandomForest model
        6. Predict on test set
        7. Compute metrics and assert basic expectations
    """
    # -------------------------------------------------------------------------
    # 1) Load raw data
    # -------------------------------------------------------------------------
    df_raw = _load_raw_sample()
    assert not df_raw.empty
    assert "Class" in df_raw.columns

    # -------------------------------------------------------------------------
    # 2) Preprocess data (outlier removal)
    # -------------------------------------------------------------------------
    processor = DataProcessor(iqr_factor=1.5)
    df_clean = processor.remove_outliers_iqr(df_raw)
    # Should keep at least some rows
    assert not df_clean.empty

    # -------------------------------------------------------------------------
    # 3) Feature/target split
    # -------------------------------------------------------------------------
    target_col = "Class"
    X = df_clean.drop(columns=[target_col])
    y = df_clean[target_col]
    n_samples, n_features = X.shape

    assert n_samples > 10  # Reasonable number of rows for a tiny integration test
    assert n_features > 0

    # -------------------------------------------------------------------------
    # 4) Encode labels + transform features
    # -------------------------------------------------------------------------
    transformer = FeatureTransformer(use_pca=True, n_components=10)

    # Encode labels
    y_encoded = transformer.encode_labels(y)
    # Basic sanity check: at least 2 classes
    assert len(np.unique(y_encoded)) >= 2

    # Train/test split inside the transformer
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    X_train_transformed, X_test_transformed = transformer.fit_transform(X_train, X_test)

    # Shapes must still align
    assert X_train_transformed.shape[0] == y_train.shape[0]
    assert X_test_transformed.shape[0] == y_test.shape[0]

    # -------------------------------------------------------------------------
    # 5) Train RandomForest model via ModelTrainer
    # -------------------------------------------------------------------------
    trainer = ModelTrainer(random_state=42)
    model = trainer.train_random_forest(X_train_transformed, y_train)

    # -------------------------------------------------------------------------
    # 6) Predict on test set
    # -------------------------------------------------------------------------
    y_pred = trainer.predict("random_forest", X_test_transformed)

    assert y_pred.shape == y_test.shape

    # -------------------------------------------------------------------------
    # 7) Compute metrics and assert expectations
    # -------------------------------------------------------------------------
    metrics = evaluate_model(y_test, y_pred)

    # Ensure required metrics are present
    for key in ["accuracy", "f1_macro", "f1_weighted", "precision_macro", "recall_macro"]:
        assert key in metrics
        # Metrics are in [0, 1]
        assert 0.0 <= metrics[key] <= 1.0

    # For an end-to-end smoke test, just ensure the model is better than random guessing
    # on at least one metric (very weak condition, but avoids flakiness).
    assert metrics["accuracy"] > 0.2

    # -------------------------------------------------------------------------
    # 8) (Optional) Save artifacts to a temp directory to verify that saving works
    # -------------------------------------------------------------------------
    # Use tmp_path to avoid touching real project artifacts
    model_path = tmp_path / "rf_integration_model.pkl"
    transformer_path = tmp_path / "rf_integration_transformer.pkl"

    import joblib

    joblib.dump(model, model_path)
    joblib.dump(transformer, transformer_path)

    assert model_path.exists()
    assert transformer_path.exists()


