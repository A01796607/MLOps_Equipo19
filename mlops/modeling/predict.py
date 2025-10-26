"""
Model prediction pipeline.
"""
from pathlib import Path

import pandas as pd
import numpy as np
from loguru import logger
import typer
import joblib

from mlops.config import MODELS_DIR, PROCESSED_DATA_DIR

app = typer.Typer()


@app.command()
def main(
    features_path: Path = PROCESSED_DATA_DIR / "test_features.csv",
    model_path: Path = MODELS_DIR / "model.pkl",
    predictions_path: Path = PROCESSED_DATA_DIR / "predictions.csv",
    transformer_path: Path = PROCESSED_DATA_DIR / "transformer.pkl",
):
    """
    Make predictions using trained model.
    
    Args:
        features_path: Path to features for prediction
        model_path: Path to trained model
        predictions_path: Path to save predictions
        transformer_path: Path to transformer for decoding labels
    """
    logger.info(f"Loading features from {features_path}")
    
    # Load data
    df = pd.read_csv(features_path)
    
    # Check if target exists
    if 'target' in df.columns:
        X = df.drop(columns=['target']).values
        y_true = df['target'].values
        has_target = True
    else:
        X = df.values
        has_target = False
    
    logger.info(f"Features shape: {X.shape}")
    
    # Load model
    logger.info(f"Loading model from {model_path}")
    model = joblib.load(model_path)
    
    # Make predictions
    logger.info("Making predictions...")
    
    if hasattr(model, 'predict'):
        y_pred = model.predict(X)
    else:
        # For LightGBM booster
        y_pred_proba = model.predict(X)
        y_pred = y_pred_proba.argmax(axis=1)
    
    # Decode labels if transformer available
    if transformer_path.exists():
        logger.info("Decoding labels...")
        transformer = joblib.load(transformer_path)
        y_pred_decoded = transformer.decode_labels(y_pred)
        
        # Save predictions with decoded labels
        results = pd.DataFrame({
            'prediction': y_pred,
            'prediction_label': y_pred_decoded
        })
        
        if has_target:
            results['true_label'] = transformer.decode_labels(y_true)
            results['correct'] = y_pred == y_true
    else:
        results = pd.DataFrame({'prediction': y_pred})
        if has_target:
            results['true_label'] = y_true
            results['correct'] = y_pred == y_true
    
    # Save predictions
    results.to_csv(predictions_path, index=False)
    
    logger.success(f"Predictions saved to {predictions_path}")
    
    if has_target:
        accuracy = results['correct'].mean()
        logger.info(f"Accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    app()
