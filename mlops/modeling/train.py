"""
Model training pipeline.
"""
from pathlib import Path

import pandas as pd
import numpy as np
from loguru import logger
import typer
import joblib

from mlops.config import MODELS_DIR, PROCESSED_DATA_DIR
from src import ModelTrainer

app = typer.Typer()


@app.command()
def main(
    train_features_path: Path = PROCESSED_DATA_DIR / "features.csv",
    test_features_path: Path = PROCESSED_DATA_DIR / "test_features.csv",
    model_path: Path = MODELS_DIR / "model.pkl",
    model_type: str = "random_forest",
    optimize: bool = False,
    n_trials: int = 50,
):
    """
    Train ML models.
    
    Args:
        train_features_path: Path to training features
        test_features_path: Path to test features
        model_path: Path to save trained model
        model_type: Type of model ('random_forest', 'lightgbm', 'lightgbm_optimized')
        optimize: Whether to optimize hyperparameters (for LightGBM)
        n_trials: Number of optimization trials
    """
    logger.info(f"Loading training features from {train_features_path}")
    
    # Load data
    df_train = pd.read_csv(train_features_path)
    df_test = pd.read_csv(test_features_path)
    
    # Split features and target
    X_train = df_train.drop(columns=['target']).values
    y_train = df_train['target'].values
    
    X_test = df_test.drop(columns=['target']).values
    y_test = df_test['target'].values
    
    logger.info(f"Train set: {X_train.shape}, Test set: {X_test.shape}")
    
    # Initialize trainer
    trainer = ModelTrainer(random_state=42)
    
    # Train models
    if model_type == "random_forest":
        logger.info("Training Random Forest...")
        model = trainer.train_random_forest(X_train, y_train)
        
    elif model_type == "lightgbm":
        logger.info("Training LightGBM...")
        model = trainer.train_lightgbm(X_train, y_train, X_test, y_test)
        
    elif model_type == "lightgbm_optimized":
        logger.info("Training LightGBM with optimization...")
        model, best_params = trainer.optimize_lightgbm(X_train, y_train, n_trials=n_trials)
        
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Make predictions
    logger.info("Making predictions...")
    y_pred = trainer.predict(model_type, X_test)
    
    # Evaluate
    logger.info("Evaluating model...")
    trainer.print_report(y_test, y_pred, model_name=model_type)
    
    # Save model
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    
    logger.success(f"Model saved to {model_path}")


if __name__ == "__main__":
    app()
