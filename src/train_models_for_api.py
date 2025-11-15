"""
Script to train models and save them in the format expected by the API.

This script:
1. Loads and preprocesses data
2. Trains Random Forest and LightGBM models
3. Saves models as models/random_forest.pkl and models/lightgbm.pkl
4. Saves transformer as data/processed/transformer.pkl
"""
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from loguru import logger

from src.data_processor import DataProcessor
from src.feature_transformer import FeatureTransformer
from src.model_trainer import ModelTrainer, LIGHTGBM_AVAILABLE
from mlops.config import RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR


def main():
    """Train models and save them for API use."""
    
    logger.info("=" * 60)
    logger.info("Training Models for API")
    logger.info("=" * 60)
    
    # ============================================================================
    # STEP 1: Load Data
    # ============================================================================
    logger.info("\nSTEP 1: Loading Data")
    data_path = RAW_DATA_DIR / "turkis_music_emotion_original.csv"
    df = pd.read_csv(data_path)
    logger.info(f"Dataset loaded: {df.shape}")
    
    # ============================================================================
    # STEP 2: Preprocess Data
    # ============================================================================
    logger.info("\nSTEP 2: Preprocessing Data")
    processor = DataProcessor(iqr_factor=1.5)
    df_clean = processor.remove_outliers_iqr(df)
    logger.info(f"Cleaned dataset: {df_clean.shape}")
    
    # ============================================================================
    # STEP 3: Prepare Features and Target
    # ============================================================================
    logger.info("\nSTEP 3: Preparing Features and Target")
    target_col = "Class"
    X = df_clean.drop(columns=[target_col])
    y = df_clean[target_col]
    logger.info(f"Features: {X.shape[1]}, Samples: {X.shape[0]}")
    
    # ============================================================================
    # STEP 4: Train-Test Split
    # ============================================================================
    logger.info("\nSTEP 4: Train-Test Split")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    logger.info(f"Train set: {X_train.shape}, Test set: {X_test.shape}")
    
    # ============================================================================
    # STEP 5: Transform Features
    # ============================================================================
    logger.info("\nSTEP 5: Transforming Features")
    transformer = FeatureTransformer(use_pca=True, n_components=50)
    
    # Encode labels
    y_train_encoded = transformer.encode_labels(y_train)
    y_test_encoded = transformer.encode_labels(y_test)
    
    # Transform features
    X_train_transformed, X_test_transformed = transformer.fit_transform(
        X_train, X_test
    )
    logger.info(f"Transformed features shape: {X_train_transformed.shape}")
    
    # ============================================================================
    # STEP 6: Train Models
    # ============================================================================
    logger.info("\nSTEP 6: Training Models")
    
    trainer = ModelTrainer(random_state=42)
    
    # Train Random Forest
    logger.info("\n--- Training Random Forest ---")
    rf_model = trainer.train_random_forest(
        X_train_transformed, 
        y_train_encoded,
        n_estimators=200
    )
    
    # Evaluate Random Forest
    y_pred_rf = trainer.predict('random_forest', X_test_transformed)
    logger.info("\nRandom Forest Evaluation:")
    logger.info(classification_report(y_test_encoded, y_pred_rf))
    
    # Train LightGBM if available
    if LIGHTGBM_AVAILABLE:
        logger.info("\n--- Training LightGBM ---")
        try:
            # Split train into train and validation for early stopping
            X_tr, X_val, y_tr, y_val = train_test_split(
                X_train_transformed, 
                y_train_encoded, 
                test_size=0.2, 
                random_state=42, 
                stratify=y_train_encoded
            )
            
            lgb_model = trainer.train_lightgbm(
                X_tr, y_tr,
                X_val, y_val,
                n_estimators=500,
                early_stopping_rounds=10
            )
            
            # Evaluate LightGBM
            y_pred_lgb = trainer.predict('lightgbm', X_test_transformed)
            logger.info("\nLightGBM Evaluation:")
            logger.info(classification_report(y_test_encoded, y_pred_lgb))
        except Exception as e:
            logger.warning(f"Error training LightGBM: {e}")
            logger.warning("Skipping LightGBM training")
    else:
        logger.warning("LightGBM not available. Skipping LightGBM training.")
    
    # ============================================================================
    # STEP 7: Save Models and Transformer
    # ============================================================================
    logger.info("\nSTEP 7: Saving Models and Transformer")
    
    # Create directories if they don't exist
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # Save Random Forest
    rf_path = MODELS_DIR / "random_forest.pkl"
    joblib.dump(rf_model, rf_path)
    logger.success(f"✓ Random Forest saved: {rf_path}")
    
    # Save LightGBM if available
    if LIGHTGBM_AVAILABLE and 'lightgbm' in trainer.models:
        lgb_path = MODELS_DIR / "lightgbm.pkl"
        joblib.dump(trainer.models['lightgbm'], lgb_path)
        logger.success(f"✓ LightGBM saved: {lgb_path}")
    else:
        logger.info("LightGBM not saved (not available or training failed)")
    
    # Save transformer (needed by API for label decoding)
    transformer_path = PROCESSED_DATA_DIR / "transformer.pkl"
    joblib.dump(transformer, transformer_path)
    logger.success(f"✓ Transformer saved: {transformer_path}")
    
    # ============================================================================
    # Summary
    # ============================================================================
    logger.info("\n" + "=" * 60)
    logger.info("Training Complete!")
    logger.info("=" * 60)
    logger.info(f"\nModels saved in: {MODELS_DIR}")
    logger.info(f"Transformer saved in: {PROCESSED_DATA_DIR}")
    logger.info("\nAvailable models:")
    if rf_path.exists():
        logger.info(f"  ✓ random_forest.pkl")
    if LIGHTGBM_AVAILABLE and (MODELS_DIR / "lightgbm.pkl").exists():
        logger.info(f"  ✓ lightgbm.pkl")
    logger.info("\nYou can now start the API with:")
    logger.info("  python3 src/api/run_server.py")
    logger.info("  or")
    logger.info("  make run-api")


if __name__ == "__main__":
    main()

