"""
Feature transformation pipeline.
"""
from pathlib import Path

import pandas as pd
import numpy as np
from loguru import logger
import typer
import joblib

from mlops.config import PROCESSED_DATA_DIR
from src import FeatureTransformer

app = typer.Typer()


@app.command()
def main(
    input_path: Path = PROCESSED_DATA_DIR / "cleaned_dataset.csv",
    output_path: Path = PROCESSED_DATA_DIR / "features.csv",
    use_pca: bool = True,
    n_components: int = 50,
    test_size: float = 0.2,
    target_col: str = "Class",
):
    """
    Transform features by encoding, scaling and applying PCA.
    
    Args:
        input_path: Path to cleaned dataset
        output_path: Path to save transformed features
        use_pca: Whether to use PCA
        n_components: Number of PCA components
        test_size: Test set size
        target_col: Name of target column
    """
    logger.info(f"Loading cleaned dataset from {input_path}")
    
    # Load data
    df = pd.read_csv(input_path)
    logger.info(f"Dataset loaded: {df.shape}")
    
    # Split features and target
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    logger.info(f"Features: {X.shape[1]}, Samples: {X.shape[0]}")
    
    # Initialize transformer
    transformer = FeatureTransformer(use_pca=use_pca, n_components=n_components)
    
    # Encode labels
    y_encoded = transformer.encode_labels(y)
    logger.info(f"Labels encoded: {len(np.unique(y_encoded))} classes")
    
    # Split data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=test_size, random_state=42, stratify=y_encoded
    )
    
    logger.info(f"Train set: {X_train.shape}, Test set: {X_test.shape}")
    
    # Transform features
    logger.info("Transforming features...")
    X_train_transformed, X_test_transformed = transformer.fit_transform(X_train, X_test)
    
    # Save transformed features
    df_train = pd.DataFrame(X_train_transformed)
    df_train['target'] = y_train
    df_train.to_csv(output_path, index=False)
    
    df_test = pd.DataFrame(X_test_transformed)
    df_test['target'] = y_test
    test_output_path = output_path.parent / f"test_{output_path.name}"
    df_test.to_csv(test_output_path, index=False)
    
    # Save transformer
    transformer_path = PROCESSED_DATA_DIR / "transformer.pkl"
    joblib.dump(transformer, transformer_path)
    
    logger.success(f"Transformed features saved to {output_path}")
    logger.success(f"Test features saved to {test_output_path}")
    logger.success(f"Transformer saved to {transformer_path}")


if __name__ == "__main__":
    app()
