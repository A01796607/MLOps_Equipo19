"""
Dataset processing pipeline for loading and cleaning data.
"""
from pathlib import Path

import pandas as pd
from loguru import logger
import typer

from mlops.config import PROCESSED_DATA_DIR, RAW_DATA_DIR
from src import DataProcessor

app = typer.Typer()


@app.command()
def main(
    input_path: Path = RAW_DATA_DIR / "turkis_music_emotion_original.csv",
    output_path: Path = PROCESSED_DATA_DIR / "cleaned_dataset.csv",
    iqr_factor: float = 1.5,
):
    """
    Process dataset by loading and cleaning outliers.
    
    Args:
        input_path: Path to raw dataset
        output_path: Path to save cleaned dataset
        iqr_factor: IQR factor for outlier detection
    """
    logger.info(f"Loading dataset from {input_path}")
    
    # Load data
    df = pd.read_csv(input_path)
    logger.info(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    
    # Initialize processor
    processor = DataProcessor(iqr_factor=iqr_factor)
    
    # Get initial statistics
    missing_pct = processor.get_missing_percentage(df)
    logger.info(f"Missing values: {missing_pct.sum()}")
    
    # Remove outliers
    logger.info("Removing outliers using IQR method...")
    df_clean = processor.remove_outliers_iqr(df)
    
    # Save cleaned data
    df_clean.to_csv(output_path, index=False)
    logger.success(f"Cleaned dataset saved to {output_path}")
    logger.info(f"Final dataset shape: {df_clean.shape}")


if __name__ == "__main__":
    app()
