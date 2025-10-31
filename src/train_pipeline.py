"""
Complete training pipeline using OOP classes.
This script demonstrates how to use the classes to train a model.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from pathlib import Path

from src.data_processor import DataProcessor
from src.feature_transformer import FeatureTransformer
from src.model_trainer import ModelTrainer
from src.plotter import Plotter
from mlops.config import RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, FIGURES_DIR


def main():
    """Complete training pipeline."""
    
    # Initialize plotter
    plotter = Plotter(figures_dir=FIGURES_DIR)
    
    # Step 1: Load data
    print("=" * 60)
    print("Step 1: Loading data")
    print("=" * 60)
    
    data_path = RAW_DATA_DIR / "turkis_music_emotion_original.csv"
    df = pd.read_csv(data_path)
    print(f"Dataset loaded: {df.shape}")
    
    # Generate exploratory plots for original data
    print("\n--- Generating plots for original data ---")
    plotter.plot_histograms(df, filename="histograms_original.png")
    plotter.plot_boxplots(df, filename="boxplots_original.png")
    
    # Step 2: Process data (remove outliers)
    print("\n" + "=" * 60)
    print("Step 2: Processing data (removing outliers)")
    print("=" * 60)
    
    processor = DataProcessor(iqr_factor=1.5)
    df_clean = processor.remove_outliers_iqr(df)
    print(f"Cleaned dataset: {df_clean.shape}")
    
    # Generate plots for cleaned data
    print("\n--- Generating plots for cleaned data ---")
    plotter.plot_boxplots(df_clean, filename="boxplots_cleaned.png")
    
    # Step 3: Prepare features and target
    print("\n" + "=" * 60)
    print("Step 3: Preparing features and target")
    print("=" * 60)
    
    target_col = "Class"
    X = df_clean.drop(columns=[target_col])
    y = df_clean[target_col]
    
    print(f"Features: {X.shape[1]}, Samples: {X.shape[0]}")
    
    # Generate plots for categorical variables and correlation
    print("\n--- Generating plots for categorical variables and correlation ---")
    plotter.plot_categorical_counts(df_clean, filename="categorical_counts.png")
    plotter.plot_correlation_heatmap(df_clean, filename="correlation_heatmap.png")
    
    # Step 4: Split data
    print("\n" + "=" * 60)
    print("Step 4: Splitting data")
    print("=" * 60)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train set: {X_train.shape}, Test set: {X_test.shape}")
    
    # Step 5: Transform features
    print("\n" + "=" * 60)
    print("Step 5: Transforming features")
    print("=" * 60)
    
    transformer = FeatureTransformer(use_pca=True, n_components=50)
    
    # Encode labels
    y_train_encoded = transformer.encode_labels(y_train)
    y_test_encoded = transformer.encode_labels(y_test)
    
    # Transform features
    X_train_transformed, X_test_transformed = transformer.fit_transform(X_train, X_test)
    print(f"Transformed features shape: {X_train_transformed.shape}")
    
    # Generate PCA variance plot if PCA is used
    if transformer.use_pca:
        print("\n--- Generating PCA variance plot ---")
        plotter.plot_pca_variance(
            transformer.pca.explained_variance_ratio_,
            filename="pca_variance.png"
        )
    
    # Step 6: Train models
    print("\n" + "=" * 60)
    print("Step 6: Training models")
    print("=" * 60)
    
    trainer = ModelTrainer(random_state=42)
    
    # Get class names for plotting
    class_names = list(transformer.label_encoder.classes_)
    
    # Train Random Forest
    print("\n--- Training Random Forest ---")
    rf_model = trainer.train_random_forest(X_train_transformed, y_train_encoded)
    y_pred_rf = trainer.predict('random_forest', X_test_transformed)
    trainer.print_report(y_test_encoded, y_pred_rf, model_name="Random Forest", class_names=class_names)
    
    # Generate confusion matrix for Random Forest
    cm_rf = confusion_matrix(y_test_encoded, y_pred_rf)
    plotter.plot_confusion_matrix(
        cm_rf, 
        class_names=class_names,
        filename="confusion_matrix_random_forest.png"
    )
    
    # Train LightGBM
    print("\n--- Training LightGBM ---")
    lgb_model = trainer.train_lightgbm(
        X_train_transformed, y_train_encoded,
        X_test_transformed, y_test_encoded
    )
    y_pred_lgb = trainer.predict('lightgbm', X_test_transformed)
    trainer.print_report(y_test_encoded, y_pred_lgb, model_name="LightGBM", class_names=class_names)
    
    # Generate confusion matrix for LightGBM
    cm_lgb = confusion_matrix(y_test_encoded, y_pred_lgb)
    plotter.plot_confusion_matrix(
        cm_lgb,
        class_names=class_names,
        filename="confusion_matrix_lightgbm.png"
    )
    
    # Step 7: Save models
    print("\n" + "=" * 60)
    print("Step 7: Saving models")
    print("=" * 60)
    
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    import joblib
    joblib.dump(rf_model, MODELS_DIR / "random_forest.pkl")
    joblib.dump(lgb_model, MODELS_DIR / "lightgbm.pkl")
    joblib.dump(transformer, PROCESSED_DATA_DIR / "transformer.pkl")
    
    print("Models saved successfully!")
    
    print("\n" + "=" * 60)
    print("Pipeline completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()

