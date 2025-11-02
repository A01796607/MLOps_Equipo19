"""
Complete training pipeline using Scikit-Learn Pipeline best practices.

This script demonstrates how to use sklearn.Pipeline to automate preprocessing,
training, and evaluation stages, ensuring reproducibility and clarity.
"""
import sys
from pathlib import Path

# Add project root to Python path to enable imports
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report, 
    confusion_matrix,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score
)

from src.data_processor import DataProcessor
from src.plotter import Plotter
from mlops.config import RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, FIGURES_DIR


def create_preprocessing_pipeline(use_pca: bool = True, n_components: int = 50):
    """
    Create a sklearn Pipeline for preprocessing features.
    
    This function creates a Pipeline that automatically chains:
    - StandardScaler: Normalizes features to zero mean and unit variance
    - PCA (optional): Reduces dimensionality while preserving variance
    
    Args:
        use_pca: Whether to apply PCA dimensionality reduction
        n_components: Number of PCA components if use_pca=True
        
    Returns:
        sklearn.Pipeline object for preprocessing
    """
    steps = [
        ('scaler', StandardScaler()),
    ]
    
    if use_pca:
        steps.append(('pca', PCA(n_components=n_components, random_state=42)))
    
    return Pipeline(steps)


def create_model_pipeline(preprocessor: Pipeline, model_type: str = 'random_forest', **model_params):
    """
    Create a complete sklearn Pipeline with preprocessing and model.
    
    This function chains preprocessing steps with a model, ensuring that:
    - fit() is called on the entire pipeline at once
    - transform() is applied consistently to train and test
    - No data leakage occurs between train/test
    
    Args:
        preprocessor: Preprocessing pipeline from create_preprocessing_pipeline()
        model_type: Type of model ('random_forest' or 'lightgbm')
        **model_params: Additional parameters for the model
        
    Returns:
        sklearn.Pipeline object with preprocessing + model
    """
    if model_type == 'random_forest':
        model = RandomForestClassifier(
            n_estimators=model_params.get('n_estimators', 200),
            random_state=42
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    
    # Combine preprocessor with model
    full_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', model)
    ])
    
    return full_pipeline


def evaluate_model(y_true, y_pred, class_names=None):
    """
    Evaluate model predictions and return comprehensive metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: Names of the classes
        
    Returns:
        Dictionary with evaluation metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'f1_macro': f1_score(y_true, y_pred, average='macro'),
        'f1_weighted': f1_score(y_true, y_pred, average='weighted'),
        'precision_macro': precision_score(y_true, y_pred, average='macro'),
        'recall_macro': recall_score(y_true, y_pred, average='macro'),
    }
    
    return metrics


def print_evaluation(y_true, y_pred, model_name, class_names=None):
    """
    Print comprehensive evaluation report.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        model_name: Name of the model
        class_names: Names of the classes
    """
    print(f"\n{'='*60}")
    print(f"Evaluation Results: {model_name}")
    print(f"{'='*60}")
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    print("\nConfusion Matrix:")
    print(cm)
    
    # Summary metrics
    metrics = evaluate_model(y_true, y_pred)
    print("\nSummary Metrics:")
    for metric_name, metric_value in metrics.items():
        print(f"  {metric_name}: {metric_value:.4f}")


def main():
    """
    Complete training pipeline using sklearn.Pipeline best practices.
    
    This pipeline demonstrates:
    1. Data loading and preprocessing
    2. sklearn.Pipeline for automated transformation
    3. Model training with proper fit/transform separation
    4. Comprehensive evaluation
    5. Model persistence
    """
    
    # Initialize plotter
    plotter = Plotter(figures_dir=FIGURES_DIR)
    
    # ============================================================================
    # STEP 1: Load Data
    # ============================================================================
    print("=" * 60)
    print("STEP 1: Loading Data")
    print("=" * 60)
    
    data_path = RAW_DATA_DIR / "turkis_music_emotion_original.csv"
    df = pd.read_csv(data_path)
    print(f"Dataset loaded: {df.shape}")
    
    # Generate exploratory plots for original data
    print("\n--- Generating plots for original data ---")
    plotter.plot_histograms(df, filename="histograms_original.png")
    plotter.plot_boxplots(df, filename="boxplots_original.png")
    
    # ============================================================================
    # STEP 2: Data Preprocessing
    # ============================================================================
    print("\n" + "=" * 60)
    print("STEP 2: Data Preprocessing")
    print("=" * 60)
    
    processor = DataProcessor(iqr_factor=1.5)
    df_clean = processor.remove_outliers_iqr(df)
    print(f"Cleaned dataset: {df_clean.shape}")
    
    # Generate plots for cleaned data
    print("\n--- Generating plots for cleaned data ---")
    plotter.plot_boxplots(df_clean, filename="boxplots_cleaned.png")
    
    # ============================================================================
    # STEP 3: Feature and Target Preparation
    # ============================================================================
    print("\n" + "=" * 60)
    print("STEP 3: Feature and Target Preparation")
    print("=" * 60)
    
    target_col = "Class"
    X = df_clean.drop(columns=[target_col])
    y = df_clean[target_col]
    
    print(f"Features: {X.shape[1]}, Samples: {X.shape[0]}")
    
    # Generate plots for categorical variables and correlation
    print("\n--- Generating plots for categorical variables and correlation ---")
    plotter.plot_categorical_counts(df_clean, filename="categorical_counts.png")
    plotter.plot_correlation_heatmap(df_clean, filename="correlation_heatmap.png")
    
    # ============================================================================
    # STEP 4: Train-Test Split
    # ============================================================================
    print("\n" + "=" * 60)
    print("STEP 4: Train-Test Split")
    print("=" * 60)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train set: {X_train.shape}, Test set: {X_test.shape}")
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)
    class_names = list(label_encoder.classes_)
    print(f"Encoded classes: {class_names}")
    
    # ============================================================================
    # STEP 5: Create sklearn Pipelines
    # ============================================================================
    print("\n" + "=" * 60)
    print("STEP 5: Creating sklearn Pipelines")
    print("=" * 60)
    
    # Create preprocessing pipeline
    preprocessor = create_preprocessing_pipeline(use_pca=True, n_components=50)
    print("\nPreprocessing Pipeline created:")
    print("  Steps: StandardScaler -> PCA (50 components)")
    
    # Create full pipelines with models
    rf_pipeline = create_model_pipeline(preprocessor, model_type='random_forest', n_estimators=200)
    print("\nRandom Forest Pipeline created:")
    print("  Steps: StandardScaler -> PCA -> RandomForestClassifier")
    
    # ============================================================================
    # STEP 6: Train Models using Pipelines
    # ============================================================================
    print("\n" + "=" * 60)
    print("STEP 6: Training Models with sklearn Pipelines")
    print("=" * 60)
    
    # Train Random Forest
    print("\n--- Training Random Forest Pipeline ---")
    print("Fitting full pipeline (preprocessing + model)...")
    rf_pipeline.fit(X_train, y_train_encoded)
    print("✓ Training complete!")
    
    # Extract preprocessor for PCA analysis
    transformer = rf_pipeline.named_steps['preprocessor']
    if 'pca' in transformer.named_steps:
        explained_variance = transformer.named_steps['pca'].explained_variance_ratio_
        cumulative_variance = explained_variance.sum()
        print(f"\nPCA Variance Analysis:")
        print(f"  Varianza acumulada: {cumulative_variance:.4f}")
        
        # Generate PCA variance plot
        plotter.plot_pca_variance(explained_variance, filename="pca_variance.png")
    
    # ============================================================================
    # STEP 7: Make Predictions
    # ============================================================================
    print("\n" + "=" * 60)
    print("STEP 7: Making Predictions")
    print("=" * 60)
    
    # Predict with Random Forest
    print("\n--- Random Forest Predictions ---")
    y_pred_rf = rf_pipeline.predict(X_test)
    print(f"Predictions shape: {y_pred_rf.shape}")
    
    # ============================================================================
    # STEP 8: Evaluate Models
    # ============================================================================
    print("\n" + "=" * 60)
    print("STEP 8: Model Evaluation")
    print("=" * 60)
    
    # Evaluate Random Forest
    print_evaluation(y_test_encoded, y_pred_rf, "Random Forest", class_names=class_names)
    
    # Generate confusion matrix plots
    cm_rf = confusion_matrix(y_test_encoded, y_pred_rf)
    plotter.plot_confusion_matrix(cm_rf, class_names=class_names, 
                                  filename="confusion_matrix_random_forest.png")
    
    # ============================================================================
    # STEP 9: Save Models and Artifacts
    # ============================================================================
    print("\n" + "=" * 60)
    print("STEP 9: Saving Models and Artifacts")
    print("=" * 60)
    
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    import joblib
    
    # Save full pipeline (preprocessing + model)
    pipeline_path = MODELS_DIR / "random_forest_pipeline.pkl"
    joblib.dump(rf_pipeline, pipeline_path)
    print(f"✓ Full pipeline saved: {pipeline_path}")
    
    # Save label encoder
    encoder_path = PROCESSED_DATA_DIR / "label_encoder.pkl"
    joblib.dump(label_encoder, encoder_path)
    print(f"✓ Label encoder saved: {encoder_path}")
    
    # Save preprocessor separately (for inference)
    preprocessor_path = PROCESSED_DATA_DIR / "preprocessor_pipeline.pkl"
    joblib.dump(preprocessor, preprocessor_path)
    print(f"✓ Preprocessor saved: {preprocessor_path}")
    
    # ============================================================================
    # Pipeline Summary
    # ============================================================================
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    
    print("\nPipeline Architecture:")
    print("  1. DataProcessor: Clean data (remove outliers)")
    print("  2. sklearn.Pipeline: Automated preprocessing + model")
    print("     - StandardScaler: Feature normalization")
    print("     - PCA: Dimensionality reduction")
    print("     - RandomForestClassifier: Model")
    print("  3. Evaluation: Comprehensive metrics")
    print("  4. Persistence: Save pipeline for inference")
    
    print("\nReproducibility Notes:")
    print("  - All transformers use random_state=42")
    print("  - Pipeline ensures consistent fit/transform separation")
    print("  - No data leakage between train/test sets")
    print("  - Full pipeline can be loaded and used for inference")
    
    print("\nTo use the saved pipeline for inference:")
    print("  pipeline = joblib.load('models/random_forest_pipeline.pkl')")
    print("  predictions = pipeline.predict(new_data)")


if __name__ == "__main__":
    main()
