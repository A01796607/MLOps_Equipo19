"""
Training pipeline with MLflow integration for experiment tracking.

This script demonstrates how to use the MLflowManager to track
experiments, log metrics, parameters, and register models.
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
from sklearn.metrics import confusion_matrix

from src.data_processor import DataProcessor
from src.feature_transformer import FeatureTransformer
from src.model_trainer import ModelTrainer
from src.plotter import Plotter
from mlops.config import RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, FIGURES_DIR
from mlops.MLFLow_Equipo19 import MLflowManager, track_training_experiment


def main():
    """Complete training pipeline with MLflow tracking."""
    
    # Initialize MLflow Manager
    mlflow_manager = MLflowManager(
        experiment_name="MusicEmotionsExperiment",
        tracking_uri=None  # Uses local file store
    )
    
    print("=" * 60)
    print("MLflow Experiment Tracking Enabled")
    print("=" * 60)
    print(f"Experiment: {mlflow_manager.experiment_name}")
    print(f"Tracking URI: {mlflow_manager.client.tracking_uri}")
    print()
    
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
    
    # Get class names
    class_names = list(transformer.label_encoder.classes_)
    
    # Generate PCA variance plot if PCA is used
    if transformer.use_pca:
        print("\n--- Generating PCA variance plot ---")
        plotter.plot_pca_variance(
            transformer.pca.explained_variance_ratio_,
            filename="pca_variance.png"
        )
        
        # Log PCA info to MLflow (we'll do this in the experiment tracking)
        pca_explained_variance = transformer.pca.explained_variance_ratio_.sum()
    
    # Step 6: Train models with MLflow tracking
    print("\n" + "=" * 60)
    print("Step 6: Training models with MLflow tracking")
    print("=" * 60)
    
    trainer = ModelTrainer(random_state=42)
    
    # ===== Train Random Forest with MLflow =====
    print("\n--- Training Random Forest with MLflow tracking ---")
    
    rf_params = {
        "model_type": "RandomForest",
        "n_estimators": 200,
        "random_state": 42,
        "use_pca": transformer.use_pca,
        "n_pca_components": transformer.n_components if transformer.use_pca else None,
        "pca_explained_variance": float(transformer.pca.explained_variance_ratio_.sum()) if transformer.use_pca else None,
        "iqr_factor": processor.iqr_factor
    }
    
    rf_model = trainer.train_random_forest(X_train_transformed, y_train_encoded)
    y_pred_rf = trainer.predict('random_forest', X_test_transformed)
    
    # Track Random Forest experiment
    rf_run_info = track_training_experiment(
        model=rf_model,
        model_name="random_forest",
        model_type="sklearn",
        X_train=X_train_transformed,
        y_train=y_train_encoded,
        X_test=X_test_transformed,
        y_test=y_test_encoded,
        params=rf_params,
        class_names=class_names,
        transformer=transformer,
        mlflow_manager=mlflow_manager,
        registered_model_name="MusicEmotions-RandomForest",
        tags={
            "algorithm": "RandomForest",
            "preprocessing": "PCA" if transformer.use_pca else "StandardScaler"
        }
    )
    
    # Log plots as artifacts
    mlflow_manager.start_run(run_name=rf_run_info['run_id'], tags={"resume": "true"})
    try:
        mlflow_manager.log_artifact(
            FIGURES_DIR / "confusion_matrix_random_forest.png",
            artifact_path="plots"
        )
        mlflow_manager.log_artifact(
            FIGURES_DIR / "pca_variance.png",
            artifact_path="plots"
        )
    finally:
        mlflow_manager.end_run()
    
    trainer.print_report(y_test_encoded, y_pred_rf, model_name="Random Forest", class_names=class_names)
    
    # Generate confusion matrix for Random Forest
    cm_rf = confusion_matrix(y_test_encoded, y_pred_rf)
    plotter.plot_confusion_matrix(
        cm_rf, 
        class_names=class_names,
        filename="confusion_matrix_random_forest.png"
    )
    
    # ===== Train LightGBM with MLflow =====
    print("\n--- Training LightGBM with MLflow tracking ---")
    
    lgb_params = {
        "model_type": "LightGBM",
        "n_estimators": 500,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "max_depth": -1,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "random_state": 42,
        "use_pca": transformer.use_pca,
        "n_pca_components": transformer.n_components if transformer.use_pca else None,
        "pca_explained_variance": float(transformer.pca.explained_variance_ratio_.sum()) if transformer.use_pca else None,
        "iqr_factor": processor.iqr_factor
    }
    
    lgb_model = trainer.train_lightgbm(
        X_train_transformed, y_train_encoded,
        X_test_transformed, y_test_encoded
    )
    y_pred_lgb = trainer.predict('lightgbm', X_test_transformed)
    
    # Track LightGBM experiment
    lgb_run_info = track_training_experiment(
        model=lgb_model,
        model_name="lightgbm",
        model_type="lightgbm",
        X_train=X_train_transformed,
        y_train=y_train_encoded,
        X_test=X_test_transformed,
        y_test=y_test_encoded,
        params=lgb_params,
        class_names=class_names,
        transformer=transformer,
        mlflow_manager=mlflow_manager,
        registered_model_name="MusicEmotions-LightGBM",
        tags={
            "algorithm": "LightGBM",
            "preprocessing": "PCA" if transformer.use_pca else "StandardScaler"
        }
    )
    
    # Log plots as artifacts
    mlflow_manager.start_run(run_name=lgb_run_info['run_id'], tags={"resume": "true"})
    try:
        mlflow_manager.log_artifact(
            FIGURES_DIR / "confusion_matrix_lightgbm.png",
            artifact_path="plots"
        )
    finally:
        mlflow_manager.end_run()
    
    trainer.print_report(y_test_encoded, y_pred_lgb, model_name="LightGBM", class_names=class_names)
    
    # Generate confusion matrix for LightGBM
    cm_lgb = confusion_matrix(y_test_encoded, y_pred_lgb)
    plotter.plot_confusion_matrix(
        cm_lgb,
        class_names=class_names,
        filename="confusion_matrix_lightgbm.png"
    )
    
    # Step 7: Save models locally
    print("\n" + "=" * 60)
    print("Step 7: Saving models locally")
    print("=" * 60)
    
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    import joblib
    joblib.dump(rf_model, MODELS_DIR / "random_forest.pkl")
    joblib.dump(lgb_model, MODELS_DIR / "lightgbm.pkl")
    joblib.dump(transformer, PROCESSED_DATA_DIR / "transformer.pkl")
    
    print("Models saved successfully!")
    
    # Step 8: Compare experiments and show best model
    print("\n" + "=" * 60)
    print("Step 8: Experiment Comparison")
    print("=" * 60)
    
    # Get best run
    best_run = mlflow_manager.get_best_run(metric="test_accuracy", ascending=False)
    if best_run:
        print(f"\nBest model based on test_accuracy:")
        print(f"  Run ID: {best_run['run_id']}")
        print(f"  Metric: {best_run['metric_name']} = {best_run['metric_value']:.4f}")
    
    # Compare runs
    runs_df = mlflow_manager.compare_runs([rf_run_info['run_id'], lgb_run_info['run_id']])
    if not runs_df.empty:
        print("\nComparison of models:")
        print("-" * 60)
        print(runs_df[['tags.mlflow.runName', 'metrics.test_accuracy', 'metrics.test_f1_macro']].to_string())
    
    print("\n" + "=" * 60)
    print("Pipeline completed successfully!")
    print("=" * 60)
    print("\nTo view MLflow UI, run:")
    print("  mlflow ui")
    print("\nThen open http://localhost:5000 in your browser")


if __name__ == "__main__":
    main()

