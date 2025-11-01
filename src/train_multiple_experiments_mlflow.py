"""
Training pipeline with MLflow integration for multiple experiments.

This script runs multiple experiments with different configurations
and tracks all of them in MLflow for comparison.
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
from itertools import product

from src.data_processor import DataProcessor
from src.feature_transformer import FeatureTransformer
from src.model_trainer import ModelTrainer
from src.plotter import Plotter
from mlops.config import RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, FIGURES_DIR
from mlops.mlflow import MLflowManager, track_training_experiment


def main():
    """Run multiple experiments with different configurations."""
    
    # Initialize MLflow Manager
    mlflow_manager = MLflowManager(
        experiment_name="MusicEmotions_Experiments" + datetime.now().strftime("%Y%m%d_%H%M%S"),
        tracking_uri=None  # Uses local file store
    )
    
    print("=" * 60)
    print("MLflow Multiple Experiments Tracking")
    print("=" * 60)
    print(f"Experiment: {mlflow_manager.experiment_name}")
    print(f"Tracking URI: {mlflow_manager.client.tracking_uri}")
    print()
    
    # Step 1: Load and prepare data (only once)
    print("=" * 60)
    print("Step 1: Loading and preparing data")
    print("=" * 60)
    
    data_path = RAW_DATA_DIR / "turkis_music_emotion_original.csv"
    df = pd.read_csv(data_path)
    print(f"Dataset loaded: {df.shape}")
    
    # Process data
    processor = DataProcessor(iqr_factor=1.5)
    df_clean = processor.remove_outliers_iqr(df)
    print(f"Cleaned dataset: {df_clean.shape}")
    
    # Prepare features and target
    target_col = "Class"
    X = df_clean.drop(columns=[target_col])
    y = df_clean[target_col]
    
    # Split data once
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train set: {X_train.shape}, Test set: {X_test.shape}")
    print()
    
    # Define experiment configurations - Similar to previous experiments
    # Generate multiple combinations of hyperparameters
    experiments = []
    
    # Random Forest configurations
    rf_configs = [
        {"n_estimators": 100, "use_pca": True, "n_components": 50},
        {"n_estimators": 200, "use_pca": True, "n_components": 50},
        {"n_estimators": 300, "use_pca": True, "n_components": 50},
        {"n_estimators": 200, "use_pca": False, "n_components": None},
        {"n_estimators": 100, "use_pca": True, "n_components": 30},
    ]
    
    for config in rf_configs:
        experiments.append({
            "model_type": "RandomForest",
            "n_estimators": config["n_estimators"],
            "use_pca": config["use_pca"],
            "n_components": config["n_components"],
            "iqr_factor": 1.5
        })
    
    # LightGBM configurations
    lgb_configs = [
        {"learning_rate": 0.05, "num_leaves": 31, "n_estimators": 500, "use_pca": True, "n_components": 50},
        {"learning_rate": 0.1, "num_leaves": 31, "n_estimators": 500, "use_pca": True, "n_components": 50},
        {"learning_rate": 0.05, "num_leaves": 50, "n_estimators": 500, "use_pca": True, "n_components": 50},
        {"learning_rate": 0.05, "num_leaves": 31, "n_estimators": 300, "use_pca": False, "n_components": None},
        {"learning_rate": 0.03, "num_leaves": 31, "n_estimators": 500, "use_pca": True, "n_components": 50},
        {"learning_rate": 0.05, "num_leaves": 20, "n_estimators": 500, "use_pca": True, "n_components": 50},
    ]
    
    for config in lgb_configs:
        experiments.append({
            "model_type": "LightGBM",
            "learning_rate": config["learning_rate"],
            "num_leaves": config["num_leaves"],
            "n_estimators": config["n_estimators"],
            "use_pca": config["use_pca"],
            "n_components": config["n_components"],
            "iqr_factor": 1.5
        })
    
    print("=" * 60)
    print(f"Step 2: Running {len(experiments)} experiments")
    print("=" * 60)
    print()
    
    run_infos = []
    
    for exp_idx, exp_config in enumerate(experiments, 1):
        print(f"\n{'='*60}")
        print(f"Experiment {exp_idx}/{len(experiments)}")
        print(f"{'='*60}")
        print(f"Configuration: {exp_config}")
        print()
        
        try:
            # Step 2: Transform features with current config
            transformer = FeatureTransformer(
                use_pca=exp_config.get("use_pca", True),
                n_components=exp_config.get("n_components", 50)
            )
            
            # Encode labels
            y_train_encoded = transformer.encode_labels(y_train)
            y_test_encoded = transformer.encode_labels(y_test)
            
            # Transform features
            X_train_transformed, X_test_transformed = transformer.fit_transform(
                X_train, X_test
            )
            
            # Get class names
            class_names = list(transformer.label_encoder.classes_)
            
            # Step 3: Train model
            trainer = ModelTrainer(random_state=42)
            model = None
            
            if exp_config["model_type"] == "RandomForest":
                model = trainer.train_random_forest(
                    X_train_transformed,
                    y_train_encoded,
                    n_estimators=exp_config["n_estimators"]
                )
                model_type_str = "sklearn"
                
                params = {
                    "model_type": "RandomForest",
                    "n_estimators": exp_config["n_estimators"],
                    "random_state": 42,
                    "use_pca": exp_config.get("use_pca", True),
                    "n_pca_components": exp_config.get("n_components"),
                    "iqr_factor": exp_config["iqr_factor"],
                    "pca_explained_variance": float(
                        transformer.pca.explained_variance_ratio_.sum()
                    ) if transformer.use_pca else None,
                }
                
            elif exp_config["model_type"] == "LightGBM":
                model = trainer.train_lightgbm(
                    X_train_transformed,
                    y_train_encoded,
                    X_test_transformed,
                    y_test_encoded,
                    n_estimators=exp_config.get("n_estimators", 500),
                    early_stopping_rounds=5
                )
                model_type_str = "lightgbm"
                
                params = {
                    "model_type": "LightGBM",
                    "learning_rate": exp_config.get("learning_rate", 0.05),
                    "num_leaves": exp_config.get("num_leaves", 31),
                    "n_estimators": exp_config.get("n_estimators", 500),
                    "random_state": 42,
                    "use_pca": exp_config.get("use_pca", True),
                    "n_pca_components": exp_config.get("n_components"),
                    "iqr_factor": exp_config["iqr_factor"],
                    "pca_explained_variance": float(
                        transformer.pca.explained_variance_ratio_.sum()
                    ) if transformer.use_pca else None,
                }
            
            # Step 4: Track experiment in MLflow
            # Create descriptive run name similar to MLflow UI format
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_name = f"{exp_config['model_type'].lower()}_{timestamp}_exp{exp_idx}"
            
            run_info = track_training_experiment(
                model=model,
                model_name=f"{exp_config['model_type'].lower()}_model",
                model_type=model_type_str,
                X_train=X_train_transformed,
                y_train=y_train_encoded,
                X_test=X_test_transformed,
                y_test=y_test_encoded,
                params=params,
                class_names=class_names,
                transformer=transformer,
                mlflow_manager=mlflow_manager,
                registered_model_name=None,  # Don't register each model
                tags={
                    "experiment_number": str(exp_idx),
                    "total_experiments": str(len(experiments)),
                    "algorithm": exp_config["model_type"],
                    "preprocessing": "PCA" if exp_config.get("use_pca") else "StandardScaler",
                    "run_type": "multiple_experiments"
                }
            )
            
            run_infos.append({
                "exp_number": exp_idx,
                "config": exp_config,
                "run_info": run_info
            })
            
            metrics = run_info['test_metrics']
            print(f"\n✓ Experiment {exp_idx} completed successfully!")
            print(f"  Run ID: {run_info['run_id'][:20]}...")
            print(f"  Test Accuracy: {metrics['test_accuracy']:.4f}")
            print(f"  Test F1 Macro: {metrics['test_f1_macro']:.4f}")
            print(f"  Test Precision: {metrics['test_precision_macro']:.4f}")
            print(f"  Test Recall: {metrics['test_recall_macro']:.4f}")
            
        except Exception as e:
            print(f"\n✗ Experiment {exp_idx} failed: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Step 5: Summary
    print("\n" + "=" * 70)
    print("EXPERIMENTS SUMMARY")
    print("=" * 70)
    
    if run_infos:
        print(f"\nTotal experiments completed: {len(run_infos)}/{len(experiments)}")
        
        # Group by model type
        rf_runs = [r for r in run_infos if r["config"]["model_type"] == "RandomForest"]
        lgb_runs = [r for r in run_infos if r["config"]["model_type"] == "LightGBM"]
        
        print(f"  - Random Forest: {len(rf_runs)} experiments")
        print(f"  - LightGBM: {len(lgb_runs)} experiments")
        
        # Best experiments by accuracy
        print("\n" + "─" * 70)
        print("TOP 5 EXPERIMENTS (by Test Accuracy)")
        print("─" * 70)
        
        sorted_runs = sorted(
            run_infos,
            key=lambda x: x["run_info"]["test_metrics"]["test_accuracy"],
            reverse=True
        )
        
        for i, run_data in enumerate(sorted_runs[:5], 1):
            run_info = run_data["run_info"]
            config = run_data["config"]
            metrics = run_info["test_metrics"]
            
            print(f"\n{i}. Experiment #{run_data['exp_number']} - {config['model_type']}")
            print(f"   Run ID: {run_info['run_id'][:20]}...")
            print(f"   Test Accuracy: {metrics['test_accuracy']:.4f}")
            print(f"   Test F1 Macro: {metrics['test_f1_macro']:.4f}")
            print(f"   Test Precision: {metrics['test_precision_macro']:.4f}")
            print(f"   Test Recall: {metrics['test_recall_macro']:.4f}")
            print(f"   Config: n_estimators={config.get('n_estimators', config.get('learning_rate', 'N/A'))}, "
                  f"PCA={config.get('use_pca', False)}")
        
        # Best by model type
        if rf_runs:
            best_rf = max(rf_runs, key=lambda x: x["run_info"]["test_metrics"]["test_accuracy"])
            print("\n" + "─" * 70)
            print("BEST RANDOM FOREST")
            print("─" * 70)
            print(f"  Run ID: {best_rf['run_info']['run_id'][:20]}...")
            print(f"  Accuracy: {best_rf['run_info']['test_metrics']['test_accuracy']:.4f}")
            print(f"  Config: {best_rf['config']}")
        
        if lgb_runs:
            best_lgb = max(lgb_runs, key=lambda x: x["run_info"]["test_metrics"]["test_accuracy"])
            print("\n" + "─" * 70)
            print("BEST LIGHTGBM")
            print("─" * 70)
            print(f"  Run ID: {best_lgb['run_info']['run_id'][:20]}...")
            print(f"  Accuracy: {best_lgb['run_info']['test_metrics']['test_accuracy']:.4f}")
            print(f"  Config: {best_lgb['config']}")
    else:
        print("\nNo experiments completed successfully.")
    
    print("\n" + "=" * 70)
    print("All experiments have been logged to MLflow!")
    print("\nTo view all experiments in MLflow UI, run:")
    print("  mlflow ui")
    print("\nThen open: http://localhost:5000")
    print("\nYou will see all runs listed with:")
    print("  - Run names (e.g., randomforest_20251031_123456_exp1)")
    print("  - Metrics (test_accuracy, test_f1_macro, etc.)")
    print("  - Parameters (n_estimators, learning_rate, etc.)")
    print("  - Model types (random_forest, lightgbm)")
    print("=" * 70)
    
    return run_infos


if __name__ == "__main__":
    main()

