"""
Grid Search with MLflow tracking - Execute multiple experiments systematically.

This script runs a grid search over multiple hyperparameters and tracks
all experiments in MLflow for comparison, similar to Optuna but with
explicit tracking of each combination.
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
from itertools import product
from datetime import datetime

from src.data_processor import DataProcessor
from src.feature_transformer import FeatureTransformer
from src.model_trainer import ModelTrainer
from mlops.config import RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR
from mlops.mlflow import MLflowManager, track_training_experiment


def main():
    """Run grid search experiments with MLflow tracking."""
    
    # Initialize MLflow Manager
    mlflow_manager = MLflowManager(
        experiment_name="MusicEmotions_GridSearch",
        tracking_uri=None
    )
    
    print("=" * 70)
    print("MLflow Grid Search - Multiple Experiments")
    print("=" * 70)
    print(f"Experiment: {mlflow_manager.experiment_name}")
    print(f"Tracking URI: {mlflow_manager.client.tracking_uri}")
    print()
    
    # Step 1: Load and prepare data (once)
    print("=" * 70)
    print("Step 1: Loading and preparing data")
    print("=" * 70)
    
    data_path = RAW_DATA_DIR / "turkis_music_emotion_original.csv"
    df = pd.read_csv(data_path)
    print(f"Dataset loaded: {df.shape}")
    
    processor = DataProcessor(iqr_factor=1.5)
    df_clean = processor.remove_outliers_iqr(df)
    print(f"Cleaned dataset: {df_clean.shape}")
    
    target_col = "Class"
    X = df_clean.drop(columns=[target_col])
    y = df_clean[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train set: {X_train.shape}, Test set: {X_test.shape}")
    print()
    
    # Step 2: Define grid search parameters
    print("=" * 70)
    print("Step 2: Defining grid search parameters")
    print("=" * 70)
    
    # Grid search configurations
    grid_configs = {
        "RandomForest": {
            "n_estimators": [100, 200, 300],
            "use_pca": [True, False],
            "n_components": [30, 50]
        },
        "LightGBM": {
            "learning_rate": [0.03, 0.05, 0.1],
            "num_leaves": [20, 31, 50],
            "n_estimators": [300, 500],
            "use_pca": [True, False],
            "n_components": [50]
        }
    }
    
    # Generate all combinations
    all_experiments = []
    
    # Random Forest experiments
    rf_params = grid_configs["RandomForest"]
    for n_est, use_pca, n_comp in product(
        rf_params["n_estimators"],
        rf_params["use_pca"],
        rf_params["n_components"]
    ):
        # Skip invalid combinations
        if not use_pca and n_comp is not None:
            continue
        
        all_experiments.append({
            "model_type": "RandomForest",
            "n_estimators": n_est,
            "use_pca": use_pca,
            "n_components": n_comp if use_pca else None,
            "iqr_factor": 1.5
        })
    
    # LightGBM experiments
    lgb_params = grid_configs["LightGBM"]
    for lr, n_leaves, n_est, use_pca, n_comp in product(
        lgb_params["learning_rate"],
        lgb_params["num_leaves"],
        lgb_params["n_estimators"],
        lgb_params["use_pca"],
        lgb_params["n_components"]
    ):
        # Skip invalid combinations
        if not use_pca and n_comp is not None:
            continue
        
        all_experiments.append({
            "model_type": "LightGBM",
            "learning_rate": lr,
            "num_leaves": n_leaves,
            "n_estimators": n_est,
            "use_pca": use_pca,
            "n_components": n_comp if use_pca else None,
            "iqr_factor": 1.5
        })
    
    print(f"Total experiments to run: {len(all_experiments)}")
    print(f"  - Random Forest: {sum(1 for e in all_experiments if e['model_type'] == 'RandomForest')}")
    print(f"  - LightGBM: {sum(1 for e in all_experiments if e['model_type'] == 'LightGBM')}")
    print()
    
    # Step 3: Run experiments
    print("=" * 70)
    print("Step 3: Running experiments")
    print("=" * 70)
    
    run_infos = []
    successful = 0
    failed = 0
    
    for exp_idx, exp_config in enumerate(all_experiments, 1):
        print(f"\n{'─' * 70}")
        print(f"Experiment {exp_idx}/{len(all_experiments)}")
        print(f"{'─' * 70}")
        print(f"Model: {exp_config['model_type']}")
        print(f"Config: {exp_config}")
        
        try:
            # Transform features
            transformer = FeatureTransformer(
                use_pca=exp_config.get("use_pca", True),
                n_components=exp_config.get("n_components", 50)
            )
            
            y_train_encoded = transformer.encode_labels(y_train)
            y_test_encoded = transformer.encode_labels(y_test)
            
            X_train_transformed, X_test_transformed = transformer.fit_transform(
                X_train, X_test
            )
            
            class_names = list(transformer.label_encoder.classes_)
            
            # Train model
            trainer = ModelTrainer(random_state=42)
            model = None
            params = {}
            
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
            
            # Track in MLflow
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_name = f"{exp_config['model_type'].lower()}_{timestamp}"
            
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
                registered_model_name=None,
                tags={
                    "experiment_number": str(exp_idx),
                    "total_experiments": str(len(all_experiments)),
                    "algorithm": exp_config["model_type"],
                    "preprocessing": "PCA" if exp_config.get("use_pca") else "StandardScaler",
                    "run_type": "grid_search"
                }
            )
            
            run_infos.append({
                "exp_number": exp_idx,
                "config": exp_config,
                "run_info": run_info
            })
            
            successful += 1
            print(f"✓ Success!")
            print(f"  Run ID: {run_info['run_id']}")
            print(f"  Test Accuracy: {run_info['test_metrics']['test_accuracy']:.4f}")
            print(f"  Test F1 Macro: {run_info['test_metrics']['test_f1_macro']:.4f}")
            
        except Exception as e:
            failed += 1
            print(f"✗ Failed: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Step 4: Summary
    print("\n" + "=" * 70)
    print("EXPERIMENTS SUMMARY")
    print("=" * 70)
    print(f"\nTotal experiments: {len(all_experiments)}")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    
    if run_infos:
        # Group by model type
        rf_runs = [r for r in run_infos if r["config"]["model_type"] == "RandomForest"]
        lgb_runs = [r for r in run_infos if r["config"]["model_type"] == "LightGBM"]
        
        print(f"\n  Random Forest: {len(rf_runs)} runs")
        print(f"  LightGBM: {len(lgb_runs)} runs")
        
        # Best overall
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
            print(f"   Run ID: {run_info['run_id']}")
            print(f"   Accuracy: {metrics['test_accuracy']:.4f}")
            print(f"   Precision (macro): {metrics['test_precision_macro']:.4f}")
            print(f"   Recall (macro): {metrics['test_recall_macro']:.4f}")
            print(f"   F1 (macro): {metrics['test_f1_macro']:.4f}")
            print(f"   Config: {config}")
        
        # Best by model type
        if rf_runs:
            best_rf = max(rf_runs, key=lambda x: x["run_info"]["test_metrics"]["test_accuracy"])
            print("\n" + "─" * 70)
            print(f"BEST RANDOM FOREST")
            print("─" * 70)
            print(f"  Run ID: {best_rf['run_info']['run_id']}")
            print(f"  Accuracy: {best_rf['run_info']['test_metrics']['test_accuracy']:.4f}")
            print(f"  Config: {best_rf['config']}")
        
        if lgb_runs:
            best_lgb = max(lgb_runs, key=lambda x: x["run_info"]["test_metrics"]["test_accuracy"])
            print("\n" + "─" * 70)
            print(f"BEST LIGHTGBM")
            print("─" * 70)
            print(f"  Run ID: {best_lgb['run_info']['run_id']}")
            print(f"  Accuracy: {best_lgb['run_info']['test_metrics']['test_accuracy']:.4f}")
            print(f"  Config: {best_lgb['config']}")
    
    print("\n" + "=" * 70)
    print("To view all experiments in MLflow UI:")
    print("  mlflow ui")
    print("\nThen open: http://localhost:5000")
    print("=" * 70)
    
    return run_infos


if __name__ == "__main__":
    main()

