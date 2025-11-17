"""
Reproducibility Validation Script.

This script validates that the ML pipeline produces identical results when run
in different environments (e.g., different machines, VMs, or containers).

It trains the model and compares metrics/artifacts with reference metrics stored
in MLflow or saved reference files.

Usage:
    python scripts/validate_reproducibility.py [--reference-run-id RUN_ID]
    python scripts/validate_reproducibility.py [--reference-metrics-file PATH]
"""
import sys
from pathlib import Path
import argparse
import json
from typing import Dict, Optional

# Add project root to Python path
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, 
    f1_score, 
    precision_score, 
    recall_score,
    classification_report
)
from loguru import logger

from src.data_processor import DataProcessor
from src.feature_transformer import FeatureTransformer
from src.model_trainer import ModelTrainer, LIGHTGBM_AVAILABLE
from src.reproducibility import ensure_reproducibility, DEFAULT_SEED
from mlops.config import RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR
from mlops.mlflow import MLflowManager


def train_model_and_get_metrics(
    model_type: str = 'random_forest',
    seed: int = DEFAULT_SEED
) -> Dict:
    """
    Train a model and return metrics and predictions.
    
    Args:
        model_type: Type of model to train ('random_forest' or 'lightgbm')
        seed: Random seed for reproducibility
    
    Returns:
        Dictionary with metrics, predictions, and model info
    """
    # Ensure reproducibility
    reprod_config = ensure_reproducibility(seed=seed, verbose=False)
    
    # Load data
    logger.info("Loading data...")
    data_path = RAW_DATA_DIR / "turkis_music_emotion_original.csv"
    df = pd.read_csv(data_path)
    
    # Preprocess
    logger.info("Preprocessing data...")
    processor = DataProcessor(iqr_factor=1.5)
    df_clean = processor.remove_outliers_iqr(df)
    
    # Prepare features
    target_col = "Class"
    X = df_clean.drop(columns=[target_col])
    y = df_clean[target_col]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, **reprod_config['split'], stratify=y
    )
    
    # Transform features
    logger.info("Transforming features...")
    transformer = FeatureTransformer(use_pca=True, n_components=50)
    y_train_encoded = transformer.encode_labels(y_train)
    y_test_encoded = transformer.encode_labels(y_test)
    X_train_transformed, X_test_transformed = transformer.fit_transform(
        X_train, X_test
    )
    
    # Train model
    logger.info(f"Training {model_type} model...")
    trainer = ModelTrainer(random_state=reprod_config['seed'])
    
    if model_type == 'random_forest':
        model = trainer.train_random_forest(
            X_train_transformed,
            y_train_encoded,
            n_estimators=200
        )
    elif model_type == 'lightgbm' and LIGHTGBM_AVAILABLE:
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train_transformed,
            y_train_encoded,
            test_size=0.2,
            **reprod_config['split'],
            stratify=y_train_encoded
        )
        model = trainer.train_lightgbm(
            X_tr, y_tr,
            X_val, y_val,
            n_estimators=500,
            early_stopping_rounds=10
        )
    else:
        raise ValueError(f"Model type {model_type} not supported")
    
    # Make predictions
    logger.info("Making predictions...")
    y_pred = trainer.predict(model_type if model_type == 'random_forest' else 'lightgbm', X_test_transformed)
    
    # Calculate metrics
    metrics = {
        'accuracy': float(accuracy_score(y_test_encoded, y_pred)),
        'f1_macro': float(f1_score(y_test_encoded, y_pred, average='macro')),
        'f1_weighted': float(f1_score(y_test_encoded, y_pred, average='weighted')),
        'precision_macro': float(precision_score(y_test_encoded, y_pred, average='macro')),
        'recall_macro': float(recall_score(y_test_encoded, y_pred, average='macro')),
    }
    
    # Get classification report as dict
    report_dict = classification_report(
        y_test_encoded, y_pred,
        output_dict=True,
        target_names=list(transformer.label_encoder.classes_)
    )
    
    # Store first few predictions for comparison
    sample_predictions = {
        'y_test_sample': y_test_encoded[:10].tolist(),
        'y_pred_sample': y_pred[:10].tolist(),
        'class_names': list(transformer.label_encoder.classes_)
    }
    
    return {
        'metrics': metrics,
        'classification_report': report_dict,
        'sample_predictions': sample_predictions,
        'seed': seed,
        'model_type': model_type,
        'data_shape': {
            'train': X_train.shape,
            'test': X_test.shape
        }
    }


def load_reference_metrics(reference_file: Path) -> Dict:
    """Load reference metrics from a JSON file."""
    with open(reference_file, 'r') as f:
        return json.load(f)


def load_reference_from_mlflow(run_id: str, mlflow_manager: MLflowManager) -> Dict:
    """Load reference metrics from MLflow run."""
    run = mlflow_manager.client.get_run(run_id)
    
    # Extract metrics
    metrics = {
        'accuracy': float(run.data.metrics.get('test_accuracy', 0)),
        'f1_macro': float(run.data.metrics.get('test_f1_macro', 0)),
        'f1_weighted': float(run.data.metrics.get('test_f1_weighted', 0)),
        'precision_macro': float(run.data.metrics.get('test_precision_macro', 0)),
        'recall_macro': float(run.data.metrics.get('test_recall_macro', 0)),
    }
    
    return {
        'metrics': metrics,
        'seed': int(run.data.params.get('random_state', DEFAULT_SEED)),
        'model_type': run.data.params.get('model_type', 'unknown')
    }


def compare_metrics(
    current_metrics: Dict,
    reference_metrics: Dict,
    tolerance: float = 1e-6
) -> Dict:
    """
    Compare current metrics with reference metrics.
    
    Args:
        current_metrics: Current run metrics
        reference_metrics: Reference metrics to compare against
        tolerance: Numerical tolerance for comparison
    
    Returns:
        Dictionary with comparison results
    """
    comparison = {
        'matches': True,
        'differences': {},
        'all_match': True
    }
    
    current = current_metrics.get('metrics', {})
    reference = reference_metrics.get('metrics', {})
    
    for metric_name in current.keys():
        if metric_name not in reference:
            comparison['differences'][metric_name] = f"Missing in reference"
            comparison['all_match'] = False
            continue
        
        curr_val = current[metric_name]
        ref_val = reference[metric_name]
        diff = abs(curr_val - ref_val)
        
        if diff > tolerance:
            comparison['matches'] = False
            comparison['all_match'] = False
            comparison['differences'][metric_name] = {
                'current': curr_val,
                'reference': ref_val,
                'difference': diff,
                'within_tolerance': False
            }
        else:
            comparison['differences'][metric_name] = {
                'current': curr_val,
                'reference': ref_val,
                'difference': diff,
                'within_tolerance': True
            }
    
    return comparison


def save_reference_metrics(metrics: Dict, output_file: Path):
    """Save metrics as reference for future comparisons."""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    logger.success(f"Reference metrics saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Validate reproducibility of ML pipeline'
    )
    parser.add_argument(
        '--model-type',
        type=str,
        default='random_forest',
        choices=['random_forest', 'lightgbm'],
        help='Model type to train and validate'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=DEFAULT_SEED,
        help=f'Random seed to use (default: {DEFAULT_SEED})'
    )
    parser.add_argument(
        '--reference-metrics-file',
        type=Path,
        help='Path to reference metrics JSON file'
    )
    parser.add_argument(
        '--reference-run-id',
        type=str,
        help='MLflow run ID to use as reference'
    )
    parser.add_argument(
        '--save-reference',
        type=Path,
        help='Save current metrics as reference to specified file'
    )
    parser.add_argument(
        '--tolerance',
        type=float,
        default=1e-6,
        help='Numerical tolerance for metric comparison (default: 1e-6)'
    )
    
    args = parser.parse_args()
    
    logger.info("=" * 70)
    logger.info("Reproducibility Validation")
    logger.info("=" * 70)
    logger.info(f"Model type: {args.model_type}")
    logger.info(f"Random seed: {args.seed}")
    logger.info(f"Tolerance: {args.tolerance}")
    logger.info("")
    
    # Train model and get metrics
    logger.info("Training model in current environment...")
    current_results = train_model_and_get_metrics(
        model_type=args.model_type,
        seed=args.seed
    )
    
    current_metrics = current_results['metrics']
    logger.info("\nCurrent Metrics:")
    for metric_name, metric_value in current_metrics.items():
        logger.info(f"  {metric_name}: {metric_value:.6f}")
    
    # If saving reference, save and exit
    if args.save_reference:
        save_reference_metrics(current_results, args.save_reference)
        logger.info("\n✓ Reference metrics saved successfully")
        return
    
    # Load reference metrics
    if args.reference_metrics_file:
        logger.info(f"\nLoading reference metrics from: {args.reference_metrics_file}")
        reference_results = load_reference_metrics(args.reference_metrics_file)
    elif args.reference_run_id:
        logger.info(f"\nLoading reference metrics from MLflow run: {args.reference_run_id}")
        mlflow_manager = MLflowManager(experiment_name="MusicEmotions_Experiments")
        reference_results = load_reference_from_mlflow(args.reference_run_id, mlflow_manager)
    else:
        logger.error("Either --reference-metrics-file or --reference-run-id must be provided")
        logger.info("\nUsage examples:")
        logger.info("  # Save reference metrics first:")
        logger.info("  python scripts/validate_reproducibility.py --save-reference reference_metrics.json")
        logger.info("\n  # Then validate against reference:")
        logger.info("  python scripts/validate_reproducibility.py --reference-metrics-file reference_metrics.json")
        return
    
    reference_metrics = reference_results.get('metrics', {})
    logger.info("\nReference Metrics:")
    for metric_name, metric_value in reference_metrics.items():
        logger.info(f"  {metric_name}: {metric_value:.6f}")
    
    # Compare metrics
    logger.info("\n" + "=" * 70)
    logger.info("Comparison Results")
    logger.info("=" * 70)
    
    comparison = compare_metrics(
        current_results,
        reference_results,
        tolerance=args.tolerance
    )
    
    logger.info(f"\nTolerance: {args.tolerance}")
    logger.info("")
    
    all_match = True
    for metric_name, result in comparison['differences'].items():
        if isinstance(result, dict) and result.get('within_tolerance', False):
            logger.success(f"✓ {metric_name}: MATCH")
            logger.info(f"    Current: {result['current']:.6f}")
            logger.info(f"    Reference: {result['reference']:.6f}")
            logger.info(f"    Difference: {result['difference']:.2e}")
        else:
            logger.error(f"✗ {metric_name}: MISMATCH")
            if isinstance(result, dict):
                logger.error(f"    Current: {result['current']:.6f}")
                logger.error(f"    Reference: {result['reference']:.6f}")
                logger.error(f"    Difference: {result['difference']:.2e}")
            else:
                logger.error(f"    {result}")
            all_match = False
    
    logger.info("\n" + "=" * 70)
    if all_match:
        logger.success("✓ REPRODUCIBILITY VALIDATION PASSED")
        logger.success("  All metrics match within tolerance!")
    else:
        logger.error("✗ REPRODUCIBILITY VALIDATION FAILED")
        logger.error("  Some metrics do not match the reference")
    logger.info("=" * 70)
    
    # Compare sample predictions
    if 'sample_predictions' in current_results and 'sample_predictions' in reference_results:
        logger.info("\nSample Predictions Comparison:")
        curr_samples = current_results['sample_predictions']['y_pred_sample']
        ref_samples = reference_results['sample_predictions']['y_pred_sample']
        
        if curr_samples == ref_samples:
            logger.success("✓ Sample predictions match exactly!")
        else:
            logger.warning("⚠ Sample predictions differ:")
            for i, (curr, ref) in enumerate(zip(curr_samples[:5], ref_samples[:5])):
                if curr == ref:
                    logger.info(f"  Sample {i}: {curr} == {ref} ✓")
                else:
                    logger.warning(f"  Sample {i}: {curr} != {ref} ✗")


if __name__ == "__main__":
    main()

