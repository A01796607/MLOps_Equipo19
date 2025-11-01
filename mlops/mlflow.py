"""
MLflow Integration for Experiment Tracking and Model Management.

This module provides comprehensive MLflow integration for:
- Experiment tracking and versioning
- Parameter and metric logging
- Model registry and versioning
- Result visualization and comparison
- Model lifecycle management
- DVC data versioning integration
"""
import subprocess
from pathlib import Path
from typing import Dict, Optional, Any, List
from datetime import datetime

import mlflow
import mlflow.sklearn
import mlflow.lightgbm
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
import pandas as pd 
import numpy as np
from loguru import logger

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# Import DVC Manager
try:
    from src.dvc import DVCManager
    DVC_AVAILABLE = True
except ImportError:
    DVC_AVAILABLE = False
    DVCManager = None
    logger.warning("DVC not available. Data versioning features will be disabled.")


class MLflowManager:
    """
    Manager class for MLflow experiment tracking and model management.
    """
    
    def __init__(
        self,
        experiment_name: str = "MusicEmotionsExperiment",
        tracking_uri: Optional[str] = None,
        artifact_path: Optional[Path] = None
    ):
        """
        Initialize MLflow Manager.
        
        Args:
            experiment_name: Name of the MLflow experiment
            tracking_uri: MLflow tracking URI (default: local file store)
            artifact_path: Path to store artifacts
        """
        # Set tracking URI
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        else:
            # Use local file store if not specified
            mlflow_dir = Path("mlruns")
            mlflow_dir.mkdir(exist_ok=True)
            mlflow.set_tracking_uri(f"file://{mlflow_dir.absolute()}")
        
        # Set experiment
        try:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment is None:
                experiment_id = mlflow.create_experiment(
                    experiment_name,
                    tags={
                        "description": "Music Emotion Recognition Experiment",
                        "created_at": datetime.now().isoformat()
                    }
                )
                logger.info(f"Created new experiment: {experiment_name} (ID: {experiment_id})")
            else:
                logger.info(f"Using existing experiment: {experiment_name} (ID: {experiment.experiment_id})")
        except Exception as e:
            logger.warning(f"Could not set experiment: {e}. Using default.")
        
        mlflow.set_experiment(experiment_name)
        
        self.experiment_name = experiment_name
        self.artifact_path = artifact_path or Path("mlflow_artifacts")
        self.artifact_path.mkdir(parents=True, exist_ok=True)
        
        self.client = MlflowClient()
        self.current_run = None
        
        # DVC Manager (lazy initialization)
        self._dvc_manager = None
        if DVC_AVAILABLE and DVCManager is not None:
            try:
                self._dvc_manager = DVCManager()
                logger.debug("DVC manager initialized")
            except Exception as e:
                logger.warning(f"DVC manager not available: {e}")
                self._dvc_manager = None
        
    def start_run(self, run_name: Optional[str] = None, tags: Optional[Dict[str, str]] = None):
        """
        Start a new MLflow run.
        
        Args:
            run_name: Name for this run
            tags: Dictionary of tags to add to the run
        """
        # End any existing run first
        if mlflow.active_run() is not None:
            logger.warning("Ending existing active run before starting new one")
            mlflow.end_run()
        
        if run_name is None:
            run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        tags = tags or {}
        tags["created_at"] = datetime.now().isoformat()
        
        self.current_run = mlflow.start_run(run_name=run_name, tags=tags)
        logger.info(f"Started MLflow run: {run_name} (ID: {self.current_run.info.run_id})")
        logger.info(f"Active run status: {mlflow.active_run() is not None}")
        
        return self.current_run
    
    def end_run(self):
        """End the current MLflow run."""
        if self.current_run:
            mlflow.end_run()
            logger.info(f"Ended MLflow run: {self.current_run.info.run_id}")
            self.current_run = None
    
    def log_params(self, params: Dict[str, Any]):
        """
        Log parameters to MLflow.
        
        Args:
            params: Dictionary of parameters to log
        """
        # Filter out None values and convert to strings if necessary
        clean_params = {k: str(v) for k, v in params.items() if v is not None}
        mlflow.log_params(clean_params)
        logger.debug(f"Logged {len(clean_params)} parameters")
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """
        Log metrics to MLflow.
        
        Args:
            metrics: Dictionary of metrics to log
            step: Optional step number for metric logging
        """
        # Check if there's an active run
        if mlflow.active_run() is None:
            logger.error("No active MLflow run! Metrics cannot be logged.")
            raise RuntimeError("No active MLflow run. Call start_run() first.")
        
        # Clean metrics: convert to float and filter out NaN/Inf values
        clean_metrics = {}
        for metric_name, metric_value in metrics.items():
            try:
                # Convert to float and check for valid values
                float_value = float(metric_value)
                if not (np.isnan(float_value) or np.isinf(float_value)):
                    clean_metrics[metric_name] = float_value
                else:
                    logger.warning(f"Skipping metric {metric_name}: invalid value {metric_value}")
            except (ValueError, TypeError) as e:
                logger.warning(f"Skipping metric {metric_name}: {e}")
        
        if not clean_metrics:
            logger.warning("No valid metrics to log")
            return
        
        try:
            if step is not None:
                for metric_name, metric_value in clean_metrics.items():
                    mlflow.log_metric(metric_name, metric_value, step=step)
            else:
                mlflow.log_metrics(clean_metrics)
            
            logger.info(f"✓ Successfully logged {len(clean_metrics)} metrics to MLflow")
            logger.debug(f"Metrics logged: {list(clean_metrics.keys())[:5]}...")  # Show first 5
        except Exception as e:
            logger.error(f"Error logging metrics to MLflow: {e}")
            raise
    
    def log_evaluation_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        prefix: str = "test",
        class_names: Optional[List[str]] = None
    ):
        """
        Log comprehensive evaluation metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels (or probabilities - will be converted to classes)
            prefix: Prefix for metric names (e.g., 'test', 'train', 'val')
            class_names: List of class names for per-class metrics
        """
        # Ensure y_true and y_pred are 1D arrays of integers (class labels)
        y_true = np.asarray(y_true).ravel().astype(int)
        
        # Convert y_pred to class labels if it's probabilities (2D array)
        y_pred = np.asarray(y_pred)
        if y_pred.ndim > 1:
            # If 2D, it's probabilities - take argmax
            y_pred = np.argmax(y_pred, axis=1)
        y_pred = y_pred.ravel().astype(int)
        
        # Ensure both arrays have the same length
        if len(y_true) != len(y_pred):
            raise ValueError(
                f"y_true and y_pred have different lengths: {len(y_true)} vs {len(y_pred)}"
            )
        
        metrics = {}
        
        try:
            # Overall metrics
            metrics[f"{prefix}_accuracy"] = float(accuracy_score(y_true, y_pred))
            metrics[f"{prefix}_precision_macro"] = float(precision_score(y_true, y_pred, average='macro', zero_division=0))
            metrics[f"{prefix}_recall_macro"] = float(recall_score(y_true, y_pred, average='macro', zero_division=0))
            metrics[f"{prefix}_f1_macro"] = float(f1_score(y_true, y_pred, average='macro', zero_division=0))
            metrics[f"{prefix}_precision_weighted"] = float(precision_score(y_true, y_pred, average='weighted', zero_division=0))
            metrics[f"{prefix}_recall_weighted"] = float(recall_score(y_true, y_pred, average='weighted', zero_division=0))
            metrics[f"{prefix}_f1_weighted"] = float(f1_score(y_true, y_pred, average='weighted', zero_division=0))
            
            logger.info(f"Calculated {len(metrics)} metrics for {prefix}")
        except Exception as e:
            logger.error(f"Error calculating metrics for {prefix}: {e}")
            raise
        
        # Per-class metrics if class names provided
        if class_names:
            report = classification_report(
                y_true, y_pred,
                target_names=class_names,
                output_dict=True,
                zero_division=0
            )
            
            for class_name in class_names:
                if class_name in report:
                    try:
                        metrics[f"{prefix}_precision_{class_name}"] = float(report[class_name]['precision'])
                        metrics[f"{prefix}_recall_{class_name}"] = float(report[class_name]['recall'])
                        metrics[f"{prefix}_f1_{class_name}"] = float(report[class_name]['f1-score'])
                    except (KeyError, ValueError, TypeError) as e:
                        logger.warning(f"Error extracting metrics for class {class_name}: {e}")
        
        # Log metrics to MLflow
        self.log_metrics(metrics)
        
        logger.info(f"Successfully logged {len(metrics)} metrics for {prefix}")
        return metrics
    
    def log_model(
        self,
        model: Any,
        model_name: str,
        signature=None,
        input_example=None,
        registered_model_name: Optional[str] = None,
        model_type: str = "sklearn"
    ):
        """
        Log model to MLflow.
        
        Args:
            model: Trained model
            model_name: Name for the model artifact
            signature: Model signature (optional, will be inferred if not provided)
            input_example: Example input for the model
            registered_model_name: Name for model registry (if None, model won't be registered)
            model_type: Type of model ('sklearn' or 'lightgbm')
            
        Returns:
            ModelInfo object
        """
        if signature is None and input_example is not None:
            try:
                # Infer signature from example
                predictions = model.predict(input_example[:5]) if hasattr(model, 'predict') else None
                if predictions is not None:
                    signature = infer_signature(input_example[:5], predictions)
            except Exception as e:
                logger.warning(f"Could not infer signature: {e}")
        
        if model_type == "sklearn":
            model_info = mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path=model_name,
                signature=signature,
                input_example=input_example,
                registered_model_name=registered_model_name
            )
        elif model_type == "lightgbm":
            model_info = mlflow.lightgbm.log_model(
                lgb_model=model,
                artifact_path=model_name,
                signature=signature,
                input_example=input_example,
                registered_model_name=registered_model_name
            )
        else:
            raise ValueError(f"Unsupported model_type: {model_type}")
        
        logger.info(f"Logged model: {model_name} (Model URI: {model_info.model_uri})")
        
        if registered_model_name:
            logger.info(f"Registered model: {registered_model_name}")
        
        return model_info
    
    def log_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        class_names: Optional[List[str]] = None,
        artifact_name: str = "confusion_matrix"
    ):
        """
        Log confusion matrix as artifact.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels (or probabilities - will be converted to classes)
            class_names: List of class names
            artifact_name: Name for the artifact
        """
        # Ensure y_true and y_pred are 1D arrays of integers (class labels)
        y_true = np.asarray(y_true).ravel().astype(int)
        
        # Convert y_pred to class labels if it's probabilities (2D array)
        y_pred = np.asarray(y_pred)
        if y_pred.ndim > 1:
            # If 2D, it's probabilities - take argmax
            y_pred = np.argmax(y_pred, axis=1)
        y_pred = y_pred.ravel().astype(int)
        
        cm = confusion_matrix(y_true, y_pred)
        
        # Save confusion matrix as CSV
        cm_df = pd.DataFrame(
            cm,
            index=class_names if class_names else [f"Class_{i}" for i in range(len(cm))],
            columns=class_names if class_names else [f"Class_{i}" for i in range(len(cm))]
        )
        
        cm_path = self.artifact_path / f"{artifact_name}.csv"
        cm_df.to_csv(cm_path)
        
        mlflow.log_artifact(str(cm_path), artifact_path="metrics")
        logger.debug(f"Logged confusion matrix: {artifact_name}")
    
    def log_classification_report(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        class_names: Optional[List[str]] = None,
        artifact_name: str = "classification_report"
    ):
        """
        Log classification report as artifact.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels (or probabilities - will be converted to classes)
            class_names: List of class names
            artifact_name: Name for the artifact
        """
        # Ensure y_true and y_pred are 1D arrays of integers (class labels)
        y_true = np.asarray(y_true).ravel().astype(int)
        
        # Convert y_pred to class labels if it's probabilities (2D array)
        y_pred = np.asarray(y_pred)
        if y_pred.ndim > 1:
            # If 2D, it's probabilities - take argmax
            y_pred = np.argmax(y_pred, axis=1)
        y_pred = y_pred.ravel().astype(int)
        
        report = classification_report(
            y_true, y_pred,
            target_names=class_names,
            output_dict=True,
            zero_division=0
        )
        
        # Convert to DataFrame
        report_df = pd.DataFrame(report).transpose()
        
        report_path = self.artifact_path / f"{artifact_name}.csv"
        report_df.to_csv(report_path)
        
        mlflow.log_artifact(str(report_path), artifact_path="metrics")
        logger.debug(f"Logged classification report: {artifact_name}")
    
    def log_artifact(self, file_path: Path, artifact_path: Optional[str] = None):
        """
        Log an artifact file.
        
        Args:
            file_path: Path to the file to log
            artifact_path: Path within the artifact store
        """
        if file_path.exists():
            mlflow.log_artifact(str(file_path), artifact_path=artifact_path)
            logger.debug(f"Logged artifact: {file_path}")
        else:
            logger.warning(f"Artifact not found: {file_path}")
    
    def log_dvc_data_version(
        self,
        data_path: Path,
        data_type: str = "dataset"
    ):
        """
        Log DVC data versioning information to MLflow.
        
        This function extracts DVC version information for tracked data files
        and registers it in MLflow for reproducibility. It also logs S3 location
        if DVC is configured with S3 remote storage.
        
        Args:
            data_path: Path to the data file/directory tracked by DVC
            data_type: Type of data (e.g., 'dataset', 'model', 'features')
            
        Returns:
            Dictionary with DVC version info (hash, git_commit, s3_location, size) or None
        """
        try:
            # Convert to relative path from project root
            data_path = Path(data_path)
            repo_root = Path.cwd()
            
            # Try to make it relative to DVC repo root
            if data_path.is_absolute():
                try:
                    data_path = data_path.relative_to(repo_root)
                except ValueError:
                    # Path is outside repo, try to find .dvc file in repo
                    pass
            
            # Check if there's a .dvc file for this path
            dvc_file = repo_root / f"{data_path}.dvc"
            if not dvc_file.exists() and data_path.is_dir():
                # Try with directory name
                dvc_file = repo_root / data_path / ".dvc"
            if not dvc_file.exists():
                logger.debug(f"No .dvc file found for {data_path}")
                return None
            
            # Read DVC file to get hash
            dvc_info = {}
            if dvc_file.exists():
                try:
                    import yaml
                    with open(dvc_file, 'r') as f:
                        dvc_content = yaml.safe_load(f)
                    
                    if 'outs' in dvc_content and len(dvc_content['outs']) > 0:
                        out = dvc_content['outs'][0]
                        dvc_info = {
                            'hash': out.get('md5', out.get('hash', 'unknown')),
                            'size': out.get('size', 0),
                            'nfiles': out.get('nfiles', 1)
                        }
                except Exception as e:
                    logger.warning(f"Error reading DVC file {dvc_file}: {e}")
                    return None
            
            # Get Git commit (if available)
            git_commit = None
            try:
                git_commit = subprocess.check_output(
                    ['git', 'rev-parse', 'HEAD'],
                    stderr=subprocess.DEVNULL,
                    cwd=repo_root
                ).decode().strip()
            except Exception:
                pass
            
            # Get DVC remote info (S3 if configured)
            s3_location = None
            try:
                # Get default remote
                try:
                    default_output = subprocess.check_output(
                        ['dvc', 'remote', 'default'],
                        stderr=subprocess.DEVNULL,
                        cwd=repo_root
                    ).decode().strip()
                    default_remote = default_output
                except Exception:
                    # Try to get from config
                    config_file = repo_root / ".dvc" / "config"
                    if config_file.exists():
                        with open(config_file, 'r') as f:
                            config_content = f.read()
                            # Try to extract default remote
                            if 'remote =' in config_content:
                                for line in config_content.split('\n'):
                                    if 'remote =' in line:
                                        default_remote = line.split('=')[-1].strip()
                                        break
                                else:
                                    default_remote = None
                            else:
                                default_remote = None
                    else:
                        default_remote = None
                
                # Get remote URL
                if default_remote:
                    try:
                        remote_url_output = subprocess.check_output(
                            ['dvc', 'remote', 'list', '-d', default_remote],
                            stderr=subprocess.DEVNULL,
                            cwd=repo_root
                        ).decode()
                        
                        # Try alternative method to get URL
                        config_file = repo_root / ".dvc" / "config"
                        if config_file.exists():
                            with open(config_file, 'r') as f:
                                config_lines = f.readlines()
                                in_remote_section = False
                                for line in config_lines:
                                    if f'["remote "{default_remote}"]' in line or f'["remote \'{default_remote}\']' in line:
                                        in_remote_section = True
                                    elif line.strip().startswith('[') and in_remote_section:
                                        break
                                    elif in_remote_section and 'url' in line:
                                        s3_base = line.split('=')[-1].strip()
                                        if s3_base.startswith('s3://'):
                                            # Construct S3 location with hash
                                            hash_val = dvc_info.get('hash', '')
                                            if hash_val and hash_val != 'unknown':
                                                hash_prefix = hash_val[:2]
                                                s3_location = f"{s3_base}/{hash_prefix}/{hash_val}"
                                            break
                    except Exception as e:
                        logger.debug(f"Could not get remote URL: {e}")
                        
            except Exception as e:
                logger.debug(f"Could not get S3 location: {e}")
            
            # Log to MLflow
            hash_val = dvc_info.get('hash', 'unknown')
            mlflow.log_param(f"{data_type}_dvc_hash", hash_val)
            mlflow.log_param(f"{data_type}_path", str(data_path))
            mlflow.log_param(f"{data_type}_size_bytes", dvc_info.get('size', 0))
            mlflow.log_param(f"{data_type}_nfiles", dvc_info.get('nfiles', 1))
            
            if git_commit:
                mlflow.log_param(f"{data_type}_git_commit", git_commit[:7])  # Short commit hash
            
            if s3_location:
                mlflow.log_param(f"{data_type}_s3_location", s3_location)
                logger.info(f"Registered {data_type} DVC version with S3 location: {s3_location}")
            else:
                logger.info(f"Registered {data_type} DVC version: {hash_val[:8] if hash_val != 'unknown' else 'unknown'}")
            
            return {
                'hash': hash_val,
                'git_commit': git_commit,
                's3_location': s3_location,
                'size': dvc_info.get('size', 0)
            }
            
        except Exception as e:
            logger.warning(f"Could not log DVC version for {data_path}: {e}")
            return None
    
    def pull_data_from_s3(self) -> bool:
        """
        Pull all DVC-tracked data from S3 remote.
        
        Returns:
            True if pull was successful, False otherwise
        """
        if not DVC_AVAILABLE or self._dvc_manager is None:
            logger.error("DVC not available. Cannot pull from S3.")
            return False
        
        return self._dvc_manager.pull_from_s3()
    
    def push_data_to_s3(self) -> bool:
        """
        Push all DVC-tracked data to S3 remote.
        
        Returns:
            True if push was successful, False otherwise
        """
        if not DVC_AVAILABLE or self._dvc_manager is None:
            logger.error("DVC not available. Cannot push to S3.")
            return False
        
        return self._dvc_manager.push_to_s3()
    
    def log_artifacts(self, directory: Path, artifact_path: Optional[str] = None):
        """
        Log all files in a directory as artifacts.
        
        Args:
            directory: Directory containing files to log
            artifact_path: Path within the artifact store
        """
        if directory.exists():
            mlflow.log_artifacts(str(directory), artifact_path=artifact_path)
            logger.debug(f"Logged artifacts from: {directory}")
        else:
            logger.warning(f"Directory not found: {directory}")
    
    def search_runs(
        self,
        filter_string: Optional[str] = None,
        max_results: int = 100,
        order_by: Optional[List[str]] = None
    ) -> List[mlflow.entities.Run]:
        """
        Search for runs in the current experiment.
        
        Args:
            filter_string: Filter string (e.g., "metrics.accuracy > 0.8")
            max_results: Maximum number of results
            order_by: List of columns to order by (e.g., ["metrics.accuracy DESC"])
            
        Returns:
            List of Run objects
        """
        experiment = mlflow.get_experiment_by_name(self.experiment_name)
        if experiment is None:
            return []
        
        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string=filter_string,
            max_results=max_results,
            order_by=order_by or []
        )
        
        return runs
    
    def get_best_run(self, metric: str = "test_accuracy", ascending: bool = False) -> Optional[Dict]:
        """
        Get the best run based on a metric.
        
        Args:
            metric: Metric name to use for comparison
            ascending: If True, lower is better; if False, higher is better
            
        Returns:
            Dictionary with run information or None
        """
        runs = self.search_runs(order_by=[f"metrics.{metric} {'ASC' if ascending else 'DESC'}"])
        
        if runs.empty:
            return None
        
        best_run = runs.iloc[0]
        return {
            "run_id": best_run["run_id"],
            "metric_value": best_run[f"metrics.{metric}"],
            "metric_name": metric
        }
    
    def compare_runs(self, run_ids: List[str]) -> pd.DataFrame:
        """
        Compare multiple runs.
        
        Args:
            run_ids: List of run IDs to compare
            
        Returns:
            DataFrame with comparison
        """
        experiment = mlflow.get_experiment_by_name(self.experiment_name)
        if experiment is None:
            return pd.DataFrame()
        
        run_ids_str = ','.join([f"'{rid}'" for rid in run_ids])
        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string=f"run_id IN ({run_ids_str})"
        )
        
        return runs


def track_training_experiment(
    model: Any,
    model_name: str,
    model_type: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    params: Dict[str, Any],
    class_names: Optional[List[str]] = None,
    transformer: Optional[Any] = None,
    mlflow_manager: Optional[MLflowManager] = None,
    registered_model_name: Optional[str] = None,
    tags: Optional[Dict[str, str]] = None
) -> Dict[str, Any]:
    """
    Complete function to track a training experiment with MLflow.
    
    Args:
        model: Trained model
        model_name: Name for the model
        model_type: Type of model ('sklearn' or 'lightgbm')
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        params: Model parameters
        class_names: List of class names
        transformer: Feature transformer (optional)
        mlflow_manager: MLflowManager instance (creates new if None)
        registered_model_name: Name for model registry
        tags: Additional tags for the run
        
    Returns:
        Dictionary with run information and metrics
    """
    if mlflow_manager is None:
        mlflow_manager = MLflowManager()
    
    run_tags = tags or {}
    run_tags["model_type"] = model_type
    run_tags["model_name"] = model_name
    
    # Start run
    mlflow_manager.start_run(run_name=f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}", tags=run_tags)
    
    try:
        # Log parameters
        mlflow_manager.log_params(params)
        
        # Log dataset info
        mlflow_manager.log_params({
            "n_train_samples": len(X_train),
            "n_test_samples": len(X_test),
            "n_features": X_train.shape[1],
            "n_classes": len(np.unique(y_train))
        })
        
        # Log DVC data versions if available
        # This tracks which version of data was used for this experiment
        from mlops.config import RAW_DATA_DIR, PROCESSED_DATA_DIR
        
        # Version del dataset raw
        if (RAW_DATA_DIR / "raw.dvc").exists() or Path("data/raw.dvc").exists():
            mlflow_manager.log_dvc_data_version(
                RAW_DATA_DIR / "raw" if (RAW_DATA_DIR / "raw").exists() else Path("data/raw"),
                data_type="raw_data"
            )
        
        # Version del transformer si está trackeado
        transformer_dvc_path = PROCESSED_DATA_DIR / "transformer.pkl.dvc"
        if transformer_dvc_path.exists():
            mlflow_manager.log_dvc_data_version(
                PROCESSED_DATA_DIR / "transformer.pkl",
                data_type="transformer"
            )
        
        # Get predictions
        if hasattr(model, 'predict'):
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
        else:
            # For LightGBM booster
            y_train_pred_proba = model.predict(X_train)
            y_train_pred = y_train_pred_proba.argmax(axis=1)
            y_test_pred_proba = model.predict(X_test)
            y_test_pred = y_test_pred_proba.argmax(axis=1)
        
        # Log metrics
        train_metrics = mlflow_manager.log_evaluation_metrics(
            y_train, y_train_pred, prefix="train", class_names=class_names
        )
        
        test_metrics = mlflow_manager.log_evaluation_metrics(
            y_test, y_test_pred, prefix="test", class_names=class_names
        )
        
        # Log confusion matrices
        mlflow_manager.log_confusion_matrix(
            y_test, y_test_pred, class_names=class_names, artifact_name="confusion_matrix_test"
        )
        
        mlflow_manager.log_confusion_matrix(
            y_train, y_train_pred, class_names=class_names, artifact_name="confusion_matrix_train"
        )
        
        # Log classification reports
        mlflow_manager.log_classification_report(
            y_test, y_test_pred, class_names=class_names, artifact_name="classification_report_test"
        )
        
        mlflow_manager.log_classification_report(
            y_train, y_train_pred, class_names=class_names, artifact_name="classification_report_train"
        )
        
        # Log model
        input_example = X_train[:5] if len(X_train) > 5 else X_train
        model_info = mlflow_manager.log_model(
            model=model,
            model_name=model_name,
            input_example=input_example,
            registered_model_name=registered_model_name,
            model_type=model_type
        )
        
        # Log transformer if provided
        if transformer is not None:
            import joblib
            transformer_path = mlflow_manager.artifact_path / "transformer.pkl"
            joblib.dump(transformer, transformer_path)
            mlflow_manager.log_artifact(transformer_path, artifact_path="transformers")
        
        run_info = {
            "run_id": mlflow_manager.current_run.info.run_id,
            "experiment_id": mlflow_manager.current_run.info.experiment_id,
            "model_uri": model_info.model_uri,
            "train_metrics": train_metrics,
            "test_metrics": test_metrics
        }
        
        logger.success(f"Experiment tracked successfully: {run_info['run_id']}")
        
        return run_info
        
    finally:
        mlflow_manager.end_run()


if __name__ == "__main__":
    """
    Example usage of MLflow integration.
    
    This is a standalone example that can be run directly:
        python mlops/MLFLow_Equipo19.py
    
    For production use with your actual pipeline, use:
        python src/train_with_mlflow.py
    """
    import sys
    from pathlib import Path
    
    # Add project root to path to ensure imports work
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.datasets import load_iris
    
    # Example with Iris dataset
    print("=" * 60)
    print("Running MLflow example with Iris dataset...")
    print("=" * 60)
    
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize MLflow Manager
    mlflow_manager = MLflowManager(experiment_name="MLFlow_MusicEmotionsExperiment")
    
    # Train model
    params = {
        "n_estimators": 100,
        "max_depth": 5,
        "random_state": 42
    }
    
    print("\nTraining Random Forest model...")
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)
    
    # Track experiment
    print("\nTracking experiment with MLflow...")
    run_info = track_training_experiment(
        model=model,
        model_name="example_random_forest",
        model_type="sklearn",
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        params=params,
        class_names=["setosa", "versicolor", "virginica"],
        mlflow_manager=mlflow_manager,
        registered_model_name="example-MusicEmotions-Model"
    )
    
    print("\n" + "=" * 60)
    print("Experiment tracked successfully!")
    print("=" * 60)
    print(f"Run ID: {run_info['run_id']}")
    print(f"Model URI: {run_info['model_uri']}")
    print(f"Test Accuracy: {run_info['test_metrics']['test_accuracy']:.4f}")
    print("\nTo view results, run:")
    print("  mlflow ui")
    print("\nThen open http://localhost:5000 in your browser")
    print("=" * 60)
