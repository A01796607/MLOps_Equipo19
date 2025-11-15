"""
Unit tests for MLflowManager class.
"""
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch, mock_open
import tempfile
import shutil

from mlops.mlflow import MLflowManager


class TestMLflowManager:
    """Test suite for MLflowManager class."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test artifacts."""
        temp_path = Path(tempfile.mkdtemp())
        yield temp_path
        shutil.rmtree(temp_path)
    
    @pytest.fixture
    def mlflow_manager(self, temp_dir):
        """Create MLflowManager instance with mocked MLflow."""
        with patch('mlops.mlflow.mlflow'):
            manager = MLflowManager(
                experiment_name="test_experiment",
                artifact_path=temp_dir / "artifacts"
            )
            return manager
    
    def test_init_default(self):
        """Test MLflowManager initialization with default parameters."""
        with patch('mlops.mlflow.mlflow'):
            manager = MLflowManager()
            assert manager.experiment_name == "MusicEmotionsExperiment"
            assert manager.artifact_path.exists()
            assert manager.current_run is None
    
    def test_init_custom_params(self, temp_dir):
        """Test MLflowManager initialization with custom parameters."""
        with patch('mlops.mlflow.mlflow'):
            manager = MLflowManager(
                experiment_name="custom_experiment",
                tracking_uri="file:///custom/path",
                artifact_path=temp_dir / "custom_artifacts"
            )
            assert manager.experiment_name == "custom_experiment"
            assert manager.artifact_path == temp_dir / "custom_artifacts"
            assert manager.artifact_path.exists()
    
    def test_start_run_default(self, mlflow_manager):
        """Test starting a run with default name."""
        with patch('mlops.mlflow.mlflow') as mock_mlflow:
            mock_run = MagicMock()
            mock_run.info.run_id = "test_run_id"
            mock_mlflow.active_run.return_value = None
            mock_mlflow.start_run.return_value = mock_run
            
            result = mlflow_manager.start_run()
            
            assert result == mock_run
            assert mlflow_manager.current_run == mock_run
            mock_mlflow.start_run.assert_called_once()
    
    def test_start_run_with_name_and_tags(self, mlflow_manager):
        """Test starting a run with custom name and tags."""
        with patch('mlops.mlflow.mlflow') as mock_mlflow:
            mock_run = MagicMock()
            mock_run.info.run_id = "test_run_id"
            mock_mlflow.active_run.return_value = None
            mock_mlflow.start_run.return_value = mock_run
            
            tags = {"model_type": "random_forest", "dataset": "music_emotions"}
            result = mlflow_manager.start_run(run_name="test_run", tags=tags)
            
            assert result == mock_run
            call_args = mock_mlflow.start_run.call_args
            assert "test_run" in str(call_args) or call_args[1]["run_name"] == "test_run"
    
    def test_start_run_ends_existing_run(self, mlflow_manager):
        """Test that starting a new run ends any existing run."""
        with patch('mlops.mlflow.mlflow') as mock_mlflow:
            mock_old_run = MagicMock()
            mock_new_run = MagicMock()
            mock_mlflow.active_run.return_value = mock_old_run
            mock_mlflow.start_run.return_value = mock_new_run
            
            mlflow_manager.start_run()
            
            mock_mlflow.end_run.assert_called_once()
    
    def test_end_run(self, mlflow_manager):
        """Test ending a run."""
        with patch('mlops.mlflow.mlflow') as mock_mlflow:
            mock_run = MagicMock()
            mock_run.info.run_id = "test_run_id"
            mlflow_manager.current_run = mock_run
            
            mlflow_manager.end_run()
            
            assert mlflow_manager.current_run is None
            mock_mlflow.end_run.assert_called_once()
    
    def test_end_run_no_active_run(self, mlflow_manager):
        """Test ending a run when no run is active."""
        mlflow_manager.current_run = None
        # Should not raise an error
        mlflow_manager.end_run()
    
    def test_log_params(self, mlflow_manager):
        """Test logging parameters."""
        with patch('mlops.mlflow.mlflow') as mock_mlflow:
            params = {
                "n_estimators": 100,
                "max_depth": 10,
                "learning_rate": 0.01,
                "none_param": None
            }
            
            mlflow_manager.log_params(params)
            
            # Verify params were logged (None should be filtered out)
            mock_mlflow.log_params.assert_called_once()
            logged_params = mock_mlflow.log_params.call_args[0][0]
            assert "none_param" not in logged_params
            assert logged_params["n_estimators"] == "100"
    
    def test_log_metrics_no_active_run(self, mlflow_manager):
        """Test that log_metrics raises error when no active run."""
        with patch('mlops.mlflow.mlflow') as mock_mlflow:
            mock_mlflow.active_run.return_value = None
            
            metrics = {"accuracy": 0.95, "f1_score": 0.87}
            
            with pytest.raises(RuntimeError, match="No active MLflow run"):
                mlflow_manager.log_metrics(metrics)
    
    def test_log_metrics_with_active_run(self, mlflow_manager):
        """Test logging metrics with active run."""
        with patch('mlops.mlflow.mlflow') as mock_mlflow:
            mock_run = MagicMock()
            mock_mlflow.active_run.return_value = mock_run
            
            metrics = {
                "accuracy": 0.95,
                "f1_score": 0.87,
                "precision": 0.92,
                "nan_metric": np.nan,
                "inf_metric": np.inf
            }
            
            mlflow_manager.log_metrics(metrics)
            
            # Verify metrics were logged (NaN and Inf should be filtered out)
            mock_mlflow.log_metrics.assert_called_once()
            logged_metrics = mock_mlflow.log_metrics.call_args[0][0]
            assert "nan_metric" not in logged_metrics
            assert "inf_metric" not in logged_metrics
            assert "accuracy" in logged_metrics
            assert logged_metrics["accuracy"] == 0.95
    
    def test_log_metrics_with_step(self, mlflow_manager):
        """Test logging metrics with step number."""
        with patch('mlops.mlflow.mlflow') as mock_mlflow:
            mock_run = MagicMock()
            mock_mlflow.active_run.return_value = mock_run
            
            metrics = {"loss": 0.5}
            mlflow_manager.log_metrics(metrics, step=10)
            
            # When step is provided, log_metric should be called for each metric
            assert mock_mlflow.log_metric.called
    
    def test_log_evaluation_metrics(self, mlflow_manager):
        """Test logging evaluation metrics."""
        with patch('mlops.mlflow.mlflow') as mock_mlflow:
            mock_run = MagicMock()
            mock_mlflow.active_run.return_value = mock_run
            
            y_true = np.array([0, 1, 2, 0, 1])
            y_pred = np.array([0, 1, 2, 0, 1])
            
            metrics = mlflow_manager.log_evaluation_metrics(
                y_true, y_pred, prefix="test"
            )
            
            # Verify metrics were calculated and logged
            assert "test_accuracy" in metrics
            assert "test_f1_macro" in metrics
            assert "test_precision_macro" in metrics
            assert "test_recall_macro" in metrics
            assert metrics["test_accuracy"] == 1.0  # Perfect predictions
    
    def test_log_evaluation_metrics_with_class_names(self, mlflow_manager):
        """Test logging evaluation metrics with class names."""
        with patch('mlops.mlflow.mlflow') as mock_mlflow:
            mock_run = MagicMock()
            mock_mlflow.active_run.return_value = mock_run
            
            y_true = np.array([0, 1, 0, 1])
            y_pred = np.array([0, 1, 0, 1])
            class_names = ["angry", "happy"]
            
            metrics = mlflow_manager.log_evaluation_metrics(
                y_true, y_pred, prefix="test", class_names=class_names
            )
            
            # Verify per-class metrics were logged
            assert "test_precision_angry" in metrics
            assert "test_recall_angry" in metrics
            assert "test_f1_angry" in metrics
    
    def test_log_evaluation_metrics_with_probabilities(self, mlflow_manager):
        """Test logging evaluation metrics when y_pred is probabilities."""
        with patch('mlops.mlflow.mlflow') as mock_mlflow:
            mock_run = MagicMock()
            mock_mlflow.active_run.return_value = mock_run
            
            y_true = np.array([0, 1, 2])
            # y_pred as probabilities (2D array)
            y_pred = np.array([[0.9, 0.1, 0.0],
                              [0.1, 0.8, 0.1],
                              [0.0, 0.2, 0.8]])
            
            metrics = mlflow_manager.log_evaluation_metrics(
                y_true, y_pred, prefix="test"
            )
            
            # Should convert probabilities to predictions
            assert "test_accuracy" in metrics
    
    def test_log_evaluation_metrics_different_lengths(self, mlflow_manager):
        """Test that log_evaluation_metrics raises error for different lengths."""
        y_true = np.array([0, 1, 2])
        y_pred = np.array([0, 1])  # Different length
        
        with pytest.raises(ValueError, match="different lengths"):
            mlflow_manager.log_evaluation_metrics(y_true, y_pred)
    
    def test_log_model_sklearn(self, mlflow_manager):
        """Test logging a sklearn model."""
        with patch('mlops.mlflow.mlflow') as mock_mlflow:
            mock_model = MagicMock()
            mock_model_info = MagicMock()
            mock_model_info.model_uri = "models:/test_model/1"
            mock_mlflow.sklearn.log_model.return_value = mock_model_info
            
            model_info = mlflow_manager.log_model(
                mock_model,
                model_name="test_model",
                model_type="sklearn"
            )
            
            assert model_info == mock_model_info
            mock_mlflow.sklearn.log_model.assert_called_once()
    
    def test_log_model_lightgbm(self, mlflow_manager):
        """Test logging a LightGBM model."""
        with patch('mlops.mlflow.mlflow') as mock_mlflow:
            mock_model = MagicMock()
            mock_model_info = MagicMock()
            mock_model_info.model_uri = "models:/test_model/1"
            mock_mlflow.lightgbm.log_model.return_value = mock_model_info
            
            model_info = mlflow_manager.log_model(
                mock_model,
                model_name="test_model",
                model_type="lightgbm"
            )
            
            assert model_info == mock_model_info
            mock_mlflow.lightgbm.log_model.assert_called_once()
    
    def test_log_model_unsupported_type(self, mlflow_manager):
        """Test that logging unsupported model type raises error."""
        mock_model = MagicMock()
        
        with pytest.raises(ValueError, match="Unsupported model_type"):
            mlflow_manager.log_model(
                mock_model,
                model_name="test_model",
                model_type="unsupported"
            )
    
    def test_log_confusion_matrix(self, mlflow_manager):
        """Test logging confusion matrix."""
        with patch('mlops.mlflow.mlflow') as mock_mlflow:
            y_true = np.array([0, 1, 2, 0, 1])
            y_pred = np.array([0, 1, 2, 0, 1])
            
            mlflow_manager.log_confusion_matrix(
                y_true, y_pred, class_names=["a", "b", "c"]
            )
            
            # Verify artifact was logged
            mock_mlflow.log_artifact.assert_called()
    
    def test_log_classification_report(self, mlflow_manager):
        """Test logging classification report."""
        with patch('mlops.mlflow.mlflow') as mock_mlflow:
            y_true = np.array([0, 1, 2, 0, 1])
            y_pred = np.array([0, 1, 2, 0, 1])
            
            mlflow_manager.log_classification_report(
                y_true, y_pred, class_names=["a", "b", "c"]
            )
            
            # Verify artifact was logged
            mock_mlflow.log_artifact.assert_called()
    
    def test_log_artifact(self, mlflow_manager, temp_dir):
        """Test logging a single artifact."""
        with patch('mlops.mlflow.mlflow') as mock_mlflow:
            # Create a test file
            test_file = temp_dir / "test.txt"
            test_file.write_text("test content")
            
            mlflow_manager.log_artifact(test_file, artifact_path="test_artifacts")
            
            mock_mlflow.log_artifact.assert_called_once()
            # Verify file path was passed
            call_args = mock_mlflow.log_artifact.call_args
            assert str(test_file) in str(call_args) or test_file in call_args[0]
    
    def test_log_artifacts(self, mlflow_manager, temp_dir):
        """Test logging multiple artifacts from directory."""
        with patch('mlops.mlflow.mlflow') as mock_mlflow:
            # Create test directory with files
            test_dir = temp_dir / "test_dir"
            test_dir.mkdir()
            (test_dir / "file1.txt").write_text("content1")
            (test_dir / "file2.txt").write_text("content2")
            
            mlflow_manager.log_artifacts(test_dir, artifact_path="test_artifacts")
            
            mock_mlflow.log_artifacts.assert_called_once()
    
    def test_log_dvc_data_version_no_dvc_file(self, mlflow_manager):
        """Test log_dvc_data_version when .dvc file doesn't exist."""
        with patch('mlops.mlflow.mlflow'):
            data_path = Path("nonexistent_data.csv")
            
            result = mlflow_manager.log_dvc_data_version(data_path)
            
            assert result is None
    
    def test_search_runs(self, mlflow_manager):
        """Test searching runs."""
        with patch('mlops.mlflow.mlflow') as mock_mlflow:
            # mlflow.search_runs returns a DataFrame
            mock_runs_df = pd.DataFrame({
                'run_id': ['run1', 'run2'],
                'metrics.accuracy': [0.95, 0.98]
            })
            mock_mlflow.get_experiment_by_name.return_value = MagicMock(experiment_id="exp1")
            mock_mlflow.search_runs.return_value = mock_runs_df
            
            runs = mlflow_manager.search_runs(max_results=10)
            
            assert isinstance(runs, pd.DataFrame)
            assert len(runs) == 2
            assert 'run1' in runs['run_id'].values
            assert 'run2' in runs['run_id'].values
    
    def test_get_best_run(self, mlflow_manager):
        """Test getting best run by metric."""
        with patch('mlops.mlflow.mlflow') as mock_mlflow:
            # mlflow.search_runs returns a DataFrame ordered by metric DESC
            mock_runs_df = pd.DataFrame({
                'run_id': ['run2', 'run1'],
                'metrics.accuracy': [0.98, 0.95]
            })
            mock_mlflow.get_experiment_by_name.return_value = MagicMock(experiment_id="exp1")
            mock_mlflow.search_runs.return_value = mock_runs_df
            
            best_run = mlflow_manager.get_best_run(metric="accuracy")
            
            assert best_run is not None
            assert best_run["run_id"] == "run2"  # Higher accuracy
            assert best_run["metric_value"] == 0.98
    
    def test_get_best_run_ascending(self, mlflow_manager):
        """Test getting best run with ascending order (lowest is best)."""
        with patch('mlops.mlflow.mlflow') as mock_mlflow:
            # mlflow.search_runs returns a DataFrame ordered by metric ASC
            mock_runs_df = pd.DataFrame({
                'run_id': ['run2', 'run1'],
                'metrics.loss': [0.3, 0.5]
            })
            mock_mlflow.get_experiment_by_name.return_value = MagicMock(experiment_id="exp1")
            mock_mlflow.search_runs.return_value = mock_runs_df
            
            best_run = mlflow_manager.get_best_run(metric="loss", ascending=True)
            
            assert best_run is not None
            assert best_run["run_id"] == "run2"  # Lower loss
            assert best_run["metric_value"] == 0.3
    
    def test_compare_runs(self, mlflow_manager):
        """Test comparing multiple runs."""
        with patch('mlops.mlflow.mlflow') as mock_mlflow:
            # mlflow.search_runs returns a DataFrame with run data
            mock_runs_df = pd.DataFrame({
                'run_id': ['run1', 'run2'],
                'metrics.accuracy': [0.95, 0.98],
                'metrics.f1': [0.87, 0.89],
                'params.n_estimators': ['100', '200']
            })
            mock_mlflow.get_experiment_by_name.return_value = MagicMock(experiment_id="exp1")
            mock_mlflow.search_runs.return_value = mock_runs_df
            
            comparison = mlflow_manager.compare_runs(["run1", "run2"])
            
            assert isinstance(comparison, pd.DataFrame)
            assert len(comparison) == 2
            assert "run_id" in comparison.columns
            assert "metrics.accuracy" in comparison.columns

