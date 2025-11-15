"""
Unit tests for ModelTrainer class.
"""
import pytest
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from src.model_trainer import ModelTrainer


class TestModelTrainer:
    """Test suite for ModelTrainer class."""
    
    def test_init_default(self):
        """Test ModelTrainer initialization with default parameters."""
        trainer = ModelTrainer()
        assert trainer.random_state == 42
        assert trainer.models == {}
        assert trainer.predictions == {}
    
    def test_init_custom_random_state(self):
        """Test ModelTrainer initialization with custom random state."""
        trainer = ModelTrainer(random_state=123)
        assert trainer.random_state == 123
    
    def test_train_random_forest(self):
        """Test Random Forest training."""
        trainer = ModelTrainer(random_state=42)
        
        # Create simple synthetic data
        np.random.seed(42)
        X_train = np.random.randn(100, 5)
        y_train = np.random.randint(0, 3, 100)
        
        model = trainer.train_random_forest(X_train, y_train)
        
        # Should return a RandomForestClassifier
        assert isinstance(model, RandomForestClassifier)
        
        # Should be stored in models dictionary
        assert 'random_forest' in trainer.models
        assert trainer.models['random_forest'] == model
    
    def test_train_random_forest_custom_estimators(self):
        """Test Random Forest training with custom number of estimators."""
        trainer = ModelTrainer(random_state=42)
        
        np.random.seed(42)
        X_train = np.random.randn(100, 5)
        y_train = np.random.randint(0, 3, 100)
        
        model = trainer.train_random_forest(X_train, y_train, n_estimators=100)
        
        assert isinstance(model, RandomForestClassifier)
        assert model.n_estimators == 100
    
    def test_predict_random_forest(self):
        """Test prediction with Random Forest."""
        trainer = ModelTrainer(random_state=42)
        
        np.random.seed(42)
        X_train = np.random.randn(100, 5)
        y_train = np.random.randint(0, 3, 100)
        X_test = np.random.randn(20, 5)
        
        # Train model first
        trainer.train_random_forest(X_train, y_train)
        
        # Make predictions
        y_pred = trainer.predict('random_forest', X_test)
        
        # Should return numpy array
        assert isinstance(y_pred, np.ndarray)
        
        # Should have correct shape
        assert len(y_pred) == len(X_test)
        
        # Should be stored in predictions dictionary
        assert 'random_forest' in trainer.predictions
        
        # Predictions should be integers (class labels)
        assert all(isinstance(pred, (np.integer, int)) for pred in y_pred)
    
    def test_predict_nonexistent_model(self):
        """Test that predict raises error for nonexistent model."""
        trainer = ModelTrainer()
        
        X_test = np.random.randn(10, 5)
        
        with pytest.raises(ValueError, match="not found"):
            trainer.predict('nonexistent_model', X_test)
    
    def test_evaluate(self):
        """Test model evaluation."""
        trainer = ModelTrainer()
        
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 1, 2, 0, 1, 2])
        
        results = trainer.evaluate(y_true, y_pred, class_names=['a', 'b', 'c'])
        
        # Should return dictionary
        assert isinstance(results, dict)
        
        # Should have expected keys
        assert 'classification_report' in results
        assert 'confusion_matrix' in results
        
        # Classification report should be a dictionary
        assert isinstance(results['classification_report'], dict)
        
        # Confusion matrix should be numpy array
        assert isinstance(results['confusion_matrix'], np.ndarray)
    
    def test_evaluate_with_perfect_predictions(self):
        """Test evaluation with perfect predictions."""
        trainer = ModelTrainer()
        
        y_true = np.array([0, 0, 1, 1, 2, 2])
        y_pred = np.array([0, 0, 1, 1, 2, 2])
        
        results = trainer.evaluate(y_true, y_pred)
        
        # Should have perfect accuracy
        accuracy = accuracy_score(y_true, y_pred)
        assert accuracy == 1.0
        
        # Confusion matrix should be diagonal
        cm = results['confusion_matrix']
        assert cm.shape == (3, 3)
        assert np.trace(cm) == len(y_true)  # All correct
    
    def test_evaluate_with_imperfect_predictions(self):
        """Test evaluation with imperfect predictions."""
        trainer = ModelTrainer()
        
        y_true = np.array([0, 0, 1, 1, 2, 2])
        y_pred = np.array([0, 1, 1, 2, 2, 0])  # Some errors
        
        results = trainer.evaluate(y_true, y_pred)
        
        # Should have less than perfect accuracy
        accuracy = accuracy_score(y_true, y_pred)
        assert accuracy < 1.0
        
        # Confusion matrix should not be diagonal
        cm = results['confusion_matrix']
        assert np.trace(cm) < len(y_true)
    
    def test_print_report(self, capsys):
        """Test print_report method."""
        trainer = ModelTrainer()
        
        y_true = np.array([0, 1, 2, 0, 1])
        y_pred = np.array([0, 1, 2, 0, 1])
        
        trainer.print_report(y_true, y_pred, class_names=['a', 'b', 'c'], model_name="Test")
        
        # Capture printed output
        captured = capsys.readouterr()
        
        # Should have printed something
        assert len(captured.out) > 0
        assert "Test" in captured.out
        assert "Classification Report" in captured.out
        assert "Confusion Matrix" in captured.out
    
    def test_train_lightgbm_basic(self):
        """Test basic LightGBM training."""
        try:
            import lightgbm as lgb
        except ImportError:
            pytest.skip("LightGBM not installed")
        
        trainer = ModelTrainer(random_state=42)
        
        np.random.seed(42)
        X_train = np.random.randn(100, 5)
        y_train = np.random.randint(0, 3, 100)
        
        model = trainer.train_lightgbm(X_train, y_train, n_estimators=10)
        
        # Should return a Booster
        assert isinstance(model, lgb.Booster)
        
        # Should be stored in models dictionary
        assert 'lightgbm' in trainer.models
    
    def test_train_lightgbm_with_validation(self):
        """Test LightGBM training with validation set."""
        try:
            import lightgbm as lgb
        except ImportError:
            pytest.skip("LightGBM not installed")
        
        trainer = ModelTrainer(random_state=42)
        
        np.random.seed(42)
        X_train = np.random.randn(100, 5)
        y_train = np.random.randint(0, 3, 100)
        X_val = np.random.randn(20, 5)
        y_val = np.random.randint(0, 3, 20)
        
        model = trainer.train_lightgbm(
            X_train, y_train, X_val, y_val, n_estimators=10, early_stopping_rounds=5
        )
        
        # Should return a Booster
        assert isinstance(model, lgb.Booster)
    
    def test_predict_lightgbm(self):
        """Test prediction with LightGBM."""
        try:
            import lightgbm as lgb
        except ImportError:
            pytest.skip("LightGBM not installed")
        
        trainer = ModelTrainer(random_state=42)
        
        np.random.seed(42)
        X_train = np.random.randn(100, 5)
        y_train = np.random.randint(0, 3, 100)
        X_test = np.random.randn(20, 5)
        
        # Train model first
        trainer.train_lightgbm(X_train, y_train, n_estimators=10)
        
        # Make predictions
        y_pred = trainer.predict('lightgbm', X_test)
        
        # Should return numpy array
        assert isinstance(y_pred, np.ndarray)
        
        # Should have correct shape
        assert len(y_pred) == len(X_test)
        
        # Should be stored in predictions dictionary
        assert 'lightgbm' in trainer.predictions
    
    def test_models_storage(self):
        """Test that models are properly stored."""
        trainer = ModelTrainer(random_state=42)
        
        np.random.seed(42)
        X_train = np.random.randn(50, 5)
        y_train = np.random.randint(0, 3, 50)
        
        # Train multiple models
        trainer.train_random_forest(X_train, y_train)
        
        # Check that model is stored
        assert len(trainer.models) == 1
        assert 'random_forest' in trainer.models
    
    def test_predictions_storage(self):
        """Test that predictions are properly stored."""
        trainer = ModelTrainer(random_state=42)
        
        np.random.seed(42)
        X_train = np.random.randn(50, 5)
        y_train = np.random.randint(0, 3, 50)
        X_test = np.random.randn(10, 5)
        
        # Train and predict
        trainer.train_random_forest(X_train, y_train)
        trainer.predict('random_forest', X_test)
        
        # Check that predictions are stored
        assert len(trainer.predictions) == 1
        assert 'random_forest' in trainer.predictions
        assert len(trainer.predictions['random_forest']) == len(X_test)

