"""
Basic tests for the test suite itself.
"""
import pytest


def test_imports():
    """Test that all main classes can be imported."""
    # Test imports that don't require optional dependencies
    from src.data_processor import DataProcessor
    from src.feature_transformer import FeatureTransformer
    from src.plotter import Plotter
    
    assert DataProcessor is not None
    assert FeatureTransformer is not None
    assert Plotter is not None
    
    # Test ModelTrainer import (may require lightgbm)
    try:
        from src.model_trainer import ModelTrainer
        assert ModelTrainer is not None
    except ImportError:
        pytest.skip("ModelTrainer requires lightgbm which is not installed")


def test_test_suite_works():
    """Basic test to verify test suite is working."""
    assert True
