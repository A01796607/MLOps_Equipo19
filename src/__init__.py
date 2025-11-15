"""
Source classes for the MLOps pipeline.
"""
from src.data_processor import DataProcessor
from src.feature_transformer import FeatureTransformer

# Lazy import for ModelTrainer to avoid import errors if lightgbm is not installed
try:
    from src.model_trainer import ModelTrainer
except ImportError:
    ModelTrainer = None  # type: ignore

# Lazy import for API to avoid import errors if fastapi is not installed
try:
    from src.api.main import app
except ImportError:
    app = None  # type: ignore

__all__ = [
    'DataProcessor',
    'FeatureTransformer',
    'ModelTrainer',
    'app',
]

