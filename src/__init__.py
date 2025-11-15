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

__all__ = [
    'DataProcessor',
    'FeatureTransformer',
    'ModelTrainer',
]

