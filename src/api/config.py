"""
Configuration for API service.
"""
from pathlib import Path
from loguru import logger

# Project root (assuming this is src/api/, go up 2 levels)
PROJ_ROOT = Path(__file__).resolve().parents[2]

# Paths
MODELS_DIR = PROJ_ROOT / "models"
PROCESSED_DATA_DIR = PROJ_ROOT / "data" / "processed"

# Default model and transformer paths
DEFAULT_MODEL_PATH = MODELS_DIR / "random_forest.pkl"
DEFAULT_TRANSFORMER_PATH = PROCESSED_DATA_DIR / "transformer.pkl"

# Model options
AVAILABLE_MODELS = {
    "random_forest": MODELS_DIR / "random_forest.pkl",
    "lightgbm": MODELS_DIR / "lightgbm.pkl",
}

logger.info(f"API Config - Models directory: {MODELS_DIR}")
logger.info(f"API Config - Transformer path: {DEFAULT_TRANSFORMER_PATH}")
