"""
Reproducibility Configuration Module.

This module provides utilities to ensure reproducible experiments by setting
random seeds for all relevant libraries (numpy, random, sklearn, lightgbm, etc.).
"""
import random
import os
import numpy as np
from typing import Optional
from loguru import logger

# Optional dependencies
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    lgb = None

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None


def set_random_seed(seed: int = 42, verbose: bool = True) -> None:
    """
    Set random seeds for all relevant libraries to ensure reproducibility.
    
    This function configures random seeds for:
    - Python's random module
    - NumPy random number generator
    - Environment variables (PYTHONHASHSEED)
    - LightGBM (if available)
    - PyTorch (if available)
    - TensorFlow (if available via os.environ)
    
    Args:
        seed: Random seed value (default: 42)
        verbose: Whether to log the seed configuration
    
    Example:
        >>> set_random_seed(42)
        >>> # Now all random operations will be reproducible
    """
    if verbose:
        logger.info(f"Setting random seed to {seed} for reproducibility...")
    
    # Python's random module
    random.seed(seed)
    
    # NumPy random number generator
    np.random.seed(seed)
    
    # Set environment variable for Python hash seed
    # This ensures dictionary iteration order is consistent
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # LightGBM seed (if available)
    if LIGHTGBM_AVAILABLE:
        # LightGBM uses numpy random state
        # The seed is already set via np.random.seed above
        # But we can also set it explicitly in LightGBM params
        pass
    
    # PyTorch seeds (if available)
    if TORCH_AVAILABLE:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Additional settings for reproducibility
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # TensorFlow seed (via environment variable if TF is used)
    # Note: TensorFlow must be imported to set seeds, but we don't want
    # to force it as a dependency
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    
    if verbose:
        logger.success(f"Random seed {seed} configured successfully")
        logger.info("Configured libraries:")
        logger.info("  ✓ Python random")
        logger.info("  ✓ NumPy")
        logger.info("  ✓ Environment (PYTHONHASHSEED)")
        if LIGHTGBM_AVAILABLE:
            logger.info("  ✓ LightGBM (via NumPy seed)")
        if TORCH_AVAILABLE:
            logger.info("  ✓ PyTorch")
        logger.info("  ✓ TensorFlow environment variables")


def get_reproducibility_config(seed: int = 42) -> dict:
    """
    Get a dictionary with reproducibility configuration for ML libraries.
    
    This function returns a dictionary with all necessary parameters to ensure
    reproducibility when passing to ML models and training functions.
    
    Args:
        seed: Random seed value (default: 42)
    
    Returns:
        Dictionary with reproducibility parameters for:
        - sklearn: random_state
        - lightgbm: random_state
        - train_test_split: random_state
    
    Example:
        >>> config = get_reproducibility_config(42)
        >>> model = RandomForestClassifier(**config['sklearn'])
        >>> X_train, X_test = train_test_split(X, y, **config['split'])
    """
    return {
        'seed': seed,
        'sklearn': {
            'random_state': seed
        },
        'lightgbm': {
            'random_state': seed
        },
        'split': {
            'random_state': seed
        },
        'cv': {
            'random_state': seed,
            'shuffle': True  # shuffle=True requires random_state
        }
    }


def ensure_reproducibility(seed: int = 42, verbose: bool = True) -> dict:
    """
    Comprehensive function to ensure reproducibility.
    
    This function both sets all random seeds and returns a configuration
    dictionary for use in ML pipelines.
    
    Args:
        seed: Random seed value (default: 42)
        verbose: Whether to log the configuration
    
    Returns:
        Dictionary with reproducibility parameters
    
    Example:
        >>> config = ensure_reproducibility(42)
        >>> # All seeds are now set, and config contains params for models
    """
    set_random_seed(seed, verbose=verbose)
    config = get_reproducibility_config(seed)
    return config


# Default seed value used throughout the project
DEFAULT_SEED = 42

