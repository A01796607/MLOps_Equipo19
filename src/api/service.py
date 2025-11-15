"""
Service layer for model inference.
"""
from pathlib import Path
from typing import Optional, Dict, Any, List
import numpy as np
import pandas as pd
import joblib
from loguru import logger

from src.api.config import (
    DEFAULT_MODEL_PATH,
    DEFAULT_TRANSFORMER_PATH,
    AVAILABLE_MODELS,
    MODELS_DIR
)

# Optional LightGBM import
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    lgb = None  # type: ignore


class ModelService:
    """
    Service for loading models and making predictions.
    """
    
    def __init__(
        self,
        model_path: Optional[Path] = None,
        transformer_path: Optional[Path] = None
    ):
        """
        Initialize ModelService.
        
        Args:
            model_path: Path to the model file (if None, uses default)
            transformer_path: Path to the transformer file (if None, uses default)
        """
        self.model_path = model_path or DEFAULT_MODEL_PATH
        self.transformer_path = transformer_path or DEFAULT_TRANSFORMER_PATH
        self.model: Optional[Any] = None
        self.transformer: Optional[Any] = None
        self.model_name: str = "unknown"
        self._load_model()
        self._load_transformer()
    
    def _load_model(self) -> None:
        """Load model from file."""
        if not self.model_path.exists():
            logger.warning(f"Model file not found: {self.model_path}")
            return
        
        try:
            logger.info(f"Loading model from {self.model_path}")
            self.model = joblib.load(self.model_path)
            self.model_name = self.model_path.stem
            logger.success(f"Model loaded successfully: {self.model_name}")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def _load_transformer(self) -> None:
        """Load transformer from file."""
        if not self.transformer_path.exists():
            logger.warning(f"Transformer file not found: {self.transformer_path}")
            return
        
        try:
            logger.info(f"Loading transformer from {self.transformer_path}")
            self.transformer = joblib.load(self.transformer_path)
            logger.success("Transformer loaded successfully")
        except Exception as e:
            logger.warning(f"Error loading transformer: {e} (predictions will be numeric only)")
    
    def load_model_by_name(self, model_name: str) -> bool:
        """
        Load a specific model by name.
        
        Args:
            model_name: Name of the model to load
            
        Returns:
            True if model loaded successfully, False otherwise
        """
        if model_name not in AVAILABLE_MODELS:
            logger.error(f"Model '{model_name}' not in available models: {list(AVAILABLE_MODELS.keys())}")
            return False
        
        model_path = AVAILABLE_MODELS[model_name]
        if not model_path.exists():
            logger.error(f"Model file not found: {model_path}")
            return False
        
        try:
            logger.info(f"Loading model '{model_name}' from {model_path}")
            self.model = joblib.load(model_path)
            self.model_path = model_path
            self.model_name = model_name
            logger.success(f"Model '{model_name}' loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Error loading model '{model_name}': {e}")
            return False
    
    def predict(
        self,
        features: List[List[float]],
        return_probabilities: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Make predictions on features.
        
        Args:
            features: List of feature vectors (each inner list is one sample)
            return_probabilities: Whether to return prediction probabilities
            
        Returns:
            List of prediction dictionaries with 'prediction', 'prediction_label', and optionally 'probabilities'
            
        Raises:
            ValueError: If model is not loaded
        """
        if self.model is None:
            raise ValueError("Model is not loaded. Cannot make predictions.")
        
        # Convert to numpy array
        X = np.array(features)
        logger.info(f"Making predictions on {len(features)} samples with shape {X.shape}")
        
        # Make predictions
        try:
            # Check if it's a LightGBM Booster
            if LIGHTGBM_AVAILABLE and lgb is not None:
                if hasattr(self.model, 'predict') and hasattr(self.model, 'num_trees'):
                    # LightGBM Booster
                    y_pred_proba = self.model.predict(X)
                    y_pred = y_pred_proba.argmax(axis=1) if y_pred_proba.ndim > 1 else (y_pred_proba > 0.5).astype(int)
                else:
                    # Regular sklearn-style model
                    y_pred = self.model.predict(X)
                    y_pred_proba = None
                    if return_probabilities and hasattr(self.model, 'predict_proba'):
                        y_pred_proba = self.model.predict_proba(X)
            else:
                # Regular sklearn-style model
                y_pred = self.model.predict(X)
                y_pred_proba = None
                if return_probabilities and hasattr(self.model, 'predict_proba'):
                    y_pred_proba = self.model.predict_proba(X)
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            raise
        
        # Decode labels if transformer available
        results = []
        for i, pred in enumerate(y_pred):
            result: Dict[str, Any] = {
                "prediction": int(pred)
            }
            
            # Decode label if transformer available
            if self.transformer is not None and hasattr(self.transformer, 'decode_labels'):
                try:
                    decoded = self.transformer.decode_labels(np.array([pred]))[0]
                    result["prediction_label"] = str(decoded)
                except Exception as e:
                    logger.warning(f"Error decoding label: {e}")
            
            # Add probabilities if available
            if return_probabilities and y_pred_proba is not None:
                try:
                    probs = y_pred_proba[i].tolist()
                    # Get class labels if available
                    if self.transformer is not None and hasattr(self.transformer, 'label_encoder'):
                        try:
                            class_names = self.transformer.label_encoder.classes_
                            result["probabilities"] = {
                                str(class_name): float(prob)
                                for class_name, prob in zip(class_names, probs)
                            }
                        except Exception:
                            result["probabilities"] = {
                                f"class_{j}": float(prob) for j, prob in enumerate(probs)
                            }
                    else:
                        result["probabilities"] = {
                            f"class_{j}": float(prob) for j, prob in enumerate(probs)
                        }
                except Exception as e:
                    logger.warning(f"Error adding probabilities: {e}")
            
            results.append(result)
        
        logger.success(f"Predictions completed: {len(results)} predictions")
        return results
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary with model information
        """
        info = {
            "model_name": self.model_name,
            "model_path": str(self.model_path),
            "model_loaded": self.model is not None,
            "transformer_loaded": self.transformer is not None,
            "available_models": list(AVAILABLE_MODELS.keys()),
            "class_labels": None
        }
        
        # Get class labels from transformer if available
        if self.transformer is not None and hasattr(self.transformer, 'label_encoder'):
            try:
                info["class_labels"] = self.transformer.label_encoder.classes_.tolist()
            except Exception:
                pass
        
        return info


# Global service instance (singleton pattern)
_service_instance: Optional[ModelService] = None


def get_model_service(
    model_name: Optional[str] = None,
    reload: bool = False
) -> ModelService:
    """
    Get or create the global ModelService instance.
    
    Args:
        model_name: Optional model name to load
        reload: Whether to reload the service
        
    Returns:
        ModelService instance
    """
    global _service_instance
    
    if _service_instance is None or reload:
        if model_name:
            _service_instance = ModelService()
            _service_instance.load_model_by_name(model_name)
        else:
            _service_instance = ModelService()
    
    return _service_instance
