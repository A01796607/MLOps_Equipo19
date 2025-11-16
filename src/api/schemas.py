"""
Pydantic schemas for API request/response models.
"""
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
try:
    # Pydantic v2
    from pydantic import field_validator  # type: ignore
except Exception:
    # Pydantic v1 fallback (to avoid breaking environments); no-op import here
    field_validator = None  # type: ignore


N_FEATURES = 50


class PredictionRequest(BaseModel):
    """
    Request schema for prediction endpoint.
    
    Attributes:
        features: List of feature vectors (each vector is a list of floats)
        model_name: Optional model name (default: "random_forest")
    """
    features: List[List[float]] = Field(
        ...,
        description="List of feature vectors for prediction. Each inner list represents one sample's features.",
        example=[[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]]
    )
    model_name: Optional[str] = Field(
        default="random_forest",
        description="Name of the model to use for prediction",
        example="random_forest"
    )
    # Validate that each sample has exactly N_FEATURES features (Pydantic v2)
    if 'field_validator' in globals() and field_validator is not None:
        @field_validator("features")
        @classmethod
        def validate_features_length(cls, v: List[List[float]]) -> List[List[float]]:
            if not v:
                raise ValueError("Features list cannot be empty")
            for i, row in enumerate(v):
                if len(row) != N_FEATURES:
                    raise ValueError(f"Sample {i} must have {N_FEATURES} features, got {len(row)}")
            return v


class PredictionItem(BaseModel):
    """Single prediction result."""
    prediction: int = Field(..., description="Predicted class index")
    prediction_label: Optional[str] = Field(None, description="Predicted class label (decoded)")
    probabilities: Optional[Dict[str, float]] = Field(None, description="Prediction probabilities per class")


class PredictionResponse(BaseModel):
    """
    Response schema for prediction endpoint.
    
    Attributes:
        prediction: Predicted class index
        prediction_label: Predicted class label (decoded)
        probabilities: Prediction probabilities per class
        model_name: Name of the model used
    """
    prediction: int = Field(..., description="Predicted class index")
    prediction_label: Optional[str] = Field(None, description="Predicted class label (decoded)")
    probabilities: Optional[Dict[str, float]] = Field(None, description="Prediction probabilities per class")
    model_name: str = Field(..., description="Name of the model used")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(..., description="Service status", example="healthy")
    version: str = Field(..., description="API version", example="1.0.0")


class ModelInfoResponse(BaseModel):
    """Model information response."""
    model_name: str = Field(..., description="Name of the model")
    model_path: str = Field(..., description="Path to the model file")
    model_loaded: bool = Field(..., description="Whether the model is loaded")
    transformer_loaded: bool = Field(..., description="Whether the transformer is loaded")
    available_models: List[str] = Field(..., description="List of available model names")
    class_labels: Optional[List[str]] = Field(None, description="Available class labels (if transformer loaded)")
