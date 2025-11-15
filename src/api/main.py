"""
FastAPI application for model inference API.
"""
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse
from loguru import logger
import sys

from src.api.schemas import (
    PredictionRequest,
    PredictionResponse,
    HealthResponse,
    ModelInfoResponse,
    PredictionItem
)
from src.api.service import get_model_service
from src.api.config import AVAILABLE_MODELS

# Configure loguru to work with uvicorn
logger.remove()
logger.add(sys.stderr, level="INFO")

# Initialize FastAPI app
app = FastAPI(
    title="MLOps Model Inference API",
    description="REST API for making predictions with trained ML models",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)


@app.on_event("startup")
async def startup_event():
    """Initialize service on startup."""
    logger.info("Starting up API service...")
    try:
        service = get_model_service()
        logger.success(f"API service started successfully with model: {service.model_name}")
    except Exception as e:
        logger.error(f"Error during startup: {e}")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Shutting down API service...")


@app.get("/", response_model=HealthResponse, tags=["Health"])
async def root():
    """
    Root endpoint - health check.
    
    Returns:
        Health status and API version
    """
    return HealthResponse(
        status="healthy",
        version="1.0.0"
    )


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Health check endpoint.
    
    Returns:
        Health status and API version
    """
    return HealthResponse(
        status="healthy",
        version="1.0.0"
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Predictions"])
async def predict(request: PredictionRequest):
    """
    Make predictions on feature data.
    
    Args:
        request: Prediction request with features and optional model name
        
    Returns:
        Prediction response with prediction result and model name
        
    Raises:
        HTTPException: If model not found or prediction fails
    """
    try:
        # Get or load service with specified model
        service = get_model_service(model_name=request.model_name, reload=False)
        
        # Check if model is loaded
        if service.model is None:
            # Try to load the requested model
            if request.model_name:
                if not service.load_model_by_name(request.model_name):
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND,
                        detail=f"Model '{request.model_name}' not found or could not be loaded"
                    )
            else:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="No model is loaded and no default model found"
                )
        
        # Validate features
        if not request.features:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Features list cannot be empty"
            )
        
        # Make predictions
        predictions_data = service.predict(
            features=request.features,
            return_probabilities=True
        )
        
        # Take the first prediction (object simple)
        if not predictions_data:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="No predictions returned"
            )
        
        first_prediction = predictions_data[0]
        
        return PredictionResponse(
            prediction=first_prediction["prediction"],
            prediction_label=first_prediction.get("prediction_label"),
            probabilities=first_prediction.get("probabilities"),
            model_name=service.model_name
        )
        
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )


@app.get("/models/info", response_model=ModelInfoResponse, tags=["Models"])
async def get_model_info(model_name: str = None):
    """
    Get information about the loaded model or a specific model.
    
    Args:
        model_name: Optional model name to get info for
        
    Returns:
        Model information response
        
    Raises:
        HTTPException: If model not found
    """
    try:
        if model_name:
            # Load specific model
            service = get_model_service(model_name=model_name, reload=True)
            if service.model is None:
                if not service.load_model_by_name(model_name):
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND,
                        detail=f"Model '{model_name}' not found"
                    )
        else:
            # Get current loaded model
            service = get_model_service()
        
        info = service.get_model_info()
        
        return ModelInfoResponse(**info)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )


@app.get("/models/available", tags=["Models"])
async def get_available_models():
    """
    Get list of available models.
    
    Returns:
        Dictionary with available model names and paths
    """
    return {
        "available_models": list(AVAILABLE_MODELS.keys()),
        "models": {
            name: str(path) for name, path in AVAILABLE_MODELS.items()
        }
    }


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error"}
    )
