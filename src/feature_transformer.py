"""
Feature Transformer for encoding, scaling and dimensionality reduction.
"""
import numpy as np
import pandas as pd
from typing import Tuple, Optional
from loguru import logger

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA

from src.reproducibility import DEFAULT_SEED


class FeatureTransformer:
    """Class for transforming features (encoding, scaling, PCA)."""
    
    def __init__(self, use_pca: bool = True, n_components: int = 50, random_state: int = DEFAULT_SEED):
        """
        Initialize FeatureTransformer.
        
        Args:
            use_pca: Whether to use PCA for dimensionality reduction
            n_components: Number of components for PCA
            random_state: Random state for PCA (default: DEFAULT_SEED)
        """
        self.use_pca = use_pca
        self.n_components = n_components
        
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=n_components, random_state=random_state)
        
        self.is_fitted = False
    
    def encode_labels(self, y: pd.Series) -> np.ndarray:
        """
        Encode categorical labels.
        
        Args:
            y: Categorical labels
            
        Returns:
            Encoded labels
        """
        y_encoded = self.label_encoder.fit_transform(y)
        logger.info(f"Clases originales: {list(self.label_encoder.classes_)}")
        return y_encoded
    
    def decode_labels(self, y_encoded: np.ndarray) -> np.ndarray:
        """
        Decode encoded labels back to original classes.
        
        Args:
            y_encoded: Encoded labels
            
        Returns:
            Decoded labels
        """
        return self.label_encoder.inverse_transform(y_encoded)
    
    def fit_transform(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fit transformer and transform both train and test sets.
        
        Args:
            X_train: Training features
            X_test: Test features
            
        Returns:
            Transformed training and test features
        """
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        logger.info("Features scaled successfully")
        
        # Apply PCA if enabled
        if self.use_pca:
            X_train_final = self.pca.fit_transform(X_train_scaled)
            X_test_final = self.pca.transform(X_test_scaled)
            
            explained_variance = self.pca.explained_variance_ratio_
            cumulative_variance = explained_variance.sum()
            
            logger.info(f"Varianza explicada por componente: {explained_variance[:5]}...")
            logger.info(f"Varianza acumulada: {cumulative_variance}")
        else:
            X_train_final = X_train_scaled
            X_test_final = X_test_scaled
        
        self.is_fitted = True
        
        return X_train_final, X_test_final
    
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        Transform features using fitted transformers.
        
        Args:
            X: Features to transform
            
        Returns:
            Transformed features
        """
        if not self.is_fitted:
            raise ValueError("Transformer must be fitted before calling transform")
        
        X_scaled = self.scaler.transform(X)
        
        if self.use_pca:
            X_final = self.pca.transform(X_scaled)
        else:
            X_final = X_scaled
        
        return X_final
    
    def get_feature_names(self) -> list:
        """
        Get feature names after transformation.
        
        Returns:
            List of feature names
        """
        if self.use_pca:
            return [f"PC{i+1}" for i in range(self.n_components)]
        else:
            return []

