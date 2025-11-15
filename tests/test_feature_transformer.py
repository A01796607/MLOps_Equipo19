"""
Unit tests for FeatureTransformer class.
"""
import pytest
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from src.feature_transformer import FeatureTransformer


class TestFeatureTransformer:
    """Test suite for FeatureTransformer class."""
    
    def test_init_default(self):
        """Test FeatureTransformer initialization with default parameters."""
        transformer = FeatureTransformer()
        assert transformer.use_pca is True
        assert transformer.n_components == 50
        assert transformer.is_fitted is False
        assert isinstance(transformer.label_encoder, LabelEncoder)
    
    def test_init_without_pca(self):
        """Test FeatureTransformer initialization without PCA."""
        transformer = FeatureTransformer(use_pca=False, n_components=30)
        assert transformer.use_pca is False
        assert transformer.n_components == 30
    
    def test_encode_labels(self):
        """Test label encoding."""
        transformer = FeatureTransformer()
        
        y = pd.Series(['a', 'b', 'c', 'a', 'b'])
        y_encoded = transformer.encode_labels(y)
        
        # Should return numpy array
        assert isinstance(y_encoded, np.ndarray)
        
        # Should have same length
        assert len(y_encoded) == len(y)
        
        # Should have unique integer labels
        unique_labels = np.unique(y_encoded)
        assert len(unique_labels) == 3  # Three unique classes
        assert all(isinstance(label, (np.integer, int)) for label in unique_labels)
    
    def test_decode_labels(self):
        """Test label decoding."""
        transformer = FeatureTransformer()
        
        y = pd.Series(['a', 'b', 'c', 'a', 'b'])
        y_encoded = transformer.encode_labels(y)
        y_decoded = transformer.decode_labels(y_encoded)
        
        # Should return numpy array
        assert isinstance(y_decoded, np.ndarray)
        
        # Should match original labels
        assert list(y_decoded) == list(y)
    
    def test_encode_decode_roundtrip(self):
        """Test that encode and decode are inverse operations."""
        transformer = FeatureTransformer()
        
        y_original = pd.Series(['angry', 'happy', 'relax', 'sad', 'angry'])
        y_encoded = transformer.encode_labels(y_original)
        y_decoded = transformer.decode_labels(y_encoded)
        
        assert list(y_decoded) == list(y_original)
    
    def test_fit_transform_with_pca(self):
        """Test fit_transform with PCA enabled."""
        transformer = FeatureTransformer(use_pca=True, n_components=2)
        
        # Create sample data
        X_train = pd.DataFrame({
            'col1': [1, 2, 3, 4, 5],
            'col2': [10, 20, 30, 40, 50],
            'col3': [100, 200, 300, 400, 500]
        })
        X_test = pd.DataFrame({
            'col1': [6, 7],
            'col2': [60, 70],
            'col3': [600, 700]
        })
        
        X_train_transformed, X_test_transformed = transformer.fit_transform(X_train, X_test)
        
        # Should return numpy arrays
        assert isinstance(X_train_transformed, np.ndarray)
        assert isinstance(X_test_transformed, np.ndarray)
        
        # Should have reduced dimensions (2 components)
        assert X_train_transformed.shape[1] == 2
        assert X_test_transformed.shape[1] == 2
        
        # Should be fitted
        assert transformer.is_fitted is True
        
        # Should have same number of rows
        assert X_train_transformed.shape[0] == len(X_train)
        assert X_test_transformed.shape[0] == len(X_test)
    
    def test_fit_transform_without_pca(self):
        """Test fit_transform without PCA."""
        transformer = FeatureTransformer(use_pca=False)
        
        X_train = pd.DataFrame({
            'col1': [1, 2, 3, 4, 5],
            'col2': [10, 20, 30, 40, 50]
        })
        X_test = pd.DataFrame({
            'col1': [6, 7],
            'col2': [60, 70]
        })
        
        X_train_transformed, X_test_transformed = transformer.fit_transform(X_train, X_test)
        
        # Should return numpy arrays
        assert isinstance(X_train_transformed, np.ndarray)
        assert isinstance(X_test_transformed, np.ndarray)
        
        # Should have same number of features (just scaled)
        assert X_train_transformed.shape[1] == X_train.shape[1]
        assert X_test_transformed.shape[1] == X_test.shape[1]
        
        # Should be fitted
        assert transformer.is_fitted is True
    
    def test_transform_before_fit(self):
        """Test that transform raises error if called before fit."""
        transformer = FeatureTransformer()
        
        X = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': [10, 20, 30]
        })
        
        with pytest.raises(ValueError, match="must be fitted"):
            transformer.transform(X)
    
    def test_transform_after_fit(self):
        """Test transform after fit_transform."""
        transformer = FeatureTransformer(use_pca=True, n_components=2)
        
        X_train = pd.DataFrame({
            'col1': [1, 2, 3, 4, 5],
            'col2': [10, 20, 30, 40, 50],
            'col3': [100, 200, 300, 400, 500]
        })
        X_test = pd.DataFrame({
            'col1': [6, 7],
            'col2': [60, 70],
            'col3': [600, 700]
        })
        
        # Fit first
        transformer.fit_transform(X_train, X_test)
        
        # Now transform new data
        X_new = pd.DataFrame({
            'col1': [8, 9],
            'col2': [80, 90],
            'col3': [800, 900]
        })
        
        X_new_transformed = transformer.transform(X_new)
        
        # Should work without error
        assert isinstance(X_new_transformed, np.ndarray)
        assert X_new_transformed.shape[1] == 2
        assert X_new_transformed.shape[0] == len(X_new)
    
    def test_get_feature_names_with_pca(self):
        """Test get_feature_names with PCA enabled."""
        transformer = FeatureTransformer(use_pca=True, n_components=5)
        
        feature_names = transformer.get_feature_names()
        
        assert len(feature_names) == 5
        assert feature_names == ['PC1', 'PC2', 'PC3', 'PC4', 'PC5']
    
    def test_get_feature_names_without_pca(self):
        """Test get_feature_names without PCA."""
        transformer = FeatureTransformer(use_pca=False)
        
        feature_names = transformer.get_feature_names()
        
        assert feature_names == []
    
    def test_pca_explained_variance(self):
        """Test that PCA explains variance correctly."""
        transformer = FeatureTransformer(use_pca=True, n_components=2)
        
        # Create data with clear structure
        np.random.seed(42)
        X_train = pd.DataFrame(np.random.randn(100, 5))
        X_test = pd.DataFrame(np.random.randn(20, 5))
        
        transformer.fit_transform(X_train, X_test)
        
        # Check that explained variance ratio exists
        explained_variance = transformer.pca.explained_variance_ratio_
        assert len(explained_variance) == 2
        assert all(0 <= var <= 1 for var in explained_variance)
        assert explained_variance.sum() <= 1.0

