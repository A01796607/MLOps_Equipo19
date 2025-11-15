"""
Unit tests for DataProcessor class.
"""
import pytest
import pandas as pd
import numpy as np
from src.data_processor import DataProcessor


class TestDataProcessor:
    """Test suite for DataProcessor class."""
    
    def test_init_default(self):
        """Test DataProcessor initialization with default parameters."""
        processor = DataProcessor()
        assert processor.iqr_factor == 1.5
        assert processor.processed_data is None
        assert processor.outliers_removed == {}
    
    def test_init_custom_iqr(self):
        """Test DataProcessor initialization with custom IQR factor."""
        processor = DataProcessor(iqr_factor=2.0)
        assert processor.iqr_factor == 2.0
    
    def test_remove_outliers_iqr_no_outliers(self):
        """Test IQR outlier removal with data that has no outliers."""
        processor = DataProcessor(iqr_factor=1.5)
        
        # Create data without outliers (all values within normal range)
        df = pd.DataFrame({
            'col1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'col2': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        })
        
        df_clean = processor.remove_outliers_iqr(df)
        
        # Should keep all rows since no outliers
        assert len(df_clean) == len(df)
        assert 'col1' in processor.outliers_removed
        assert 'col2' in processor.outliers_removed
    
    def test_remove_outliers_iqr_with_outliers(self):
        """Test IQR outlier removal with data that has outliers."""
        processor = DataProcessor(iqr_factor=1.5)
        
        # Create data with outliers
        df = pd.DataFrame({
            'col1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1000]  # 1000 is an outlier
        })
        
        df_clean = processor.remove_outliers_iqr(df)
        
        # Should remove the outlier
        assert len(df_clean) < len(df)
        assert 1000 not in df_clean['col1'].values
        assert processor.processed_data is not None
    
    def test_remove_outliers_iqr_specific_columns(self):
        """Test IQR outlier removal on specific columns."""
        processor = DataProcessor(iqr_factor=1.5)
        
        df = pd.DataFrame({
            'col1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1000],
            'col2': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110],
            'col3': ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k']
        })
        
        df_clean = processor.remove_outliers_iqr(df, columns=['col1'])
        
        # Should only process col1
        assert 'col1' in processor.outliers_removed
        # col2 and col3 should still be in the dataframe
        assert 'col2' in df_clean.columns
        assert 'col3' in df_clean.columns
    
    def test_remove_outliers_iqr_empty_dataframe(self):
        """Test IQR outlier removal with empty dataframe."""
        processor = DataProcessor(iqr_factor=1.5)
        
        df = pd.DataFrame()
        df_clean = processor.remove_outliers_iqr(df)
        
        assert len(df_clean) == 0
    
    def test_get_stats_numeric(self):
        """Test getting numeric statistics."""
        processor = DataProcessor()
        
        df = pd.DataFrame({
            'col1': [1, 2, 3, 4, 5],
            'col2': [10, 20, 30, 40, 50],
            'col3': ['a', 'b', 'c', 'd', 'e']  # Categorical, should be ignored
        })
        
        stats = processor.get_stats_numeric(df)
        
        # Should only include numeric columns
        assert 'col1' in stats.columns
        assert 'col2' in stats.columns
        assert 'col3' not in stats.columns
        
        # Should have expected statistics
        assert 'Media' in stats.index
        assert 'Mediana/Cuartil 50%' in stats.index
        assert 'Min' in stats.index
        assert 'Max' in stats.index
    
    def test_get_stats_categorical(self):
        """Test getting categorical statistics."""
        processor = DataProcessor()
        
        df = pd.DataFrame({
            'col1': ['a', 'b', 'c', 'a', 'b'],
            'col2': ['x', 'y', 'x', 'y', 'x'],
            'col3': [1, 2, 3, 4, 5]  # Numeric, should be ignored
        })
        
        stats = processor.get_stats_categorical(df)
        
        # Should only include categorical columns
        assert 'col1' in stats.columns
        assert 'col2' in stats.columns
        assert 'col3' not in stats.columns
        
        # Should have expected statistics
        assert 'Moda' in stats.index
        assert 'Cardinalidad' in stats.index
    
    def test_get_missing_percentage_no_missing(self):
        """Test missing percentage calculation with no missing values."""
        processor = DataProcessor()
        
        df = pd.DataFrame({
            'col1': [1, 2, 3, 4, 5],
            'col2': [10, 20, 30, 40, 50]
        })
        
        missing_pct = processor.get_missing_percentage(df)
        
        # Should return 0 for all columns
        assert (missing_pct == 0).all()
    
    def test_get_missing_percentage_with_missing(self):
        """Test missing percentage calculation with missing values."""
        processor = DataProcessor()
        
        df = pd.DataFrame({
            'col1': [1, 2, np.nan, 4, 5],
            'col2': [10, 20, 30, 40, 50]
        })
        
        missing_pct = processor.get_missing_percentage(df)
        
        # col1 should have 20% missing (1 out of 5)
        assert missing_pct['col1'] == 20.0
        assert missing_pct['col2'] == 0.0
    
    def test_outliers_removed_tracking(self):
        """Test that outliers_removed dictionary is properly populated."""
        processor = DataProcessor(iqr_factor=1.5)
        
        df = pd.DataFrame({
            'col1': [1, 2, 3, 4, 5, 1000]  # 1000 is an outlier
        })
        
        processor.remove_outliers_iqr(df)
        
        # Check that outliers_removed is populated
        assert 'col1' in processor.outliers_removed
        assert 'before' in processor.outliers_removed['col1']
        assert 'after' in processor.outliers_removed['col1']
        assert 'removed' in processor.outliers_removed['col1']
        
        # Verify counts are correct
        assert processor.outliers_removed['col1']['before'] == 6
        assert processor.outliers_removed['col1']['after'] == 5
        assert processor.outliers_removed['col1']['removed'] == 1

