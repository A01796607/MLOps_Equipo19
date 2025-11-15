"""
Unit tests for Plotter class.
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import shutil
from src.plotter import Plotter


# Fixtures
@pytest.fixture
def temp_dir():
    """Create a temporary directory for test plots."""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    shutil.rmtree(temp_path)


@pytest.fixture
def sample_dataframe():
    """Create a sample dataframe for testing."""
    np.random.seed(42)
    return pd.DataFrame({
        'numeric1': np.random.randn(100),
        'numeric2': np.random.randn(100),
        'numeric3': np.random.randn(100),
        'categorical': np.random.choice(['a', 'b', 'c'], 100),
        'target': np.random.choice(['class1', 'class2', 'class3'], 100)
    })


class TestPlotter:
    """Test suite for Plotter class."""
    
    def test_init_default(self):
        """Test Plotter initialization with default directory."""
        plotter = Plotter()
        assert plotter.figures_dir == Path("reports/figures")
        assert plotter.figures_dir.exists()
    
    def test_init_custom_directory(self, temp_dir):
        """Test Plotter initialization with custom directory."""
        plotter = Plotter(figures_dir=temp_dir)
        assert plotter.figures_dir == temp_dir
        assert plotter.figures_dir.exists()
    
    def test_plot_histograms(self, temp_dir, sample_dataframe):
        """Test histogram plotting."""
        plotter = Plotter(figures_dir=temp_dir)
        
        output_path = plotter.plot_histograms(
            sample_dataframe,
            filename="test_histograms.png"
        )
        
        # Should return a Path
        assert isinstance(output_path, Path)
        
        # File should exist
        assert output_path.exists()
        assert output_path.name == "test_histograms.png"
    
    def test_plot_histograms_specific_columns(self, temp_dir, sample_dataframe):
        """Test histogram plotting with specific columns."""
        plotter = Plotter(figures_dir=temp_dir)
        
        output_path = plotter.plot_histograms(
            sample_dataframe,
            numeric_columns=['numeric1', 'numeric2'],
            filename="test_histograms_specific.png"
        )
        
        assert output_path.exists()
    
    def test_plot_boxplots(self, temp_dir, sample_dataframe):
        """Test boxplot plotting."""
        plotter = Plotter(figures_dir=temp_dir)
        
        output_path = plotter.plot_boxplots(
            sample_dataframe,
            filename="test_boxplots.png"
        )
        
        # Should return a Path
        assert isinstance(output_path, Path)
        
        # File should exist
        assert output_path.exists()
        assert output_path.name == "test_boxplots.png"
    
    def test_plot_categorical_counts(self, temp_dir, sample_dataframe):
        """Test categorical counts plotting."""
        plotter = Plotter(figures_dir=temp_dir)
        
        output_path = plotter.plot_categorical_counts(
            sample_dataframe,
            filename="test_categorical.png"
        )
        
        # Should return a Path
        assert isinstance(output_path, Path)
        
        # File should exist
        assert output_path.exists()
        assert output_path.name == "test_categorical.png"
    
    def test_plot_categorical_counts_no_categorical(self, temp_dir):
        """Test categorical counts plotting with no categorical columns."""
        plotter = Plotter(figures_dir=temp_dir)
        
        # Create dataframe with only numeric columns
        df = pd.DataFrame({
            'col1': [1, 2, 3, 4, 5],
            'col2': [10, 20, 30, 40, 50]
        })
        
        with pytest.raises(ValueError, match="No categorical columns"):
            plotter.plot_categorical_counts(df)
    
    def test_plot_correlation_heatmap(self, temp_dir, sample_dataframe):
        """Test correlation heatmap plotting."""
        plotter = Plotter(figures_dir=temp_dir)
        
        output_path = plotter.plot_correlation_heatmap(
            sample_dataframe,
            filename="test_correlation.png"
        )
        
        # Should return a Path
        assert isinstance(output_path, Path)
        
        # File should exist
        assert output_path.exists()
        assert output_path.name == "test_correlation.png"
    
    def test_plot_correlation_heatmap_no_numeric(self, temp_dir):
        """Test correlation heatmap plotting with no numeric columns."""
        plotter = Plotter(figures_dir=temp_dir)
        
        # Create dataframe with only categorical columns
        df = pd.DataFrame({
            'col1': ['a', 'b', 'c', 'd', 'e'],
            'col2': ['x', 'y', 'z', 'x', 'y']
        })
        
        with pytest.raises(ValueError, match="No numeric columns"):
            plotter.plot_correlation_heatmap(df)
    
    def test_plot_confusion_matrix(self, temp_dir):
        """Test confusion matrix plotting."""
        plotter = Plotter(figures_dir=temp_dir)
        
        # Create sample confusion matrix
        cm = np.array([[10, 2, 1],
                       [1, 15, 3],
                       [2, 1, 12]])
        
        class_names = ['class1', 'class2', 'class3']
        
        output_path = plotter.plot_confusion_matrix(
            cm,
            class_names=class_names,
            filename="test_confusion.png"
        )
        
        # Should return a Path
        assert isinstance(output_path, Path)
        
        # File should exist
        assert output_path.exists()
        assert output_path.name == "test_confusion.png"
    
    def test_plot_confusion_matrix_normalized(self, temp_dir):
        """Test normalized confusion matrix plotting."""
        plotter = Plotter(figures_dir=temp_dir)
        
        cm = np.array([[10, 2, 1],
                       [1, 15, 3],
                       [2, 1, 12]])
        
        output_path = plotter.plot_confusion_matrix(
            cm,
            normalize=True,
            filename="test_confusion_normalized.png"
        )
        
        assert output_path.exists()
    
    def test_plot_pca_variance(self, temp_dir):
        """Test PCA variance plotting."""
        plotter = Plotter(figures_dir=temp_dir)
        
        # Create sample explained variance ratio
        explained_variance = np.array([0.5, 0.3, 0.1, 0.05, 0.05])
        
        output_path = plotter.plot_pca_variance(
            explained_variance,
            filename="test_pca_variance.png"
        )
        
        # Should return a Path
        assert isinstance(output_path, Path)
        
        # File should exist
        assert output_path.exists()
        assert output_path.name == "test_pca_variance.png"
    
    def test_save_method(self, temp_dir):
        """Test internal _save method."""
        plotter = Plotter(figures_dir=temp_dir)
        
        import matplotlib.pyplot as plt
        
        # Create a simple figure
        fig = plt.figure()
        plt.plot([1, 2, 3], [1, 2, 3])
        
        output_path = plotter._save(fig, "test_save.png")
        
        # Should return a Path
        assert isinstance(output_path, Path)
        
        # File should exist
        assert output_path.exists()
        assert output_path.name == "test_save.png"
        
        plt.close(fig)
    
    def test_multiple_plots_same_directory(self, temp_dir, sample_dataframe):
        """Test that multiple plots can be saved to the same directory."""
        plotter = Plotter(figures_dir=temp_dir)
        
        # Generate multiple plots
        plotter.plot_histograms(sample_dataframe, filename="hist1.png")
        plotter.plot_boxplots(sample_dataframe, filename="box1.png")
        plotter.plot_correlation_heatmap(sample_dataframe, filename="corr1.png")
        
        # All files should exist
        assert (temp_dir / "hist1.png").exists()
        assert (temp_dir / "box1.png").exists()
        assert (temp_dir / "corr1.png").exists()

