"""
Data Processor for cleaning and preprocessing data.
"""
import pandas as pd
import numpy as np
from typing import Optional
from loguru import logger


class DataProcessor:
    """Class for processing and cleaning datasets."""
    
    def __init__(self, iqr_factor: float = 1.5):
        """
        Initialize DataProcessor.
        
        Args:
            iqr_factor: Factor for IQR outlier detection (default: 1.5)
        """
        self.iqr_factor = iqr_factor
        self.processed_data: Optional[pd.DataFrame] = None
        self.outliers_removed: dict = {}
    
    def remove_outliers_iqr(self, df: pd.DataFrame, columns: Optional[list] = None) -> pd.DataFrame:
        """
        Remove outliers using IQR method.
        
        Args:
            df: Input dataframe
            columns: List of numeric columns to process. If None, uses all numeric columns.
            
        Returns:
            Cleaned dataframe
        """
        df_clean = df.copy()
        
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        initial_rows = len(df_clean)
        
        for col in columns:
            rows_before = len(df_clean)
            
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - self.iqr_factor * IQR
            upper_bound = Q3 + self.iqr_factor * IQR
            
            # Filter outliers
            df_clean = df_clean[
                (df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)
            ]
            
            rows_after = len(df_clean)
            outliers_removed = rows_before - rows_after
            
            self.outliers_removed[col] = {
                'before': rows_before,
                'after': rows_after,
                'removed': outliers_removed
            }
            
            logger.info(f"{col}: Filas restantes {rows_after}")
        
        final_rows = len(df_clean)
        total_removed = initial_rows - final_rows
        
        logger.info(f"Total de filas eliminadas: {total_removed} de {initial_rows}")
        
        self.processed_data = df_clean
        return df_clean
    
    def get_stats_numeric(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Get descriptive statistics for numeric columns.
        
        Args:
            df: Input dataframe
            
        Returns:
            DataFrame with statistics
        """
        numeric_cols = df.select_dtypes(include=[np.number])
        describe_df = numeric_cols.describe()
        
        nom_estadisticas = {
            'mean': 'Media',
            '50%': 'Mediana/Cuartil 50%',
            'min': 'Min',
            'max': 'Max',
            'std': 'Desviacion estandar',
            '25%': 'Cuartil 25%',
            '75%': 'Cuartil 75%'
        }
        
        estadisticas = describe_df.rename(index=nom_estadisticas).drop(index='count')
        return estadisticas
    
    def get_stats_categorical(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Get descriptive statistics for categorical columns.
        
        Args:
            df: Input dataframe
            
        Returns:
            DataFrame with statistics
        """
        categorical_cols = df.select_dtypes(exclude=[np.number])
        stats = categorical_cols.describe(include='object')
        
        nom_estadisticas = {
            'top': 'Moda',
            'unique': 'Cardinalidad'
        }
        
        estadisticas = stats.rename(index=nom_estadisticas).drop(index=['count', 'freq'])
        return estadisticas
    
    def get_missing_percentage(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate percentage of missing values per column.
        
        Args:
            df: Input dataframe
            
        Returns:
            Series with missing percentages
        """
        return df.isna().mean() * 100

