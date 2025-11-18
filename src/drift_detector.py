"""
Data Drift Detection Module.

This module provides functionality to detect data drift between reference
(training) data and production data.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union
from pathlib import Path
from loguru import logger
import joblib
from scipy.stats import ks_2samp
from scipy.spatial.distance import jensenshannon


class DriftDetector:
    """
    Detecta data drift comparando datos de producción con datos de referencia.
    """
    
    def __init__(
        self,
        reference_data: Optional[np.ndarray] = None,
        transformer_path: Optional[Union[str, Path]] = None,
        method: str = 'psi',
        threshold: float = 0.25
    ):
        """
        Inicializa el detector de drift.
        
        Args:
            reference_data: Datos de referencia (train/test) ya transformados
            transformer_path: Path al transformer guardado (si reference_data no está transformado)
            method: Método de detección ('psi', 'ks', 'js')
            threshold: Umbral para considerar drift significativo
        """
        self.method = method
        self.threshold = threshold
        self.transformer = None
        
        if transformer_path:
            self.transformer = joblib.load(transformer_path)
            logger.info(f"Transformer cargado desde {transformer_path}")
        
        if reference_data is not None:
            self.set_reference_data(reference_data)
    
    def set_reference_data(self, reference_data: np.ndarray):
        """
        Establece los datos de referencia.
        
        Args:
            reference_data: Datos de referencia (ya transformados o sin transformar)
        """
        # Si hay transformer, transformar los datos
        if self.transformer is not None:
            if isinstance(reference_data, pd.DataFrame):
                self.reference_data = self.transformer.transform(reference_data)
            else:
                # Convertir a DataFrame temporal para el transformer
                ref_df = pd.DataFrame(reference_data)
                self.reference_data = self.transformer.transform(ref_df)
        else:
            self.reference_data = reference_data
        
        # Guardar estadísticas de referencia
        self.reference_stats = {
            'mean': np.mean(self.reference_data, axis=0),
            'std': np.std(self.reference_data, axis=0),
            'min': np.min(self.reference_data, axis=0),
            'max': np.max(self.reference_data, axis=0)
        }
        
        logger.info(f"Datos de referencia establecidos: shape {self.reference_data.shape}")
    
    def detect_drift(
        self,
        production_data: Union[np.ndarray, pd.DataFrame],
        return_details: bool = True
    ) -> Dict:
        """
        Detecta drift en datos de producción.
        
        Args:
            production_data: Datos de producción (raw o transformados)
            return_details: Si True, retorna detalles por feature
            
        Returns:
            dict: Resultados de detección con scores y flags de drift
        """
        if self.reference_data is None:
            raise ValueError("Debe establecer datos de referencia primero usando set_reference_data()")
        
        # Transformar datos de producción si es necesario
        if self.transformer is not None:
            if isinstance(production_data, pd.DataFrame):
                prod_transformed = self.transformer.transform(production_data)
            else:
                prod_df = pd.DataFrame(production_data)
                prod_transformed = self.transformer.transform(prod_df)
        else:
            prod_transformed = production_data
        
        # Detectar drift por feature
        drift_results = {}
        n_features = self.reference_data.shape[1]
        
        for feature_idx in range(n_features):
            ref_feature = self.reference_data[:, feature_idx]
            prod_feature = prod_transformed[:, feature_idx]
            
            if self.method == 'psi':
                score = self._calculate_psi(ref_feature, prod_feature)
            elif self.method == 'ks':
                _, p_value = ks_2samp(ref_feature, prod_feature)
                score = 1 - p_value  # Convertir a score de drift
            elif self.method == 'js':
                score = jensenshannon(ref_feature, prod_feature)
            else:
                raise ValueError(f"Método {self.method} no soportado. Use 'psi', 'ks', o 'js'")
            
            drift_results[feature_idx] = {
                'score': float(score),
                'drift_detected': score > self.threshold,
                'reference_mean': float(self.reference_stats['mean'][feature_idx]),
                'production_mean': float(np.mean(prod_feature)),
                'reference_std': float(self.reference_stats['std'][feature_idx]),
                'production_std': float(np.std(prod_feature))
            }
        
        # Resumen agregado
        drift_scores = [r['score'] for r in drift_results.values()]
        summary = {
            'max_drift_score': float(max(drift_scores)),
            'mean_drift_score': float(np.mean(drift_scores)),
            'min_drift_score': float(min(drift_scores)),
            'features_with_drift': sum(1 for r in drift_results.values() if r['drift_detected']),
            'total_features': len(drift_results),
            'drift_detected': any(r['drift_detected'] for r in drift_results.values()),
            'drift_percentage': (sum(1 for r in drift_results.values() if r['drift_detected']) / len(drift_results)) * 100
        }
        
        if return_details:
            summary['feature_details'] = drift_results
        
        return summary
    
    def _calculate_psi(self, expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
        """
        Calcula el Population Stability Index entre dos distribuciones.
        
        PSI < 0.1: Sin cambio significativo
        0.1 <= PSI < 0.25: Cambio moderado
        PSI >= 0.25: Cambio significativo (drift)
        
        Args:
            expected: Distribución esperada (referencia)
            actual: Distribución actual (producción)
            bins: Número de bins para discretizar
            
        Returns:
            PSI score
        """
        # Normalizar a [0, 1] para bins consistentes
        min_val = min(np.min(expected), np.min(actual))
        max_val = max(np.max(expected), np.max(actual))
        
        if max_val == min_val:
            return 0.0
        
        expected_norm = (expected - min_val) / (max_val - min_val)
        actual_norm = (actual - min_val) / (max_val - min_val)
        
        # Crear bins
        breakpoints = np.linspace(0, 1, bins + 1)
        
        expected_percents = np.histogram(expected_norm, bins=breakpoints)[0] / len(expected)
        actual_percents = np.histogram(actual_norm, bins=breakpoints)[0] / len(actual)
        
        # Evitar división por cero
        expected_percents = np.where(expected_percents == 0, 0.0001, expected_percents)
        actual_percents = np.where(actual_percents == 0, 0.0001, actual_percents)
        
        psi = np.sum((actual_percents - expected_percents) * 
                     np.log(actual_percents / expected_percents))
        return psi
    
    def get_drift_severity(self, drift_score: float) -> str:
        """
        Determina la severidad del drift basado en el score.
        
        Args:
            drift_score: Score de drift
            
        Returns:
            Severidad: 'none', 'low', 'moderate', 'high'
        """
        if self.method == 'psi':
            if drift_score < 0.1:
                return 'none'
            elif drift_score < 0.25:
                return 'low'
            elif drift_score < 0.5:
                return 'moderate'
            else:
                return 'high'
        elif self.method == 'ks':
            p_value = 1 - drift_score
            if p_value >= 0.05:
                return 'none'
            elif p_value >= 0.01:
                return 'low'
            elif p_value >= 0.001:
                return 'moderate'
            else:
                return 'high'
        elif self.method == 'js':
            if drift_score < 0.1:
                return 'none'
            elif drift_score < 0.3:
                return 'low'
            elif drift_score < 0.5:
                return 'moderate'
            else:
                return 'high'
        else:
            return 'unknown'

