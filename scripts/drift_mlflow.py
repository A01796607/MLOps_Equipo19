"""
Script para detectar drift y registrar resultados en MLflow.

Ejemplo de uso:
    python scripts/drift_mlflow.py \
        --reference-data data/processed/features.csv \
        --production-data data/production/sample.csv \
        --transformer-path data/processed/transformer.pkl \
        --method psi
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
import json
from datetime import datetime
from loguru import logger
import typer
import mlflow
from typing import Optional

from src.drift_detector import DriftDetector
from mlops.config import PROCESSED_DATA_DIR, RAW_DATA_DIR
from src.data_processor import DataProcessor
from src.feature_transformer import FeatureTransformer
from sklearn.model_selection import train_test_split
import joblib

app = typer.Typer()


def log_drift_to_mlflow(
    drift_results: dict,
    reference_data_shape: tuple,
    production_data_shape: tuple,
    method: str = "psi",
    threshold: float = 0.25,
    experiment_name: str = "Data_Drift_Monitoring"
):
    """
    Registra resultados de drift detection en MLflow.
    
    Args:
        drift_results: Resultados de detección de drift
        reference_data_shape: Shape de los datos de referencia
        production_data_shape: Shape de los datos de producción
        method: Método de detección usado
        threshold: Umbral usado
        experiment_name: Nombre del experimento en MLflow
    """
    # Configurar experimento
    mlflow.set_experiment(experiment_name)
    
    with mlflow.start_run(run_name=f"drift_detection_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        # Parámetros
        mlflow.log_param("detection_method", method)
        mlflow.log_param("threshold", threshold)
        mlflow.log_param("reference_samples", reference_data_shape[0])
        mlflow.log_param("reference_features", reference_data_shape[1])
        mlflow.log_param("production_samples", production_data_shape[0])
        mlflow.log_param("production_features", production_data_shape[1])
        
        # Métricas principales
        mlflow.log_metric("max_drift_score", drift_results['max_drift_score'])
        mlflow.log_metric("mean_drift_score", drift_results['mean_drift_score'])
        mlflow.log_metric("min_drift_score", drift_results['min_drift_score'])
        mlflow.log_metric("features_with_drift", drift_results['features_with_drift'])
        mlflow.log_metric("total_features", drift_results['total_features'])
        mlflow.log_metric("drift_percentage", drift_results['drift_percentage'])
        mlflow.log_metric("drift_detected", 1 if drift_results['drift_detected'] else 0)
        
        # Severidad como tag
        detector = DriftDetector(method=method, threshold=threshold)
        severity = detector.get_drift_severity(drift_results['max_drift_score'])
        mlflow.set_tag("drift_severity", severity)
        mlflow.set_tag("drift_status", "detected" if drift_results['drift_detected'] else "no_drift")
        
        # Guardar detalles de features con drift
        if 'feature_details' in drift_results:
            feature_details = drift_results['feature_details']
            
            # Crear DataFrame con detalles de drift
            drift_df = pd.DataFrame([
                {
                    'feature_idx': idx,
                    'drift_score': details['score'],
                    'drift_detected': details['drift_detected'],
                    'reference_mean': details['reference_mean'],
                    'production_mean': details['production_mean'],
                    'reference_std': details.get('reference_std', 0),
                    'production_std': details.get('production_std', 0)
                }
                for idx, details in feature_details.items()
            ])
            
            # Guardar como artifact CSV
            drift_csv_path = "drift_features_details.csv"
            drift_df.to_csv(drift_csv_path, index=False)
            mlflow.log_artifact(drift_csv_path)
            
            # Guardar top features con drift como JSON
            top_drift_features = drift_df.nlargest(10, 'drift_score')
            top_drift_json = {
                'top_drift_features': top_drift_features.to_dict('records')
            }
            mlflow.log_dict(top_drift_json, "top_drift_features.json")
        
        # Guardar resumen completo como JSON
        summary_json = {
            'detection_summary': {
                'method': method,
                'threshold': threshold,
                'max_drift_score': drift_results['max_drift_score'],
                'mean_drift_score': drift_results['mean_drift_score'],
                'features_with_drift': drift_results['features_with_drift'],
                'total_features': drift_results['total_features'],
                'drift_detected': drift_results['drift_detected'],
                'severity': severity
            }
        }
        mlflow.log_dict(summary_json, "drift_summary.json")
        
        logger.success(f"Resultados registrados en MLflow - Run ID: {mlflow.active_run().info.run_id}")
        logger.info(f"Ver resultados: mlflow ui")


@app.command()
def main(
    reference_data_path: Optional[Path] = typer.Option(
        None,
        "--reference-data",
        "-r",
        help="Path a los datos de referencia (train). Si no se proporciona, se generarán."
    ),
    production_data_path: Optional[Path] = typer.Option(
        None,
        "--production-data",
        "-p",
        help="Path a los datos de producción. Si no se proporciona, se simularán."
    ),
    transformer_path: Path = typer.Option(
        PROCESSED_DATA_DIR / "transformer.pkl",
        "--transformer-path",
        "-t",
        help="Path al transformer guardado"
    ),
    method: str = typer.Option(
        "psi",
        "--method",
        "-m",
        help="Método de detección: 'psi', 'ks', o 'js'"
    ),
    threshold: float = typer.Option(
        0.25,
        "--threshold",
        help="Umbral para considerar drift significativo"
    ),
    experiment_name: str = typer.Option(
        "Data_Drift_Monitoring",
        "--experiment",
        "-e",
        help="Nombre del experimento en MLflow"
    ),
    simulate_production: bool = typer.Option(
        False,
        "--simulate",
        help="Simular datos de producción con drift artificial"
    )
):
    """
    Detecta data drift y registra resultados en MLflow.
    """
    logger.info("=" * 70)
    logger.info("Data Drift Detection con MLflow")
    logger.info("=" * 70)
    
    # Cargar o generar datos de referencia
    if reference_data_path and reference_data_path.exists():
        logger.info(f"Cargando datos de referencia desde {reference_data_path}")
        reference_df = pd.read_csv(reference_data_path)
        if 'target' in reference_df.columns:
            reference_df = reference_df.drop(columns=['target'])
        reference_data = reference_df.values
    else:
        logger.info("Generando datos de referencia desde datos raw...")
        raw_data_path = RAW_DATA_DIR / "turkis_music_emotion_original.csv"
        if not raw_data_path.exists():
            logger.error("No se pueden generar datos de referencia sin datos raw")
            raise typer.Exit(1)
        
        df = pd.read_csv(raw_data_path)
        processor = DataProcessor()
        df_clean = processor.remove_outliers_iqr(df)
        X = df_clean.drop(columns=["Class"])
        
        transformer = joblib.load(transformer_path)
        X_sample = X.sample(n=min(100, len(X)), random_state=42)
        reference_data = transformer.transform(X_sample)
    
    # Cargar o simular datos de producción
    if simulate_production or production_data_path is None:
        logger.info("Simulando datos de producción con drift...")
        production_data = reference_data.copy()
        
        n_features = production_data.shape[1]
        n_features_to_drift = max(1, int(n_features * 0.15))
        
        np.random.seed(42)
        features_to_drift = np.random.choice(n_features, n_features_to_drift, replace=False)
        
        for feat_idx in features_to_drift:
            shift = np.random.normal(0.5, 0.2)
            noise = np.random.normal(shift, 0.3, len(production_data))
            production_data[:, feat_idx] += noise
        
        logger.info(f"Drift simulado en {len(features_to_drift)} features")
    else:
        logger.info(f"Cargando datos de producción desde {production_data_path}")
        if not production_data_path.exists():
            logger.error(f"Archivo no encontrado: {production_data_path}")
            raise typer.Exit(1)
        prod_df = pd.read_csv(production_data_path)
        transformer = joblib.load(transformer_path)
        production_data = transformer.transform(prod_df)
    
    # Detectar drift
    logger.info(f"\nDetectando drift usando método: {method}")
    detector = DriftDetector(
        reference_data=reference_data,
        method=method,
        threshold=threshold
    )
    
    drift_results = detector.detect_drift(production_data, return_details=True)
    
    # Mostrar resultados
    logger.info("\n" + "=" * 70)
    logger.info("RESULTADOS DE DETECCIÓN")
    logger.info("=" * 70)
    logger.info(f"Score máximo: {drift_results['max_drift_score']:.4f}")
    logger.info(f"Score promedio: {drift_results['mean_drift_score']:.4f}")
    logger.info(f"Features con drift: {drift_results['features_with_drift']}/{drift_results['total_features']}")
    
    severity = detector.get_drift_severity(drift_results['max_drift_score'])
    logger.info(f"Severidad: {severity.upper()}")
    
    # Registrar en MLflow
    logger.info("\n" + "-" * 70)
    logger.info("Registrando resultados en MLflow...")
    logger.info("-" * 70)
    
    log_drift_to_mlflow(
        drift_results=drift_results,
        reference_data_shape=reference_data.shape,
        production_data_shape=production_data.shape,
        method=method,
        threshold=threshold,
        experiment_name=experiment_name
    )
    
    logger.info("\n" + "=" * 70)
    logger.info("✅ Proceso completado")
    logger.info("=" * 70)
    logger.info("Para ver los resultados en MLflow UI:")
    logger.info("  mlflow ui")
    logger.info("  Luego abre http://localhost:5000 en tu navegador")


if __name__ == "__main__":
    app()

