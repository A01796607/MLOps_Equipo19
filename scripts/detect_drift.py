"""
Script para detectar data drift en datos de producción.

Ejemplo de uso:
    python scripts/detect_drift.py \
        --reference-data data/processed/features.csv \
        --production-data data/production/sample_data.csv \
        --transformer-path data/processed/transformer.pkl \
        --method psi \
        --threshold 0.25
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
import typer
from loguru import logger
from typing import Optional

from src.drift_detector import DriftDetector
from mlops.config import PROCESSED_DATA_DIR

app = typer.Typer()


@app.command()
def main(
    reference_data_path: Path = typer.Option(
        PROCESSED_DATA_DIR / "features.csv",
        "--reference-data",
        "-r",
        help="Path a los datos de referencia (train)"
    ),
    production_data_path: Optional[Path] = typer.Option(
        None,
        "--production-data",
        "-p",
        help="Path a los datos de producción. Si no se proporciona, se simularán datos."
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
    output_path: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Path para guardar los resultados (JSON)"
    ),
    simulate_production: bool = typer.Option(
        False,
        "--simulate",
        help="Simular datos de producción con drift artificial"
    )
):
    """
    Detecta data drift comparando datos de producción con datos de referencia.
    """
    logger.info("=" * 60)
    logger.info("Data Drift Detection")
    logger.info("=" * 60)
    
    # Cargar datos de referencia
    logger.info(f"Cargando datos de referencia desde {reference_data_path}")
    if not reference_data_path.exists():
        logger.error(f"Archivo de referencia no encontrado: {reference_data_path}")
        raise typer.Exit(1)
    
    reference_df = pd.read_csv(reference_data_path)
    logger.info(f"Datos de referencia cargados: {reference_df.shape}")
    
    # Cargar transformer
    logger.info(f"Cargando transformer desde {transformer_path}")
    if not transformer_path.exists():
        logger.error(f"Transformer no encontrado: {transformer_path}")
        logger.info("Nota: El transformer debe ser generado primero ejecutando el pipeline de entrenamiento")
        raise typer.Exit(1)
    
    # Inicializar detector
    detector = DriftDetector(
        transformer_path=transformer_path,
        method=method,
        threshold=threshold
    )
    
    # Establecer datos de referencia
    detector.set_reference_data(reference_df)
    
    # Preparar datos de producción
    if simulate_production or production_data_path is None:
        logger.info("Simulando datos de producción...")
        # Simular datos con algún drift
        production_df = reference_df.copy()
        
        # Introducir drift artificial en algunas features
        n_features = production_df.shape[1]
        n_features_to_drift = int(n_features * 0.2)  # 20% de las features
        
        np.random.seed(42)
        features_to_drift = np.random.choice(n_features, n_features_to_drift, replace=False)
        
        for feat_idx in features_to_drift:
            # Agregar ruido y shift
            production_df.iloc[:, feat_idx] += np.random.normal(0.5, 0.2, len(production_df))
        
        logger.info(f"Drift simulado en {len(features_to_drift)} features")
    else:
        logger.info(f"Cargando datos de producción desde {production_data_path}")
        if not production_data_path.exists():
            logger.error(f"Archivo de producción no encontrado: {production_data_path}")
            raise typer.Exit(1)
        production_df = pd.read_csv(production_data_path)
        logger.info(f"Datos de producción cargados: {production_df.shape}")
    
    # Detectar drift
    logger.info(f"\nDetectando drift usando método: {method}")
    logger.info(f"Umbral: {threshold}")
    logger.info("-" * 60)
    
    drift_results = detector.detect_drift(production_df, return_details=True)
    
    # Mostrar resultados
    logger.info("\n📊 RESULTADOS DE DETECCIÓN DE DRIFT")
    logger.info("=" * 60)
    logger.info(f"Método utilizado: {method.upper()}")
    logger.info(f"Score máximo de drift: {drift_results['max_drift_score']:.4f}")
    logger.info(f"Score promedio de drift: {drift_results['mean_drift_score']:.4f}")
    logger.info(f"Score mínimo de drift: {drift_results['min_drift_score']:.4f}")
    logger.info(f"Features con drift detectado: {drift_results['features_with_drift']}/{drift_results['total_features']}")
    logger.info(f"Porcentaje de features con drift: {drift_results['drift_percentage']:.2f}%")
    
    # Severidad
    severity = detector.get_drift_severity(drift_results['max_drift_score'])
    logger.info(f"\n🚨 SEVERIDAD DEL DRIFT: {severity.upper()}")
    
    if drift_results['drift_detected']:
        logger.warning("⚠️  DRIFT DETECTADO - Se recomienda evaluar el modelo")
        
        # Mostrar top features con drift
        feature_details = drift_results['feature_details']
        sorted_features = sorted(
            feature_details.items(),
            key=lambda x: x[1]['score'],
            reverse=True
        )
        
        logger.info("\n📈 Top 10 features con mayor drift:")
        logger.info("-" * 60)
        logger.info(f"{'Feature':<10} {'Score':<12} {'Drift':<8} {'Ref Mean':<12} {'Prod Mean':<12}")
        logger.info("-" * 60)
        
        for feat_idx, details in sorted_features[:10]:
            drift_status = "✅" if not details['drift_detected'] else "🚨"
            logger.info(
                f"{feat_idx:<10} {details['score']:<12.4f} {drift_status:<8} "
                f"{details['reference_mean']:<12.4f} {details['production_mean']:<12.4f}"
            )
        
        # Recomendaciones
        logger.info("\n💡 RECOMENDACIONES:")
        if severity == 'high':
            logger.warning("  - Retrenar el modelo inmediatamente")
            logger.warning("  - Recopilar nuevos datos de entrenamiento")
            logger.warning("  - Validar el modelo antes de desplegar")
        elif severity == 'moderate':
            logger.info("  - Evaluar el rendimiento del modelo en producción")
            logger.info("  - Considerar retrenar si el rendimiento ha degradado")
            logger.info("  - Monitorear más frecuentemente")
        else:
            logger.info("  - Continuar monitoreando")
            logger.info("  - No se requiere acción inmediata")
    else:
        logger.success("✅ No se detectó drift significativo")
        logger.info("  - Los datos de producción son consistentes con los de entrenamiento")
    
    # Guardar resultados si se especifica output
    if output_path:
        import json
        # Convertir numpy types a Python types para JSON
        results_json = {
            'method': method,
            'threshold': threshold,
            'max_drift_score': drift_results['max_drift_score'],
            'mean_drift_score': drift_results['mean_drift_score'],
            'min_drift_score': drift_results['min_drift_score'],
            'features_with_drift': drift_results['features_with_drift'],
            'total_features': drift_results['total_features'],
            'drift_detected': drift_results['drift_detected'],
            'drift_percentage': drift_results['drift_percentage'],
            'severity': severity,
            'feature_details': {
                str(k): {
                    'score': v['score'],
                    'drift_detected': v['drift_detected'],
                    'reference_mean': v['reference_mean'],
                    'production_mean': v['production_mean']
                }
                for k, v in drift_results['feature_details'].items()
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(results_json, f, indent=2)
        logger.info(f"\n💾 Resultados guardados en {output_path}")
    
    logger.info("\n" + "=" * 60)


if __name__ == "__main__":
    app()

