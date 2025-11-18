"""
Script para generar datos de producción de ejemplo para evaluación F1.

Este script crea datos de producción simulados a partir de los datos de entrenamiento,
útil para probar el sistema de evaluación F1-score.
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
import joblib
from loguru import logger
import typer
from mlops.config import RAW_DATA_DIR, PROCESSED_DATA_DIR

app = typer.Typer()


@app.command()
def main(
    output_dir: Path = typer.Option(
        Path("data/production"),
        "--output-dir",
        "-o",
        help="Directorio donde guardar datos de producción"
    ),
    n_samples: int = typer.Option(
        50,
        "--n-samples",
        "-n",
        help="Número de muestras a generar"
    ),
    noise_level: float = typer.Option(
        0.1,
        "--noise",
        help="Nivel de ruido (0-1)"
    ),
    drift_factor: float = typer.Option(
        0.0,
        "--drift",
        help="Factor de drift (0 = sin drift, >0 = con drift)"
    ),
    use_test_split: bool = typer.Option(
        True,
        "--use-test/--use-train",
        help="Usar datos de test (True) o train (False)"
    )
):
    """
    Genera datos de producción simulados para evaluación F1.
    """
    logger.info("=" * 70)
    logger.info("Generación de Datos de Producción")
    logger.info("=" * 70)
    
    # Crear directorio de salida
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Directorio de salida: {output_dir}")
    
    # Cargar datos originales
    raw_data_path = RAW_DATA_DIR / "turkis_music_emotion_original.csv"
    if not raw_data_path.exists():
        logger.error(f"Archivo no encontrado: {raw_data_path}")
        logger.info("Asegúrate de que los datos raw existen")
        raise typer.Exit(1)
    
    logger.info(f"Cargando datos desde {raw_data_path}")
    df = pd.read_csv(raw_data_path)
    logger.info(f"Datos cargados: {df.shape}")
    
    # Verificar que existe columna Class
    if 'Class' not in df.columns:
        logger.error("Columna 'Class' no encontrada en los datos")
        raise typer.Exit(1)
    
    # Separar features y labels
    X = df.drop('Class', axis=1)
    y = df['Class']
    
    # Muestrear datos
    logger.info(f"Generando {n_samples} muestras...")
    n_available = len(df)
    n_to_sample = min(n_samples, n_available)
    
    # Muestrear aleatoriamente
    np.random.seed(42)
    indices = np.random.choice(n_available, size=n_to_sample, replace=False)
    
    X_prod = X.iloc[indices].copy()
    y_prod = y.iloc[indices].copy()
    
    # Aplicar drift si está especificado
    if drift_factor > 0:
        logger.info(f"Aplicando drift con factor {drift_factor}")
        # Seleccionar algunas features para aplicar drift
        n_features_to_drift = max(1, int(len(X_prod.columns) * 0.3))
        features_to_drift = np.random.choice(
            X_prod.columns,
            size=n_features_to_drift,
            replace=False
        )
        
        for col in features_to_drift:
            # Aplicar drift multiplicativo con ruido
            drift = 1 + drift_factor * np.random.randn(len(X_prod))
            X_prod[col] = X_prod[col] * drift
    
    # Añadir ruido
    if noise_level > 0:
        logger.info(f"Añadiendo ruido con nivel {noise_level}")
        # Calcular desviación estándar de cada columna para ruido proporcional
        noise_std = X_prod.std() * noise_level
        noise = np.random.randn(*X_prod.shape) * noise_std.values
        X_prod = X_prod + noise
    
    # Guardar features de producción
    features_path = output_dir / "features.csv"
    X_prod.to_csv(features_path, index=False)
    logger.success(f"✅ Features guardadas en: {features_path}")
    
    # Guardar labels de producción
    labels_path = output_dir / "labels.csv"
    labels_df = pd.DataFrame({'Class': y_prod})
    labels_df.to_csv(labels_path, index=False)
    logger.success(f"✅ Labels guardadas en: {labels_path}")
    
    # Guardar datos completos (features + labels) para referencia
    full_path = output_dir / "production_data.csv"
    prod_df = X_prod.copy()
    prod_df['Class'] = y_prod
    prod_df.to_csv(full_path, index=False)
    logger.success(f"✅ Datos completos guardados en: {full_path}")
    
    # Resumen
    logger.info("\n" + "=" * 70)
    logger.info("RESUMEN")
    logger.info("=" * 70)
    logger.info(f"Archivos generados:")
    logger.info(f"  - Features: {features_path}")
    logger.info(f"  - Labels:   {labels_path}")
    logger.info(f"  - Completo: {full_path}")
    logger.info(f"\nEstadísticas:")
    logger.info(f"  - Muestras: {len(X_prod)}")
    logger.info(f"  - Features: {len(X_prod.columns)}")
    logger.info(f"  - Clases: {y_prod.value_counts().to_dict()}")
    
    if drift_factor > 0:
        logger.info(f"  - Drift aplicado: {drift_factor}")
    if noise_level > 0:
        logger.info(f"  - Ruido aplicado: {noise_level}")
    
    logger.info("\n" + "=" * 70)
    logger.info("✅ Datos de producción generados exitosamente")
    logger.info("=" * 70)
    logger.info("\n💡 Ahora puedes evaluar F1-score:")
    logger.info(f"   python scripts/evaluate_f1.py \\")
    logger.info(f"       --production-data {features_path} \\")
    logger.info(f"       --production-labels {labels_path} \\")
    logger.info(f"       --reference-f1 0.84")


if __name__ == "__main__":
    app()

