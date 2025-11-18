"""
Script para evaluar F1-score del modelo en producción y compararlo con entrenamiento.

Este script permite:
- Evaluar F1-score en datos de producción (si hay labels)
- Comparar con F1 de entrenamiento
- Registrar métricas en MLflow
- Detectar degradación de rendimiento

Ejemplo de uso:
    python scripts/evaluate_f1.py \
        --model-path models/random_forest.pkl \
        --transformer-path data/processed/transformer.pkl \
        --production-data data/production/sample.csv \
        --production-labels data/production/labels.csv \
        --reference-f1 0.84
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
from typing import Optional
from datetime import datetime
import mlflow

from mlops.config import MODELS_DIR, PROCESSED_DATA_DIR
from src.feature_transformer import FeatureTransformer
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix
)

app = typer.Typer()


def evaluate_model_f1(
    model_path: Path,
    transformer_path: Path,
    production_features: np.ndarray,
    production_labels: Optional[np.ndarray] = None,
    reference_f1: Optional[float] = None,
    class_names: Optional[list] = None
) -> dict:
    """
    Evalúa F1-score del modelo en datos de producción.
    
    Args:
        model_path: Path al modelo entrenado
        transformer_path: Path al transformer
        production_features: Features de producción (ya transformadas)
        production_labels: Labels de producción (opcional)
        reference_f1: F1-score de referencia (entrenamiento)
        class_names: Nombres de las clases
        
    Returns:
        dict: Diccionario con métricas calculadas
    """
    # Cargar modelo
    logger.info(f"Cargando modelo desde {model_path}")
    model = joblib.load(model_path)
    
    # Hacer predicciones
    logger.info("Generando predicciones...")
    if hasattr(model, 'predict'):
        y_pred = model.predict(production_features)
        
        # Obtener probabilidades si están disponibles
        y_pred_proba = None
        if hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(production_features)
    else:
        # Para LightGBM booster
        y_pred_proba = model.predict(production_features)
        y_pred = y_pred_proba.argmax(axis=1) if y_pred_proba.ndim > 1 else (y_pred_proba > 0.5).astype(int)
    
    # Calcular métricas si hay labels
    metrics = {
        'n_samples': len(production_features),
        'predictions_generated': True
    }
    
    if production_labels is not None:
        logger.info("Calculando métricas F1...")
        
        # Asegurar que son arrays 1D de enteros
        y_true = np.asarray(production_labels).ravel().astype(int)
        y_pred = np.asarray(y_pred).ravel().astype(int)
        
        # Métricas generales
        metrics['f1_macro'] = float(f1_score(y_true, y_pred, average='macro', zero_division=0))
        metrics['f1_weighted'] = float(f1_score(y_true, y_pred, average='weighted', zero_division=0))
        metrics['f1_micro'] = float(f1_score(y_true, y_pred, average='micro', zero_division=0))
        metrics['accuracy'] = float(accuracy_score(y_true, y_pred))
        metrics['precision_macro'] = float(precision_score(y_true, y_pred, average='macro', zero_division=0))
        metrics['recall_macro'] = float(recall_score(y_true, y_pred, average='macro', zero_division=0))
        
        # F1 por clase si hay class_names
        if class_names:
            f1_per_class = {}
            for i, class_name in enumerate(class_names):
                try:
                    f1_per_class[class_name] = float(f1_score(
                        y_true == i,
                        y_pred == i,
                        zero_division=0
                    ))
                except Exception as e:
                    logger.warning(f"Error calculando F1 para clase {class_name}: {e}")
            metrics['f1_per_class'] = f1_per_class
        
        # Comparar con referencia
        if reference_f1 is not None:
            f1_diff = metrics['f1_macro'] - reference_f1
            f1_degradation = f1_diff < 0
            metrics['f1_difference'] = float(f1_diff)
            metrics['f1_degradation'] = f1_degradation
            metrics['f1_degradation_pct'] = float((f1_diff / reference_f1) * 100) if reference_f1 > 0 else 0.0
            
            logger.info(f"F1 de referencia: {reference_f1:.4f}")
            logger.info(f"F1 en producción: {metrics['f1_macro']:.4f}")
            logger.info(f"Diferencia: {f1_diff:+.4f} ({metrics['f1_degradation_pct']:+.2f}%)")
            
            if f1_degradation:
                logger.warning(f"⚠️  Degradación detectada: F1 disminuyó en {abs(f1_diff):.4f}")
            else:
                logger.success(f"✅ F1 mejoró o se mantuvo: {f1_diff:+.4f}")
        
        # Classification report
        metrics['classification_report'] = classification_report(
            y_true, y_pred,
            target_names=class_names,
            output_dict=True,
            zero_division=0
        )
        
        # Confusion matrix
        metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred).tolist()
    else:
        logger.warning("No se proporcionaron labels de producción. Solo se generaron predicciones.")
    
    return metrics


@app.command()
def main(
    model_path: Path = typer.Option(
        MODELS_DIR / "random_forest.pkl",
        "--model-path",
        "-m",
        help="Path al modelo entrenado"
    ),
    transformer_path: Path = typer.Option(
        PROCESSED_DATA_DIR / "transformer.pkl",
        "--transformer-path",
        "-t",
        help="Path al transformer"
    ),
    production_data_path: Path = typer.Option(
        ...,
        "--production-data",
        "-p",
        help="Path a datos de producción (features)"
    ),
    production_labels_path: Optional[Path] = typer.Option(
        None,
        "--production-labels",
        "-l",
        help="Path a labels de producción (opcional, necesario para calcular F1)"
    ),
    reference_f1: Optional[float] = typer.Option(
        None,
        "--reference-f1",
        "-r",
        help="F1-score de referencia (del entrenamiento)"
    ),
    experiment_name: str = typer.Option(
        "F1_Evaluation",
        "--experiment",
        "-e",
        help="Nombre del experimento en MLflow"
    ),
    log_to_mlflow: bool = typer.Option(
        True,
        "--log-mlflow/--no-log-mlflow",
        help="Registrar métricas en MLflow"
    ),
    output_path: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Path para guardar reporte JSON"
    )
):
    """
    Evalúa F1-score del modelo en datos de producción.
    """
    logger.info("=" * 70)
    logger.info("Evaluación de F1-Score en Producción")
    logger.info("=" * 70)
    
    # Cargar datos de producción
    logger.info(f"Cargando datos de producción desde {production_data_path}")
    if not production_data_path.exists():
        logger.error(f"Archivo no encontrado: {production_data_path}")
        raise typer.Exit(1)
    
    prod_df = pd.read_csv(production_data_path)
    logger.info(f"Datos cargados: {prod_df.shape}")
    
    # Cargar transformer
    logger.info(f"Cargando transformer desde {transformer_path}")
    if not transformer_path.exists():
        logger.error("Transformer no encontrado")
        raise typer.Exit(1)
    
    transformer = joblib.load(transformer_path)
    
    # Transformar features
    logger.info("Transformando features...")
    production_features = transformer.transform(prod_df)
    
    # Cargar labels si están disponibles
    production_labels = None
    if production_labels_path and production_labels_path.exists():
        logger.info(f"Cargando labels desde {production_labels_path}")
        labels_df = pd.read_csv(production_labels_path)
        if 'Class' in labels_df.columns:
            labels_series = labels_df['Class']
        elif len(labels_df.columns) == 1:
            labels_series = labels_df.iloc[:, 0]
        else:
            logger.error("No se pudo identificar la columna de labels")
            raise typer.Exit(1)
        
        # Codificar labels si es necesario
        if hasattr(transformer, 'label_encoder'):
            try:
                production_labels = transformer.label_encoder.transform(labels_series)
            except:
                production_labels = transformer.encode_labels(labels_series)
        else:
            # Asumir que ya están codificados o usar LabelEncoder
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            production_labels = le.fit_transform(labels_series)
        
        logger.info(f"Labels cargados: {len(production_labels)} muestras")
    else:
        logger.warning("No se proporcionaron labels. Solo se generarán predicciones.")
    
    # Obtener nombres de clases del transformer
    class_names = None
    if hasattr(transformer, 'label_encoder'):
        class_names = transformer.label_encoder.classes_.tolist()
        logger.info(f"Clases detectadas: {class_names}")
    
    # Evaluar modelo
    logger.info("\n" + "-" * 70)
    logger.info("EVALUANDO MODELO")
    logger.info("-" * 70)
    
    metrics = evaluate_model_f1(
        model_path=model_path,
        transformer_path=transformer_path,
        production_features=production_features,
        production_labels=production_labels,
        reference_f1=reference_f1,
        class_names=class_names
    )
    
    # Mostrar resultados
    logger.info("\n" + "=" * 70)
    logger.info("RESULTADOS DE EVALUACIÓN")
    logger.info("=" * 70)
    
    if production_labels is not None:
        logger.info(f"\n📊 MÉTRICAS:")
        logger.info(f"  F1-Score (Macro):     {metrics['f1_macro']:.4f}")
        logger.info(f"  F1-Score (Weighted):  {metrics['f1_weighted']:.4f}")
        logger.info(f"  Accuracy:             {metrics['accuracy']:.4f}")
        logger.info(f"  Precision (Macro):    {metrics['precision_macro']:.4f}")
        logger.info(f"  Recall (Macro):       {metrics['recall_macro']:.4f}")
        
        if 'f1_per_class' in metrics:
            logger.info(f"\n📈 F1 POR CLASE:")
            for class_name, f1_val in metrics['f1_per_class'].items():
                logger.info(f"  {class_name}: {f1_val:.4f}")
        
        if reference_f1 is not None:
            logger.info(f"\n🔍 COMPARACIÓN CON REFERENCIA:")
            logger.info(f"  F1 Referencia:       {reference_f1:.4f}")
            logger.info(f"  F1 Producción:       {metrics['f1_macro']:.4f}")
            logger.info(f"  Diferencia:          {metrics['f1_difference']:+.4f}")
            logger.info(f"  Cambio porcentual:   {metrics['f1_degradation_pct']:+.2f}%")
            
            if metrics['f1_degradation']:
                logger.warning(f"\n⚠️  DEGRADACIÓN DETECTADA")
                logger.warning(f"   El modelo ha perdido {abs(metrics['f1_difference']):.4f} puntos de F1")
                logger.warning(f"   Considera retrenar el modelo o investigar data drift")
            else:
                logger.success(f"\n✅ RENDIMIENTO ACEPTABLE")
                logger.info(f"   El modelo mantiene o mejora su rendimiento")
        
        # Classification report detallado
        logger.info(f"\n📋 REPORTE DE CLASIFICACIÓN:")
        report = metrics['classification_report']
        for class_name in class_names if class_names else report.keys():
            if class_name in report and isinstance(report[class_name], dict):
                class_metrics = report[class_name]
                logger.info(f"  {class_name}:")
                logger.info(f"    Precision: {class_metrics.get('precision', 0):.4f}")
                logger.info(f"    Recall:    {class_metrics.get('recall', 0):.4f}")
                logger.info(f"    F1-Score:  {class_metrics.get('f1-score', 0):.4f}")
    else:
        logger.info(f"\n✅ Predicciones generadas para {metrics['n_samples']} muestras")
        logger.info("   Para calcular F1-score, proporciona labels con --production-labels")
    
    # Registrar en MLflow
    if log_to_mlflow and production_labels is not None:
        logger.info("\n" + "-" * 70)
        logger.info("REGISTRANDO EN MLFLOW")
        logger.info("-" * 70)
        
        mlflow.set_experiment(experiment_name)
        
        with mlflow.start_run(run_name=f"f1_evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            # Parámetros
            mlflow.log_param("model_path", str(model_path))
            mlflow.log_param("production_samples", metrics['n_samples'])
            if reference_f1 is not None:
                mlflow.log_param("reference_f1", reference_f1)
            
            # Métricas
            mlflow.log_metric("f1_macro", metrics['f1_macro'])
            mlflow.log_metric("f1_weighted", metrics['f1_weighted'])
            mlflow.log_metric("f1_micro", metrics['f1_micro'])
            mlflow.log_metric("accuracy", metrics['accuracy'])
            mlflow.log_metric("precision_macro", metrics['precision_macro'])
            mlflow.log_metric("recall_macro", metrics['recall_macro'])
            
            if reference_f1 is not None:
                mlflow.log_metric("f1_difference", metrics['f1_difference'])
                mlflow.log_metric("f1_degradation_pct", metrics['f1_degradation_pct'])
                mlflow.set_tag("degradation_detected", str(metrics['f1_degradation']))
            
            # F1 por clase
            if 'f1_per_class' in metrics:
                for class_name, f1_val in metrics['f1_per_class'].items():
                    mlflow.log_metric(f"f1_{class_name}", f1_val)
            
            logger.success(f"✅ Métricas registradas en MLflow")
            logger.info(f"   Run ID: {mlflow.active_run().info.run_id}")
            logger.info(f"   Ver resultados: mlflow ui")
    
    # Guardar reporte JSON
    if output_path:
        import json
        # Preparar JSON serializable
        report_json = {
            'timestamp': datetime.now().isoformat(),
            'model_path': str(model_path),
            'metrics': {k: v for k, v in metrics.items() if k not in ['classification_report', 'confusion_matrix']}
        }
        
        # Agregar classification report y confusion matrix
        if 'classification_report' in metrics:
            report_json['classification_report'] = metrics['classification_report']
        if 'confusion_matrix' in metrics:
            report_json['confusion_matrix'] = metrics['confusion_matrix']
        
        with open(output_path, 'w') as f:
            json.dump(report_json, f, indent=2)
        logger.info(f"\n💾 Reporte guardado en: {output_path}")
    
    logger.info("\n" + "=" * 70)
    logger.info("✅ Evaluación completada")
    logger.info("=" * 70)
    
    return metrics


if __name__ == "__main__":
    app()

