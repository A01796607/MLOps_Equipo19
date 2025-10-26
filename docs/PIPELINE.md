# Pipeline de Machine Learning - Orientado a Objetos

Este documento describe cómo usar las clases orientadas a objetos para el pipeline de machine learning basado en el notebook `MLOPS_FASE1.ipynb`.

## Estructura del Proyecto

```
MLOps_Equipo19/
├── src/                          # Clases principales del pipeline
│   ├── __init__.py              # Exporta las clases principales
│   ├── data_processor.py        # Clase DataProcessor
│   ├── feature_transformer.py   # Clase FeatureTransformer
│   ├── model_trainer.py         # Clase ModelTrainer
│   └── train_pipeline.py        # Ejemplo de uso completo
├── mlops/                        # Scripts CLI del pipeline
│   ├── config.py                # Configuración de paths
│   ├── dataset.py               # CLI para procesar datos
│   ├── features.py              # CLI para transformar features
│   └── modeling/
│       ├── train.py             # CLI para entrenar modelos
│       └── predict.py           # CLI para hacer predicciones
└── docs/
    └── PIPELINE.md              # Esta documentación
```

## Estructura del Pipeline

El pipeline está compuesto por tres clases principales ubicadas en `src/`:

1. **DataProcessor**: Limpieza y procesamiento de datos
2. **FeatureTransformer**: Transformación de features (encoding, scaling, PCA)
3. **ModelTrainer**: Entrenamiento y optimización de modelos

## Uso de las Clases

Las clases están ubicadas en la carpeta `src/` del proyecto.

### 1. DataProcessor

Clase para procesar y limpiar datasets.

```python
from src import DataProcessor

# Inicializar processor
processor = DataProcessor(iqr_factor=1.5)

# Remover outliers usando IQR
df_clean = processor.remove_outliers_iqr(df)

# Obtener estadísticas
stats_numeric = processor.get_stats_numeric(df)
stats_categorical = processor.get_stats_categorical(df)
missing_pct = processor.get_missing_percentage(df)
```

**Métodos principales:**
- `remove_outliers_iqr()`: Elimina outliers usando el método IQR
- `get_stats_numeric()`: Estadísticas descriptivas para columnas numéricas
- `get_stats_categorical()`: Estadísticas para columnas categóricas
- `get_missing_percentage()`: Porcentaje de valores faltantes

### 2. FeatureTransformer

Clase para transformar features (encoding, scaling, PCA).

```python
from src import FeatureTransformer

# Inicializar transformer
transformer = FeatureTransformer(use_pca=True, n_components=50)

# Codificar labels
y_encoded = transformer.encode_labels(y)

# Transformar features
X_train_transformed, X_test_transformed = transformer.fit_transform(X_train, X_test)

# Decodificar labels
y_decoded = transformer.decode_labels(y_encoded)
```

**Métodos principales:**
- `encode_labels()`: Codifica labels categóricas
- `decode_labels()`: Decodifica labels a clases originales
- `fit_transform()`: Ajusta y transforma train y test
- `transform()`: Transforma nuevos datos usando transformers ajustados

### 3. ModelTrainer

Clase para entrenar y optimizar modelos.

```python
from src import ModelTrainer

# Inicializar trainer
trainer = ModelTrainer(random_state=42)

# Entrenar Random Forest
rf_model = trainer.train_random_forest(X_train, y_train)

# Entrenar LightGBM
lgb_model = trainer.train_lightgbm(X_train, y_train, X_val, y_val)

# Optimizar LightGBM con Optuna
lgb_optimized, best_params = trainer.optimize_lightgbm(X_train, y_train, n_trials=50)

# Hacer predicciones
y_pred = trainer.predict('random_forest', X_test)

# Evaluar modelo
trainer.print_report(y_test, y_pred, model_name="Random Forest")
```

**Métodos principales:**
- `train_random_forest()`: Entrena Random Forest
- `train_lightgbm()`: Entrena LightGBM básico
- `optimize_lightgbm()`: Optimiza LightGBM con Optuna
- `predict()`: Hace predicciones con un modelo entrenado
- `evaluate()`: Evalúa predicciones y retorna métricas
- `print_report()`: Imprime reporte de clasificación

## Pipeline Completo

### Usando las clases directamente

Ver ejemplo completo en `src/train_pipeline.py`:

```python
from src import DataProcessor, FeatureTransformer, ModelTrainer

# 1. Cargar datos
df = pd.read_csv("data/raw/turkis_music_emotion_original.csv")

# 2. Procesar datos
processor = DataProcessor()
df_clean = processor.remove_outliers_iqr(df)

# 3. Preparar features
X = df_clean.drop(columns=["Class"])
y = df_clean["Class"]

# 4. Dividir datos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Transformar features
transformer = FeatureTransformer(use_pca=True, n_components=50)
y_train_encoded = transformer.encode_labels(y_train)
y_test_encoded = transformer.encode_labels(y_test)
X_train_transformed, X_test_transformed = transformer.fit_transform(X_train, X_test)

# 6. Entrenar modelos
trainer = ModelTrainer()
rf_model = trainer.train_random_forest(X_train_transformed, y_train_encoded)
y_pred = trainer.predict('random_forest', X_test_transformed)

# 7. Evaluar
trainer.print_report(y_test_encoded, y_pred, model_name="Random Forest")
```

### Usando los scripts CLI

El pipeline también se puede ejecutar usando los scripts CLI:

```bash
# 1. Procesar dataset
python -m mlops.dataset

# 2. Transformar features
python -m mlops.features

# 3. Entrenar modelo
python -m mlops.modeling.train --model-type random_forest

# 4. Hacer predicciones
python -m mlops.modeling.predict
```

## Modelos Disponibles

1. **Random Forest**: Clasificador de árboles aleatorios
2. **LightGBM**: Clasificador basado en gradient boosting
3. **LightGBM Optimizado**: LightGBM con optimización de hiperparámetros usando Optuna

## Configuración

Los paths y configuraciones están definidos en `mlops/config.py`:

- `RAW_DATA_DIR`: Datos crudos
- `PROCESSED_DATA_DIR`: Datos procesados
- `MODELS_DIR`: Modelos entrenados
- `REPORTS_DIR`: Reportes y figuras

## Ventajas del Enfoque OOP

1. **Modularidad**: Cada clase tiene una responsabilidad específica
2. **Reutilización**: Las clases se pueden usar en diferentes contextos
3. **Mantenibilidad**: Más fácil de mantener y extender
4. **Testabilidad**: Fácil de escribir tests unitarios
5. **Encapsulación**: Estado y comportamiento juntos

## Comparación con el Notebook

El código del notebook se ha refactorizado en:

| Notebook | OOP |
|----------|-----|
| Procesamiento manual de outliers | `DataProcessor.remove_outliers_iqr()` |
| Codificación manual de labels | `FeatureTransformer.encode_labels()` |
| Scaling manual | `FeatureTransformer` (internamente) |
| PCA manual | `FeatureTransformer` (internamente) |
| Entrenamiento de modelos | `ModelTrainer.train_*()` |
| Optimización Optuna | `ModelTrainer.optimize_lightgbm()` |

