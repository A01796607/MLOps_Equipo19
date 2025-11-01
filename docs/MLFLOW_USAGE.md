# Guía de Uso de MLflow

Esta guía explica cómo usar MLflow para el seguimiento de experimentos, registro de modelos y visualización de resultados.

## Instalación

MLflow ya está incluido en `requirements.txt`:

```bash
pip install -r requirements.txt
```

## Inicio Rápido

### 1. Entrenar modelos con seguimiento MLflow

```bash
python src/train_with_mlflow.py
```

Este script entrenará los modelos (Random Forest y LightGBM) y registrará automáticamente:
- Parámetros del modelo
- Métricas de entrenamiento y prueba
- Matrices de confusión
- Reportes de clasificación
- Modelos versionados
- Gráficas y artefactos

### 2. Iniciar la interfaz de MLflow

En una terminal separada:

```bash
mlflow ui
```

O si prefieres especificar un puerto:

```bash
mlflow ui --port 5000
```

Luego abre tu navegador en: `http://localhost:5000`

### 3. Ver experimentos y comparar modelos

En la interfaz web de MLflow podrás:
- Ver todos los experimentos ejecutados
- Comparar diferentes ejecuciones (runs)
- Ver métricas en gráficas
- Descargar modelos registrados
- Ver parámetros y configuraciones

## Uso Programático

### Clase MLflowManager

La clase `MLflowManager` proporciona una interfaz completa para gestionar experimentos:

```python
from mlops.MLFLow_Equipo19 import MLflowManager

# Inicializar manager
mlflow_manager = MLflowManager(
    experiment_name="MiExperimento",
    tracking_uri=None  # Usa almacenamiento local
)

# Iniciar un run
mlflow_manager.start_run(run_name="mi_ejecucion")

# Registrar parámetros
mlflow_manager.log_params({
    "learning_rate": 0.01,
    "n_estimators": 100,
    "max_depth": 5
})

# Registrar métricas
mlflow_manager.log_metrics({
    "accuracy": 0.95,
    "f1_score": 0.93
})

# Registrar modelo
mlflow_manager.log_model(
    model=mi_modelo,
    model_name="mi_modelo",
    model_type="sklearn",
    registered_model_name="MiModeloRegistrado"
)

# Finalizar run
mlflow_manager.end_run()
```

### Función track_training_experiment

Función de alto nivel que registra todo un experimento completo:

```python
from mlops.MLFLow_Equipo19 import track_training_experiment

run_info = track_training_experiment(
    model=modelo_entrenado,
    model_name="random_forest",
    model_type="sklearn",
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
    params=parametros_del_modelo,
    class_names=["angry", "happy", "relax", "sad"],
    transformer=transformer,
    registered_model_name="MusicEmotions-RandomForest"
)
```

## Características Implementadas

### ✅ Registro y Versionado de Experimentos

- Cada ejecución del pipeline crea un nuevo "run" en MLflow
- Cada run tiene un ID único y timestamp
- Los experimentos se organizan por nombre
- Soporte para tags y metadatos personalizados

### ✅ Logging de Parámetros

Se registran automáticamente:
- Parámetros del modelo (n_estimators, learning_rate, etc.)
- Parámetros de preprocesamiento (use_pca, n_components, etc.)
- Información del dataset (n_samples, n_features, n_classes)

### ✅ Logging de Métricas

Métricas registradas:

**Métricas generales:**
- `test_accuracy`: Precisión en el conjunto de prueba
- `test_precision_macro`: Precisión macro promedio
- `test_recall_macro`: Recall macro promedio
- `test_f1_macro`: F1-score macro promedio
- `test_precision_weighted`: Precisión ponderada
- `test_recall_weighted`: Recall ponderado
- `test_f1_weighted`: F1-score ponderado

**Métricas por clase:**
- `test_precision_{class_name}`: Precisión por clase
- `test_recall_{class_name}`: Recall por clase
- `test_f1_{class_name}`: F1-score por clase

Las mismas métricas se registran para el conjunto de entrenamiento (con prefijo `train_`).

### ✅ Registro de Modelos

- Modelos guardados con versionado automático
- Soporte para modelos de scikit-learn y LightGBM
- Modelos pueden registrarse en el Model Registry
- Incluye signature (firma) del modelo para validación
- Incluye input_example para pruebas

### ✅ Artefactos

Se registran los siguientes artefactos:
- Matriz de confusión (train y test) en formato CSV
- Reporte de clasificación (train y test) en formato CSV
- Gráficas de confusión en formato PNG
- Transformador (transformer.pkl) para preprocesamiento
- Gráficas de PCA (si se usa)

### ✅ Visualización

La interfaz web de MLflow proporciona:
- Gráficas interactivas de métricas
- Comparación de múltiples runs
- Tablas comparativas de parámetros y métricas
- Búsqueda y filtrado de runs
- Descarga de modelos y artefactos

## Comparación de Modelos

### Usando la interfaz web

1. Abre MLflow UI: `mlflow ui`
2. Selecciona un experimento
3. Selecciona múltiples runs usando los checkboxes
4. Haz clic en "Compare" para ver comparación lado a lado

### Usando código Python

```python
from mlops.MLFLow_Equipo19 import MLflowManager

mlflow_manager = MLflowManager()

# Obtener el mejor run
best_run = mlflow_manager.get_best_run(
    metric="test_accuracy",
    ascending=False  # False = mayor es mejor
)

# Comparar múltiples runs
runs_df = mlflow_manager.compare_runs([
    "run_id_1",
    "run_id_2",
    "run_id_3"
])

# Buscar runs con filtros
runs = mlflow_manager.search_runs(
    filter_string="metrics.test_accuracy > 0.8",
    order_by=["metrics.test_accuracy DESC"]
)
```

## Estructura de Datos

### Ubicación de Datos MLflow

Por defecto, MLflow almacena datos localmente en:
- `./mlruns/` - Metadatos y tracking
- `./mlflow_artifacts/` - Artefactos temporales

### Model Registry

Los modelos registrados se almacenan en el Model Registry y pueden:
- Tener múltiples versiones
- Ser promovidos a "Production" o "Staging"
- Incluir descripciones y metadatos
- Tener tags y anotaciones

## Comandos Útiles de MLflow CLI

```bash
# Ver experimentos
mlflow experiments list

# Ver runs de un experimento
mlflow runs list --experiment-id <experiment_id>

# Descargar modelo
mlflow models download -r <run_id> -A <artifact_path>

# Servir modelo para predicción
mlflow models serve -m runs:/<run_id>/<model_path> --port 5001

# Iniciar UI en puerto específico
mlflow ui --port 5000 --host 0.0.0.0
```

## Integración con el Pipeline

El script `src/train_with_mlflow.py` integra MLflow completamente con el pipeline:

1. **Preprocesamiento**: Se registran los parámetros de limpieza de datos
2. **Transformación**: Se registran parámetros de PCA y scaling
3. **Entrenamiento**: Cada modelo se registra con sus hiperparámetros
4. **Evaluación**: Todas las métricas se registran automáticamente
5. **Artefactos**: Gráficas y reportes se guardan como artefactos

## Mejores Prácticas

1. **Nombres descriptivos**: Usa nombres descriptivos para experimentos y runs
2. **Tags útiles**: Agrega tags que te ayuden a filtrar y organizar
3. **Registrar modelos**: Usa el Model Registry para modelos importantes
4. **Comparar sistemáticamente**: Compara diferentes configuraciones antes de decidir
5. **Documentar cambios**: Usa tags o notas para documentar cambios importantes
6. **Backup de datos**: Considera hacer backup del directorio `mlruns/`

## Solución de Problemas

### Error: "MLflow tracking URI not set"

Solución: El tracking URI se configura automáticamente. Si persiste, verifica que el directorio `mlruns/` exista.

### Error: "Experiment does not exist"

Solución: MLflow crea experimentos automáticamente. Si usas un nombre personalizado, asegúrate de que sea consistente.

### Modelo no se registra

Solución: Verifica que `registered_model_name` esté especificado y que tengas permisos de escritura.

### UI no muestra datos

Solución: 
1. Verifica que `mlflow ui` se ejecute desde el directorio raíz del proyecto
2. Verifica que el directorio `mlruns/` exista y contenga datos
3. Intenta limpiar el cache: `rm -rf mlruns/.trash`

## Recursos Adicionales

- [Documentación oficial de MLflow](https://www.mlflow.org/docs/latest/index.html)
- [MLflow Model Registry](https://www.mlflow.org/docs/latest/model-registry.html)
- [MLflow Tracking](https://www.mlflow.org/docs/latest/tracking.html)

