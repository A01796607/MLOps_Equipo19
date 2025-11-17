# Reproducibilidad en el Pipeline de ML

Este documento describe cómo garantizar la reproducibilidad del pipeline de ML en diferentes entornos.

## Resumen

La reproducibilidad es un requisito fundamental en MLOps para:
- **Auditar** experimentos y decisiones
- **Depurar** problemas en producción
- **Cumplir** estándares de calidad y gobernanza
- **Validar** que el mismo modelo produce resultados consistentes en diferentes entornos

## Componentes de Reproducibilidad

### 1. Fijación de Dependencias

Todas las dependencias críticas están fijadas con versiones exactas en `requirements.txt`:

```bash
# Instalar dependencias exactas
pip install -r requirements.txt
```

Las librerías core (numpy, pandas, scikit-learn, etc.) están fijadas con `==` en lugar de `~=` para garantizar versiones exactas.

### 2. Configuración de Semillas Aleatorias

El módulo `src/reproducibility.py` configura semillas para todas las librerías relevantes:

- **Python random**: `random.seed(42)`
- **NumPy**: `np.random.seed(42)`
- **Scikit-learn**: `random_state=42` en todos los modelos y splits
- **LightGBM**: `random_state=42` en parámetros del modelo
- **Variables de entorno**: `PYTHONHASHSEED=42` para consistencia en diccionarios

### 3. Uso del Módulo de Reproducibilidad

Todos los scripts principales usan el módulo de reproducibilidad:

```python
from src.reproducibility import ensure_reproducibility, DEFAULT_SEED

# Al inicio del script
reprod_config = ensure_reproducibility(seed=DEFAULT_SEED, verbose=True)

# Uso en train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, **reprod_config['split'], stratify=y
)

# Uso en modelos
trainer = ModelTrainer(random_state=reprod_config['seed'])
```

## Validación de Reproducibilidad

### Método 1: Usando el Script de Validación

El script `scripts/validate_reproducibility.py` permite validar que el pipeline produce resultados idénticos en diferentes entornos.

#### Paso 1: Generar Métricas de Referencia

En el entorno de referencia (entorno de entrenamiento original):

```bash
python scripts/validate_reproducibility.py \
    --model-type random_forest \
    --save-reference reference_metrics.json
```

Esto guarda las métricas de referencia en `reference_metrics.json`.

#### Paso 2: Validar en Entorno Limpio

En un entorno diferente (nueva máquina, VM o contenedor):

1. **Clonar el repositorio**:
   ```bash
   git clone <repository-url>
   cd MLOps_Equipo19
   ```

2. **Configurar entorno**:
   ```bash
   # Crear entorno virtual
   python3 -m venv venv
   source venv/bin/activate  # En Windows: venv\Scripts\activate
   
   # Instalar dependencias exactas
   pip install -r requirements.txt
   ```

3. **Descargar datos con DVC**:
   ```bash
   dvc pull data/raw/turkis_music_emotion_original.csv.dvc
   ```

4. **Ejecutar validación**:
   ```bash
   python scripts/validate_reproducibility.py \
       --model-type random_forest \
       --reference-metrics-file reference_metrics.json \
       --tolerance 1e-6
   ```

El script comparará las métricas actuales con las de referencia y reportará si coinciden dentro de la tolerancia especificada.

#### Método 2: Usando MLflow

Si las métricas de referencia están en MLflow:

```bash
python scripts/validate_reproducibility.py \
    --model-type random_forest \
    --reference-run-id <MLFLOW_RUN_ID> \
    --tolerance 1e-6
```

### Métricas Comparadas

El script compara las siguientes métricas:
- `accuracy`: Precisión del modelo
- `f1_macro`: F1-score macro promedio
- `f1_weighted`: F1-score ponderado
- `precision_macro`: Precisión macro promedio
- `recall_macro`: Recall macro promedio

También compara una muestra de predicciones para verificación adicional.

## Versionamiento de Artefactos

### DVC (Data Version Control)

Los siguientes artefactos están versionados con DVC:

- **Datos raw**: `data/raw/turkis_music_emotion_original.csv`
- **Transformer**: `data/processed/transformer.pkl`
- **Modelos**: `models/random_forest.pkl`, `models/lightgbm.pkl`

Para descargar artefactos versionados:

```bash
# Descargar todos los artefactos DVC
dvc pull

# Descargar un artefacto específico
dvc pull data/raw/turkis_music_emotion_original.csv.dvc
```

### MLflow

Los experimentos están registrados en MLflow con:

- **Parámetros**: Hiperparámetros, semillas, configuraciones
- **Métricas**: Métricas de entrenamiento y validación
- **Artifacts**: Modelos entrenados, reportes, matrices de confusión
- **Tags**: Información adicional sobre el experimento

Para ver experimentos:

```bash
mlflow ui
# Abrir http://localhost:5000 en el navegador
```

## Ejecución en Entorno Limpio

### Requisitos Previos

1. **Python 3.8+** instalado
2. **Git** para clonar el repositorio
3. **DVC** instalado (se instala con `pip install -r requirements.txt`)
4. **Acceso a S3** (configurado en `.dvc/config`) para descargar datos

### Pasos Completos

```bash
# 1. Clonar repositorio
git clone <repository-url>
cd MLOps_Equipo19

# 2. Crear entorno virtual
python3 -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Configurar DVC (si es necesario)
# Editar .dvc/config con credenciales S3

# 5. Descargar datos
dvc pull data/raw/turkis_music_emotion_original.csv.dvc

# 6. Entrenar modelo
python src/train_models_for_api.py

# 7. Validar reproducibilidad (si hay métricas de referencia)
python scripts/validate_reproducibility.py \
    --reference-metrics-file reference_metrics.json
```

## Tolerancia y Diferencias Numéricas

Debido a diferencias en implementaciones de punto flotante entre plataformas (CPU, OS), pequeñas diferencias numéricas pueden ocurrir. El script de validación permite especificar una tolerancia:

```bash
# Tolerancia por defecto: 1e-6
python scripts/validate_reproducibility.py \
    --reference-metrics-file reference_metrics.json \
    --tolerance 1e-5  # Tolerancia más relajada
```

**Nota**: Si las diferencias son mayores que la tolerancia, verificar:
1. Versiones de dependencias
2. Configuración de semillas
3. Orden de operaciones
4. Diferencias en hardware (CPU vs GPU, diferentes versiones de BLAS)

## Scripts con Reproducibilidad Habilitada

Los siguientes scripts incluyen configuración de reproducibilidad:

- `src/experiments_mlflow.py`: Pipeline de experimentos con MLflow
- `src/train_pipeline.py`: Pipeline de entrenamiento con sklearn.Pipeline
- `src/train_models_for_api.py`: Entrenamiento de modelos para API

Todos estos scripts configuran semillas automáticamente al inicio.

## Buenas Prácticas

1. **Siempre usar el módulo de reproducibilidad**: Importar `ensure_reproducibility` al inicio de cualquier script de entrenamiento
2. **No hardcodear semillas**: Usar `DEFAULT_SEED` o `reprod_config['seed']` en lugar de valores mágicos
3. **Versionar artefactos**: Usar DVC para datos y modelos, MLflow para experimentos
4. **Documentar cambios**: Si se cambia la semilla o configuraciones, documentar en commits y MLflow
5. **Validar en CI/CD**: Incluir validación de reproducibilidad en pipelines de CI/CD

## Troubleshooting

### Problema: Métricas no coinciden

**Solución**: 
- Verificar que todas las dependencias estén instaladas con versiones exactas
- Asegurar que `PYTHONHASHSEED` esté configurado
- Verificar que la semilla sea la misma en ambos entornos

### Problema: Datos diferentes

**Solución**:
- Usar `dvc pull` para descargar la versión exacta de los datos
- Verificar que el hash DVC coincida: `dvc status`

### Problema: Diferencias en predicciones

**Solución**:
- Verificar versión de NumPy y BLAS
- En algunos casos, diferencias menores son esperadas debido a operaciones de punto flotante
- Ajustar tolerancia si es necesario

## Referencias

- [DVC Documentation](https://dvc.org/doc)
- [MLflow Documentation](https://www.mlflow.org/docs/latest/index.html)
- [Scikit-learn Random State](https://scikit-learn.org/stable/glossary.html#term-random_state)
- [Reproducibility in Machine Learning](https://www.pytorchlightning.ai/blog/reproducibility-in-machine-learning)

