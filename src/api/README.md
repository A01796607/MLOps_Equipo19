# API REST - Servicio de Inferencia de Modelos

API REST construida con FastAPI para realizar predicciones con modelos de machine learning entrenados.

## Estructura Modular

```
src/api/
├── __init__.py          # Exporta la app FastAPI
├── main.py              # Aplicación FastAPI y endpoints
├── schemas.py           # Modelos Pydantic (Request/Response)
├── service.py           # Lógica de negocio (carga modelos, predicciones)
├── config.py            # Configuración (rutas, modelos disponibles)
├── run_server.py        # Script para ejecutar el servidor
└── README.md            # Esta documentación
```

## Instalación

```bash
# Instalar dependencias
pip install -r requirements.txt

# O instalar solo FastAPI y dependencias
pip install fastapi uvicorn[standard] pydantic
```

## Ejecutar el Servidor

### Opción 1: Usando el script de Python

```bash
python src/api/run_server.py
```

### Opción 2: Usando uvicorn directamente

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

### Opción 3: Usando Make (si está configurado)

```bash
make run-api
```

El servidor estará disponible en: `http://localhost:8000`

## Endpoints

### 1. Health Check

```http
GET /health
GET /
```

**Respuesta:**
```json
{
  "status": "healthy",
  "version": "1.0.0"
}
```

### 2. Predicción

```http
POST /predict
```

**Request Body:**
```json
{
  "features": [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]],
  "model_name": "random_forest"  // opcional, default: "random_forest"
}
```

**Response:**
```json
{
  "predictions": [
    {
      "prediction": 0,
      "prediction_label": "happy",
      "probabilities": {
        "happy": 0.85,
        "sad": 0.10,
        "angry": 0.05
      }
    }
  ],
  "model_name": "random_forest",
  "model_info": {
    "model_loaded": true,
    "transformer_loaded": true,
    "available_models": ["random_forest", "lightgbm"],
    "class_labels": ["happy", "sad", "angry"]
  }
}
```

### 3. Información del Modelo

```http
GET /models/info?model_name=random_forest
```

**Response:**
```json
{
  "model_name": "random_forest",
  "model_path": "/path/to/models/random_forest.pkl",
  "model_loaded": true,
  "transformer_loaded": true,
  "available_models": ["random_forest", "lightgbm"],
  "class_labels": ["happy", "sad", "angry"]
}
```

### 4. Modelos Disponibles

```http
GET /models/available
```

**Response:**
```json
{
  "available_models": ["random_forest", "lightgbm"],
  "models": {
    "random_forest": "/path/to/models/random_forest.pkl",
    "lightgbm": "/path/to/models/lightgbm.pkl"
  }
}
```

## Documentación Interactiva

FastAPI genera automáticamente documentación interactiva:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## Ejemplo de Uso con curl

```bash
# Health check
curl http://localhost:8000/health

# Predicción
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "features": [[0.1, 0.2, 0.3, 0.4]],
    "model_name": "random_forest"
  }'

# Información del modelo
curl http://localhost:8000/models/info?model_name=random_forest
```

## Ejemplo de Uso con Python

```python
import requests

# Health check
response = requests.get("http://localhost:8000/health")
print(response.json())

# Predicción
data = {
    "features": [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]],
    "model_name": "random_forest"
}
response = requests.post("http://localhost:8000/predict", json=data)
print(response.json())
```

## Configuración

Los modelos y transformers se cargan desde:

- **Modelos**: `models/` (configurado en `src/api/config.py`)
- **Transformer**: `data/processed/transformer.pkl` (configurado en `src/api/config.py`)

Puedes modificar las rutas en `src/api/config.py`:

```python
AVAILABLE_MODELS = {
    "random_forest": MODELS_DIR / "random_forest.pkl",
    "lightgbm": MODELS_DIR / "lightgbm.pkl",
}
```

## Arquitectura

### Separación de Responsabilidades

1. **`main.py`**: Endpoints FastAPI, validación de requests, manejo de errores
2. **`schemas.py`**: Modelos Pydantic para validación de datos (Request/Response)
3. **`service.py`**: Lógica de negocio (carga de modelos, predicciones)
4. **`config.py`**: Configuración centralizada (rutas, modelos disponibles)

### Flujo de Predicción

1. Cliente envía request a `/predict` con features
2. FastAPI valida el request usando `PredictionRequest` (Pydantic)
3. `ModelService` carga el modelo solicitado (si no está cargado)
4. `ModelService.predict()` hace la predicción
5. Se decodifican las etiquetas usando el transformer (si está disponible)
6. Se retorna la respuesta usando `PredictionResponse` (Pydantic)

## Notas

- El servicio carga el modelo en memoria al iniciarse (singleton pattern)
- Los modelos pueden cambiarse dinámicamente usando el parámetro `model_name`
- El transformer es opcional; si no está disponible, solo se retornan índices numéricos
- Soporta modelos de scikit-learn y LightGBM
- Las probabilidades se retornan si el modelo soporta `predict_proba()`

## Troubleshooting

### Error: "Model is not loaded"
- Verifica que el modelo exista en `models/random_forest.pkl` o `models/lightgbm.pkl`
- Verifica los permisos de lectura del archivo

### Error: "Transformer file not found"
- El transformer es opcional
- Sin transformer, las predicciones solo incluirán índices numéricos

### Error: "Model 'X' not found"
- Verifica que el modelo esté en `AVAILABLE_MODELS` en `config.py`
- Verifica que el archivo del modelo exista en la ruta especificada
