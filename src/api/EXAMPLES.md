# Ejemplos de uso de la API

## Endpoint: `/predict`

### Ejemplo 1: Predicción con Random Forest (modelo por defecto)

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "features": [
      [0.5, 0.3, 0.2, 0.1, 0.4, 0.6, 0.7, 0.8, 0.9, 0.1,
       0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.1, 0.2,
       0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.1, 0.2, 0.3,
       0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.1, 0.2, 0.3, 0.4,
       0.5, 0.6, 0.7, 0.8, 0.9, 0.1, 0.2, 0.3, 0.4, 0.5]
    ]
  }'
```

### Ejemplo 2: Predicción con LightGBM

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "lightgbm",
    "features": [
      [0.5, 0.3, 0.2, 0.1, 0.4, 0.6, 0.7, 0.8, 0.9, 0.1,
       0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.1, 0.2,
       0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.1, 0.2, 0.3,
       0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.1, 0.2, 0.3, 0.4,
       0.5, 0.6, 0.7, 0.8, 0.9, 0.1, 0.2, 0.3, 0.4, 0.5]
    ]
  }'
```

### Ejemplo 3: Predicción múltiple (múltiples muestras)

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "random_forest",
    "features": [
      [0.5, 0.3, 0.2, 0.1, 0.4, 0.6, 0.7, 0.8, 0.9, 0.1,
       0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.1, 0.2,
       0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.1, 0.2, 0.3,
       0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.1, 0.2, 0.3, 0.4,
       0.5, 0.6, 0.7, 0.8, 0.9, 0.1, 0.2, 0.3, 0.4, 0.5],
      [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.1,
       0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.1, 0.2,
       0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.1, 0.2, 0.3,
       0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.1, 0.2, 0.3, 0.4,
       0.5, 0.6, 0.7, 0.8, 0.9, 0.1, 0.2, 0.3, 0.4, 0.5]
    ]
  }'
```

### Ejemplo 4: Usando un archivo JSON

Crea un archivo `request.json`:

```json
{
  "model_name": "random_forest",
  "features": [
    [0.5, 0.3, 0.2, 0.1, 0.4, 0.6, 0.7, 0.8, 0.9, 0.1,
     0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.1, 0.2,
     0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.1, 0.2, 0.3,
     0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.1, 0.2, 0.3, 0.4,
     0.5, 0.6, 0.7, 0.8, 0.9, 0.1, 0.2, 0.3, 0.4, 0.5]
  ]
}
```

Luego ejecuta:

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d @request.json
```

### Ejemplo 5: Formato más legible (usando jq)

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "random_forest",
    "features": [
      [0.5, 0.3, 0.2, 0.1, 0.4, 0.6, 0.7, 0.8, 0.9, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.1, 0.2, 0.3, 0.4, 0.5]
    ]
  }' | jq '.'
```

## Respuesta esperada

```json
{
  "prediction": 2,
  "prediction_label": "relax",
  "probabilities": {
    "angry": 0.1,
    "happy": 0.2,
    "relax": 0.6,
    "sad": 0.1
  },
  "model_name": "random_forest"
}
```

## Otros endpoints

### Health Check

```bash
curl http://localhost:8000/health
```

### Información del modelo

```bash
# Información del modelo actual
curl http://localhost:8000/models/info

# Información de un modelo específico
curl "http://localhost:8000/models/info?model_name=lightgbm"
```

### Modelos disponibles

```bash
curl http://localhost:8000/models/available
```

## Notas importantes

- El modelo espera **50 features** (valores numéricos) por muestra
- Cada muestra debe ser un array de 50 números flotantes
- Puedes enviar múltiples muestras en una sola petición
- Los modelos disponibles son: `random_forest` (por defecto) y `lightgbm`
- Las clases predichas son: `angry`, `happy`, `relax`, `sad`

