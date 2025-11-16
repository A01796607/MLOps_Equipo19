# Pruebas automatizadas (unitarias e integración)

Este directorio contiene:
- Pruebas unitarias de los componentes principales.
- Pruebas de integración que validan el flujo extremo a extremo del pipeline (carga de datos → preprocesamiento → predicción → métricas).

## Ejecutar Tests

### Comando único (quiet)
```bash
pytest -q
```

### Todos los tests (detallado)
```bash
python3 -m pytest tests/ -v
```

### Un archivo específico
```bash
python3 -m pytest tests/test_data_processor.py -v
python3 -m pytest tests/test_feature_transformer.py -v
python3 -m pytest tests/test_model_trainer.py -v
python3 -m pytest tests/test_plotter.py -v
python3 -m pytest tests/e2e/test_integration_pipeline.py -v
```

### Una clase específica
```bash
python3 -m pytest tests/test_data_processor.py::TestDataProcessor -v
```

### Un test específico
```bash
python3 -m pytest tests/test_data_processor.py::TestDataProcessor::test_init_default -v
```

### Opciones útiles

```bash
# Output muy detallado
python3 -m pytest tests/ -vv

# Mostrar prints (stdout)
python3 -m pytest tests/ -v -s

# Solo resumen (quiet)
python3 -m pytest tests/ -q

# Detener en el primer error
python3 -m pytest tests/ -x

# Ver qué tests se ejecutarían sin ejecutarlos
python3 -m pytest tests/ --co

# Ejecutar solo los tests que fallaron la última vez
python3 -m pytest tests/ --lf
```

## Estructura de Tests

- `test_data.py`: Tests básicos de imports
- `test_data_processor.py`: Tests para `DataProcessor` (11 tests)
- `test_feature_transformer.py`: Tests para `FeatureTransformer` (12 tests)
- `test_model_trainer.py`: Tests para `ModelTrainer` (15 tests, 3 opcionales si lightgbm no está instalado)
- `test_plotter.py`: Tests para `Plotter` (14 tests)
- `e2e/test_integration_pipeline.py`: Prueba de integración end-to-end (datos → preprocesamiento → predicción → métricas)

## Resultados Esperados

```
======================== 52 passed in XX.XXs =========================
```

Nota: Algunos tests de LightGBM pueden omitirse automáticamente si la librería no está instalada.

## Code Coverage

Para calcular la cobertura de código:

### Instalar pytest-cov (si no está instalado)
```bash
pip install pytest-cov
```

### Ejecutar tests con coverage

```bash
# Coverage en terminal con líneas faltantes
python3 -m pytest tests/ --cov=src --cov-report=term-missing

# Coverage solo en terminal (resumen)
python3 -m pytest tests/ --cov=src --cov-report=term

# Coverage en HTML (se genera en htmlcov/)
python3 -m pytest tests/ --cov=src --cov-report=html

# Abrir el reporte HTML en el navegador
open htmlcov/index.html  # macOS
# xdg-open htmlcov/index.html  # Linux
```

### Opciones adicionales

```bash
# Coverage solo para archivos específicos
python3 -m pytest tests/ --cov=src.data_processor --cov-report=term-missing

# Coverage con umbral mínimo (falla si coverage < 80%)
python3 -m pytest tests/ --cov=src --cov-report=term-missing --cov-fail-under=80

# Ver archivos no cubiertos
python3 -m pytest tests/ --cov=src --cov-report=term-missing | grep "TOTAL"
```

También puedes usar Makefile:
```bash
make test
```

