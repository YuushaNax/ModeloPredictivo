# Pruebas Unitarias - ModeloPredictivo

## Descripción

Este directorio contiene todas las pruebas unitarias para el proyecto ModeloPredictivo. Las pruebas están organizadas para verificar la funcionalidad de cada módulo del sistema.

## Estructura de Pruebas

### `conftest.py`
Archivo de configuración de pytest que contiene fixtures compartidas:
- `sample_dataframe`: DataFrame completo con datos de muestra
- `minimal_dataframe`: DataFrame mínimo para pruebas básicas
- `temp_excel_file`: Archivo Excel temporal para pruebas
- `temp_csv_file`: Archivo CSV temporal para pruebas
- `temp_models_dir`: Directorio temporal para guardar modelos

### `test_utils.py`
Pruebas para `src/utils.py`:
- `TestParsePercentColumn`: Pruebas de parseo de porcentajes
- `TestParseTimeColumn`: Pruebas de parseo de tiempos
- `TestLoadAndPrepare`: Pruebas de carga y preparación de datos

### `test_data.py`
Pruebas para `src/data.py`:
- `TestLoadDataset`: Pruebas de carga de datasets
- `TestValidateColumns`: Pruebas de validación de columnas
- `TestPrepareDataset`: Pruebas de preparación de datos
- `TestSaveDataset`: Pruebas de guardado de datos

### `test_train.py`
Pruebas para `src/train.py`:
- `TestBuildPreprocessor`: Pruebas del preprocessor
- `TestTrainModels`: Pruebas del entrenamiento de modelos

### `test_predict.py`
Pruebas para `src/predict.py`:
- `TestParseSampleString`: Pruebas de parseo de strings de muestra
- `TestSampleToDF`: Pruebas de conversión a DataFrame
- `TestIntegration`: Pruebas de integración

### `test_visualize.py`
Pruebas para `src/visualize.py`:
- `TestMakePlots`: Pruebas de generación de gráficos

## Instalación de Dependencias

Para ejecutar las pruebas, primero instala las dependencias necesarias:

```bash
pip install pytest pytest-cov
```

## Ejecutar las Pruebas

### Ejecutar todas las pruebas:
```bash
pytest
```

### Ejecutar pruebas con salida detallada:
```bash
pytest -v
```

### Ejecutar pruebas de un módulo específico:
```bash
pytest tests/test_utils.py
```

### Ejecutar pruebas de una clase específica:
```bash
pytest tests/test_utils.py::TestParsePercentColumn
```

### Ejecutar una prueba específica:
```bash
pytest tests/test_utils.py::TestParsePercentColumn::test_parse_numeric_value
```

### Ejecutar con reporte de cobertura:
```bash
pytest --cov=src --cov-report=html
```

Esto generará un reporte de cobertura en `htmlcov/index.html`

### Ejecutar solo pruebas rápidas:
```bash
pytest -m "not slow"
```

## Cobertura de Código

Los objetivos de cobertura mínima son:
- `src/utils.py`: 95%
- `src/data.py`: 90%
- `src/train.py`: 85%
- `src/predict.py`: 90%
- `src/visualize.py`: 80%

Para verificar la cobertura actual:
```bash
pytest --cov=src --cov-report=term-missing
```

## Mejores Prácticas

1. **Aislamiento**: Cada test debe ser independiente y no afectar a otros
2. **Claridad**: Los nombres de las pruebas deben describir qué prueban
3. **Rapidez**: Las pruebas deben ejecutarse rápidamente
4. **Fixtures**: Usa las fixtures en `conftest.py` para datos compartidos
5. **Mocking**: Usa mocks cuando sea necesario para aislar dependencias

## Ejemplo de Ejecución

```bash
# Ejecutar todas las pruebas con reporte detallado
pytest -v --cov=src --cov-report=html

# Ver el reporte en un navegador
start htmlcov/index.html
```

## Solución de Problemas

### Error: "No module named 'src'"
Asegúrate de ejecutar pytest desde el directorio raíz del proyecto.

### Error: "ImportError: cannot import name..."
Verifica que todas las dependencias estén instaladas en `requirements.txt`

### Las pruebas no detectan cambios
Limpia el directorio `__pycache__`:
```bash
find . -type d -name __pycache__ -exec rm -rf {} +
```

## Contribución

Al agregar nuevas características, asegúrate de:
1. Escribir pruebas unitarias
2. Mantener o mejorar la cobertura
3. Ejecutar las pruebas antes de hacer commit
4. Documentar cualquier nueva fixture en este archivo
