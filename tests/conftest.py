"""
Fixtures compartidas para todas las pruebas unitarias
"""
import pytest
import pandas as pd
import numpy as np
import tempfile
import os


@pytest.fixture
def sample_dataframe():
    """Crea un DataFrame de muestra para pruebas"""
    data = {
        'Mes': [1, 2, 3, 4, 5, 6],
        'Hora de Salida': ['08:30', '09:15', '10:45', '11:20', '14:30', '18:00'],
        'Distancia Kilometros': [15.5, 22.3, 10.1, 35.7, 5.2, 50.0],
        'Tipo de Vehiculo': ['Auto', 'Moto', 'Auto', 'Auto', 'Bus', 'Auto'],
        'Clima': ['Soleado', 'Lluvia', 'Nublado', 'Lluvia', 'Soleado', 'Nublado'],
        'Dia de la Semana': [1, 2, 3, 4, 5, 6],
        'Tipo de Carretera': ['Autopista', 'Urbana', 'Urbana', 'Autopista', 'Urbana', 'Autopista'],
        'Velocidad Promedio': [60.0, 40.0, 50.0, 70.0, 35.0, 85.0],
        'Edad Conductor': [25, 35, 28, 45, 30, 50],
        'Experiencia Conductor': [5, 10, 3, 15, 7, 20],
        'Alcohol en Sangre': [0.0, 0.0, 0.1, 0.0, 0.0, 0.05],
        'Visibilidad': [100.0, 50.0, 80.0, 40.0, 100.0, 70.0],
        'Estado de la Via': ['Buena', 'Regular', 'Buena', 'Mala', 'Buena', 'Regular'],
        'Iluminacion': ['Diurna', 'Nocturna', 'Diurna', 'Nocturna', 'Diurna', 'Nocturna'],
        'Cinturon': ['Sí', 'No', 'Sí', 'Sí', 'No', 'Sí'],
        'Probabilidad de Accidente': ['10%', '25%', '5%', '45%', '8%', '60%'],
        'Accidente': ['No', 'Sí', 'No', 'Sí', 'No', 'Sí'],
        'Condicion del Vehiculo': ['95%', '75%', '98%', '60%', '85%', '40%'],
        'Probabilidad de accidente (%)': [10.0, 25.0, 5.0, 45.0, 8.0, 60.0],
        'Condicion del Vehiculo (%)': [95.0, 75.0, 98.0, 60.0, 85.0, 40.0],
        'Accidente_bin': [0, 1, 0, 1, 0, 1],
        'Hora_hour': [8, 9, 10, 11, 14, 18],
        'Hora_min': [30, 15, 45, 20, 30, 0]
    }
    return pd.DataFrame(data)


@pytest.fixture
def minimal_dataframe():
    """DataFrame mínimo para pruebas de columnas requeridas"""
    data = {
        'Mes': [1, 2],
        'Hora de Salida': ['08:30', '09:15'],
        'Distancia Kilometros': [15.5, 22.3],
        'Tipo de Vehiculo': ['Auto', 'Moto'],
        'Clima': ['Soleado', 'Lluvia'],
        'Dia de la Semana': [1, 2],
        'Tipo de Carretera': ['Autopista', 'Urbana'],
        'Velocidad Promedio': [60.0, 40.0],
        'Edad Conductor': [25, 35],
        'Experiencia Conductor': [5, 10],
        'Alcohol en Sangre': [0.0, 0.0],
        'Visibilidad': [100.0, 50.0],
        'Estado de la Via': ['Buena', 'Regular'],
        'Iluminacion': ['Diurna', 'Nocturna'],
        'Cinturon': ['Sí', 'No'],
        'Probabilidad de Accidente': ['10%', '25%'],
        'Accidente': ['No', 'Sí'],
        'Condicion del Vehiculo': ['95%', '75%']
    }
    return pd.DataFrame(data)


@pytest.fixture
def temp_excel_file(sample_dataframe):
    """Crea un archivo Excel temporal para pruebas"""
    with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as tmp:
        sample_dataframe.to_excel(tmp.name, index=False)
        yield tmp.name
    os.unlink(tmp.name)


@pytest.fixture
def temp_csv_file(sample_dataframe):
    """Crea un archivo CSV temporal para pruebas"""
    with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as tmp:
        sample_dataframe.to_csv(tmp.name, index=False)
        yield tmp.name
    os.unlink(tmp.name)


@pytest.fixture
def temp_models_dir():
    """Crea un directorio temporal para guardar modelos"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    # Limpieza
    import shutil
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
