"""
Pruebas unitarias para src/data.py
"""
import pytest
import pandas as pd
import numpy as np
import os
import tempfile
from src.data import load_dataset, validate_columns, prepare_dataset, save_dataset


class TestLoadDataset:
    """Pruebas para la función load_dataset"""

    def test_load_existing_excel_file(self, temp_excel_file):
        """Debe cargar correctamente un archivo Excel existente"""
        df = load_dataset(temp_excel_file)
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_load_nonexistent_file(self):
        """Debe lanzar FileNotFoundError para archivos inexistentes"""
        with pytest.raises(FileNotFoundError):
            load_dataset('archivo_inexistente.xlsx')


class TestValidateColumns:
    """Pruebas para la función validate_columns"""

    def test_valid_columns(self, sample_dataframe):
        """No debe lanzar excepción si todas las columnas requeridas existen"""
        # No debería lanzar ninguna excepción
        validate_columns(sample_dataframe)

    def test_missing_columns(self):
        """Debe lanzar ValueError si faltan columnas requeridas"""
        # Crear un DataFrame sin las columnas requeridas
        df = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]})
        with pytest.raises(ValueError):
            validate_columns(df)

    def test_error_message_content(self):
        """El mensaje de error debe listar las columnas faltantes"""
        df = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]})
        with pytest.raises(ValueError) as exc_info:
            validate_columns(df)
        error_msg = str(exc_info.value)
        assert 'Faltan columnas' in error_msg


class TestPrepareDataset:
    """Pruebas para la función prepare_dataset"""

    def test_prepare_removes_nulls(self, sample_dataframe):
        """Debe remover filas con valores nulos"""
        df_with_nulls = sample_dataframe.copy()
        df_with_nulls.loc[0, 'Mes'] = np.nan
        result = prepare_dataset(df_with_nulls)
        assert len(result) < len(df_with_nulls)

    def test_prepare_converts_numeric(self, sample_dataframe):
        """Debe convertir columnas a tipos numéricos"""
        result = prepare_dataset(sample_dataframe)
        numeric_cols = ['Hora de Salida', 'Distancia Kilometros', 'Velocidad Promedio',
                       'Edad Conductor', 'Experiencia Conductor', 'Alcohol en Sangre']
        for col in numeric_cols:
            if col in result.columns:
                assert result[col].dtype in [np.float64, np.int64, float, int]

    def test_prepare_resets_index(self, sample_dataframe):
        """Debe resetear el índice después de remover nulos"""
        result = prepare_dataset(sample_dataframe)
        assert result.index.tolist() == list(range(len(result)))

    def test_prepare_validates_columns_first(self):
        """Debe validar columnas antes de procesar"""
        df_invalid = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]})
        with pytest.raises(ValueError):
            prepare_dataset(df_invalid)


class TestSaveDataset:
    """Pruebas para la función save_dataset"""

    def test_save_dataset_to_excel(self, sample_dataframe):
        """Debe guardar DataFrame a archivo Excel"""
        with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as tmp:
            temp_path = tmp.name
        
        try:
            save_dataset(sample_dataframe, temp_path)
            assert os.path.exists(temp_path)
            
            # Verificar que se guardó correctamente
            loaded = pd.read_excel(temp_path)
            assert loaded.shape == sample_dataframe.shape
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_save_dataset_default_path(self, sample_dataframe):
        """Debe usar la ruta por defecto si no se especifica"""
        # Guardamos en un archivo temporal para esta prueba
        with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as tmp:
            temp_path = tmp.name
        
        try:
            save_dataset(sample_dataframe, temp_path)
            assert os.path.exists(temp_path)
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_saved_file_can_be_loaded(self, sample_dataframe):
        """El archivo guardado debe poder cargarse nuevamente"""
        with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as tmp:
            temp_path = tmp.name
        
        try:
            save_dataset(sample_dataframe, temp_path)
            loaded = load_dataset(temp_path)
            assert len(loaded) == len(sample_dataframe)
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
