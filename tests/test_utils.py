"""
Pruebas unitarias para src/utils.py
"""
import pytest
import pandas as pd
import numpy as np
from src.utils import parse_percent_column, parse_time_column, load_and_prepare


class TestParsePercentColumn:
    """Pruebas para la función parse_percent_column"""

    def test_parse_numeric_value(self):
        """Debe convertir valores numéricos correctamente"""
        assert parse_percent_column(12.3) == 12.3
        assert parse_percent_column(0) == 0.0
        assert parse_percent_column(100) == 100.0

    def test_parse_percent_string(self):
        """Debe parseEAr strings con símbolo de porcentaje"""
        assert parse_percent_column('12.3%') == 12.3
        assert parse_percent_column('0%') == 0.0
        assert parse_percent_column('100%') == 100.0
        assert parse_percent_column(' 50.5% ') == 50.5

    def test_parse_nan_value(self):
        """Debe retornar NaN para valores faltantes"""
        assert np.isnan(parse_percent_column(np.nan))
        assert np.isnan(parse_percent_column(None))

    def test_parse_invalid_string(self):
        """Debe retornar NaN para strings inválidos"""
        assert np.isnan(parse_percent_column('invalid'))
        assert np.isnan(parse_percent_column(''))


class TestParseTimeColumn:
    """Pruebas para la función parse_time_column"""

    def test_parse_valid_time(self):
        """Debe parseAr tiempos válidos en formato HH:MM"""
        assert parse_time_column('13:45') == (13, 45)
        assert parse_time_column('08:30') == (8, 30)
        assert parse_time_column('23:59') == (23, 59)

    def test_parse_single_digit_hour(self):
        """Debe manejar horas con un solo dígito"""
        assert parse_time_column('9:05') == (9, 5)
        assert parse_time_column('1:00') == (1, 0)

    def test_parse_time_without_minutes(self):
        """Debe asumir minutos = 0 si no están presentes"""
        assert parse_time_column('14') == (14, 0)

    def test_parse_invalid_time(self):
        """Debe retornar (NaN, NaN) para tiempos inválidos"""
        hour, minute = parse_time_column('invalid')
        assert np.isnan(hour) and np.isnan(minute)

    def test_parse_nan_time(self):
        """Debe retornar (NaN, NaN) para valores NaN"""
        hour, minute = parse_time_column(np.nan)
        assert np.isnan(hour) and np.isnan(minute)


class TestLoadAndPrepare:
    """Pruebas para la función load_and_prepare"""

    def test_load_dataframe_input(self, sample_dataframe):
        """Debe procesar un DataFrame directamente"""
        result = load_and_prepare(sample_dataframe)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(sample_dataframe)

    def test_percent_columns_parsed(self, sample_dataframe):
        """Debe crear columnas numéricas para porcentajes"""
        result = load_and_prepare(sample_dataframe)
        # Puede haber Probabilidad de accidente (%) o solo la versión sin (%)
        has_percent_col = any('Probabilidad' in col and '%' in col for col in result.columns)
        assert has_percent_col or 'Probabilidad de accidente (%)' in result.columns or 'Probabilidad de Accidente' in result.columns
        assert 'Condicion del Vehiculo (%)' in result.columns or 'Condicion del Vehiculo' in result.columns

    def test_accident_binary_column(self, sample_dataframe):
        """Debe crear columna binaria para accidentes"""
        result = load_and_prepare(sample_dataframe)
        assert 'Accidente_bin' in result.columns
        assert set(result['Accidente_bin'].dropna().unique()).issubset({0, 1})

    def test_time_parsing(self, sample_dataframe):
        """Debe parsEAr hora de salida en componentes"""
        result = load_and_prepare(sample_dataframe)
        assert 'Hora_hour' in result.columns
        assert 'Hora_min' in result.columns
        assert result['Hora_hour'].dtype in [np.float64, float, np.int64, int]

    def test_numeric_conversion(self, sample_dataframe):
        """Debe convertir columnas a tipos numéricos"""
        result = load_and_prepare(sample_dataframe)
        assert result['Mes'].dtype in [np.float64, np.int64, float, int]
        assert result['Distancia Kilometros'].dtype in [np.float64, np.int64, float, int]

    def test_categorical_normalization(self, sample_dataframe):
        """Debe normalizar campos categóricos (strip)"""
        sample_dataframe_with_spaces = sample_dataframe.copy()
        sample_dataframe_with_spaces['Tipo de Carretera'] = '  ' + sample_dataframe_with_spaces['Tipo de Carretera']
        result = load_and_prepare(sample_dataframe_with_spaces)
        assert result['Tipo de Carretera'].iloc[0] == result['Tipo de Carretera'].iloc[0].strip()

    def test_column_name_normalization(self, sample_dataframe):
        """Debe normalizar nombres de columnas (strip)"""
        df_with_spaces = sample_dataframe.copy()
        df_with_spaces.columns = [f'  {col}  ' for col in df_with_spaces.columns]
        result = load_and_prepare(df_with_spaces)
        # Las columnas después del procesamiento no deben tener espacios extras
        assert all(col == col.strip() for col in result.columns)

    def test_different_accident_representations(self, minimal_dataframe):
        """Debe manejar diferentes representaciones de accidentes"""
        test_values = ['Sí', 'sí', 'Si', 'si', 's', 'S', 'yes', 'YES']
        for val in test_values:
            df = minimal_dataframe.copy()
            df['Accidente'].iloc[0] = val
            result = load_and_prepare(df)
            assert result['Accidente_bin'].iloc[0] == 1, f"Falló para {val}"

        test_values = ['No', 'no', 'No', 'n', 'N', 'no', 'NO']
        for val in test_values:
            df = minimal_dataframe.copy()
            df['Accidente'].iloc[0] = val
            result = load_and_prepare(df)
            assert result['Accidente_bin'].iloc[0] == 0, f"Falló para {val}"
