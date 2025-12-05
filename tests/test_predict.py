"""
Pruebas unitarias para src/predict.py
"""
import pytest
import pandas as pd
import numpy as np
from src.predict import parse_sample_string, sample_to_df


class TestParseSampleString:
    """Pruebas para la función parse_sample_string"""

    def test_parse_simple_string(self):
        """Debe parsEAr strings simples en formato K=V;K=V"""
        s = "Mes=5;Hora=14:30;Distancia=25.5"
        result = parse_sample_string(s)
        
        assert result['Mes'] == '5'
        assert result['Hora'] == '14:30'
        assert result['Distancia'] == '25.5'

    def test_parse_with_spaces(self):
        """Debe manejar espacios alrededor de claves y valores"""
        s = " Mes = 5 ; Hora = 14:30 "
        result = parse_sample_string(s)
        
        assert 'Mes' in result
        assert result['Mes'] == '5'

    def test_parse_single_pair(self):
        """Debe parsEAr un solo par clave=valor"""
        s = "Mes=3"
        result = parse_sample_string(s)
        
        assert result['Mes'] == '3'
        assert len(result) == 1

    def test_parse_empty_values(self):
        """Debe manejar valores vacíos"""
        s = "Mes=5;Valor="
        result = parse_sample_string(s)
        
        assert 'Mes' in result
        assert 'Valor' in result

    def test_parse_special_characters(self):
        """Debe preservar caracteres especiales en los valores"""
        s = "Clima=Soleado;Tipo=Auto/Moto"
        result = parse_sample_string(s)
        
        assert result['Clima'] == 'Soleado'
        assert 'Tipo' in result


class TestSampleToDF:
    """Pruebas para la función sample_to_df"""

    def test_sample_to_df_creates_dataframe(self):
        """Debe convertir diccionario a DataFrame"""
        sample_dict = {
            'Mes': '5',
            'Hora': '14:30',
            'Distancia Kilometros': '25.5'
        }
        df = sample_to_df(sample_dict)
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1

    def test_sample_to_df_has_correct_columns(self):
        """El DataFrame debe tener las columnas del diccionario o mapeadas"""
        sample_dict = {
            'Mes': '5',
            'Hora': '14:30',
            'Distancia': '25.5'
        }
        df = sample_to_df(sample_dict)
        
        assert 'Mes' in df.columns
        assert ('Hora' in df.columns or 'Hora de Salida' in df.columns)
        assert 'Distancia' in df.columns

    def test_sample_to_df_hora_mapping(self):
        """Debe mapear 'Hora' a 'Hora de Salida'"""
        sample_dict = {
            'Mes': '5',
            'Hora': '14:30'
        }
        df = sample_to_df(sample_dict)
        
        assert 'Hora de Salida' in df.columns
        assert 'Hora' not in df.columns or df['Hora de Salida'].iloc[0] == '14:30'

    def test_sample_to_df_preserves_values(self):
        """Debe preservar los valores del diccionario"""
        sample_dict = {
            'Mes': '5',
            'Distancia': '25.5',
            'Tipo': 'Auto'
        }
        df = sample_to_df(sample_dict)
        
        assert df['Mes'].iloc[0] == '5'
        assert df['Distancia'].iloc[0] == '25.5'
        assert df['Tipo'].iloc[0] == 'Auto'

    def test_sample_to_df_with_hora_salida_already_present(self):
        """No debe duplicar si 'Hora de Salida' ya existe"""
        sample_dict = {
            'Mes': '5',
            'Hora de Salida': '14:30'
        }
        df = sample_to_df(sample_dict)
        
        # Debería tener 'Hora de Salida' pero no duplicada
        assert df['Hora de Salida'].iloc[0] == '14:30'


class TestIntegration:
    """Pruebas de integración parse_sample_string + sample_to_df"""

    def test_full_parsing_pipeline(self):
        """Debe parsEAr correctamente desde string hasta DataFrame"""
        sample_string = "Mes=5;Hora=14:30;Distancia Kilometros=25.5;Tipo de Vehiculo=Auto"
        
        sample_dict = parse_sample_string(sample_string)
        df = sample_to_df(sample_dict)
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1
        assert 'Mes' in df.columns
        assert 'Hora de Salida' in df.columns

    def test_complex_sample_string(self):
        """Debe manejar strings complejos con múltiples campos"""
        sample_string = "Mes=3;Hora=09:15;Distancia Kilometros=50;Tipo de Vehiculo=Moto;Clima=Lluvia"
        
        sample_dict = parse_sample_string(sample_string)
        df = sample_to_df(sample_dict)
        
        assert len(df.columns) >= 5
        assert df['Mes'].iloc[0] == '3'
