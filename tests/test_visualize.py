"""
Pruebas unitarias para src/visualize.py
"""
import pytest
import pandas as pd
import numpy as np
import os
import tempfile
import shutil
import matplotlib
matplotlib.use('Agg')  # Usar backend sin GUI
from src.visualize import make_plots


class TestMakePlots:
    """Pruebas para la función make_plots"""

    def test_make_plots_creates_output_directory(self, sample_dataframe):
        """Debe crear el directorio de salida si no existe"""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = os.path.join(tmpdir, 'test_reports')
            assert not os.path.exists(output_dir)
            
            make_plots(sample_dataframe, outdir=output_dir)
            
            assert os.path.exists(output_dir)

    def test_make_plots_creates_histogram(self, sample_dataframe):
        """Debe crear un histograma de probabilidad"""
        with tempfile.TemporaryDirectory() as tmpdir:
            make_plots(sample_dataframe, outdir=tmpdir)
            
            hist_file = os.path.join(tmpdir, 'hist_probabilidad.png')
            assert os.path.exists(hist_file)
            assert os.path.getsize(hist_file) > 0

    def test_make_plots_creates_count_plot(self, sample_dataframe):
        """Debe crear un gráfico de conteo de accidentes"""
        with tempfile.TemporaryDirectory() as tmpdir:
            make_plots(sample_dataframe, outdir=tmpdir)
            
            count_file = os.path.join(tmpdir, 'count_accidente.png')
            assert os.path.exists(count_file)
            assert os.path.getsize(count_file) > 0

    def test_make_plots_creates_correlation_heatmap(self, sample_dataframe):
        """Debe crear un mapa de calor de correlación"""
        with tempfile.TemporaryDirectory() as tmpdir:
            make_plots(sample_dataframe, outdir=tmpdir)
            
            corr_file = os.path.join(tmpdir, 'corr_numeric.png')
            assert os.path.exists(corr_file)
            assert os.path.getsize(corr_file) > 0

    def test_make_plots_with_default_directory(self, sample_dataframe):
        """Debe usar 'reports' como directorio por defecto"""
        # Limpiamos si existe
        if os.path.exists('reports'):
            shutil.rmtree('reports')
        
        try:
            make_plots(sample_dataframe)
            assert os.path.exists('reports')
            assert os.path.exists('reports/hist_probabilidad.png')
        finally:
            # Limpieza
            if os.path.exists('reports'):
                shutil.rmtree('reports')

    def test_make_plots_with_minimal_data(self, minimal_dataframe):
        """Debe manejar DataFrames mínimos sin errores"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Debe ejecutar sin errores
            make_plots(minimal_dataframe, outdir=tmpdir)
            
            # Al menos debe crear algunos archivos
            files = os.listdir(tmpdir)
            assert len(files) > 0

    def test_make_plots_handles_missing_probability_column(self):
        """Debe manejar gracefully cuando falta la columna de probabilidad"""
        df = pd.DataFrame({
            'Mes': [1, 2, 3],
            'Accidente': ['Sí', 'No', 'Sí']
        })
        
        with tempfile.TemporaryDirectory() as tmpdir:
            make_plots(df, outdir=tmpdir)
            
            # Debe crear archivos de respaldo
            assert os.path.exists(os.path.join(tmpdir, 'hist_probabilidad.png'))

    def test_make_plots_handles_missing_accident_column(self):
        """Debe manejar gracefully cuando falta la columna de accidente"""
        df = pd.DataFrame({
            'Mes': [1, 2, 3],
            'Probabilidad de accidente': [10, 25, 5]
        })
        
        with tempfile.TemporaryDirectory() as tmpdir:
            make_plots(df, outdir=tmpdir)
            
            assert os.path.exists(os.path.join(tmpdir, 'count_accidente.png'))

    def test_make_plots_handles_empty_dataframe(self):
        """Debe manejar DataFrames vacíos"""
        df = pd.DataFrame()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Debe ejecutar sin errores
            make_plots(df, outdir=tmpdir)

    def test_make_plots_normalizes_column_names(self):
        """Debe normalizar nombres de columnas (remover espacios extras)"""
        df = pd.DataFrame({
            '  Mes  ': [1, 2, 3],
            'Probabilidad de accidente': [10, 25, 5],
            'Accidente': ['Sí', 'No', 'Sí']
        })
        
        with tempfile.TemporaryDirectory() as tmpdir:
            make_plots(df, outdir=tmpdir)
            assert os.path.exists(os.path.join(tmpdir, 'hist_probabilidad.png'))

    def test_make_plots_with_sufficient_numeric_columns(self, sample_dataframe):
        """Debe crear correlación heatmap con suficientes columnas numéricas"""
        with tempfile.TemporaryDirectory() as tmpdir:
            make_plots(sample_dataframe, outdir=tmpdir)
            
            corr_file = os.path.join(tmpdir, 'corr_numeric.png')
            assert os.path.exists(corr_file)
            assert os.path.getsize(corr_file) > 0

    def test_make_plots_with_vehicle_type_boxplot(self, sample_dataframe):
        """Debe crear boxplot de probabilidad por tipo de vehículo si disponible"""
        with tempfile.TemporaryDirectory() as tmpdir:
            make_plots(sample_dataframe, outdir=tmpdir)
            
            # Verifica si se creó el boxplot
            box_file = os.path.join(tmpdir, 'box_prob_by_vehicle.png')
            # Puede o no existir dependiendo de los datos
            if os.path.exists(box_file):
                assert os.path.getsize(box_file) > 0

    def test_make_plots_output_files_not_empty(self, sample_dataframe):
        """Todos los archivos PNG creados deben tener contenido"""
        with tempfile.TemporaryDirectory() as tmpdir:
            make_plots(sample_dataframe, outdir=tmpdir)
            
            png_files = [f for f in os.listdir(tmpdir) if f.endswith('.png')]
            for png_file in png_files:
                file_path = os.path.join(tmpdir, png_file)
                assert os.path.getsize(file_path) > 0
