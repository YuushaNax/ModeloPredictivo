"""
Pruebas unitarias para src/train.py
"""
import pytest
import pandas as pd
import numpy as np
import os
import joblib
from sklearn.pipeline import Pipeline
from src.train import build_preprocessor, train_models
from src.utils import load_and_prepare


class TestBuildPreprocessor:
    """Pruebas para la función build_preprocessor"""

    def test_build_preprocessor_returns_column_transformer(self):
        """Debe retornar un ColumnTransformer"""
        num_cols = ['Mes', 'Distancia Kilometros']
        cat_cols = ['Tipo de Vehiculo', 'Clima']
        preprocessor = build_preprocessor(num_cols, cat_cols)
        
        from sklearn.compose import ColumnTransformer
        assert isinstance(preprocessor, ColumnTransformer)

    def test_preprocessor_has_num_and_cat_transformers(self):
        """El preprocessor debe tener transformadores para numéricas y categóricas"""
        num_cols = ['Mes', 'Distancia Kilometros']
        cat_cols = ['Tipo de Vehiculo', 'Clima']
        preprocessor = build_preprocessor(num_cols, cat_cols)
        
        # Verificar que tiene los transformadores
        assert hasattr(preprocessor, 'transformers') or hasattr(preprocessor, 'named_transformers_')
        
        # Después de fit, podemos acceder a los transformadores
        try:
            transformers = dict(preprocessor.transformers_)
        except AttributeError:
            # En algunas versiones de sklearn
            transformers = {name: trans for name, trans, _ in preprocessor.transformers}
        
        assert 'num' in transformers or any('num' in str(t) for t in transformers)
        assert 'cat' in transformers or any('cat' in str(t) for t in transformers)

    def test_preprocessor_fit_transform(self, sample_dataframe):
        """El preprocessor debe poder ajustarse y transformar datos"""
        num_cols = ['Mes', 'Distancia Kilometros']
        cat_cols = ['Tipo de Vehiculo', 'Clima']
        
        X = sample_dataframe[num_cols + cat_cols]
        
        preprocessor = build_preprocessor(num_cols, cat_cols)
        X_transformed = preprocessor.fit_transform(X)
        
        assert X_transformed.shape[0] == len(X)
        assert X_transformed.shape[1] > 0


class TestTrainModels:
    """Pruebas para la función train_models"""

    def test_train_models_creates_three_pipelines(self, sample_dataframe, temp_models_dir):
        """Debe crear tres archivos de modelos guardados"""
        # Expandir datos para tener suficientes muestras
        extended_df = pd.concat([sample_dataframe] * 20, ignore_index=True)
        train_models(extended_df, out_dir=temp_models_dir)
        
        assert os.path.exists(os.path.join(temp_models_dir, 'prob_pipeline.joblib'))
        assert os.path.exists(os.path.join(temp_models_dir, 'cond_pipeline.joblib'))
        assert os.path.exists(os.path.join(temp_models_dir, 'acc_pipeline.joblib'))

    def test_train_models_saves_valid_pipelines(self, sample_dataframe, temp_models_dir):
        """Los modelos guardados deben ser pipelines válidos"""
        extended_df = pd.concat([sample_dataframe] * 20, ignore_index=True)
        train_models(extended_df, out_dir=temp_models_dir)
        
        prob_pipe = joblib.load(os.path.join(temp_models_dir, 'prob_pipeline.joblib'))
        cond_pipe = joblib.load(os.path.join(temp_models_dir, 'cond_pipeline.joblib'))
        acc_pipe = joblib.load(os.path.join(temp_models_dir, 'acc_pipeline.joblib'))
        
        assert hasattr(prob_pipe, 'predict')
        assert hasattr(cond_pipe, 'predict')
        assert hasattr(acc_pipe, 'predict')

    def test_train_models_with_sufficient_data(self, sample_dataframe, temp_models_dir):
        """Debe entrenar exitosamente con datos suficientes"""
        extended_df = pd.concat([sample_dataframe] * 30, ignore_index=True)
        
        train_models(extended_df, out_dir=temp_models_dir)
        
        assert os.path.exists(os.path.join(temp_models_dir, 'prob_pipeline.joblib'))

    def test_prob_model_makes_predictions(self, sample_dataframe, temp_models_dir):
        """El modelo de probabilidad debe hacer predicciones válidas"""
        extended_df = pd.concat([sample_dataframe] * 30, ignore_index=True)
        train_models(extended_df, out_dir=temp_models_dir)
        
        prob_pipe = joblib.load(os.path.join(temp_models_dir, 'prob_pipeline.joblib'))
        
        # Preparar datos de prueba
        features = ['Mes', 'Hora_hour', 'Hora_min', 'Distancia Kilometros', 
                   'Tipo de Vehiculo', 'Clima']
        
        test_sample = extended_df[features].iloc[:1]
        prediction = prob_pipe.predict(test_sample)
        
        assert len(prediction) == 1
        assert isinstance(prediction[0], (float, np.floating))
        assert 0 <= prediction[0] <= 100

    def test_acc_model_makes_predictions(self, sample_dataframe, temp_models_dir):
        """El modelo de accidente debe hacer predicciones de clase"""
        extended_df = pd.concat([sample_dataframe] * 30, ignore_index=True)
        train_models(extended_df, out_dir=temp_models_dir)
        
        acc_pipe = joblib.load(os.path.join(temp_models_dir, 'acc_pipeline.joblib'))
        
        features = ['Mes', 'Hora_hour', 'Hora_min', 'Distancia Kilometros', 
                   'Tipo de Vehiculo', 'Clima']
        
        test_sample = extended_df[features].iloc[:1]
        prediction = acc_pipe.predict(test_sample)
        
        assert len(prediction) == 1
        assert prediction[0] in [0, 1]

    def test_train_models_handles_missing_values(self, sample_dataframe, temp_models_dir):
        """Debe manejar correctamente valores faltantes"""
        df_with_nan = sample_dataframe.copy()
        df_with_nan.loc[0, 'Velocidad Promedio'] = np.nan
        df_with_nan.loc[1, 'Edad Conductor'] = np.nan
        
        extended_df = pd.concat([df_with_nan] * 30, ignore_index=True)
        
        # Debe ejecutar sin errores
        train_models(extended_df, out_dir=temp_models_dir)
        assert os.path.exists(os.path.join(temp_models_dir, 'prob_pipeline.joblib'))

    def test_train_models_with_minimal_rows(self, minimal_dataframe, temp_models_dir):
        """Debe entrenar incluso con número mínimo de filas"""
        extended_df = pd.concat([minimal_dataframe] * 15, ignore_index=True)
        
        train_models(extended_df, out_dir=temp_models_dir)
        assert os.path.exists(os.path.join(temp_models_dir, 'prob_pipeline.joblib'))
