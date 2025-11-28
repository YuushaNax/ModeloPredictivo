import pandas as pd

# Valores permitidos
VALID_CLIMA = ["Despejado", "Lluvia", "Nublado", "Tormenta", "Nieve"]
VALID_TIPO_VEHICULO = ["Auto", "Moto", "Camion", "Bus"]
VALID_CARRETERA = ["Urbana", "Autopista", "Rural"]
VALID_DIA = ["Lunes", "Martes", "Miercoles", "Jueves", "Viernes", "Sabado", "Domingo"]
VALID_ILUMINACION = ["Dia", "Noche"]
VALID_ESTADO_VIA = ["Seca", "Mojada", "Nevada", "Congelada"]

def validate_features(df: pd.DataFrame):
    """
    Verifica que todos los valores categóricos estén dentro del rango permitido.
    """
    if "Clima" in df:
        if not df["Clima"].isin(VALID_CLIMA).all():
            raise ValueError("Clima contiene valores inválidos.")

    if "Tipo de Vehiculo" in df:
        if not df["Tipo de Vehiculo"].isin(VALID_TIPO_VEHICULO).all():
            raise ValueError("Tipo de Vehiculo contiene valores inválidos.")

    if "Tipo de Carretera" in df:
        if not df["Tipo de Carretera"].isin(VALID_CARRETERA).all():
            raise ValueError("Tipo de Carretera contiene valores inválidos.")


def clean_features(df: pd.DataFrame):
    # Convertir Hora a número (ya viene como número)
    df["Hora de Salida"] = pd.to_numeric(df["Hora de Salida"], errors="coerce")

    return df


def encode_categoricals(df: pd.DataFrame):
    """
    Transformación de variables categóricas.
    """
    cat_cols = [
        "Mes", "Tipo de Vehiculo", "Clima", "Dia de la Semana",
        "Tipo de Carretera", "Estado de la Via", "Iluminacion", "Cinturon"
    ]

    df = pd.get_dummies(df, columns=cat_cols)
    return df


def extract_features(df: pd.DataFrame):
    df = clean_features(df)
    validate_features(df)
    df = encode_categoricals(df)
    return df
