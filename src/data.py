import pandas as pd
import os

DEFAULT_PATH = "dataset.xlsx"

REQUIRED_COLUMNS = [
    "Mes", "Hora de Salida", "Distancia Kilometros", "Tipo de Vehiculo",
    "Clima", "Dia de la Semana", "Tipo de Carretera", "Velocidad Promedio",
    "Edad Conductor", "Experiencia Conductor", "Alcohol en Sangre",
    "Visibilidad", "Estado de la Via", "Iluminacion", "Cinturon",
    "Probabilidad de Accidente", "Accidente", "Condicion del Vehiculo"
]

def load_dataset(path: str = DEFAULT_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset no encontrado: {path}")

    df = pd.read_excel(path)
    return df


def validate_columns(df: pd.DataFrame):
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas obligatorias en dataset: {missing}")


def prepare_dataset(df: pd.DataFrame):
    validate_columns(df)

    # Limpieza básica
    df = df.dropna().reset_index(drop=True)

    # Conversión a tipos numéricos donde aplique
    numeric_cols = [
        "Hora de Salida", "Distancia Kilometros", "Velocidad Promedio",
        "Edad Conductor", "Experiencia Conductor", "Alcohol en Sangre",
        "Visibilidad"
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna().reset_index(drop=True)
    return df


def save_dataset(df: pd.DataFrame, path: str = DEFAULT_PATH):
    df.to_excel(path, index=False)
