import pandas as pd
import numpy as np
import re


def parse_percent_column(s):
    # Accept values like '12.3%' or numeric
    if pd.isna(s):
        return np.nan
    if isinstance(s, (int, float)):
        return float(s)
    try:
        return float(str(s).strip().replace('%',''))
    except Exception:
        return np.nan


def parse_time_column(t):
    # Expect formats like '13:45' or '9:05'
    if pd.isna(t):
        return (np.nan, np.nan)
    try:
        parts = str(t).split(':')
        hour = int(parts[0])
        minute = int(parts[1]) if len(parts) > 1 else 0
        return (hour, minute)
    except Exception:
        return (np.nan, np.nan)


def load_and_prepare(path_or_df):
    """Load a CSV or Excel file (or accept a DataFrame) and return a cleaned DataFrame.

    Expected columns (any language variations):
    - 'Mes'
    - 'Hora de Salida'
    - 'Distancia Kilometros'
    - 'Tipo de Vehiculo'
    - 'Clima'
    - 'Probabilidad de accidente' (percent string)
    - 'Accidente' ('Sí'/'No')
    - 'Condicion del Vehiculo' (percent string)
    """
    if isinstance(path_or_df, pd.DataFrame):
        df = path_or_df.copy()
    else:
        if str(path_or_df).lower().endswith('.xlsx') or str(path_or_df).lower().endswith('.xls'):
            df = pd.read_excel(path_or_df)
        else:
            df = pd.read_csv(path_or_df)

    # Normalize column names (strip)
    df.columns = [c.strip() for c in df.columns]

    # Percent columns
    if 'Probabilidad de accidente' in df.columns:
        df['Probabilidad de accidente (%)'] = df['Probabilidad de accidente'].apply(parse_percent_column)
    if 'Condicion del Vehiculo' in df.columns:
        df['Condicion del Vehiculo (%)'] = df['Condicion del Vehiculo'].apply(parse_percent_column)

    # Accident binary
    if 'Accidente' in df.columns:
        df['Accidente_bin'] = df['Accidente'].astype(str).str.strip().str.lower().map({'sí':1, 'si':1, 's':1, 'yes':1, 'no':0, 'n':0})

    # Time split
    if 'Hora de Salida' in df.columns:
        times = df['Hora de Salida'].apply(parse_time_column)
        df['Hora_hour'] = times.apply(lambda x: x[0])
        df['Hora_min'] = times.apply(lambda x: x[1])

    # Ensure numeric types
    if 'Mes' in df.columns:
        df['Mes'] = pd.to_numeric(df['Mes'], errors='coerce')
    if 'Distancia Kilometros' in df.columns:
        df['Distancia Kilometros'] = pd.to_numeric(df['Distancia Kilometros'], errors='coerce')
    
        # New numeric columns
        if 'Velocidad Promedio' in df.columns:
            df['Velocidad Promedio'] = pd.to_numeric(df['Velocidad Promedio'], errors='coerce')
        if 'Edad Conductor' in df.columns:
            df['Edad Conductor'] = pd.to_numeric(df['Edad Conductor'], errors='coerce')
        if 'Experiencia Conductor' in df.columns:
            df['Experiencia Conductor'] = pd.to_numeric(df['Experiencia Conductor'], errors='coerce')
        if 'Dia de la Semana' in df.columns:
            df['Dia de la Semana'] = pd.to_numeric(df['Dia de la Semana'], errors='coerce')

        # Normalize some categorical fields (strip)
        for c in ['Tipo de Carretera','Visibilidad','Estado de la Via','Iluminacion','Cinturon']:
            if c in df.columns:
                df[c] = df[c].astype(str).str.strip()

    return df
