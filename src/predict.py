import argparse
import joblib
import pandas as pd
import numpy as np

from src.utils import load_and_prepare


def parse_sample_string(s):
    # Expect format: "K=V;K=V;..."
    d = {}
    parts = s.split(';')
    for p in parts:
        if '=' in p:
            k,v = p.split('=',1)
            d[k.strip()] = v.strip()
    return d


def sample_to_df(sample_dict):
    # Build a single-row dataframe with expected column names
    df = pd.DataFrame([sample_dict])
    # If Hora provided as '13:45', map to 'Hora de Salida'
    if 'Hora' in df.columns and 'Hora de Salida' not in df.columns:
        df['Hora de Salida'] = df['Hora']
        df = df.drop(columns=['Hora'])
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample', required=True, help='Sample as K=V;K=V string')
    parser.add_argument('--models_dir', default='models', help='Folder where models were saved')
    args = parser.parse_args()

    d = parse_sample_string(args.sample)
    df = sample_to_df(d)
    df = load_and_prepare(df)

    prob_pipe = joblib.load(f"{args.models_dir}/prob_pipeline.joblib")
    cond_pipe = joblib.load(f"{args.models_dir}/cond_pipeline.joblib")
    acc_pipe = joblib.load(f"{args.models_dir}/acc_pipeline.joblib")

    # Determine the columns expected by the preprocessor inside the saved pipeline
    try:
        pre = prob_pipe.named_steps['pre']
        num_cols = pre.transformers[0][2]
        cat_cols = pre.transformers[1][2]
        expected_cols = list(num_cols) + list(cat_cols)
    except Exception:
        # fallback to a minimal set
        expected_cols = ['Mes', 'Hora_hour', 'Hora_min', 'Distancia Kilometros', 'Tipo de Vehiculo', 'Clima']

    # Ensure all expected columns exist in the dataframe; fill missing with NaN
    for c in expected_cols:
        if c not in df.columns:
            df[c] = np.nan

    X = df[expected_cols]

    # Predict
    p = float(prob_pipe.predict(X)[0])
    cval = float(cond_pipe.predict(X)[0])
    a = acc_pipe.predict(X)[0]

    # Try to get probability for the positive class if available
    a_proba = None
    try:
        if hasattr(acc_pipe, 'predict_proba'):
            a_proba = float(acc_pipe.predict_proba(X)[0, 1])
    except Exception:
        a_proba = None

    # Post-process / clip regression outputs to sensible ranges
    p = float(np.clip(p, 0.0, 100.0))
    cval = float(np.clip(cval, 0.0, 100.0))

    # Print nicely
    print(f'Predicted Probabilidad de accidente: {p:.1f}%')
    print(f'Predicted Condicion del Vehiculo: {cval:.1f}%')
    print(f'Predicted Accidente: {"Sí" if int(a) == 1 else "No"} ({int(a)})')
    if a_proba is not None:
        print(f'Predicted Accidente_proba: {a_proba*100:.1f}%')


if __name__ == '__main__':
    main()
