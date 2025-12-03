import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from src.utils import load_and_prepare


def make_plots(df, outdir='reports'):
    os.makedirs(outdir, exist_ok=True)

    # Normalize column names: strip spaces
    df.columns = [str(c).strip() for c in df.columns]
    print('Nombres de columnas:', df.columns.tolist())
    print('Número de filas:', len(df))

    # Basic histograms
    # Try multiple possible column names for probability
    prob_cols = [c for c in df.columns if 'Probabilidad de accidente' in c]
    print('Columnas posibles para probabilidad:', prob_cols)
    col_prob = prob_cols[0] if prob_cols else None
    if col_prob and len(df) > 1:
        # Try to parse percent strings if needed
        vals = df[col_prob]
        if vals.dtype == object:
            # Remove % and convert
            vals = vals.astype(str).str.replace('%', '').str.replace(',', '.').str.strip()
        vals = pd.to_numeric(vals, errors='coerce').dropna()
        print(f'Histograma: {len(vals)} valores válidos')
        if len(vals) > 1:
            plt.figure(figsize=(8, 4))
            vals.hist(bins=30)
            plt.title(f'Distribución: {col_prob}')
            plt.savefig(os.path.join(outdir, 'hist_probabilidad.png'))
            plt.close()
        else:
            plt.figure(figsize=(8, 4))
            plt.text(0.5, 0.5, 'No hay suficientes datos para el histograma', ha='center', va='center')
            plt.title(f'Distribución: {col_prob}')
            plt.savefig(os.path.join(outdir, 'hist_probabilidad.png'))
            plt.close()
    else:
        plt.figure(figsize=(8, 4))
        plt.text(0.5, 0.5, 'Columna faltante o datos insuficientes', ha='center', va='center')
        plt.title('Distribución: Probabilidad de accidente (%)')
        plt.savefig(os.path.join(outdir, 'hist_probabilidad.png'))
        plt.close()

    # Accident counts
    col_acc = [c for c in df.columns if c.lower().strip() == 'accidente']
    col_acc = col_acc[0] if col_acc else None
    if col_acc and len(df) > 1:
        vals = df[col_acc].dropna()
        print(f'Conteo: {len(vals)} valores válidos')
        if len(vals) > 1:
            plt.figure(figsize=(6, 4))
            sns.countplot(x=col_acc, data=df)
            plt.title('Conteo: Accidente (Sí/No)')
            plt.savefig(os.path.join(outdir, 'count_accidente.png'))
            plt.close()
        else:
            plt.figure(figsize=(6, 4))
            plt.text(0.5, 0.5, 'No hay suficientes datos para el conteo', ha='center', va='center')
            plt.title('Conteo: Accidente (Sí/No)')
            plt.savefig(os.path.join(outdir, 'count_accidente.png'))
            plt.close()
    else:
        plt.figure(figsize=(6, 4))
        plt.text(0.5, 0.5, 'Columna faltante o datos insuficientes', ha='center', va='center')
        plt.title('Conteo: Accidente (Sí/No)')
        plt.savefig(os.path.join(outdir, 'count_accidente.png'))
        plt.close()

    # Correlation heatmap for numeric columns
    nums = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    if len(nums) > 1 and len(df) > 1:
        corr = df[nums].corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm')
        plt.title('Correlación (numéricas)')
        plt.savefig(os.path.join(outdir, 'corr_numeric.png'))
        plt.close()
    else:
        plt.figure(figsize=(10, 8))
        plt.text(0.5, 0.5, 'No hay suficientes datos para la correlación', ha='center', va='center')
        plt.title('Correlación (numéricas)')
        plt.savefig(os.path.join(outdir, 'corr_numeric.png'))
        plt.close()

    # Boxplot of probability by vehicle type if available and valid
    if ('Tipo de Vehiculo' in df.columns and col_prob):
        valid = df[['Tipo de Vehiculo', col_prob]].dropna()
        if not valid.empty and valid['Tipo de Vehiculo'].nunique() > 1:
            plt.figure(figsize=(8, 4))
            sns.boxplot(x='Tipo de Vehiculo', y=col_prob, data=valid)
            plt.title('Probabilidad por Tipo de Vehículo')
            plt.savefig(os.path.join(outdir, 'box_prob_by_vehicle.png'))
            plt.close()

    print(f'Gráficas guardadas en {outdir}')


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--out', default='reports')
    args = parser.parse_args()

    df = load_and_prepare(args.data)
    make_plots(df, outdir=args.out)


if __name__ == '__main__':
    main()
