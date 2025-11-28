import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from src.utils import load_and_prepare


def make_plots(df, outdir='reports'):
    os.makedirs(outdir, exist_ok=True)

    # Basic histograms
    plt.figure(figsize=(8,4))
    df['Probabilidad de accidente (%)'].hist(bins=30)
    plt.title('Distribución: Probabilidad de accidente (%)')
    plt.savefig(os.path.join(outdir,'hist_probabilidad.png'))
    plt.close()

    # Accident counts
    plt.figure(figsize=(6,4))
    sns.countplot(x='Accidente', data=df)
    plt.title('Conteo: Accidente (Sí/No)')
    plt.savefig(os.path.join(outdir,'count_accidente.png'))
    plt.close()

    # Correlation heatmap for numeric columns
    nums = df.select_dtypes(include=['float64','int64']).columns.tolist()
    if len(nums) > 1:
        corr = df[nums].corr()
        plt.figure(figsize=(10,8))
        sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm')
        plt.title('Correlación (numéricas)')
        plt.savefig(os.path.join(outdir,'corr_numeric.png'))
        plt.close()

    # Boxplot of probability by vehicle type if available
    if 'Tipo de Vehiculo' in df.columns:
        plt.figure(figsize=(8,4))
        sns.boxplot(x='Tipo de Vehiculo', y='Probabilidad de accidente (%)', data=df)
        plt.title('Probabilidad por Tipo de Vehiculo')
        plt.savefig(os.path.join(outdir,'box_prob_by_vehicle.png'))
        plt.close()

    print(f'Plots written to {outdir}')


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
