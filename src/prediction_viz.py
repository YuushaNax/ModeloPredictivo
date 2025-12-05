"""
Módulo para mejorar la visualización de predicciones
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns
import os


def create_prediction_summary_chart(predictions_df, output_path=None):
    """
    Crea un gráfico de resumen de predicciones con métricas principales.
    
    Args:
        predictions_df: DataFrame con predicciones
        output_path: Ruta donde guardar la imagen
    
    Returns:
        Ruta de la imagen guardada
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Resumen de Predicciones', fontsize=16, fontweight='bold')
    
    # 1. Distribución de Probabilidad de Accidente
    if 'Probabilidad de accidente' in predictions_df.columns:
        ax = axes[0, 0]
        data = predictions_df['Probabilidad de accidente'].dropna()
        ax.hist(data, bins=30, color='#FF6B6B', alpha=0.7, edgecolor='black')
        ax.axvline(data.mean(), color='red', linestyle='--', linewidth=2, label=f'Media: {data.mean():.1f}%')
        ax.set_xlabel('Probabilidad de Accidente (%)', fontsize=11)
        ax.set_ylabel('Frecuencia', fontsize=11)
        ax.set_title('Distribución: Probabilidad de Accidente', fontweight='bold')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
    
    # 2. Distribución de Condición del Vehículo
    if 'Condicion del Vehiculo' in predictions_df.columns:
        ax = axes[0, 1]
        data = predictions_df['Condicion del Vehiculo'].dropna()
        ax.hist(data, bins=30, color='#4ECDC4', alpha=0.7, edgecolor='black')
        ax.axvline(data.mean(), color='darkgreen', linestyle='--', linewidth=2, label=f'Media: {data.mean():.1f}%')
        ax.set_xlabel('Condición del Vehículo (%)', fontsize=11)
        ax.set_ylabel('Frecuencia', fontsize=11)
        ax.set_title('Distribución: Condición del Vehículo', fontweight='bold')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
    
    # 3. Conteo de Accidentes Predichos
    if 'Accidente' in predictions_df.columns:
        ax = axes[1, 0]
        data = predictions_df['Accidente'].value_counts()
        colors = ['#FFD93D', '#FF6B6B']
        ax.bar(data.index, data.values, color=colors[:len(data)], edgecolor='black', linewidth=2)
        ax.set_xlabel('Accidente', fontsize=11)
        ax.set_ylabel('Cantidad', fontsize=11)
        ax.set_title('Conteo: Accidentes Predichos', fontweight='bold')
        for i, v in enumerate(data.values):
            ax.text(i, v + 5, str(v), ha='center', fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
    
    # 4. Scatter: Probabilidad vs Condición
    if 'Probabilidad de accidente' in predictions_df.columns and 'Condicion del Vehiculo' in predictions_df.columns:
        ax = axes[1, 1]
        data = predictions_df[['Probabilidad de accidente', 'Condicion del Vehiculo']].dropna()
        
        if 'Accidente' in predictions_df.columns:
            # Colorear según accidente
            colors = ['#FFD93D' if x == 'No' else '#FF6B6B' for x in predictions_df.loc[data.index, 'Accidente']]
        else:
            colors = '#4ECDC4'
        
        ax.scatter(data['Probabilidad de accidente'], data['Condicion del Vehiculo'], 
                  alpha=0.6, s=100, c=colors, edgecolors='black', linewidth=1)
        ax.set_xlabel('Probabilidad de Accidente (%)', fontsize=11)
        ax.set_ylabel('Condición del Vehículo (%)', fontsize=11)
        ax.set_title('Relación: Probabilidad vs Condición', fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Agregar leyenda si hay datos de accidente
        if 'Accidente' in predictions_df.columns:
            yellow_patch = mpatches.Patch(color='#FFD93D', label='No Accidente')
            red_patch = mpatches.Patch(color='#FF6B6B', label='Accidente')
            ax.legend(handles=[yellow_patch, red_patch], loc='best')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        return output_path
    
    return fig


def create_detailed_metrics_chart(predictions_df, output_path=None):
    """
    Crea un gráfico con métricas detalladas de las predicciones.
    
    Args:
        predictions_df: DataFrame con predicciones
        output_path: Ruta donde guardar la imagen
    
    Returns:
        Ruta de la imagen guardada
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle('Métricas Detalladas de Predicciones', fontsize=14, fontweight='bold')
    
    # 1. Box plot de Probabilidad de Accidente por categoría
    if 'Probabilidad de accidente' in predictions_df.columns and 'Accidente' in predictions_df.columns:
        ax = axes[0]
        data_to_plot = [predictions_df[predictions_df['Accidente'] == 'No']['Probabilidad de accidente'].dropna(),
                       predictions_df[predictions_df['Accidente'] == 'Sí']['Probabilidad de accidente'].dropna()]
        bp = ax.boxplot(data_to_plot, labels=['No Accidente', 'Accidente'], patch_artist=True)
        
        colors = ['#FFD93D', '#FF6B6B']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax.set_ylabel('Probabilidad de Accidente (%)', fontsize=11)
        ax.set_title('Box Plot: Probabilidad por Tipo', fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
    
    # 2. Box plot de Condición del Vehículo
    if 'Condicion del Vehiculo' in predictions_df.columns and 'Accidente' in predictions_df.columns:
        ax = axes[1]
        data_to_plot = [predictions_df[predictions_df['Accidente'] == 'No']['Condicion del Vehiculo'].dropna(),
                       predictions_df[predictions_df['Accidente'] == 'Sí']['Condicion del Vehiculo'].dropna()]
        bp = ax.boxplot(data_to_plot, labels=['No Accidente', 'Accidente'], patch_artist=True)
        
        colors = ['#FFD93D', '#FF6B6B']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax.set_ylabel('Condición del Vehículo (%)', fontsize=11)
        ax.set_title('Box Plot: Condición por Tipo', fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
    
    # 3. Tabla de resumen de estadísticas
    ax = axes[2]
    ax.axis('off')
    
    stats_data = []
    
    if 'Probabilidad de accidente' in predictions_df.columns:
        data = predictions_df['Probabilidad de accidente'].dropna()
        stats_data.append(['Probabilidad Accidente', 'Mín', f'{data.min():.1f}%'])
        stats_data.append(['', 'Máx', f'{data.max():.1f}%'])
        stats_data.append(['', 'Media', f'{data.mean():.1f}%'])
        stats_data.append(['', 'Std Dev', f'{data.std():.1f}%'])
    
    if 'Condicion del Vehiculo' in predictions_df.columns:
        data = predictions_df['Condicion del Vehiculo'].dropna()
        stats_data.append(['Condición Vehículo', 'Mín', f'{data.min():.1f}%'])
        stats_data.append(['', 'Máx', f'{data.max():.1f}%'])
        stats_data.append(['', 'Media', f'{data.mean():.1f}%'])
        stats_data.append(['', 'Std Dev', f'{data.std():.1f}%'])
    
    if 'Accidente' in predictions_df.columns:
        total = len(predictions_df)
        si_count = (predictions_df['Accidente'] == 'Sí').sum()
        no_count = (predictions_df['Accidente'] == 'No').sum()
        si_pct = (si_count / total * 100) if total > 0 else 0
        no_pct = (no_count / total * 100) if total > 0 else 0
        
        stats_data.append(['Accidentes', 'Total', f'{total}'])
        stats_data.append(['', 'Sí (Accidente)', f'{si_count} ({si_pct:.1f}%)'])
        stats_data.append(['', 'No (Seguro)', f'{no_count} ({no_pct:.1f}%)'])
    
    if stats_data:
        table = ax.table(cellText=stats_data, colLabels=['Métrica', 'Estadística', 'Valor'],
                        cellLoc='left', loc='center', colWidths=[0.35, 0.3, 0.35])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # Colorear encabezado
        for i in range(3):
            table[(0, i)].set_facecolor('#4ECDC4')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Alternar colores de filas
        for i in range(1, len(stats_data) + 1):
            for j in range(3):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#F0F0F0')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        return output_path
    
    return fig


def create_prediction_quality_chart(predictions_df, output_path=None):
    """
    Crea un gráfico de calidad de predicciones.
    
    Args:
        predictions_df: DataFrame con predicciones
        output_path: Ruta donde guardar la imagen
    
    Returns:
        Ruta de la imagen guardada
    """
    fig = plt.figure(figsize=(12, 6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1])
    
    fig.suptitle('Calidad de Predicciones', fontsize=14, fontweight='bold')
    
    # 1. Rango de probabilidad de accidente
    ax1 = fig.add_subplot(gs[0])
    if 'Probabilidad de accidente' in predictions_df.columns:
        data = predictions_df['Probabilidad de accidente'].dropna()
        
        ranges = ['0-20%', '20-40%', '40-60%', '60-80%', '80-100%']
        bins = [0, 20, 40, 60, 80, 100]
        counts = pd.cut(data, bins=bins).value_counts().sort_index()
        
        colors_range = ['#FFD93D', '#FFB84D', '#FF9B4D', '#FF7B4D', '#FF6B6B']
        ax1.bar(range(len(counts)), counts.values, color=colors_range, edgecolor='black', linewidth=2)
        ax1.set_xticks(range(len(ranges)))
        ax1.set_xticklabels(ranges)
        ax1.set_xlabel('Rango de Probabilidad', fontsize=11)
        ax1.set_ylabel('Cantidad', fontsize=11)
        ax1.set_title('Distribución de Riesgos', fontweight='bold')
        
        for i, v in enumerate(counts.values):
            ax1.text(i, v + 2, str(v), ha='center', fontweight='bold')
        
        ax1.grid(axis='y', alpha=0.3)
    
    # 2. Matriz de confusión si tenemos datos binarios
    ax2 = fig.add_subplot(gs[1])
    
    if 'Accidente' in predictions_df.columns and 'Accidente_proba' in predictions_df.columns:
        # Crear predicciones binarias basadas en probabilidad
        predictions = (predictions_df['Accidente_proba'] >= 0.5).astype(int)
        actual = (predictions_df['Accidente'] == 'Sí').astype(int)
        
        # Calcular matriz de confusión
        from sklearn.metrics import confusion_matrix
        try:
            cm = confusion_matrix(actual, predictions)
            sns.heatmap(cm, annot=True, fmt='d', cmap='RdYlGn_r', ax=ax2, cbar=False,
                       xticklabels=['Predicho: No', 'Predicho: Sí'],
                       yticklabels=['Real: No', 'Real: Sí'])
            ax2.set_title('Matriz de Confusión (Umbral: 50%)', fontweight='bold')
            ax2.set_ylabel('Valor Real')
            ax2.set_xlabel('Valor Predicho')
        except Exception:
            ax2.text(0.5, 0.5, 'No hay suficientes datos', ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('Matriz de Confusión', fontweight='bold')
    else:
        ax2.text(0.5, 0.5, 'Datos insuficientes', ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('Matriz de Confusión', fontweight='bold')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        return output_path
    
    return fig


def generate_all_prediction_visualizations(predictions_df, output_dir='reports/gui_temp'):
    """
    Genera todos los gráficos de predicción.
    
    Args:
        predictions_df: DataFrame con predicciones
        output_dir: Directorio de salida
    
    Returns:
        Lista de rutas de las imágenes generadas
    """
    os.makedirs(output_dir, exist_ok=True)
    
    generated_files = []
    
    try:
        path = os.path.join(output_dir, 'prediction_summary.png')
        create_prediction_summary_chart(predictions_df, path)
        generated_files.append(path)
    except Exception as e:
        print(f"Error generando summary chart: {e}")
    
    try:
        path = os.path.join(output_dir, 'detailed_metrics.png')
        create_detailed_metrics_chart(predictions_df, path)
        generated_files.append(path)
    except Exception as e:
        print(f"Error generando detailed metrics: {e}")
    
    try:
        path = os.path.join(output_dir, 'prediction_quality.png')
        create_prediction_quality_chart(predictions_df, path)
        generated_files.append(path)
    except Exception as e:
        print(f"Error generando quality chart: {e}")
    
    return generated_files
