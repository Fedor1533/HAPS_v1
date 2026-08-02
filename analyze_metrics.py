import argparse
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

from scripts.eval_metrics import calculate_smart_auc



def load_and_prepare_data(csv_path, metric_columns):
    """
    Загружает CSV и подготавливает данные для анализа.
    
    Returns:
        df: DataFrame с данными
        metrics_dict: словарь {metric_name: scores_array}
        classes_3: массив классов (0, 1, 2)
        classes_2: массив классов (0 объединён с 1 -> 0, 2 -> 1)
    """
    df = pd.read_csv(csv_path)
    
    # Ищем колонку с классами (может называться 'class' или 'class3')
    class_col = None
    for possible_name in ['class', 'class3', 'Class', 'label', 'target']:
        if possible_name in df.columns:
            class_col = possible_name
            break
    
    if class_col is None:
        print(f"ОШИБКА: CSV должен содержать колонку с классами!")
        print(f"Доступные колонки: {list(df.columns)}")
        raise ValueError("CSV должен содержать колонку 'class' или 'class3'")
    
    print(f"Используется колонка классов: '{class_col}'")
    print(f"Уникальные значения классов: {sorted(df[class_col].unique())}")
    
    # Проверяем наличие метрик
    missing_metrics = [col for col in metric_columns if col not in df.columns]
    if missing_metrics:
        print(f"ВНИМАНИЕ: Следующие метрики не найдены в CSV: {missing_metrics}")
        metric_columns = [col for col in metric_columns if col in df.columns]
    
    if not metric_columns:
        print("ОШИБКА: Не найдено ни одной метрики!")
        print(f"Доступные колонки: {list(df.columns)}")
        raise ValueError("Не найдено метрик для анализа")
    
    # Извлекаем метрики
    metrics_dict = {}
    for col in metric_columns:
        # Проверяем на NaN
        values = df[col].values
        nan_count = np.sum(np.isnan(values))
        if nan_count > 0:
            print(f"ВНИМАНИЕ: В метрике '{col}' найдено {nan_count} NaN значений. Они будут удалены.")
            # Удаляем строки с NaN для этой метрики
            valid_mask = ~np.isnan(values)
            values = values[valid_mask]
        metrics_dict[col] = values
    
    # Классы для 3-классовой классификации
    classes_3 = df[class_col].values
    
    # Проверяем количество классов
    unique_classes = np.unique(classes_3)
    print(f"Количество классов: {len(unique_classes)}")
    print(f"Распределение классов:")
    for cls in unique_classes:
        count = np.sum(classes_3 == cls)
        print(f"  Класс {cls}: {count} записей ({100*count/len(classes_3):.1f}%)")
    
    # Классы для 2-классовой классификации 
    # Если классов 3: 0 и 1 объединяем в 0, 2 -> 1
    # Если классов 2: используем как есть
    if len(unique_classes) == 3:
        classes_2 = np.where(classes_3 == 2, 1, 0)
        print(f"\nДля 2-классовой классификации: классы 0 и 1 объединены в 0, класс 2 -> 1")
        print(f"  Класс 0 (good+norm): {np.sum(classes_2 == 0)} записей")
        print(f"  Класс 1 (other): {np.sum(classes_2 == 1)} записей")
    else:
        classes_2 = classes_3.copy()
        print(f"\nДля 2-классовой классификации используются исходные классы")
    
    return df, metrics_dict, classes_3, classes_2



def plot_metrics_boxplots(metrics_dict, classes, class_type='3class', 
                         figsize=(14, 8), save_dir='plots', dpi=150, 
                         format='png', palette='Set2'):
    """
    Строит боксплоты для всех метрик по классам.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    unique_classes = np.unique(classes)
    n_metrics = len(metrics_dict)
    
    if n_metrics == 0:
        print("Нет метрик для визуализации")
        return None
    
    # Создаем фигуру с подграфиками
    n_cols = min(3, n_metrics)
    n_rows = (n_metrics + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, 
                            figsize=(6 * n_cols, 5 * n_rows),
                            squeeze=False)
    
    metric_names = list(metrics_dict.keys())
    
    for idx, metric_name in enumerate(metric_names):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row][col]
        
        scores = np.array(metrics_dict[metric_name])
        
        # Подготавливаем данные для боксплота
        boxplot_data = []
        boxplot_labels = []
        
        for cls in unique_classes:
            class_scores = scores[classes == cls]
            # Убираем NaN
            class_scores = class_scores[~np.isnan(class_scores)]
            
            if len(class_scores) >= 3:
                boxplot_data.append(class_scores)
                if class_type == '2class':
                    label = 'Good+Norm (0)' if cls == 0 else f'Class {cls}'
                else:
                    class_names = {0: 'Good (0)', 1: 'Norm (1)', 2: 'Other (2)'}
                    label = class_names.get(cls, f'Class {cls}')
                boxplot_labels.append(label)
        
        if boxplot_data:
            bp = ax.boxplot(boxplot_data, labels=boxplot_labels,
                          patch_artist=True, showfliers=True,
                          showmeans=True, meanprops=dict(marker='D', 
                                                       markerfacecolor='red',
                                                       markersize=6))
            
            colors = plt.cm.get_cmap(palette, len(boxplot_data))
            for i, box in enumerate(bp['boxes']):
                box.set_facecolor(colors(i))
                box.set_alpha(0.7)
            
            # Добавляем jittered scatter для отдельных точек
            for i, data in enumerate(boxplot_data):
                jitter = np.random.normal(0, 0.04, size=len(data))
                ax.scatter(np.ones(len(data)) * (i + 1) + jitter, data, 
                         alpha=0.3, s=10, color='black')
        
        # Укорачиваем название метрики для заголовка
        short_name = metric_name.replace('_mist_mpp', '').replace('_', ' ')
        ax.set_title(short_name, fontsize=12, fontweight='bold')
        ax.set_ylabel('Score', fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
    
    # Убираем пустые subplots
    for idx in range(n_metrics, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        fig.delaxes(axes[row][col])
    
    plt.suptitle(f'Boxplots of Metrics by Class ({class_type})', 
                fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Сохраняем
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(save_dir, f'metrics_boxplots_{class_type}_{timestamp}.{format}')
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Сохранен боксплот: {save_path}")
    return save_path


def plot_auc_comparison(auc_results, save_dir='plots', figsize=(14, 6), 
                       dpi=150, format='png'):
    """
    Строит bar chart сравнения AUC для разных метрик.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # График для 3-классового AUC
    ax1 = axes[0]
    metrics_3 = [k for k, v in auc_results.items() if '3class' in k]
    metrics_3_names = [k.replace('_3class', '').replace('_mist_mpp', '').replace('_', ' ') for k in metrics_3]
    auc_3_values = [auc_results[k] for k in metrics_3]
    
    if metrics_3:
        colors_3 = plt.cm.get_cmap('viridis', len(metrics_3))
        bars = ax1.bar(range(len(metrics_3_names)), auc_3_values, 
                      color=[colors_3(i) for i in range(len(metrics_3))])
        ax1.set_xticks(range(len(metrics_3_names)))
        ax1.set_xticklabels(metrics_3_names, rotation=45, ha='right')
        ax1.set_ylabel('Macro-AUC')
        ax1.set_title('3-Class AUC (Good vs Norm vs Other)')
        ax1.set_ylim([0.4, 1.0])
        ax1.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Random')
        ax1.grid(True, alpha=0.3, axis='y')
        ax1.legend()
        
        # Добавляем значения над барами
        for bar, val in zip(bars, auc_3_values):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{val:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # График для 2-классового AUC
    ax2 = axes[1]
    metrics_2 = [k for k, v in auc_results.items() if '2class' in k]
    metrics_2_names = [k.replace('_2class', '').replace('_mist_mpp', '').replace('_', ' ') for k in metrics_2]
    auc_2_values = [auc_results[k] for k in metrics_2]
    
    if metrics_2:
        colors_2 = plt.cm.get_cmap('plasma', len(metrics_2))
        bars = ax2.bar(range(len(metrics_2_names)), auc_2_values,
                      color=[colors_2(i) for i in range(len(metrics_2))])
        ax2.set_xticks(range(len(metrics_2_names)))
        ax2.set_xticklabels(metrics_2_names, rotation=45, ha='right')
        ax2.set_ylabel('AUC')
        ax2.set_title('2-Class AUC (Good+Norm vs Other)')
        ax2.set_ylim([0.4, 1.0])
        ax2.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Random')
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.legend()
        
        for bar, val in zip(bars, auc_2_values):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{val:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.suptitle('AUC Comparison Across Metrics', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(save_dir, f'auc_comparison_{timestamp}.{format}')
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Сохранено сравнение AUC: {save_path}")
    return save_path


# --------------------------------------------------
# Main analysis function
# --------------------------------------------------

def analyze_metrics(csv_path, metric_columns=None, output_dir='analysis_results'):
    """
    Полный анализ метрик: AUC для 2 и 3 классов, визуализация.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Если метрики не указаны, пытаемся найти их автоматически
    if metric_columns is None:
        df_temp = pd.read_csv(csv_path)
        excluded = ['pname', 'patch_id', 'dist', 'rdist', 'class', 'class3', 
                   'Similarity_Score', 'fixed_path', 'warped_path']
        metric_columns = [col for col in df_temp.columns if col not in excluded]
        print(f"Автоматически найдены метрики: {metric_columns}")
    
    if not metric_columns:
        raise ValueError("Не найдено ни одной метрики в CSV файле")
    
    print(f"\nМетрики для анализа ({len(metric_columns)}):")
    for m in metric_columns:
        print(f"  - {m}")
    
    # Загружаем данные
    df, metrics_dict, classes_3, classes_2 = load_and_prepare_data(csv_path, metric_columns)
    
    # Словарь для хранения результатов AUC
    auc_results = {}
    
    print("\n" + "="*60)
    print("РАСЧЁТ AUC")
    print("="*60)
    
    # Рассчитываем AUC для каждой метрики
    for metric_name, scores in metrics_dict.items():
        print(f"\n{'─'*40}")
        print(f"Метрика: {metric_name}")
        print(f"  Mean ± Std: {np.mean(scores):.4f} ± {np.std(scores):.4f}")
        
        # 3-классовый AUC
        print("\n  3-классовая классификация:")
        try:
            auc_3 = calculate_smart_auc(classes_3, scores, test_direction=True, verbose=True)
            auc_results[f"{metric_name}_3class"] = auc_3
            print(f"  → Macro-AUC (3 класса): {auc_3:.4f}")
        except Exception as e:
            print(f"  ✗ Ошибка при расчете 3-классового AUC: {e}")
            auc_results[f"{metric_name}_3class"] = np.nan
        
        # 2-классовый AUC
        print("\n  2-классовая классификация:")
        try:
            auc_2 = calculate_smart_auc(classes_2, scores, test_direction=True, verbose=True)
            auc_results[f"{metric_name}_2class"] = auc_2
            print(f"  → AUC (2 класса): {auc_2:.4f}")
        except Exception as e:
            print(f"  ✗ Ошибка при расчете 2-классового AUC: {e}")
            auc_results[f"{metric_name}_2class"] = np.nan
    
    # Выводим сводную таблицу
    print("\n" + "="*70)
    print("СВОДНАЯ ТАБЛИЦА AUC")
    print("="*70)
    print(f"{'Метрика':<45} {'AUC (3 класса)':<18} {'AUC (2 класса)':<18}")
    print("-"*81)
    
    for metric_name in metrics_dict.keys():
        auc_3 = auc_results.get(f"{metric_name}_3class", np.nan)
        auc_2 = auc_results.get(f"{metric_name}_2class", np.nan)
        print(f"{metric_name:<45} {auc_3:<18.4f} {auc_2:<18.4f}")
    
    # Находим лучшую метрику
    valid_3class = [(k, v) for k, v in auc_results.items() if '3class' in k and not np.isnan(v)]
    valid_2class = [(k, v) for k, v in auc_results.items() if '2class' in k and not np.isnan(v)]
    
    if valid_3class:
        best_3class = max(valid_3class, key=lambda x: x[1])
        print(f"\nЛучшая метрика (3 класса): {best_3class[0]} с AUC = {best_3class[1]:.4f}")
    
    if valid_2class:
        best_2class = max(valid_2class, key=lambda x: x[1])
        print(f"Лучшая метрика (2 класса): {best_2class[0]} с AUC = {best_2class[1]:.4f}")
    
    # Визуализация
    print("\n" + "="*60)
    print("ВИЗУАЛИЗАЦИЯ")
    print("="*60)
    
    # Боксплоты для 3 классов
    plot_metrics_boxplots(metrics_dict, classes_3, class_type='3class', 
                         save_dir=os.path.join(output_dir, 'boxplots'))
    
    # Боксплоты для 2 классов
    plot_metrics_boxplots(metrics_dict, classes_2, class_type='2class',
                         save_dir=os.path.join(output_dir, 'boxplots'))
    
    # Сравнение AUC
    plot_auc_comparison(auc_results, save_dir=os.path.join(output_dir, 'auc'))
    
    # Сохраняем результаты в CSV
    results_data = []
    for metric_name, scores in metrics_dict.items():
        results_data.append({
            'metric': metric_name,
            'mean_score': np.mean(scores),
            'std_score': np.std(scores),
            'auc_3class': auc_results.get(f"{metric_name}_3class", np.nan),
            'auc_2class': auc_results.get(f"{metric_name}_2class", np.nan)
        })
    
    results_df = pd.DataFrame(results_data)
    results_df = results_df.sort_values('auc_3class', ascending=False)
    
    results_csv = os.path.join(output_dir, 'auc_results.csv')
    results_df.to_csv(results_csv, index=False)
    print(f"\nРезультаты сохранены в: {results_csv}")
    
    return auc_results, results_df


# --------------------------------------------------
# Argument parser
# --------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Analyze and compare metrics with AUC")
    
    parser.add_argument('--csv', type=str, required=True,
                       help='Path to CSV file with metrics')
    parser.add_argument('--output-dir', type=str, default='analysis_results',
                       help='Output directory for results and plots')
    parser.add_argument('--metrics', type=str, nargs='+', default=None,
                       help='Metric columns to analyze (if not specified, auto-detect)')
    
    return parser.parse_args()


# --------------------------------------------------
# Main
# --------------------------------------------------

if __name__ == "__main__":
    args = parse_args()
    
    analyze_metrics(
        csv_path=args.csv,
        metric_columns=args.metrics,
        output_dir=args.output_dir
    )