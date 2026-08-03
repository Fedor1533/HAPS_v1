# merge_metrics.py
# Объединяет все CSV с метриками в один файл

import pandas as pd
import sys
import os
import glob
import argparse


def merge_metrics(results_dir, output_csv):
    """
    Объединяет все CSV файлы с метриками из results_dir в один файл.
    
    Args:
        results_dir: директория с CSV файлами метрик
        output_csv: путь для сохранения объединенного файла
    """
    
    # Базовые колонки, которые должны быть во всех файлах
    base_columns = ['pname', 'patch_id', 'dist', 'rdist', 'class', 'fixed_path', 'warped_path']
    
    # # Собираем все файлы с метриками
    # pattern = os.path.join(results_dir, "mist_*_test.csv")
    # all_files = sorted(glob.glob(pattern))
    
    # print(f"Поиск файлов по паттерну: {pattern}")
    # print(f"Найдено файлов: {len(all_files)}")
    

    patterns = [
        os.path.join(results_dir, "mist_*.csv"),
        os.path.join(results_dir, "generic_*.csv"),
    ]

    all_files = []
    for pattern in patterns:
        all_files.extend(glob.glob(pattern))
    all_files = sorted(set(all_files))

    print(f"Найдено файлов: {len(all_files)}")

    # Исключаем уже объединённый файл
    all_files = [f for f in all_files if 'all_metrics_merged' not in f]
    print(f"После фильтрации: {len(all_files)} файлов")
    
    if not all_files:
        print("\nОШИБКА: Не найдено файлов для объединения!")
        print(f"\nСодержимое директории {results_dir}:")
        for f in sorted(os.listdir(results_dir)):
            print(f"  {f}")
        sys.exit(1)
    
    print(f"\nФайлы для объединения:")
    for f in all_files:
        print(f"  - {os.path.basename(f)}")
    
    # Загружаем первый файл как базовый
    base_df = pd.read_csv(all_files[0])
    print(f"\nБазовый файл: {os.path.basename(all_files[0])}")
    print(f"  Записей: {len(base_df)}")
    print(f"  Колонок: {len(base_df.columns)}")
    print(f"  Колонки: {list(base_df.columns)}")
    
    # Проверяем наличие базовых колонок
    missing_base = [col for col in base_columns if col not in base_df.columns]
    if missing_base:
        print(f"  ВНИМАНИЕ: Отсутствуют базовые колонки: {missing_base}")
    
    # Словарь для отслеживания добавленных метрик
    added_metrics = []
    
    # Обрабатываем остальные файлы
    for file_path in all_files[1:]:
        if not os.path.exists(file_path):
            print(f"  Пропущен (не найден): {os.path.basename(file_path)}")
            continue
        
        df = pd.read_csv(file_path)
        
        # Находим колонки с метриками (те, которых нет в base_columns)
        metric_cols = [col for col in df.columns if col not in base_columns]
        
        if len(metric_cols) == 0:
            print(f"  Пропущен (нет метрик): {os.path.basename(file_path)}")
            continue
        
        # Берем все колонки с метриками
        for metric_col in metric_cols:
            if metric_col in base_df.columns:
                print(f"  Пропущена (уже существует): {metric_col}")
                continue
            
            # Проверяем, что размеры совпадают
            if len(df) == len(base_df):
                base_df[metric_col] = df[metric_col].values
                added_metrics.append(metric_col)
                print(f"  ✓ Добавлена: {metric_col} (из {os.path.basename(file_path)})")
            else:
                print(f"  ✗ Размеры не совпадают: {metric_col} ({len(df)} vs {len(base_df)})")
    
    # Проверяем результат
    print(f"\nРезультат объединения:")
    print(f"  Всего записей: {len(base_df)}")
    print(f"  Всего колонок: {len(base_df.columns)}")
    print(f"  Добавлено метрик: {len(added_metrics)}")
    print(f"  Метрики: {added_metrics}")
    
    # Убеждаемся, что базовые колонки идут первыми
    existing_base = [col for col in base_columns if col in base_df.columns]
    other_columns = [col for col in base_df.columns if col not in base_columns]
    base_df = base_df[existing_base + other_columns]
    
    # Сохраняем
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    base_df.to_csv(output_csv, index=False)
    
    print(f"\n✓ Объединенный CSV сохранен: {output_csv}")
    print(f"  Размер файла: {os.path.getsize(output_csv) / 1024:.1f} KB")
    
    # Показываем первые строки
    print(f"\nПервые 5 строк:")
    print(base_df.head())
    
    return base_df


def main():
    parser = argparse.ArgumentParser(
        description="Объединяет все CSV файлы с метриками в один файл"
    )
    
    parser.add_argument(
        '--results-dir',
        type=str,
        required=True,
        help='Директория с CSV файлами метрик'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Путь для сохранения объединенного CSV'
    )
    
    args = parser.parse_args()
    
    merge_metrics(args.results_dir, args.output)


if __name__ == "__main__":
    main()