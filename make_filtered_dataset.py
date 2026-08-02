# make_filtered_dataset.py

import argparse
import os
from pathlib import Path

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description="Filter dataset by metric score and create symlink dataset."
    )

    parser.add_argument(
        "--csv",
        type=str,
        required=True,
        help="CSV file with metric scores.",
    )

    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Source dataset directory.",
    )

    parser.add_argument(
        "--target",
        type=str,
        required=True,
        help="Output dataset directory.",
    )

    parser.add_argument(
        "--score-column",
        type=str,
        default="ncc_hed",
        help="Column used for filtering.",
    )

    parser.add_argument(
        "--remove-percent",
        type=float,
        default=15,
        help="Percentage of worst samples to remove.",
    )

    parser.add_argument(
        "--descending",
        action="store_true",
        help="Treat larger values as worse (higher score = worse quality).",
    )

    return parser.parse_args()


def create_filtered_dataset(
    csv_file_path,
    source_dataset_dir,
    target_dir,
    score_column="ncc_hed",
    remove_percent=15,
    ascending=True,
):
    """
    Parameters
    ----------
    csv_file_path : str
        CSV с колонкой score_column.

    source_dataset_dir : str
        Исходный датасет:
            trainA/
            trainB/
            testA/
            testB/

    target_dir : str

    score_column : str
        По какой колонке фильтровать.

    remove_percent : float
        Какой процент удалить.

    ascending : bool
        True  -> маленькие значения хуже (например NCC, SSIM)
        False -> большие значения хуже (например LPIPS)
    """
    os.makedirs(target_dir, exist_ok=True)
    
    df = pd.read_csv(csv_file_path)

    if score_column not in df.columns:
        raise ValueError(f"Column '{score_column}' not found in CSV.")
    
    print(f"Using score column: '{score_column}'")

    print(f"Loaded {len(df)} samples")

    #
    # ---------- фильтрация ----------
    #

    df = df.sort_values(score_column, ascending=ascending)

    n_remove = int(len(df) * remove_percent / 100)
    n_keep = len(df) - n_remove

    filtered_df = df.iloc[n_remove:].copy()

    print(f"Removed {n_remove} worst samples (bottom {remove_percent}%)")
    print(f"Kept {len(filtered_df)} samples ({100-remove_percent}%)")
    
    # Показываем статистику по оставшимся
    print(f"\nScore statistics (kept samples):")
    print(f"  Min: {filtered_df[score_column].min():.4f}")
    print(f"  Max: {filtered_df[score_column].max():.4f}")
    print(f"  Mean: {filtered_df[score_column].mean():.4f}")
    print(f"  Std: {filtered_df[score_column].std():.4f}")

    #
    # ---------- сохранить CSV ----------
    #

    os.makedirs(target_dir, exist_ok=True)

    filtered_csv = os.path.join(
        target_dir,
        f"filtered_{100-remove_percent:.0f}percent.csv",
    )

    filtered_df.to_csv(filtered_csv, index=False)

    print(f"\nSaved filtered csv: {filtered_csv}")

    #
    # ---------- создать директории ----------
    #

    # train
    os.makedirs(os.path.join(target_dir, "trainA", "HE"), exist_ok=True)
    os.makedirs(os.path.join(target_dir, "trainB", "IHC"), exist_ok=True)

    # test
    os.makedirs(os.path.join(target_dir, "testA"), exist_ok=True)
    os.makedirs(os.path.join(target_dir, "testB"), exist_ok=True)

    #
    # ---------- train симлинки ----------
    #

    print(f"\nCreating train symlinks...")
    train_count = 0

    for _, row in filtered_df.iterrows():

        fixed_src = row["fixed_path"]
        warped_src = row["warped_path"]

        # Проверяем существование исходных файлов
        if not os.path.exists(fixed_src):
            print(f"  WARNING: fixed_path not found: {fixed_src}")
            continue
        
        if not os.path.exists(warped_src):
            print(f"  WARNING: warped_path not found: {warped_src}")
            continue

        fixed_dst = os.path.join(
            target_dir,
            "trainA",
            "HE",
            Path(fixed_src).name,
        )

        warped_dst = os.path.join(
            target_dir,
            "trainB",
            "IHC",
            Path(warped_src).name,
        )

        # Удаляем старые симлинки, если есть
        if os.path.lexists(fixed_dst):
            os.remove(fixed_dst)

        if os.path.lexists(warped_dst):
            os.remove(warped_dst)

        os.symlink(fixed_src, fixed_dst)
        os.symlink(warped_src, warped_dst)
        train_count += 1

    print(f"Train links created: {train_count}")

    #
    # ---------- test симлинки ----------
    #

    print(f"\nCreating test symlinks...")

    test_dirs = ["testA", "testB"]

    for folder in test_dirs:

        src_dir = os.path.join(source_dataset_dir, folder)
        dst_dir = os.path.join(target_dir, folder)

        if not os.path.isdir(src_dir):
            print(f"  WARNING: Source directory not found: {src_dir}")
            continue

        cnt = 0

        for file in sorted(os.listdir(src_dir)):
            # Пропускаем служебные файлы
            if file.startswith('.') or file.endswith('~'):
                continue
                
            src = os.path.join(src_dir, file)
            dst = os.path.join(dst_dir, file)

            if not os.path.isfile(src):
                continue

            if os.path.lexists(dst):
                os.remove(dst)

            os.symlink(src, dst)
            cnt += 1

        print(f"  {folder}: {cnt} links")

    print(f"\n{'='*60}")
    print(f"Done! Filtered dataset created at: {target_dir}")
    print(f"{'='*60}")

    return filtered_df


if __name__ == "__main__":

    args = parse_args()

    create_filtered_dataset(
        csv_file_path=os.path.expanduser(args.csv),
        source_dataset_dir=os.path.expanduser(args.source),
        target_dir=os.path.expanduser(args.target),
        score_column=args.score_column,
        remove_percent=args.remove_percent,
        ascending=not args.descending,
    )