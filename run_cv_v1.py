import argparse, os, sys
import time
import pandas as pd

sys.path.append('/beegfs/home/f.gubanov/f.gubanov/bimai_lab/metrics_code_v2')
from scripts.dataset_class import SimpleDataset
# from nested_cv_metrics import run_outer_cv, save_artifact, add_preproc_key
import nested_cv_metrics_opt as n_cv_opt

def parse_args():
    parser = argparse.ArgumentParser(description="Run nested CV for metric evaluation")
    parser.add_argument("--identities", type=str, default="ncc,psnr,mi,ssim", help="Comma-separated metric identities")
    parser.add_argument("--csv", type=str, default="/beegfs/home/f.gubanov/f.gubanov/bimai_lab/filtration_imgs/exp_full.csv", help="Path to full dataset CSV")
    parser.add_argument("--output", type=str, required=True, help="Path to output .pkl file")
    parser.add_argument("--outer-seed", type=int, default=155, help="Outer CV random seed")
    parser.add_argument("--n-jobs", type=int, default=1, help="Parallel outer folds (1=sequential)")
    return parser.parse_args()

def main():
    args = parse_args()
    identities = args.identities.split(",")
    print(f"Run CV for specified metrics: {identities}")

    # Не перезаписываем существующие результаты: чтобы пересчитать — удалите .pkl вручную.
    if os.path.exists(args.output):
        print(
            f"SKIP: output already exists: {args.output}\n"
            f"  Удалите файл вручную, если хотите перезапустить эксперимент."
        )
        sys.exit(0)

    out_dir = os.path.dirname(os.path.abspath(args.output))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    dataset = SimpleDataset(args.csv, dist_col='Similarity_Score', class_col='class3', return_meta=True)
    print(f"Loaded dataset: {len(dataset)}")

    t0 = time.time()
    print(f"\nStarted nested CV (n_jobs={args.n_jobs}):\n")
    oof_preds, fold_meta, meta_df = n_cv_opt.run_outer_cv(dataset, identities=identities, outer_seed=args.outer_seed, n_jobs=args.n_jobs)
    elapsed = time.time() - t0
    print(f"\nFinished nested CV in {elapsed:.1f}s.\n")
    
    fm_df = pd.DataFrame(fold_meta).sort_values(by='metric')
    fm_df = n_cv_opt.add_preproc_key(fm_df)
    print(fm_df.groupby("metric")["preproc_cfg"].value_counts())

    print(f"\nSaving to {args.output}")
    # после run_outer_cv сохраняем
    # Внутри .pkl четыре поля: oof_predictions, fold_meta, meta_df, run_info — этого достаточно, чтобы пересчитать любые метрики и проверить воспроизводимость.
    n_cv_opt.save_artifact(oof_preds, fold_meta, meta_df, outer_seed=args.outer_seed, path=args.output, identities=identities)


if __name__ == "__main__":
    main()