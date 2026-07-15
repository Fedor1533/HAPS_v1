import argparse, os, sys
import time
import pandas as pd

sys.path.append('/beegfs/home/f.gubanov/f.gubanov/bimai_lab/metrics_code_v2')
from scripts.dataset_class import SimpleDataset
from nested_cv_metrics import run_outer_cv, save_artifact, add_preproc_key

def parse_args():
    parser = argparse.ArgumentParser(description="Run nested CV for metric evaluation")
    parser.add_argument("--identities", type=str, default="ncc,psnr,mi,ssim", help="Comma-separated metric identities")
    parser.add_argument("--csv", type=str, default="/beegfs/home/f.gubanov/f.gubanov/bimai_lab/filtration_imgs/exp_full.csv", help="Path to full dataset CSV")
    parser.add_argument("--output", type=str, required=True, help="Path to output .pkl file")
    parser.add_argument("--outer-seed", type=int, default=155, help="Outer CV random seed")
    return parser.parse_args()

def main():
    args = parse_args()
    identities = args.identities.split(",")
    print(f"Run CV for specified metrics: {identities}")

    dataset = SimpleDataset(args.csv, dist_col='Similarity_Score', class_col='class3', return_meta=True)
    print(f"Loaded dataset: {len(dataset)}")

    t0 = time.time()
    print("\nStarted nested CV:\n")
    oof_preds, fold_meta, meta_df = run_outer_cv(dataset, identities=identities, outer_seed=args.outer_seed)
    elapsed = time.time() - t0
    print(f"\nFinished nested CV in {elapsed:.1f}s.\n")

    fm_df = pd.DataFrame(fold_meta).sort_values(by='metric')
    fm_df = add_preproc_key(fm_df)
    print(fm_df.groupby("metric")["preproc_cfg"].value_counts())

    print(f"\nSaving to {args.output}")
    # после run_outer_cv сохраняем
    # Внутри .pkl четыре поля: oof_predictions, fold_meta, meta_df, run_info — этого достаточно, чтобы пересчитать любые метрики и проверить воспроизводимость.
    save_artifact(oof_preds, fold_meta, meta_df, outer_seed=args.outer_seed, path=args.output, identities=identities)


if __name__ == "__main__":
    main()
