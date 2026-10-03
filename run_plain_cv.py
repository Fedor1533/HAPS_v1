#!/usr/bin/env python3
"""
Plain 5-fold CV для зафиксированного конфига метрики (без inner CV / выбора конфига).

Оценивает «чистую» цифру для proposed-метрики: тот же outer-сплит, что и в 
nested CV (StratifiedGroupKFold, seed=155), но конфиг ФИКСИРОВАН и не перевыбирается
per-fold. Аналог nested-CV, только без inner-перебора.

Запуск (нужен .venv-foundation для HF foundation-бэкбонов + кэш моделей):
    source slurm/env.sh
    .venv-foundation/bin/python run_plain_cv.py \
        --metric hibou_l_spec \
        --csv /gpfs/gpfs0/gubanov-lab/virtual_stain/filtration_imgs/exp_full.csv \
        --output results/plain_hibou_l_spec.pkl

Фиксированный препроцессинг по умолчанию соответствует выигравшему конфигу
hibou_l_spec (0 | rgb | 0 | 0 | 0 | 0 | cos | layer_20).
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, "/beegfs/home/f.gubanov/f.gubanov/bimai_lab/HAPS_v1_Vladislav")

from scripts.dataset_class import SimpleDataset  # noqa: E402
import nested_cv_metrics_opt as ncv  # noqa: E402

# Дефолтный препроцессинг — см. win-config hibou_l_spec.
DEFAULT_PREPROC = {
    "normalization": False,
    "channel_mode": "rgb",
    "flip_intensity": False,
    "match_histogram": False,
    "clahe": False,
    "smoothing": False,
}


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--metric", required=True, help="identity метрики, напр. hibou_l_spec")
    ap.add_argument("--csv", default="/gpfs/gpfs0/gubanov-lab/virtual_stain/filtration_imgs/exp_full.csv")
    ap.add_argument("--output", required=True, help="Путь к .pkl результата")
    ap.add_argument("--config-json", default='{"feature": "layer_20", "agg": "cos"}', help="metric-specific параметры (feature/agg/...) как JSON")
    ap.add_argument("--outer-seed", type=int, default=155)
    ap.add_argument("--n-boot", type=int, default=1000)
    return ap.parse_args()


def main():
    args = parse_args()
    metric = args.metric

    fixed_cfg = dict(DEFAULT_PREPROC)
    fixed_cfg.update(json.loads(args.config_json))
    fixed_cfg["metric"] = metric

    print(f"Plain 5-fold CV: metric={metric}")
    print(f"  fixed config: {fixed_cfg}")
    print(f"  csv={args.csv}  outer_seed={args.outer_seed}")

    ds = SimpleDataset(args.csv, dist_col="Similarity_Score", class_col="class3", return_meta=True)
    print(f"Loaded dataset: {len(ds)}")
    all_pairs, meta_df = ncv._cache_dataset(ds)

    y = meta_df["class3"].to_numpy(dtype=int)
    groups = meta_df["pname"].astype(str).to_numpy()
    splitter = ncv.StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=args.outer_seed)
    splits = list(splitter.split(np.zeros(len(meta_df)), y, groups))

    oof_preds = []
    fold_meta = []
    for fold_id, (_, va) in enumerate(splits):
        val_pairs = [all_pairs[i] for i in va]
        scores = ncv.compute_scores_for_config(fixed_cfg, val_pairs, metric)
        for j, idx in enumerate(va):
            oof_preds.append({
                "sample_idx": int(idx),
                "metric": metric,
                "fold": int(fold_id),
                "similarity_score": float(scores[j]),
                "config": fixed_cfg,
            })
        fold_meta.append({"fold": fold_id, "metric": metric, "config": fixed_cfg})
        print(f"  fold {fold_id}: {len(va)} pairs")

    print("\n=== evaluate_oof ===")
    print(ncv.evaluate_oof(oof_preds, meta_df).to_string(index=False))

    print(f"\n=== bootstrap_oof (n_boot={args.n_boot}) ===")
    boot = ncv.bootstrap_oof(oof_preds, meta_df, n_boot=args.n_boot)
    print(boot.to_string(index=False))

    ncv.save_artifact(oof_preds, fold_meta, meta_df, args.outer_seed, args.output, identities=[metric])
    print(f"\nSaved -> {args.output}")


if __name__ == "__main__":
    main()
