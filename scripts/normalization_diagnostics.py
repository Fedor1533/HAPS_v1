#!/usr/bin/env python3
"""
Диагностика межфолдового разброса OOF-скор и рекомендация per-fold нормализации.

Помогает решить, для каких метрик применять `normalize_oof_per_fold(metrics=...)`:
считает разброс средних по outer-фолдам, отношение «разброс средних / max within-fold std»
и сопоставляет его с отличающимися параметрами конфига у фолдов max/min.

Правило (см. nested_cv_protocol.md §7): нормализовать, если ratio > threshold
(по умолчанию 1.5).

Использование как модуль:
    from scripts.normalization_diagnostics import diagnose_normalization
    diag = diagnose_normalization(oof, fold_meta, threshold=1.5)
    norm_metrics = diag.loc[diag["normalize"], "metric"].tolist()

CLI:
    python scripts/normalization_diagnostics.py --pkl results/x.pkl [--threshold 1.5]
"""

import ast
import numpy as np
import pandas as pd


def fold_spread(oof, stat="mean"):
    """Для каждой метрики — максимальный разброс по фолдам и пара фолдов (max, min).

    Возвращает DataFrame (index=metric) с колонками:
      fold_{stat}_max / fold_{stat}_min, {stat}_max / {stat}_min,
      {stat}_abs_diff, std_max, std_min.
    """
    df = pd.DataFrame(oof).copy()
    by = ["metric", "fold"]

    g = df.groupby(by)["similarity_score"].agg([stat, "std"]).unstack("fold")
    g_stat, g_std = g[stat], g["std"]

    out = pd.DataFrame({
        f"fold_{stat}_max": g_stat.idxmax(axis=1),
        f"fold_{stat}_min": g_stat.idxmin(axis=1),
        f"{stat}_max":      g_stat.max(axis=1).round(4),
        f"{stat}_min":      g_stat.min(axis=1).round(4),
        f"{stat}_abs_diff": (g_stat.max(axis=1) - g_stat.min(axis=1)).round(4),
        "std_max":          g_std.max(axis=1).round(4),
        "std_min":          g_std.min(axis=1).round(4),
    })
    return out.sort_values(f"{stat}_abs_diff", ascending=False)


def compare_fold_configs(spread, cfg_df, stat="mean", config_col="config"):
    """Дополняет spread колонкой cfg_diff_desc — какие параметры конфига различаются
    у фолдов fold_{stat}_max и fold_{stat}_min.
    """
    def _as_dict(c):
        return ast.literal_eval(c) if isinstance(c, str) else dict(c)

    lookup = (
        cfg_df.drop_duplicates(["metric", "fold"])
              .assign(**{config_col: lambda d: d[config_col].map(_as_dict)})
              .set_index(["metric", "fold"])[config_col]
    )

    f_max, f_min = f"fold_{stat}_max", f"fold_{stat}_min"

    def _diff(row):
        metric = row.name[0] if isinstance(row.name, tuple) else row.name
        cfg_a = lookup.loc[(metric, row[f_max])]
        cfg_b = lookup.loc[(metric, row[f_min])]
        d = {
            k: (cfg_a.get(k), cfg_b.get(k))
            for k in set(cfg_a) | set(cfg_b)
            if cfg_a.get(k) != cfg_b.get(k)
        }
        return pd.Series({
            "cfg_diff_desc": "; ".join(f"{k}: {a}→{b}" for k, (a, b) in sorted(d.items())),
        })

    return spread.join(spread.apply(_diff, axis=1))


def diagnose_normalization(oof, fold_meta, threshold=1.5, stat="mean"):
    """Рекомендация per-fold нормализации.

    Возвращает DataFrame (metric как колонка), отсортированный по ratio desc, с колонками:
      fold_{stat}_max/min, {stat}_max/min, {stat}_abs_diff, std_max, std_min,
      {stat}_diff/std, normalize, cfg_diff_desc.
    normalize = ratio > threshold.
    """
    spread = fold_spread(oof, stat=stat)

    ratio = spread[f"{stat}_abs_diff"] / spread["std_max"].replace(0, np.nan)
    spread[f"{stat}_diff/std"] = ratio.round(4)
    spread["normalize"] = ratio > threshold

    cfg_df = pd.DataFrame(fold_meta)
    result = compare_fold_configs(spread, cfg_df, stat=stat)

    cols = [
        f"fold_{stat}_max", f"fold_{stat}_min",
        f"{stat}_max", f"{stat}_min", f"{stat}_abs_diff",
        "std_max", "std_min", f"{stat}_diff/std", "normalize", "cfg_diff_desc",
    ]
    return result[cols].reset_index()


def _load_artifact(path):
    """Лёгкая загрузка .pkl без импорта nested_cv_metrics_opt (не тянет torch/lpips)."""
    import pickle
    with open(path, "rb") as f:
        art = pickle.load(f)
    return art["oof_predictions"], art["fold_meta"]


def main():
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pkl", nargs="+", required=True, help="Один или несколько .pkl артефактов")
    ap.add_argument("--threshold", type=float, default=1.5, help="Порог ratio для рекомендации нормализации (по умолчанию 1.5)")
    ap.add_argument("--stat", default="mean", choices=["mean", "median"])
    ap.add_argument("--out", default=None, help="Куда сохранить CSV (опционально)")
    args = ap.parse_args()

    oof_all, fold_all = [], []
    for p in args.pkl:
        oof, fold = _load_artifact(p)
        oof_all.extend(oof)
        fold_all.extend(fold)

    diag = diagnose_normalization(oof_all, fold_all, threshold=args.threshold, stat=args.stat)

    print("=== Normalization diagnostics (ratio = mean_abs_diff / std_max) ===")
    print(diag.to_string(index=False))

    norm_metrics = diag.loc[diag["normalize"], "metric"].tolist()
    print("\nРекомендовано нормализовать:", norm_metrics if norm_metrics else "(нет)")

    if args.out:
        diag.to_csv(args.out, index=False)
        print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()
