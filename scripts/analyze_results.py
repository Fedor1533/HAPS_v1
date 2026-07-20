#!/usr/bin/env python3
"""
Процессор результатов nested CV: загружает .pkl, считает финальные метрики,
bootstrap CI, стабильность выбранных конфигов и рисует сводные графики.

Формат входного .pkl (см. save_artifact):
    oof_predictions : list[{sample_idx, metric, fold, similarity_score, config}]
    fold_meta       : list[{fold, metric, config, inner_sp_mean, ...}]
    meta_df         : DataFrame(pname, patch_id, Similarity_Score, class3)
    run_info        : dict(timestamp, outer_seed, identities, ...)

Запуск:
    uv run python scripts/analyze_results.py results/deep.pkl
    uv run python scripts/analyze_results.py results/*.pkl --outdir reports
"""
import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import nested_cv_metrics_opt as ncv  # noqa: E402


def _load_many(paths):
    """Объединяет oof/fold_meta из нескольких .pkl (meta_df общий)."""
    oof_all, fold_all, meta_df, run_infos = [], [], None, []
    for p in paths:
        oof, fold, mdf, info = ncv.load_artifact(str(p))
        oof_all.extend(oof)
        fold_all.extend(fold)
        run_infos.append(info)
        if meta_df is None:
            meta_df = mdf
    return oof_all, fold_all, meta_df, run_infos


def _plot_ranking(bootstrap_df, out_png):
    """Bar chart: Spearman(class3) и AUC Bad-vs-Rest по метрикам с CI95."""
    df = bootstrap_df.sort_values("full_spearman_class3", ascending=True)
    metrics = df["metric"].tolist()
    y = np.arange(len(metrics))

    fig, axes = plt.subplots(1, 2, figsize=(12, max(3, 0.5 * len(metrics) + 1)))

    def err(col_ci, col_mean):
        means = df[col_mean].to_numpy(dtype=float)
        lo = np.array([c[0] for c in df[col_ci]])
        hi = np.array([c[1] for c in df[col_ci]])
        return means, np.abs(np.vstack([means - lo, hi - means]))

    sp_mean, sp_err = err("bs_sp_ci", "bs_sp_mean")
    axes[0].barh(y, sp_mean, xerr=sp_err, color="#4C72B0", capsize=3)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(metrics)
    axes[0].set_title("Spearman vs class3 (|corr|, CI95)")
    axes[0].axvline(0, color="k", lw=0.6)

    auc_mean, auc_err = err("bs_auc_bad_ci", "bs_auc_bad_mean")
    axes[1].barh(y, auc_mean, xerr=auc_err, color="#C44E52", capsize=3)
    axes[1].set_yticks(y)
    axes[1].set_yticklabels(metrics)
    axes[1].set_title("AUROC Bad-vs-Rest (CI95)")
    axes[1].axvline(0.5, color="k", lw=0.6, ls="--")

    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    plt.close(fig)


def _plot_score_distributions(oof, meta_df, out_png, top_metrics):
    """Распределение similarity_score по классам (0=Good,1=Border,2=Bad) для top-метрик."""
    df = pd.DataFrame(oof)
    cls3 = meta_df["class3"].to_numpy(dtype=int)
    labels = {0: "Good", 1: "Border", 2: "Bad"}
    n = len(top_metrics)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), squeeze=False)
    for ax, metric in zip(axes[0], top_metrics):
        g = df[df["metric"] == metric].sort_values("sample_idx")
        scores = g["similarity_score"].to_numpy(dtype=float)
        data = [scores[cls3 == c] for c in (0, 1, 2)]
        ax.boxplot(data, labels=[labels[c] for c in (0, 1, 2)], showmeans=True)
        ax.set_title(metric)
        ax.set_ylabel("similarity score")
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pkl", nargs="+", help="Один или несколько .pkl артефактов")
    ap.add_argument("--outdir", default=None, help="Куда писать отчёт (по умолчанию рядом с первым pkl)")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--top", type=int, default=3, help="Сколько top-метрик для распределений")
    args = ap.parse_args()

    paths = [Path(p) for p in args.pkl]
    outdir = Path(args.outdir) if args.outdir else paths[0].parent / (paths[0].stem + "_analysis")
    outdir.mkdir(parents=True, exist_ok=True)

    oof, fold_meta, meta_df, _ = _load_many(paths)

    # 1. Финальные метрики
    summary = ncv.evaluate_oof(oof, meta_df)
    print("\n=== Ranking (evaluate_oof) ===")
    print(summary.to_string(index=False))
    summary.to_csv(outdir / "summary.csv", index=False)

    # 2. Bootstrap CI
    boot = ncv.bootstrap_oof(oof, meta_df, n_boot=args.n_boot)
    print("\n=== Bootstrap (CI95) ===")
    cols = ["metric", "bs_sp_mean", "bs_sp_ci", "bs_auc_bad_mean", "bs_auc_bad_ci"]
    print(boot[cols].to_string(index=False))
    boot.to_csv(outdir / "bootstrap.csv", index=False)

    # 3. Стабильность выбранных конфигов по outer folds
    fm = pd.DataFrame(fold_meta).sort_values("metric")
    fm = ncv.add_preproc_key(fm)
    stab = fm.groupby("metric")["preproc_cfg"].value_counts()
    print("\n=== Config stability (по outer folds) ===")
    print(stab.to_string())
    stab.to_csv(outdir / "config_stability.csv")

    # 4. Графики
    _plot_ranking(boot, outdir / "ranking.png")
    top_metrics = summary["metric"].head(args.top).tolist()
    _plot_score_distributions(oof, meta_df, outdir / "score_distributions.png", top_metrics)

    print(f"\nОтчёт сохранён в: {outdir}")
    print(f"  - summary.csv / bootstrap.csv / config_stability.csv")
    print(f"  - ranking.png / score_distributions.png")


if __name__ == "__main__":
    main()
