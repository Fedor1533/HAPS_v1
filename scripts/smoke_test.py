#!/usr/bin/env python3
"""
Быстрый smoke-тест метрик на нескольких парах (без полного nested CV).

Проверяет, что preprocess + каждая metric identity отрабатывают без ошибок и
возвращают конечные числа. Deep-метрики (cellpose/lpips_cellpose/transpath)
можно гонять и на CPU.

Запуск:
    uv run python scripts/smoke_test.py --csv <path> --n 6 \
        --metrics ncc,ssim,cellpose,lpips_cellpose
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.dataset_class import SimpleDataset  # noqa: E402
import nested_cv_metrics_opt as ncv  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--n", type=int, default=6, help="Сколько пар взять")
    ap.add_argument("--metrics", default="ncc,ssim,cellpose,lpips_cellpose")
    args = ap.parse_args()

    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]
    print(f"Device: {ncv.device}")
    print(f"Metrics: {metrics}")

    ds = SimpleDataset(
        args.csv, dist_col="Similarity_Score", class_col="class3", return_meta=True
    )
    pairs = []
    for i in range(min(args.n, len(ds))):
        f, w, _, _, _ = ds.unpack_item(ds[i])
        pairs.append((f, w))
    print(f"Loaded {len(pairs)} pairs from {args.csv}")

    base_cfg = {
        "normalization": True,
        "channel_mode": "gray",
        "flip_intensity": False,
        "match_histogram": False,
        "clahe": False,
        "smoothing": False,
        "binary_mask": False,
    }

    def build_configs(name, info):
        """Базовый конфиг + по одному варианту на каждое значение каждого параметра.

        Полное произведение гонять незачем, а вот каждое значение каждого params
        задеть надо: разные stat/agg — это разные ветки кода.
        """
        cfg = dict(base_cfg)
        cfg["channel_mode"] = info["channel_mode"][0]
        for k, v in info.get("flags", {}).items():
            cfg[k] = v[0]
        for k, v in info.get("params", {}).items():
            cfg[k] = v[0]
        cfg["metric"] = name

        out = [cfg]
        for k, values in info.get("params", {}).items():
            for v in values[1:]:
                variant = dict(cfg)
                variant[k] = v
                out.append(variant)
        # маска ткани — отдельная ветка препроцессинга
        if True in info.get("flags", {}).get("binary_mask", []):
            variant = dict(cfg)
            variant["binary_mask"] = True
            out.append(variant)
        return out

    def describe(cfg, info):
        keys = ["channel_mode", "binary_mask"] + list(info.get("params", {}).keys())
        return ",".join(f"{k}={cfg[k]}" for k in keys if k in cfg)

    failed = []
    for name in metrics:
        info = ncv.METRIC_IDENTITIES.get(name)
        if info is None:
            print(f"[SKIP] {name}: не зарегистрирована")
            continue
        for cfg in build_configs(name, info):
            tag = describe(cfg, info)
            try:
                scores = ncv.compute_scores_for_config(cfg, pairs, name)
                arr = np.asarray(scores, dtype=float)
                if not np.isfinite(arr).all():
                    print(f"[NON-FINITE] {name:16s} {tag}")
                    failed.append(f"{name}[{tag}]")
                    continue
                spread = "CONSTANT" if np.allclose(arr, arr[0]) else "OK"
                print(f"[{spread:8s}] {name:16s} {tag}\n"
                      f"{'':11s} scores={np.round(arr, 4).tolist()}")
                if spread == "CONSTANT":
                    # константа по всем парам = метрика ничего не различает
                    failed.append(f"{name}[{tag}] constant")
            except Exception as e:
                print(f"[FAIL    ] {name:16s} {tag}: {type(e).__name__}: {e}")
                failed.append(f"{name}[{tag}]")

    print("\n" + ("SMOKE OK" if not failed else f"SMOKE FAILED: {failed}"))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
