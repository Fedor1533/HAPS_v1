import re
import pickle
import time
from datetime import datetime
import itertools
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
import scipy

from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

import torch
import sys
sys.path.append('/beegfs/home/f.gubanov/f.gubanov/bimai_lab/metrics_code_v2')
from scripts.preprocessing import Preprocessor
from scripts.metrics import calc_ncc, calc_psnr, calc_mi, calc_ssim, calc_ms_ssim, calc_fsim, calc_lpips, calc_dists
from scripts.eval_metrics import calculate_smart_auc

device = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------------------------------
# Инициализация перцептуальных моделей (LPIPS / DISTS) — ленивая
# ---------------------------------------------------------------------------
import lpips
from DISTS_pytorch import DISTS

_LPIPS_MODELS = None

def _get_lpips_model(backbone, aggregation):
    global _LPIPS_MODELS
    if _LPIPS_MODELS is None:
        _LPIPS_MODELS = {}
        for b in ['alex', 'vgg', 'squeeze']:
            for linear in [True, False]:
                tag = f"{b}_{'lin' if linear else 'avg'}"
                _LPIPS_MODELS[tag] = lpips.LPIPS(net=b, lpips=linear).eval().to(device)
    return _LPIPS_MODELS[f"{backbone}_{aggregation}"]


_DISTS_MODEL = None

def _get_dists_model():
    global _DISTS_MODEL
    if _DISTS_MODEL is None:
        _DISTS_MODEL = DISTS().eval().to(device)
    return _DISTS_MODEL


def _make_lpips_wrapper(backbone):
    """Фабрика lambda-обёртки для LPIPS: извлекает lpips_aggregation из конфига."""
    def wrapper(inp, lpips_aggregation="avg", **kw):
        loss_fn = _get_lpips_model(backbone, lpips_aggregation)
        return -calc_lpips(inp.src_t, inp.trg_t, None, 0.0, loss_fn)
    return wrapper

# ---------------------------------------------------------------------------
# Утилиты (из старого test_metrics.py)
# ---------------------------------------------------------------------------

def add_preproc_key(df, out_col="preproc_cfg", sep=" | ", config_col="config"):
    """Extract preproc config from dict column into a readable string.
    
    Для каждой строки берёт все ключи из config (кроме 'metric'), сортирует,
    форматирует: булевы -> 0/1, остальные как есть.
    """
    df = df.copy()
    df.insert(0, out_col, np.nan)
    df[out_col] = df[config_col].apply(
        lambda cfg: sep.join(
            f"{int(cfg[c])}" if isinstance(cfg.get(c), bool)
            else str(cfg.get(c, ""))
            for c in [k for k in cfg.keys() if k != "metric"] # берем ключи для данной метрик кроме "metric"
        )
    )
    return df


def extract_value(val):
    if isinstance(val, (int, float)):
        return val
    if isinstance(val, str):
        try:
            return float(val)
        except (ValueError, TypeError):
            pass
    match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", str(val))
    if match:
        try:
            return float(match.group())
        except (ValueError, TypeError):
            return None
    return None

# ---------------------------------------------------------------------------
# Конфигурации: metric identities + search space
# ---------------------------------------------------------------------------

COMMON_FLAGS = {
    "normalization":    [False, True],
    "channel_mode":     ["gray", "hed"],
    "flip_intensity":   [False, True], # [False]
    "match_histogram":  [False, True],
    "clahe":            [False, True], # [False]
    "smoothing":        [False, True],
}

METRIC_IDENTITIES = {
    "ncc": {
        "channel_mode": ["gray", "hed"],
    },
    "psnr": {
        "channel_mode": ["gray", "hed"],
    },
    "mi": {
        "channel_mode": ["gray", "hed"],
    },
    "ssim": {
        "channel_mode": ["gray", "hed"],
        "params": {
            "win_size": [7, 31],
        },
    },
    "ms-ssim": {
        "channel_mode": ["gray", "hed"],
    },
    "fsim": {
        "channel_mode": ["gray", "hed"],
    },
    "fsimc": {
        "channel_mode": ["rgb"],
    },
    "lpips_alex": {
        "channel_mode": ["gray", "hed", "rgb"],
        "params": {"lpips_aggregation": ["avg", "lin"]},
    },
    "lpips_vgg": {
        "channel_mode": ["gray", "hed", "rgb"],
        "params": {"lpips_aggregation": ["avg", "lin"]},
    },
    "lpips_squeeze": {
        "channel_mode": ["gray", "hed", "rgb"],
        "params": {"lpips_aggregation": ["avg", "lin"]},
    },
    "dists": {
        "channel_mode": ["gray", "hed", "rgb"],
    },
}

METRICS_MAP = {
    "ncc":     lambda inp, **kw: calc_ncc(inp.src_np, inp.trg_np, None, **kw),
    "psnr":    lambda inp, **kw: calc_psnr(inp.src_np, inp.trg_np, None, **kw),
    "mi":      lambda inp, **kw: calc_mi(inp.src_np, inp.trg_np, None, **kw),
    "ssim":    lambda inp, **kw: calc_ssim(inp.src_np, inp.trg_np, None, **kw),
    "ms-ssim": lambda inp, **kw: calc_ms_ssim(inp.src_t, inp.trg_t, None, **kw),
    "fsim":    lambda inp, **kw: calc_fsim(inp.src_t, inp.trg_t, chromatic=False),
    "fsimc":   lambda inp, **kw: calc_fsim(inp.src_t, inp.trg_t, chromatic=True),
    "lpips_alex":    _make_lpips_wrapper("alex"),
    "lpips_vgg":     _make_lpips_wrapper("vgg"),
    "lpips_squeeze": _make_lpips_wrapper("squeeze"),
    "dists":         lambda inp, **kw: -calc_dists(inp.src_t, inp.trg_t, None, 0.0, _get_dists_model()),
}

# ---------------------------------------------------------------------------
# Генерация конфигов для одной metric identity
# ---------------------------------------------------------------------------

def generate_configs(identity_name):
    """Генерирует список config dicts для заданной metric identity."""
    info = METRIC_IDENTITIES[identity_name]
    identity_channel_modes = info["channel_mode"]
    identity_params = info.get("params", {})

    # общие флаги, но channel_mode заменяем на identity-specific
    flags = dict(COMMON_FLAGS)
    flags["channel_mode"] = list(identity_channel_modes)

    # metric-specific hyperparams
    for k, v in identity_params.items():
        flags[k] = list(v)

    keys = list(flags.keys())
    value_lists = [flags[k] for k in keys]

    configs = []
    for combo in itertools.product(*value_lists):
        cfg = dict(zip(keys, combo))
        cfg["metric"] = identity_name
        configs.append(cfg)

    return configs


# ---------------------------------------------------------------------------
# Вычисление scores для одного конфига
# ---------------------------------------------------------------------------

def _extra_kwargs_for_metric(config, metric_name):
    """Извлекает metric-specific kwargs из config, используя params из METRIC_IDENTITIES."""
    extra = {}
    params = METRIC_IDENTITIES.get(metric_name, {}).get("params", {})
    for k in params:
        if k in config:
            extra[k] = config[k]
    return extra


def compute_scores_for_config(config, pairs_subset, metric_names):
    """
    Вычисляет similarity_scores для одного конфига на заданном подмножестве пар.
    
    metric_names: str или list[str]
        str  -> возвращает np.ndarray
        list -> возвращает dict[str, np.ndarray]
    
    Препроцессинг выполняется ОДИН раз на пару, затем метрики считаются из MetricInput.
    """
    if isinstance(metric_names, str):
        metric_names = [metric_names]
        return_single = True
    else:
        return_single = False

    preprocessor = Preprocessor(device=device)

    # предвычисляем metric_fn + extra_kwargs для каждой метрики в группе
    fn_kwargs = {}
    for name in metric_names:
        fn_kwargs[name] = {
            "fn": METRICS_MAP[name],
            "kw": _extra_kwargs_for_metric(config, name),
        }

    # инициализируем буферы scores
    all_scores = {name: [] for name in metric_names}

    for f_patch, w_patch in pairs_subset:
        inp = preprocessor.process(f_patch, w_patch, config)
        for name in metric_names:
            info = fn_kwargs[name]
            val = info["fn"](inp, **info["kw"])
            all_scores[name].append(extract_value(val))

    if return_single:
        return np.asarray(all_scores[metric_names[0]], dtype=float)
    return {name: np.asarray(arr, dtype=float) for name, arr in all_scores.items()}


# ---------------------------------------------------------------------------
# Eval: метрики по scores + labels
# ---------------------------------------------------------------------------

def evaluate_scores(similarity_scores, class3_arr, scores_1_5_arr):
    """
    similarity_scores: np.ndarray, higher = better pair
    class3_arr: 0=Good, 1=Borderline, 2=Bad
    scores_1_5_arr: raw expert scores 1-5
    """
    y_bad = (class3_arr == 2).astype(int)
    y_good = (class3_arr == 0).astype(int)
    badness = -similarity_scores

    sp_cls3 = spearmanr(similarity_scores, class3_arr).correlation
    sp_1_5 = spearmanr(similarity_scores, scores_1_5_arr).correlation

    return {
        "spearman_class3":    -sp_cls3 if not np.isnan(sp_cls3) else np.nan,
        "auc_bad_vs_rest":    roc_auc_score(y_bad, badness),
        "auc_good_vs_rest":   roc_auc_score(y_good, similarity_scores),
        "auc_3class_ovo":     calculate_smart_auc(class3_arr, badness, test_direction=False),
        "spearman_raw_1_5":   sp_1_5,
    }

# ---------------------------------------------------------------------------
# Группировка метрик с одинаковым search space
# ---------------------------------------------------------------------------

def _params_key(params):
    """Хешируемое представление params (list -> tuple)."""
    items = []
    for k, v in sorted(params.items()):
        if isinstance(v, list):
            v = tuple(v)
        items.append((k, v))
    return tuple(items)


def _group_identities(identities):
    """
    Группирует метрики по сигнатуре search space:
      (tuple(channel_modes), _params_key(params))
    Метрики в одной группе используют одинаковые конфиги preprocess'а.
    """
    groups = {}
    for name in identities:
        info = METRIC_IDENTITIES[name]
        params = info.get("params", {})
        key = (tuple(info["channel_mode"]), _params_key(params))
        groups.setdefault(key, []).append(name)
    return list(groups.values())


# ---------------------------------------------------------------------------
# Inner CV: перебор конфигов для группы метрик (или одной)
# ---------------------------------------------------------------------------

def _eval_config_on_folds(scores_all, inner_splits, y_class3, y_1_5):
    """Eval scores на inner val фолдах, возвращает mean/std Spearman и AUC Bad."""
    fold_sp, fold_auc_bad = [], []
    for inner_tr, inner_va in inner_splits:
        inner_scores = scores_all[inner_va]
        inner_cls3 = y_class3[inner_va]
        inner_1_5 = y_1_5[inner_va]
        ev = evaluate_scores(inner_scores, inner_cls3, inner_1_5)
        fold_sp.append(ev["spearman_class3"])
        fold_auc_bad.append(ev["auc_bad_vs_rest"])
    return {
        "inner_sp_mean": float(np.nanmean(fold_sp)),
        "inner_sp_std": float(np.nanstd(fold_sp, ddof=1)),
        "inner_auc_bad_mean": float(np.nanmean(fold_auc_bad)),
        "inner_auc_bad_std": float(np.nanstd(fold_auc_bad, ddof=1)),
    }


def run_inner_cv_group(identity_names, configs, all_pairs, meta_df, outer_train_idx, seed=42, use_tqdm=True):
    """
    Для группы метрик с одинаковым search space и одного outer fold:
      - Для каждого конфига: ОДИН проход preprocessor.process -> MetricInput
      - Затем для каждой identity в группе: metric_fn(inp) -> scores -> inner fold eval
    
    Возвращает: {identity_name: [{cfg_idx, config, inner_sp_mean, ...}, ...]}
    """
    train_meta = meta_df.iloc[outer_train_idx].copy()
    train_pairs = [all_pairs[i] for i in outer_train_idx]

    y_class3 = train_meta["class3"].to_numpy(dtype=int)
    y_1_5 = train_meta["Similarity_Score"].to_numpy(dtype=float)
    groups = train_meta["pname"].astype(str).to_numpy()

    inner_splitter = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=seed)
    inner_splits = list(inner_splitter.split(np.zeros(len(train_meta)), y_class3, groups))

    all_results = {name: [] for name in identity_names}

    iterator = tqdm(configs, desc="  configs", leave=False) if use_tqdm else configs
    for cfg_idx, config in enumerate(iterator):
        # ОДИН проход: preproc + все метрики группы
        scores_dict = compute_scores_for_config(config, train_pairs, identity_names)

        for name in identity_names:
            scores_all = scores_dict[name]
            agg = _eval_config_on_folds(scores_all, inner_splits, y_class3, y_1_5)
            all_results[name].append({
                "cfg_idx": cfg_idx,
                "config": {**config, "metric": name},
                **agg,
            })

    return all_results


# ---------------------------------------------------------------------------
# Выбор лучшего конфига по результатам inner CV
# ---------------------------------------------------------------------------

def select_best_config(inner_results):
    """
    primary:   inner_sp_mean (максимизируем)
    secondary: inner_auc_bad_mean (tie-break)

    Возвращает dict: {config, inner_sp_mean, inner_auc_bad_mean, ...}
    """
    best = max(inner_results, key=lambda r: (r["inner_sp_mean"], r["inner_auc_bad_mean"]))
    return {
        "config": best["config"],
        "inner_sp_mean": best["inner_sp_mean"],
        "inner_sp_std": best["inner_sp_std"],
        "inner_auc_bad_mean": best["inner_auc_bad_mean"],
        "inner_auc_bad_std": best["inner_auc_bad_std"],
    }


# ---------------------------------------------------------------------------
# Кэширование датасета (вынесено из run_outer_cv для переиспользования)
# ---------------------------------------------------------------------------

def _cache_dataset(dataset):
    """Загружает все пары изображений и метаданные в память."""
    all_pairs = []
    meta_rows = []
    for item in tqdm(dataset, desc="Caching"):
        f_patch, w_patch, dist, target, meta = dataset.unpack_item(item)
        all_pairs.append((f_patch, w_patch))
        meta_rows.append(meta)

    meta_df = pd.DataFrame(meta_rows)
    meta_df["Similarity_Score"] = [dataset.annotations.iloc[i][dataset.dist_col] for i in range(len(dataset))]
    meta_df["class3"] = [dataset.annotations.iloc[i][dataset.class_col] for i in range(len(dataset))]

    print(f"Cached {len(all_pairs)} pairs, {meta_df['pname'].nunique()} WSI")
    return all_pairs, meta_df


# ---------------------------------------------------------------------------
# Обработка одного outer fold (вызывается как sequentially, так и из ProcessPoolExecutor)
# ---------------------------------------------------------------------------

def _process_single_fold(fold_id, all_pairs, meta_df, outer_tr, outer_va, identities, outer_seed, use_tqdm=True):
    """
    Один outer fold: inner CV по группам метрик, выбор конфигов, predict outer_val.
    
    Возвращает dict: {oof_predictions, fold_meta}
    """
    groups_arr = meta_df["pname"].astype(str).to_numpy()

    print(f"\n=== Outer fold {fold_id + 1}/5 ===")
    print(f"  outer_train: {len(outer_tr)} pairs, {len(np.unique(groups_arr[outer_tr]))} WSI")
    print(f"  outer_val:   {len(outer_va)} pairs, {len(np.unique(groups_arr[outer_va]))} WSI")

    # группируем метрики с одинаковым search space
    metric_groups = _group_identities(identities)

    oof_preds = []
    fold_results = []

    for id_group in metric_groups:
        t0 = time.time()
        print(f"  \n--- Group: {id_group} ---")

        # конфиги от первой метрики в группе (search space идентичен)
        first_name = id_group[0]
        configs = generate_configs(first_name)
        for cfg in configs:
            cfg.pop("metric", None)
        print(f"    {len(configs)} configs")

        # inner CV для всей группы (один preproc-проход на конфиг)
        inner_seed = outer_seed + fold_id
        group_results = run_inner_cv_group(
            id_group, configs, all_pairs, meta_df, outer_tr, seed=inner_seed, use_tqdm=use_tqdm
        )

        t_inner = time.time()
        print(f"    Time of inner CV: {t_inner - t0:.1f}s")

        for identity_name in id_group:
            # выбираем лучший конфиг для этой метрики
            best = select_best_config(group_results[identity_name])
            fold_results.append({
                "fold": fold_id,
                "metric": identity_name,
                **best,
            })
            print(f"    {identity_name}: best sp={best['inner_sp_mean']:.4f}, auc_bad={best['inner_auc_bad_mean']:.4f}")
            print(f"    Config: {best['config']}\n")

            # предсказание на outer_val
            val_pairs = [all_pairs[i] for i in outer_va]
            val_scores = compute_scores_for_config(best["config"], val_pairs, identity_name)

            for i, sample_idx in enumerate(outer_va):
                oof_preds.append({
                    "sample_idx": int(sample_idx),
                    "metric": identity_name,
                    "fold": int(fold_id),
                    "similarity_score": float(val_scores[i]),
                    "config": best["config"],
                })

    return {"oof_predictions": oof_preds, "fold_meta": fold_results}


# ---------------------------------------------------------------------------
# Outer CV: оркестратор (sequential / parallel)
# ---------------------------------------------------------------------------

_GPU_METRICS = {"lpips_alex", "lpips_vgg", "lpips_squeeze", "dists"}

def run_outer_cv(dataset, identities=None, outer_seed=155, n_jobs=1):
    """
    Основной nested CV цикл.
    
    dataset: SimpleDataset с return_meta=True
    identities: список metric identity names (по умолчанию все из METRIC_IDENTITIES)
    n_jobs: 1 = последовательно, >1 = параллельно outer folds (ProcessPoolExecutor, spawn)
    
    Возвращает:
      oof_predictions, fold_meta, meta_df
    """
    if identities is None:
        identities = list(METRIC_IDENTITIES.keys())

    print(f"\nTest metrics: {identities}")
    print(f"n_jobs: {n_jobs}\n")

    if n_jobs > 1:
        gpu_ids = [m for m in identities if m in _GPU_METRICS]
        if gpu_ids:
            print(f"  [WARNING] GPU metrics detected: {gpu_ids}.")
            print(f"  Parallel workers on one GPU may cause contention (slowdown).")
            print(f"  Consider n_jobs=1 or 2 for GPU-heavy runs.\n")

    # кэширование (один раз в родительском процессе)
    print("Caching dataset...")
    all_pairs, meta_df = _cache_dataset(dataset)

    # outer splits
    y = meta_df["class3"].to_numpy(dtype=int)
    groups = meta_df["pname"].astype(str).to_numpy()
    outer_splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=outer_seed)
    outer_splits = list(outer_splitter.split(np.zeros(len(meta_df)), y, groups))

    if n_jobs == 1:
        # --- sequential ---
        oof_predictions = []
        fold_meta = []
        for fold_id, (outer_tr, outer_va) in enumerate(outer_splits):
            result = _process_single_fold(fold_id, all_pairs, meta_df, outer_tr, outer_va, identities, outer_seed)
            oof_predictions.extend(result["oof_predictions"])
            fold_meta.extend(result["fold_meta"])
    else:
        # --- parallel ---
        with ProcessPoolExecutor(max_workers=min(n_jobs, len(outer_splits)), mp_context=mp.get_context("spawn")) as pool:
            futures = []
            for fold_id, (outer_tr, outer_va) in enumerate(outer_splits):
                f = pool.submit(_process_single_fold, fold_id, all_pairs, meta_df,
                                outer_tr, outer_va, identities, outer_seed, False)
                futures.append(f)

            oof_predictions = []
            fold_meta = []
            for f in futures:
                result = f.result()
                oof_predictions.extend(result["oof_predictions"])
                fold_meta.extend(result["fold_meta"])

    return oof_predictions, fold_meta, meta_df


# ---------------------------------------------------------------------------
# Per-fold нормализация OOF predictions (rank → [0, 1])
# ---------------------------------------------------------------------------

def normalize_oof_per_fold(oof_predictions):
    """
    Rank-нормализация similarity_score внутри каждой (metric, fold) пары.
    Решает проблему несопоставимого масштаба между фолдами (например, lin vs avg в LPIPS).
    Spearman/AUC инвариантны к этой трансформации внутри fold.
    """
    df = pd.DataFrame(oof_predictions)
    df["similarity_score"] = df.groupby(["metric", "fold"])["similarity_score"].transform(
        lambda x: (scipy.stats.rankdata(x) - 1) / (len(x) - 1) if len(x) > 1 else x
    )
    return df.to_dict("records")


# ---------------------------------------------------------------------------
# OOF агрегация: финальные метрики по всем парам
# ---------------------------------------------------------------------------

def evaluate_oof(oof_predictions, meta_df):
    """
    Принимает OOF predictions и meta_df.
    Для каждой метрики вычисляет финальные метрики на всех 522 предсказаниях.
    """
    df = pd.DataFrame(oof_predictions)
    results = []

    for metric_name, group in df.groupby("metric"):
        # убеждаемся в порядке sample_idx
        group = group.sort_values("sample_idx")
        scores = group["similarity_score"].to_numpy(dtype=float)

        cls3 = meta_df["class3"].to_numpy(dtype=int)
        score_1_5 = meta_df["Similarity_Score"].to_numpy(dtype=float)

        # проверка: число предсказаний
        assert len(scores) == len(meta_df), \
            f"Metric {metric_name}: got {len(scores)} predictions, expected {len(meta_df)}"

        ev = evaluate_scores(scores, cls3, score_1_5)

        results.append({
            "metric": metric_name,
            "spearman_class3": ev["spearman_class3"],
            "auc_bad_vs_rest": ev["auc_bad_vs_rest"],
            "auc_good_vs_rest": ev["auc_good_vs_rest"],
            "auc_3class_ovo": ev["auc_3class_ovo"],
            "spearman_raw_1_5": ev["spearman_raw_1_5"],
        })

    return pd.DataFrame(results).sort_values("spearman_class3", ascending=False).reset_index(drop=True)

# ---------------------------------------------------------------------------
# WSI Bootstrap на OOF predictions
# ---------------------------------------------------------------------------/uno

def bootstrap_oof(oof_predictions, meta_df, n_boot=1000, seed=143):
    """
    WSI-level bootstrap на OOF predictions.
    
    Семплирует WSI с возвращением, для каждого bootstrap-сэмпла пересчитывает
    все метрики. Возвращает DataFrame с mean, std, CI95.
    """
    df = pd.DataFrame(oof_predictions)
    pnames = meta_df["pname"].astype(str).to_numpy()
    uniq_pnames = np.unique(pnames)
    n_wsi = len(uniq_pnames)

    # индексы патчей по WSI
    idx_by_wsi = {p: np.flatnonzero(pnames == p) for p in uniq_pnames}

    rng = np.random.default_rng(seed)
    boot_wsi_choices = rng.integers(0, n_wsi, size=(n_boot, n_wsi), endpoint=False)

    results = []

    for metric_name, group in df.groupby("metric"):
        group = group.sort_values("sample_idx")
        scores_full = group["similarity_score"].to_numpy(dtype=float)

        cls3 = meta_df["class3"].to_numpy(dtype=int)
        s15 = meta_df["Similarity_Score"].to_numpy(dtype=float)

        # full (point estimate)
        full = evaluate_scores(scores_full, cls3, s15)

        # bootstrap
        sp_b, auc_bad_b, auc_good_b, auc_ovo_b, sp15_b = [], [], [], [], []
        for b in range(n_boot):
            chosen = [idx_by_wsi[uniq_pnames[j]] for j in boot_wsi_choices[b]]
            boot_idx = np.concatenate(chosen, axis=0)
            boot_scores = scores_full[boot_idx]
            boot_cls3 = cls3[boot_idx]
            boot_s15 = s15[boot_idx]
            ev = evaluate_scores(boot_scores, boot_cls3, boot_s15)
            sp_b.append(ev["spearman_class3"])
            auc_bad_b.append(ev["auc_bad_vs_rest"])
            auc_good_b.append(ev["auc_good_vs_rest"])
            auc_ovo_b.append(ev["auc_3class_ovo"])
            sp15_b.append(ev["spearman_raw_1_5"])

        def ci(arr):
            arr = np.asarray(arr, dtype=float)
            lo = float(np.nanpercentile(arr, 2.5))
            hi = float(np.nanpercentile(arr, 97.5))
            return (round(lo, 3), round(hi, 3))

        results.append({
            "metric": metric_name,
            "full_spearman_class3": full["spearman_class3"],
            "full_auc_bad_vs_rest": full["auc_bad_vs_rest"],
            "full_auc_good_vs_rest": full["auc_good_vs_rest"],
            "full_auc_3class_ovo": full["auc_3class_ovo"],
            "full_spearman_raw_1_5": full["spearman_raw_1_5"],
            "bs_sp_mean": float(np.nanmean(sp_b)),
            "bs_sp_std": float(np.nanstd(sp_b, ddof=1)),
            "bs_sp_ci": ci(sp_b),
            "bs_auc_bad_mean": float(np.nanmean(auc_bad_b)),
            "bs_auc_bad_std": float(np.nanstd(auc_bad_b, ddof=1)),
            "bs_auc_bad_ci": ci(auc_bad_b),
            "bs_auc_good_mean": float(np.nanmean(auc_good_b)),
            "bs_auc_good_ci": ci(auc_good_b),
            "bs_auc_ovo_mean": float(np.nanmean(auc_ovo_b)),
            "bs_auc_ovo_ci": ci(auc_ovo_b),
            "n_boot": n_boot,
            "seed": seed,
        })

    return pd.DataFrame(results).sort_values("full_spearman_class3", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Сохранение / загрузка артефакта
# ---------------------------------------------------------------------------

def save_artifact(oof_predictions, fold_meta, meta_df, outer_seed, path, identities=None):
    """
    Сохраняет результаты nested CV в .pkl для последующего анализа.
    """
    if identities is None:
        identities = list(METRIC_IDENTITIES.keys())
        
    sep = " | " # for str with config parameters
    artifact = {
        "oof_predictions": oof_predictions,
        "fold_meta": fold_meta,
        "meta_df": meta_df,
        "run_info": {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "outer_seed": outer_seed,
            "identities": identities,
            "configs": sep.join([f'{k}:{v}' for k, v in COMMON_FLAGS.items()]), # save possible config params
            "n_outer_splits": 5,
            "n_inner_splits": 4,
        },
    }
    with open(path, "wb") as f:
        pickle.dump(artifact, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Artifact saved to {path}")


def load_artifact(path):
    """
    Загружает артефакт из .pkl.
    Возвращает: (oof_predictions, fold_meta, meta_df, run_info)
    """
    with open(path, "rb") as f:
        artifact = pickle.load(f)
        
    print(f"Loaded artifact from {path}")
    print(f"  run_info: {artifact['run_info']}")
    return (
        artifact["oof_predictions"],
        artifact["fold_meta"],
        artifact["meta_df"],
        artifact["run_info"],
    )
