"""Устойчивость метрик сходства H&E к геометрическим искажениям.

Переиспользуемая логика эксперимента из geom_transforms_exp/Geom_Transform_v2.ipynb:
  - искажения (shift / rotate / elastic) через albumentations;
  - прогон метрик по уровням искажений с per-metric preproc-конфигом;
  - численные индексы ES (Early Saturation) и LSR (Late Sensitivity Ratio).

Протокол эксперимента см. geom_transforms_exp/geom_transform_exp.md.
"""
from __future__ import annotations

import re

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

import albumentations as A


def crop_borders(img, margin):
    """Обрезает изображение по краям на заданное количество пикселей."""
    if margin <= 0:
        return img
    h, w = img.shape[:2]
    if 2 * margin >= max(h, w):
        raise ValueError(f"Размер кропа ({margin}) не соотв. изображению ({h}x{w})")
    return img[margin:-margin, margin:-margin]


def apply_distortion(image, dist_type, magnitude, seed=42):
    """Детерминированное геом. искажение к одному изображению.

    Args:
        image: (H, W, C) numpy array.
        dist_type: 'shift' | 'rotate' | 'elastic'.
        magnitude: уровень искажения (shift — доля размера, rotate — градусы,
                   elastic — сила GridElasticDeform).
        seed: влияет на знак (+/-) и случайную сетку (elastic).

    Returns:
        (distorted_image, applied_param)
    """
    rng = np.random.default_rng(seed)
    sign = rng.choice([-1, 1])
    val = magnitude * sign

    if dist_type == "shift":
        transforms = [A.ShiftScaleRotate(
            shift_limit_x=(val, val),
            shift_limit_y=(val, val),
            scale_limit=(0, 0),
            rotate_limit=(0, 0),
            border_mode=cv2.BORDER_REFLECT_101,
            interpolation=cv2.INTER_LINEAR,
            p=1.0)]
    elif dist_type == "rotate":
        transforms = [A.ShiftScaleRotate(
            shift_limit=(0, 0),
            scale_limit=(0, 0),
            rotate_limit=(val, val),
            border_mode=cv2.BORDER_REFLECT_101,
            interpolation=cv2.INTER_LINEAR,
            p=1.0)]
    elif dist_type == "elastic":
        val = magnitude
        # NOTE: GridElasticDeform в albumentations 2.x не принимает border_mode
        # (краевые артефакты убираются кропом 16px после искажения).
        transforms = [A.GridElasticDeform(
            num_grid_xy=(5, 5),
            magnitude=magnitude,
            interpolation=cv2.INTER_LINEAR,
            p=1.0)]
    else:
        raise ValueError(f"Unknown dist_type: {dist_type}")

    pipeline = A.Compose(transforms, seed=seed, save_applied_params=True)
    data = pipeline(image=image)
    return data["image"], val


def extract_value(val):
    """Приводит выход метрики к float (защита от str / 0-d тензоров)."""
    if isinstance(val, (int, float, np.number)):
        return float(val)
    if isinstance(val, str):
        m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", val)
        return float(m.group()) if m else None
    m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", str(val))
    return float(m.group()) if m else None


def _config_key(config):
    return tuple(sorted((k, str(v)) for k, v in config.items()))


def _group_metrics(metric_specs):
    """Группирует метрики по preproc-конфигу, чтобы считать один MetricInput на группу."""
    groups = {}
    for name, spec in metric_specs.items():
        key = _config_key(spec["config"])
        if key not in groups:
            groups[key] = {"config": spec["config"], "metrics": {}}
        groups[key]["metrics"][name] = spec["fn"]
    return groups


def run_sensitivity_benchmark(dataset, metric_specs, preprocessor, stages,
                              crop_margin=16, n_seeds=5, n_patches=None):
    """Прогон метрик по уровням геом. искажений.

    Args:
        dataset: SimpleDataset (берётся H&E-патч = f_patch).
        metric_specs: dict {name: {"fn": callable(MetricInput)->float, "config": dict}}.
        preprocessor: class Preprocessor.
        stages: dict {dist_type: [levels]}.
        crop_margin: сколько пикселей отрезать по краям.
        n_seeds: кол-во повторов на уровень (для усреднения по seed).
        n_patches: сколько патчей обработать (None = все).

    Returns:
        pd.DataFrame с колонками: patch_idx, metric, param_type, level, seed, value, actual_params.
    """
    if n_patches is None:
        n_patches = len(dataset)
    indices = range(min(len(dataset), n_patches))
    groups = _group_metrics(metric_specs)

    results = []
    for idx in tqdm(indices, desc="Benchmarking Patches"):
        # 1. Загрузка сырого патча
        try:
            item = dataset[idx]
            raw_patch, _, _, _, _ = dataset.unpack_item(item)
        except Exception as e:
            print(f"Skipping patch {idx} due to error: {e}")
            continue
            
        # 2. Подготовка эталона (Reference)
        ref_patch = crop_borders(raw_patch, crop_margin)

        # --- LEVEL 0: Baseline, metric(I_0, I_0) ---
        for key, g in groups.items():
            inp_0 = preprocessor.process(ref_patch, ref_patch, g["config"])
            for m_name, m_fn in g["metrics"].items():
                try:
                    val = m_fn(inp_0)
                except Exception as e:
                    print(f"Error metric {m_name} at baseline: {e}")
                    continue
                    
                results.append({
                    "patch_idx": idx, "metric": m_name, "param_type": "baseline",
                    "level": 0.0, "seed": 0, "value": extract_value(val), "actual_params": None,
                })

        # --- LEVEL > 0: Distortions ---
        for dist_type, levels in stages.items():
            for level in levels:
                for seed in range(n_seeds):
                    # Важно: меняем seed для каждого прогона, чтобы варьировать знак и случайность
                    current_seed = (seed + idx) * 1000
                    try:
                        dist_full, params = apply_distortion(raw_patch, dist_type, level, seed=current_seed)
                        dist_cropped = crop_borders(dist_full, crop_margin)
                        
                        for key, g in groups.items():
                            # process(f=dist_cropped, w=ref_patch) -> src_np=ref (I_0), trg_np=dist (I_k).
                            # Т.е. метрика считает metric(I_0, I_k).
                            inp_k = preprocessor.process(dist_cropped, ref_patch, g["config"])
                            for m_name, m_fn in g["metrics"].items():
                                val = m_fn(inp_k)
                                
                                results.append({
                                    "patch_idx": idx, "metric": m_name, "param_type": dist_type, "level": float(level), 
                                    "seed": seed, "value": extract_value(val), "actual_params": params,
                                })

                    except Exception as e:
                        print(f"Error at {dist_type}={level}, seed={seed}: {e}")
                        continue

    return pd.DataFrame(results)


def prepare_plotting_data(df):
    """Дублирует baseline (level=0) для каждого типа искажения, чтобы линии шли из 0."""
    df_base = df[df["level"] == 0].copy()
    df_dist = df[df["level"] > 0].copy()
    dfs = [df_dist]
    for d_type in df_dist["param_type"].unique():
        tmp = df_base.copy()
        tmp["param_type"] = d_type
        dfs.append(tmp)
    return pd.concat(dfs, ignore_index=True)


def calculate_sensitivity_indices(df):
    """Численные индексы ES (Early Saturation) и LSR (Late Sensitivity Ratio).

    Считает по метрике × типу искажения (медиана по level):
      - ES_k1..ES_k3 : доля общего изменения, накопленная уже к уровню k;
      - LSR_penult / LSR_last : отношение локального наклона (последние шаги) к глобальному среднему наклону.
        Считается для последнего (last) и предпоследнего (penult) интервалов.
    """
    # Добавляем 0-й уровень ко всем типам искажений
    df_proc = prepare_plotting_data(df)
    results_list = []

    for (metric, dist_type), group in df_proc.groupby(["metric", "param_type"]):
        # Агрегируем по уровням (берем медиану, чтобы убрать шум сидов)
        stats = group.groupby("level")["value"].median().sort_index()
        
        levels = stats.index.values.astype(float) # [0.0, 0.005, 0.01, ...]
        values = stats.values                     # [V0, V1, V2, ...]
        
        # --- 0. Global Params ---
        V0, V_max = values[0], values[-1]  # Baseline, Max distortion
        range_val = abs(V_max - V0)        # Global Delta Y
        range_lvl = levels[-1] - levels[0] # Global Delta X
        
        # Средняя глобальная скорость (Global Slope)
        slope_global = range_val / range_lvl if range_lvl > 1e-9 else 0.0
        if range_val < 1e-9:
            range_val = 1e-9

        row = {"Metric": metric, "Distortion": dist_type, "Max_Range": range_val}
        
        # --- 1. Early Saturation (ES) ---
        # k - это индекс в массиве levels (0 - это baseline)
        for k in [1, 2, 3]:
            row[f"ES_k{k}"] = abs(values[k] - V0) / range_val
                
        # --- 2. Late Sensitivity Ratio (LSR) ---
        # B. Penultimate Interval (k_max-2 -> k_max)
        dy_pen = abs(values[-1] - values[-3])
        dx_pen = levels[-1] - levels[-3]
        slope_pen = dy_pen / dx_pen if dx_pen > 1e-9 else 0.0
        row["LSR_penult"] = slope_pen / slope_global if slope_global > 1e-9 else np.nan
        
        # A. Last Interval (k_max-1 -> k_max)
        dy_last = abs(values[-1] - values[-2])
        dx_last = levels[-1] - levels[-2]
        slope_last = dy_last / dx_last if dx_last > 1e-9 else 0.0
        row["LSR_last"] = slope_last / slope_global if slope_global > 1e-9 else np.nan

        results_list.append(row)

    return pd.DataFrame(results_list)
