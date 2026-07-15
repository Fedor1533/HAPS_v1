# Nested CV Metric Evaluation

Инструмент для оценки качества метрик структурной пригодности H&E–HER2 пар через nested WSI-grouped cross-validation.

## Файлы

| Файл | Назначение |
|------|-----------|
| `nested_cv_metrics_opt.py` | **Рекомендуемая версия.** Поддерживает sequential (`n_jobs=1`) и parallel (`n_jobs>1`) outer folds через `ProcessPoolExecutor(spawn)` |
| `nested_cv_metrics.py` | Стабильная sequential-версия (без изменений протокола) |
| `run_cv_v1.py` | CLI-скрипт для запуска (использует `nested_cv_metrics_opt`) |

## Быстрый старт

```bash
# Последовательно
python run_cv_v1.py \
    --identities "ncc,psnr,mi,ssim" \
    --output results/my_run.pkl

# Параллельно (4 воркера)
python run_cv_v1.py \
    --identities "ncc,psnr,mi,ssim" \
    --output results/my_run.pkl \
    --n-jobs 4
```

Аргументы:

| Флаг | По умолчанию | Описание |
|------|-------------|----------|
| `--identities` | `ncc,psnr,mi,ssim` | Имена метрик через запятую |
| `--csv` | `filtration_imgs/exp_full.csv` | Путь к CSV с данными |
| `--output` | **обязательный** | Путь к `.pkl` с результатами |
| `--outer-seed` | `155` | Random seed для outer CV |
| `--n-jobs` | `1` | Параллельные outer folds: `1` = последовательно, `2-5` = параллельно |

**Примечание по `--n-jobs`**: для GPU-метрик (lpips_*, dists) несколько воркеров на одном GPU вызывают замедление. Для CPU-метрик (ncc, psnr, mi, ssim, ms-ssim) — почти линейное ускорение.

Результат — `.pkl` с полями: `oof_predictions`, `fold_meta`, `meta_df`, `run_info`.

---

## Добавление своей метрики

Все изменения — в `nested_cv_metrics_opt.py`. Три шага:

### Шаг 1. Написать функцию подсчёта метрики

Функция принимает тензоры `(1, C, H, W) float [0, 1]` и **возвращает distance** (меньше — лучше):

```python
# Можно добавить в scripts/metrics.py или прямо в nested_cv_metrics_opt.py

def calc_my_metric(src_t, trg_t, my_model, **kw):
    """
    src_t, trg_t: Tensor (1, C, H, W), float [0, 1], на GPU/CPU
    my_model:     предзагруженная модель-бэкбон
    **kw:         metric-specific параметры из конфига

    Возвращает: скаляр (float) — distance, где 0 = идеальное совпадение.
    """
    # ... сравнение фича-мапов между src_t и trg_t ...
    return float(distance)
```

Если метрика возвращает **similarity** (чем больше — тем лучше), см. примечание в конце шага 2.

### Шаг 2. Зарегистрировать identity и маппинг

В `nested_cv_metrics_opt.py` найти секции `METRIC_IDENTITIES` и `METRICS_MAP`:

```python
# --- Инициализация модели (лениво или при импорте) ---
_MY_MODEL = None

def _get_my_model():
    global _MY_MODEL
    if _MY_MODEL is None:
        _MY_MODEL = load_my_backbone().eval().to(device)
    return _MY_MODEL

# --- METRIC_IDENTITIES: search space для identity ---
METRIC_IDENTITIES["my_metric"] = {
    "channel_mode": ["gray", "hed", "rgb"],    # режимы каналов для preprocess'а
    "params": {                                  # metric-specific гиперпараметры
        "layer_aggregation": ["mean", "max"],
    },
    # params — опционально. Если метрика не имеет своих параметров — НЕ писать ключ "params".
}

# --- METRICS_MAP: связь identity → функция ---
METRICS_MAP["my_metric"] = lambda inp, **kw: -calc_my_metric(inp.src_t, inp.trg_t, _get_my_model(), **kw)
```

**ВАЖНО**: минус перед `calc_my_metric` инвертирует distance → similarity, как того требует протокол. Если ваша метрика возвращает similarity — не ставьте минус.

### Шаг 3. Запустить

```bash
python run_cv_v1.py --identities "my_metric" --output results/my_metric.pkl --n-jobs 4
```

---

## Анализ результатов в Python

```python
import pandas as pd
from nested_cv_metrics_opt import load_artifact, evaluate_oof, bootstrap_oof, add_preproc_key

oof_preds, fold_meta, meta_df, run_info = load_artifact("results/my_metric.pkl")

# Финальные метрики
results = evaluate_oof(oof_preds, meta_df)

# Bootstrap CI
bs = bootstrap_oof(oof_preds, meta_df, n_boot=1000, seed=143)

# Стабильность выбора конфигов по outer folds
fm_df = pd.DataFrame(fold_meta).sort_values(by='metric')
fm_df = add_preproc_key(fm_df)
print(fm_df.groupby("metric")["preproc_cfg"].value_counts())
```

---

## Справочник по `MetricInput`

Все поля доступны в лямбде `METRICS_MAP`:

```
inp.src_np    # np.ndarray (H, W) или (H, W, C) — предобработанное HER2
inp.trg_np    # np.ndarray — предобработанное H&E
inp.mask_np   # np.ndarray (H, W) — маска ткани (не используется, всегда None)
inp.src_t     # torch.Tensor (1, C, H, W) float [0, 1] — версия для GPU
inp.trg_t     # torch.Tensor
inp.mask_t    # torch.Tensor (не используется, всегда None)
inp.bg_val    # float — 0.0 если flip_intensity=True, иначе 1.0 (не используется, всегда 0.0)
```

Классические метрики (NCC, PSNR, MI, SSIM) используют `src_np, trg_np`.  
Перцептуальные (LPIPS, DISTS, FSIM) используют `src_t, trg_t`.

---

## Как устроен протокол

1. **Outer CV**: 5-fold StratifiedGroupKFold по WSI (стратификация по class3). При `n_jobs > 1` folds выполняются параллельно через `ProcessPoolExecutor(spawn)`
2. **Inner CV** (внутри каждого outer fold): 4-fold StratifiedGroupKFold на outer_train
3. **Выбор конфига**: для каждой metric identity перебираются все комбинации `COMMON_FLAGS × channel_mode × params`. Лучший конфиг — по Spearman (inner CV), tie-break — AUC Bad vs Rest
4. **OOF predictions**: выбранный конфиг применяется к outer_val → собираются предсказания по всем парам
5. **Primary метрики**: Spearman vs class3, AUROC Bad vs Rest
6. **Bootstrap**: WSI-level семплирование для CI95

Метрики с одинаковым search space обрабатываются группой (один preprocess-проход на пару) для оптимизации.
