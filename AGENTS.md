# AGENTS.md — HAPS_v1

Выжимка для агентов, работающих с этим репозиторием.

## Что это

Оценка качества метрик схожести H&E–IHC пар через **nested WSI-grouped CV**.
Каждая метрика даёт similarity-скор на пару; отбор конфига препроцессинга — по
inner CV (Spearman vs class3), финал — OOF на outer folds + bootstrap CI.

## Структура

| Путь | Назначение |
|------|-----------|
| `run_cv_v1.py` | CLI-вход (nested CV); skip, если `--output` уже существует |
| `nested_cv_metrics_opt.py` | протокол CV + `METRIC_IDENTITIES`/`METRICS_MAP` |
| `scripts/metrics.py` | пиксельные/piq метрики (ncc, psnr, mi, ssim, ms-ssim, fsim, lpips, dists) |
| `scripts/deep_metrics.py` | feature-based: `cellpose`, `lpips_cellpose`, `transpath` |
| `scripts/preprocessing.py` | `Preprocessor` → `MetricInput` |
| `scripts/dataset_class.py` | `SimpleDataset` (читает CSV с `fixed_path`/`warped_path`) |
| `scripts/analyze_results.py` | процессор результатов: таблицы + графики |
| `slurm/` | sbatch-обёртки (`run_cv.sbatch`, `env.sh`) |

## Окружение (uv)

Целевая ОС — **CentOS 7.9, glibc 2.17**: версии в `pyproject.toml` ограничены
сверху ради `manylinux2014`-колёс; `no-build = true` запрещает сборку из
исходников (древний gcc 4.8). Python 3.10 (uv скачивает сам).

```bash
cd <repo> && uv sync          # на login-ноде (нужна сеть)
```

Веса моделей предзагрузить на login-ноде (compute-ноды могут быть без сети):
```bash
uv run python - <<'PY'
import lpips; from DISTS_pytorch import DISTS; from cellpose import models
[lpips.LPIPS(net=n, lpips=l) for n in ("alex","vgg","squeeze") for l in (True,False)]
DISTS(); models.CellposeModel(gpu=False, model_type="cyto2")
PY
```

## Запуск — ТОЛЬКО через Slurm

Тяжёлый compute на login-ноде запрещён. GPU только на compute-нодах.
Подробности — в скилле `.cursor/skills/haps-slurm/SKILL.md`.

```bash
cd slurm
sbatch run_cv.sbatch "cellpose,lpips_cellpose" ../results/deep.pkl   # $1=метрики, $2=output
squeue -u $USER
```

> Список метрик — только позиционным `$1`, НЕ через `--export` (там запятая =
> разделитель переменных sbatch).

Если `output` уже существует — запуск пропускается (чтобы пересчитать, удалите `.pkl`).

GPU-метрики (`lpips_*`, `dists`, `cellpose`, `lpips_cellpose`, `transpath`) — `NJOBS=1`.
CPU-метрики (`ncc`, `psnr`, `mi`, `ssim`, `ms-ssim`) — можно `NJOBS>1`.

## Данные

CSV: колонки `fixed_path`, `warped_path` (абсолютные пути к PNG), `class3`
(0=Good,1=Border,2=Bad), `Similarity_Score` (1–5), `pname` (WSI-группа).
Локальная копия с рабочими путями:
`/trinity/home/vladislav.kozlovskiy/archieve/filtration_imgs/exp_full_local.csv`.
`SimpleDataset.base_dir` игнорируется, если путь в CSV абсолютный.

## Результаты

`.pkl` = `{oof_predictions, fold_meta, meta_df, run_info}`. Анализ:
```bash
uv run python scripts/analyze_results.py results/<name>.pkl
```
→ `summary.csv`, `bootstrap.csv`, `config_stability.csv`, `ranking.png`,
`score_distributions.png` в `<name>_analysis/`.

## Добавление метрики

1. Функция в `scripts/metrics.py` или `scripts/deep_metrics.py`, возвращает
   **similarity** (больше = лучше; для distance верните `-distance`).
2. Тяжёлые модели — ленивая загрузка через `_get_*` (глобальный кэш; важно для
   `n_jobs>1`/spawn).
3. Записать identity в `METRIC_IDENTITIES` + лямбду в `METRICS_MAP`.

## Готчи

- Не полагаться на системный `python` (2.7) / `python3` (3.6) — только `.venv`.
- `MetricInput.src_t/trg_t` — `(1,3,H,W)` float [0,1]; gray/hed дублируются в 3 канала.
- `transpath` требует ручной установки весов (`scripts/setup_transpath.sh`).
