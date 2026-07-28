# Nested CV Metric Evaluation

Оценка качества метрик схожести H&E–IHC пар через nested WSI-grouped
cross-validation. Каждая метрика даёт similarity-скор на пару, лучший
препроцессинг подбирается по inner CV, финал — OOF + bootstrap CI.

## Установка (uv)

Окружение собирается через [uv](https://docs.astral.sh/uv/). На login-ноде
кластера (там есть сеть):

```bash
uv sync
```

> ОС кластера — CentOS 7.9 (glibc 2.17), поэтому версии пакетов в `pyproject.toml`
> зафиксированы под `manylinux2014`-колёса. Python 3.10 уже установленный не нужен —
> uv скачает сам.

## Запуск — через Slurm

Полные прогоны запускаются **на compute-нодах через Slurm** (GPU только там):

```bash
cd slurm
sbatch run_cv.sbatch "cellpose" ../results/cellpose.pkl      # $1=метрики, $2=output
squeue -u $USER
```

Если `output` уже существует — запуск пропускается (чтобы пересчитать, удалите `.pkl` вручную).

Основные метрики: `ncc, psnr, mi, ssim, ms-ssim, fsim, fsimc, lpips_alex/vgg/squeeze,
dists, cellpose, lpips_cellpose, transpath`.

## Анализ результатов

Результат каждого прогона — `.pkl`. Сводка + графики:

```bash
uv run python scripts/analyze_results.py results/cellpose.pkl
```

Создаёт `summary.csv`, `bootstrap.csv`, `config_stability.csv`, `ranking.png`,
`score_distributions.png`.

Для deep-метрик с config-dependent scale (LPIPS `lin`/`avg` и т.п.) перед
OOF-оценкой полезна per-fold rank-нормализация:

```python
from nested_cv_metrics_opt import load_artifact, evaluate_oof, normalize_oof_per_fold

oof, fold_meta, meta_df, info = load_artifact("results/my_run.pkl")
oof_norm = normalize_oof_per_fold(oof)  # для классических метрик обычно не нужна
results = evaluate_oof(oof_norm, meta_df)
```

## Документация

- **`AGENTS.md`** — устройство репозитория, окружение, запуск (для агентов и разработчиков).
- **`.cursor/skills/haps-slurm/`** — скилл по запуску через Slurm.
- **`scripts/setup_transpath.sh`** — настройка весов CTransPath для метрики `transpath`.
