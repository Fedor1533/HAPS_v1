---
name: haps-slurm
description: Run HAPS_v1 metric experiments on the Slurm cluster (CentOS 7, gpu partitions) using the uv environment. Use when running run_cv_v1.py, nested CV metric experiments, submitting sbatch jobs, or when the user asks to launch anything in HAPS_v1.
---

# HAPS_v1 через Slurm

Эксперименты HAPS_v1 запускаются **только через Slurm** (не на login-ноде).
GPU есть лишь на compute-нодах. Окружение — `uv` (`.venv` в корне репо).

## Правила

- НИКОГДА не запускать тяжёлый compute на login-ноде — только `sbatch`/`srun`.
- Зависимости (`uv sync`) ставятся на login-ноде (там есть сеть). Веса моделей
  (lpips, dists, cellpose cyto2) должны быть предзагружены в `~/.cellpose` и
  `~/.cache/torch` на login-ноде — compute-ноды могут быть без интернета.
- Внутри задач используется `.venv/bin/python` напрямую (без `uv run`, без сети).
- Если `--output` / `.pkl` уже существует — запуск пропускается; чтобы
  пересчитать, удалите файл вручную.

## Партиции

| Партиция | Лимит | Для чего |
|----------|-------|----------|
| `gpu_devel` | 12ч | отладка (быстрая очередь) |
| `gpu` | 6 дней | полные прогоны nested CV |
| `gpu_a100` / `ais-gpu` | 1–6 дней | тяжёлые/приоритетные |

## Готовые скрипты (в `slurm/`)

- `slurm/env.sh` — общее окружение (REPO, `.venv`, кэши весов).
- `slurm/run_cv.sbatch` — полный nested CV одной/нескольких метрик.

## Workflow

```
- [ ] 1. (однократно) на login-ноде: cd <repo> && uv sync
- [ ] 2. (однократно) предзагрузить веса на login-ноде (см. AGENTS.md)
- [ ] 3. эксперимент: sbatch slurm/run_cv.sbatch "<metrics>" <output.pkl>
- [ ] 4. мониторинг: squeue -u $USER
- [ ] 5. анализ: uv run python scripts/analyze_results.py results/<name>.pkl
```

### Полный эксперимент

```bash
cd <repo>/slurm
sbatch run_cv.sbatch "cellpose,lpips_cellpose" ../results/deep.pkl
```
Позиционные: `$1`=метрики (через запятую), `$2`=выходной `.pkl` (необяз.).
Прочее — через `--export=ALL,CSV=...,NJOBS=...,OUTER_SEED=...` (для CPU-метрик
`NJOBS>1`, GPU-метрики держать `NJOBS=1`).

> НЕ передавайте список метрик через `--export=ALL,IDENTITIES=...`: `sbatch`
> трактует запятую как разделитель переменных. Только позиционный `$1`.

GPU-метрики (`lpips_*`, `dists`, `cellpose`, `lpips_cellpose`, `transpath`) —
`NJOBS=1`. CPU-метрики (`ncc`, `psnr`, `mi`, `ssim`, `ms-ssim`) — можно `NJOBS=4`.

### Мониторинг и результат

```bash
squeue -u $USER -o "%.10i %.12j %.10P %.2t %.10M %.20R"   # статус
tail -f slurm/haps_cv_<jobid>.out                          # прогресс
uv run python scripts/analyze_results.py results/deep.pkl  # анализ + графики
```

## transpath

Метрика `transpath` требует отдельной настройки весов CTransPath.
На login-ноде: `bash scripts/setup_transpath.sh`, затем задать `TRANSPATH_ROOT`
и `TRANSPATH_WEIGHTS` (можно раскомментировать в `slurm/env.sh`).

## Отладка (интерактивно)

```bash
srun -p gpu_devel --gres=gpu:1 --cpus-per-task=4 --mem=24G --time=00:30:00 --pty bash
# затем: cd <repo> && .venv/bin/python -c "import torch; print(torch.cuda.is_available())"
```
