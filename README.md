# HAPS_v1 — nested CV для метрик H&E–IHC сходства

Реализация оценки метрик структурного сходства на экспертно размеченных парах патчей
H&E ↔ HER2-IHC через nested cross-validation. Код развивает протокол из
`metrics_code_v2/` и добавляет feature-based метрики на предобученных гистологических
бэкбонах, структурные метрики и foundation-модели.

> **Статус.** Автор кода (коллега) передал проект без README. Этот файл — описание
> реализации и логики, восстановленное по коду и результатам. Проект больше не
> развивается автором; продолжение — на стороне `f.gubanov`.

---

## 1. Структура проекта

| Файл / каталог | Назначение |
|----------------|-----------|
| `run_cv_v1.py` | CLI-скрипт запуска nested CV для набора метрик |
| `nested_cv_metrics_opt.py` | Ядро: outer/inner CV, search space, выбор конфига, OOF, bootstrap |
| `scripts/metrics.py` | Классические/перцептуальные метрики (NCC, PSNR, MI, SSIM, MS-SSIM, FSIM, LPIPS, DISTS) |
| `scripts/structural_metrics.py` | Структурные метрики: NGF, тайловые NCC/MI/NGF, остаточное смещение (phase correlation), ядерная геометрия |
| `scripts/deep_metrics.py` | Cellpose-энкодер, LPIPS-Cellpose, локальный CTransPath (TransPath) |
| `scripts/foundation_metrics.py` | Реестр HF foundation-бэкбонов + извлечение фич по слоям |
| `scripts/preprocessing.py` | `Preprocessor`, `MetricInput`, конвейер предобработки |
| `scripts/dataset_class.py` | `SimpleDataset` (загрузка пар + метаданные) |
| `scripts/eval_metrics.py` | `calculate_smart_auc` (бинарный + ординальный OVO AUC) |
| `scripts/eval_deep_noval.py` | No-validation прогон deep-метрик (cellpose/lpips_cellpose/transpath) |
| `scripts/eval_foundation_noval.py` | No-validation прогон foundation-метрик |
| `scripts/eval_layers_noval.py` | No-validation layer-sweep (deep + foundation), один forward на пару |
| `scripts/eval_layers_fullgrid.py` | Полный грид preprocess × слои (оптимистичный потолок, без CV) |
| `scripts/analyze_results.py` | Постобработка `.pkl`: summary, bootstrap, стабильность конфигов, графики |
| `scripts/verify_foundation.py` | Проверка foundation-бэкбонов (веса, data-config, эмбеддинг) |
| `scripts/smoke_test.py` | Smoke-тест метрик на нескольких парах |
| `scripts/diag_nuclei.py` | Диагностика сегментации ядер для метрики `nuclei` |
| `scripts/ab_foundation_old_vs_new.py` | A/B пересчёт скоров на старой vs новой версии кода |
| `scripts/setup_foundation.sh` | Сборка `.venv-foundation` (modern timm + cpsam) |
| `scripts/setup_transpath.sh` | Настройка локального CTransPath (код + веса + patched timm) |
| `slurm/` | sbatch-скрипты и submit-обёртки для кластера |
| `TransPath/` | Локальный клон CTransPath (`ctran.py`, `ctranspath.pth`, `timm-0.5.4.tar`) |
| `pyproject.toml` | Зависимости через uv (Python 3.10, torch cu121) |
| `results/` | Артефакты прогонов (`.pkl`) и сводки (`*.csv`, `excel_report/`) |

### 1.1. Скрипты `slurm/`

| Файл | Назначение |
|------|-----------|
| `env.sh` | Общая настройка окружения: `REPO`, `VENV_PY` (=`$REPO/.venv/bin/python`), кэши (cellpose/torch/HF), HF-токен, `TRANSPATH_ROOT/WEIGHTS`. Все sbatch делают `source env.sh`. |
| `run_cv.sbatch` | Основной runner nested CV (GPU-партиция): `$1`=identities, `$2`=output; зовёт `run_cv_v1.py`; skip-if-exists. |
| `run_cv_cpu.sbatch` | То же на CPU-партиции (`htc`): `NJOBS=5` параллельных outer-фолдов, прижимает BLAS/OpenCV к 1 потоку. Для структурных + классики без GPU. |
| `smoke.sbatch` | Smoke-тест (`smoke_test.py`) на нескольких парах — проверка окружения/кода. |
| `run_deep_noval.sbatch` | `eval_deep_noval.py` (фикс. конфиг, без CV) для `lpips_cellpose,cellpose,transpath`. |
| `run_deep_noval_fast.sbatch` | То же, но без layer-sweep (быстрее). |
| `run_deep_bestcfg.sbatch` | inline-Python: прогон заранее подобранных конфигов deep-метрик → `results/deep_noval_bestcfg.csv`. |
| `run_fullgrid.sbatch` | `eval_layers_fullgrid.py` (полный грид preprocess × слои) для foundation, через `.venv-foundation`. |
| `run_layers_noval.sbatch` | `eval_layers_noval.py --family deep|foundation` (layer-sweep, без CV). |
| `diag.sbatch` | Generic GPU-обёртка: запускает `"$@"` (использовался для `diag_nuclei.py`). |
| `submit_foundation.sh` | По одному `run_cv.sbatch` на каждый foundation-бэкбон, через `.venv-foundation`. |
| `submit_structural.sh` | Сабмит CPU-джобов для структурных метрик + классики с маской. |
| `submit_layers.sh` | Сабмит layer-sweep: noval + nested CV (`*_spec` / cellpose / transpath), бэкап `.pkl`. |
| `submit_layers_rest.sh` | Досабмит оставшихся (`cpsam_spec`). |

> **Внимание.** Все скрипты захардкожены на машину автора: `REPO=/trinity/home/
> vladislav.kozlovskiy/HAPS/HAPS_v1` и `CSV=/trinity/.../exp_full_local.csv` (522 примера).
> Перед использованием переопределить `REPO` (на эту копию) и `CSV` (на
> `filtration_imgs/exp_full.csv`, 512). Партиции (`gpu`, `gpu_devel`, `htc`) сверять с
> `sinfo`/QOS кластера.

---

## 2. Логика работы (протокол)

Протокол унаследован из `metrics_code_v2/` и расширен. Общая схема:

1. **Датасет.** Пары патчей (H&E + IHC) с экспертной оценкой: `Similarity_Score` (1–5)
   и ординальный класс `class3` (0=Good, 1=Borderline, 2=Bad). Группировка по слайду
   `pname`.
2. **Outer CV.** 5-fold `StratifiedGroupKFold` (группы = `pname`, стратификация по
   `class3`). Для каждой метрики на outer_val собираются out-of-fold предсказания.
3. **Inner CV** (внутри каждого outer fold). 4-fold `StratifiedGroupKFold` на
   outer_train. Перебираются все комбинации препроцессинга и metric-specific
   параметров (см. §3). Лучший конфиг — по среднему inner Spearman vs `class3`,
   tie-break — AUC Bad vs Rest.
4. **OOF-предсказания.** Выбранный конфиг применяется к outer_val → `similarity_score`
   по всем парам. Для метрик с нестабильным масштабом между фолдами применяется
   per-fold rank-нормализация `normalize_oof_per_fold()`.
5. **Финальные метрики.** `evaluate_oof`: Spearman vs `class3`, AUC Bad vs Rest,
   AUC Good vs Rest, 3-class OVO AUC, Spearman vs raw 1–5.
6. **Bootstrap.** WSI-level семплирование (1000 итераций, seed=143) → CI95.

Оптимизации:

- **Групповая обработка** — метрики с одинаковым search space считаются из одного
  `MetricInput` (один проход препроцессинга на пару).
- **Layer-aware fast path** (`run_inner_cv_group`) — для метрик с параметром
  `feature` (layer-sweep) один forward бэкбона на пару даёт скоры сразу по всем
  слоям/агрегациям (`_preproc_signature` + `_layer_scores_for_pairs`).
- **Параллелизация** — outer folds через `ProcessPoolExecutor` (`--n-jobs`).

Направление score-vs-class: все метрики приводятся к `similarity` (больше = лучше пара);
distance-метрики (LPIPS/DISTS/LPIPS-Cellpose) инвертируются.

---

## 3. Метрики и search space

Общие флаги препроцессинга (`COMMON_FLAGS`):

```
normalization, channel_mode, flip_intensity, match_histogram, clahe, smoothing, binary_mask
```

Для каждой identity можно сузить набор флагов через ключ `flags` (фиксированные
значения, не попадающие в перебор) и добавить metric-specific параметры через `params`.

| Семейство | Identity | Параметры поиска |
|-----------|----------|------------------|
| Классика | `ncc`, `psnr`, `mi`, `ssim` | `ssim` + `win_size` [7, 31]; у всех включён `binary_mask` |
| | `ms-ssim`, `fsim`, `fsimc` | — |
| Перцептуальные | `lpips_alex`, `lpips_vgg`, `lpips_squeeze` | `lpips_aggregation` [avg, lin] |
| | `dists` | — |
| Deep | `lpips_cellpose` | — |
| | `cellpose` | `feature` × `agg` [cos, dist] |
| | `transpath` | `feature` × `agg` [cos, dist] (rgb, фикс. флаги) |
| Структурные | `ngf` | `eta` [0.25, 1.0] |
| | `tile_ncc`, `tile_mi`, `tile_ngf` | `tile` × `agg` [min, p10, q25, mean] |
| | `flow_residual` | `tile`, `stat`, `on` [grad, raw], `min_resp` |
| | `nuclei` | `stat`, `radius`, `diameter`, `channel` [gray, hem] |
| Foundation | `<name>` и `<name>_spec` | `feature` (layer_k/neck/mean) × `agg` [cos, dist] |

**Foundation-бэкбоны** (`scripts/foundation_metrics.py`, `FOUNDATION_BACKBONES`):
`uni2h` (UNI2-h), `gigapath` (prov-gigapath), `virchow2` (Virchow2), `hoptimus0`
(H-optimus-0), `hibou_l` (hibou-L), `phikon` (owkin/phikon), `ctranspath_hf`
(HF-зеркало CTransPath), `cpsam` (Cellpose-SAM). Большинство — gated (нужен
HF-токен и принятие лицензии).

Разница plain vs `_spec`:

- **plain** — полный препроцесс-грид (`rgb`/`hed` + `COMMON_FLAGS`) + layer sweep;
- **`_spec`** — mean/std/crop из спеки модели, канал `rgb`, без `flip_intensity`
  (инверсия ломает ImageNet/histology-нормализацию), остальные флаги — в поиске.

### 3.1. Как устроены feature-метрики (принцип работы)

Все deep/foundation-метрики устроены одинаково по форме: обе картинки пары прогоняются
через **замороженный** бэкбон → извлекаются фичи → фичи сравниваются. Разница — в том,
**что сравнивается** и **как**:

1. **LPIPS-формулировка** — сравнение карт признаков попозиционно (только `lpips_cellpose`).
2. **Cosine / -L2 по сплющенному вектору** — вся карта фич/токенов сворачивается в один
   длинный вектор и сравнивается целиком (все остальные: `cellpose`, `transpath`,
   foundation-бэкбоны).

#### `cellpose`

- **Энкодер**: downsample-часть Cellpose `cyto2` (ResUNet). RGB усредняется в grayscale и
  дублируется в 2 канала (Cellpose ждёт 2 канала). Выход — 4 карты `[32, 64, 128, 256]`.
- **Фичи** (`extract_cellpose_features`): `layer_0..layer_3` + `neck` (= `layer_3`).
- **Сравнение** (`calc_cellpose`): `layer_k`/`neck` → `reshape(1, -1)` → `cos` (косинус по
  сплющенному вектору) или `dist` (`-‖normalize(a)-normalize(b)‖`); `mean` — усреднение по
  `layer_*`.

#### `lpips_cellpose` — единственная «настоящая LPIPS»

Тот же энкодер `cyto2`, но LPIPS-формулировка **без** обучаемых весов (эквивалент
`lpips=False`): по каждой из 4 карт `normalize_tensor` (L2-норма по каналам) →
`(f0-f1)²` → сумма по каналам → среднее по пространству → скаляр на слой; слои
суммируются с весом 1.0, итог `-dist`.

#### `transpath` (локальный CTransPath, Swin-T)

- **Вход**: resize 224 + ImageNet mean/std-нормализация.
- **Фичи** (`extract_transpath_features`): ручной проход `patch_embed` → 12 Swin-блоков;
  после каждого блока сохраняется выход как `layer_0..layer_11`, `neck = mean` по токенам
  после `model.norm`.
- **Формат**: timm-Swin работает в **BNC `(B, L, C)`** — 3D-последовательность патч-токенов
  (batch × токены × каналы), а не пространственная карта `(H, W)`. В NHWC Swin переходит
  **только внутри блока** для window-attention и возвращает обратно BNC (`x.view(B, H*W, C)`).
  `layer_k` по стадиям Swin-T: `(B, 3136, 96)` → `(B, 784, 192)` → `(B, 196, 384)` →
  `(B, 49, 768)`; `neck = z.mean(dim=1)` → `(B, 768)`. 
  **Flatten при сравнении** происходит в `_cosine`/`_normalized_dist` через `reshape(1, -1)`.

#### Foundation-бэкбоны

- **Вход** (`prepare_input`): resize до `spec.img_size` (+ center-crop, если `crop_pct<1`)
  и модель-специфичная нормализация: mean/std (ImageNet или histology, напр. H-optimus,
  hibou-L) либо percentile-нормализация для `cpsam`.
- **Два представления** (`calc_foundation`):
  - `neck` — глобальный эмбеддинг (`spec.embed`): 
      - timm-ViT → CLS-токен;
      - virchow2 → CLS ‖ mean(patch-токены, пропуская 4 register-токенов);
      - HF-ViT (phikon/hibou_l) → CLS из last_hidden_state;
      - cpsam → GAP по neck-фичам энкодера SAM;
      - Swin (ctranspath_hf) → mean по токенам после norm.,
  - `layer_k` — послойные токены (`extract_foundation_features`): берутся выходные токены
    каждого блока (`get_intermediate_layers` / `forward_intermediates` / ручной walk по `blocks`), затем
    `_pool_tokens(pool="flat")` **сплющивает их в один вектор** `(B, N*D)`.
- **Сравнение** (`_score_feature_pair`): `cos` / `dist` по сплющенному вектору; `mean` —
  усреднение по `layer_*`.

**Важно про ViT.** Промежуточное представление ViT — это **последовательность
патч-токенов `(B, N, D)`**. Для `uni2h`(`embed_dim=1536, patch_size=14, img_size=224`) это `D=1536`, 
`N=256` патч-токенов (`(224/14)²=16×16`), поэтому `layer_k` после flatten — вектор длины `1536 × 256 = 393216`.
Косинус считается по нему целиком — это один глобальный косинус по всей раскладке токенов, а не усреднение попозиционных косинусов (в этом отличие от LPIPS).

---

## 4. Отличия от `metrics_code_v2/`

Часть скриптов взята из `metrics_code_v2/` и изменена. Ниже — точная сверка.

### 4.1. Идентичные файлы (изменений нет)

- `scripts/metrics.py`
- `scripts/eval_metrics.py`
- `scripts/dataset_class.py`

### 4.2. Изменённые файлы

**`run_cv_v1.py`** — только добавлена защита от перезаписи:
- пропуск, если выходной `.pkl` уже существует (skip-if-exists);
- `os.makedirs` для выходной директории.

**`scripts/preprocessing.py`**:
- включена тканевая маска `binary_mask` (в оригинале закомментирована);
- порог отката маски: `mask_np.mean() < 0.05` → маска отбрасывается (в оригинале был
  warning при `< 0.1`);
- `MetricInput` расширен полями `raw_src_np` / `raw_trg_np` (сырые патчи до
  предобработки, нужны для метрики `nuclei`, где Cellpose нормализует сам).

**`nested_cv_metrics_opt.py`** — основные изменения:
- в `COMMON_FLAGS` добавлен `binary_mask` (по умолчанию `[False]`), для классики
  введён `_MASK_FLAG = {"binary_mask": [False, True]}`;
- классические метрики теперь передают маску: `calc_*(inp.src_np, inp.trg_np, inp.mask_np, …)`
  вместо `None` — **главная причина расхождения результатов с оригиналом**;
- добавлены импорты и ленивые геттеры deep/structural/foundation метрик;
- добавлены новые identity (`lpips_cellpose`, `cellpose`, `transpath`, `ngf`,
  `tile_*`, `flow_residual`, `nuclei`, foundation `*` и `*_spec`) и соответствующие
  записи в `METRICS_MAP`;
- в `generate_configs` добавлена обработка ключа `flags` (сужение search space);
- добавлен layer-aware fast path: `_preproc_signature`, `_layer_scores_for_pairs`,
  ветка в `run_inner_cv_group`;
- расширено множество `_GPU_METRICS` (deep/structural/foundation метрики).

### 4.3. Не перенесено из `metrics_code_v2/`

- `scripts/threshold_optimization.py` (пороговая оптимизация, не используется);
- `nested_cv_metrics.py` (последовательная версия, заменена на `_opt`);
- ноутбуки (`test_nested_cv.ipynb`, `Optimal_threshold.ipynb`, `Test_RetCCL.ipynb`).

### 4.4. Рудимент, унаследованный из оригинала

`sys.path.append('.../metrics_code_v2')` в `nested_cv_metrics_opt.py:19` и
`run_cv_v1.py:5` — безвреден (локальный `scripts/` имеет приоритет), но это ссылка
на старый каталог; при объединении кодовой базы следует удалить.

---

## 5. Окружение

- **uv**, Python 3.10, PyTorch 2.2.2 (cu121); `no-build = true` (совместимость с
  glibc 2.17 / CentOS 7.9 — только manylinux2014-колёса).
- Два окружения:
  - `.venv` — основной (cellpose 2.x `cyto2`, patched timm-0.5.4 для локального
    `transpath`);
  - `.venv-foundation` — modern `timm>=1.0` для HF foundation-бэкбонов +
    `cellpose>=4`/`segment-anything` для `cpsam`.
- Foundation-модели требуют `huggingface-cli login` + принятия лицензии на странице
  каждой gated-модели.

---

## 6. Пути и переносимость

Многие пути захардкожены на машину автора (`/trinity/home/vladislav.kozlovskiy/...`) и
переопределяются через переменные окружения / аргументы:

| Параметр | Где задаётся | Как переопределить |
|----------|--------------|--------------------|
| `REPO` | `slurm/env.sh`, `*.sbatch`, `submit_*.sh` | `REPO=…` |
| `CSV` | `run_cv.sbatch`, `run_cv_cpu.sbatch` | `--csv` / `CSV=…` |
| `TRANSPATH_ROOT` / `TRANSPATH_WEIGHTS` | `deep_metrics.py`, `slurm/env.sh` | env-переменные |
| HF-токен | `foundation_metrics.py`, `slurm/env.sh` | `HF_TOKEN` / `~/.cache/huggingface/token` |

Для запуска на другой машине нужно: пересоздать окружение (`uv sync`,
`uv sync --extra foundation`), выставить `REPO`/`CSV` и убедиться в наличии данных
(`filtration_imgs/exp_full.csv` или локальной копии).

---

## 7. Итоговые результаты (справочно)

`01_oof_ranking.csv` (nested CV, Spearman vs class3 / AUC Bad vs Rest):

| metric | spearman_class3 | auc_bad_vs_rest |
|--------|-----------------|-----------------|
| hibou_l_spec | 0.656 | 0.876 |
| ncc | 0.648 | 0.862 |
| nuclei | 0.613 | 0.839 |
| phikon_spec | 0.593 | 0.835 |
| ssim | 0.588 | 0.832 |
| mi | 0.569 | 0.819 |
| lpips_cellpose | 0.562 | 0.822 |

Полные данные: `02_bootstrap.csv` (bootstrap mean/std/CI), `03_config_stability.csv`
(устойчивость выбранных конфигов по outer folds), `haps_deep_foundation_results.xlsx`.

> **Важно.** Эти результаты посчитаны на **старой версии датасета**
> (`exp_full_local.csv`, 522 примера — без удаления 10 примеров, в актуальной версии
> 512) **и с включённой тканевой маской `binary_mask`**. Поэтому классические метрики
> (ncc/ssim/mi) здесь не сопоставимы с результатами `metrics_code_v2/`, где маска
> отключена. Расхождения (`binary_mask` и версия датасета) — ожидаемы и объяснимы.
