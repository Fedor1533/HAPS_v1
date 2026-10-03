"""
Структурные метрики качества кросс-модальной регистрации (H&E ↔ IHC).

Мотивация: глобальные эмбеддинги (в том числе pathology foundation models) на этих
данных работают около случайного, потому что косинус между H&E и IHC определяется
окраской и типом ткани, а не геометрическим соответствием. Единственный сигнал,
честно общий для двух модальностей, — геометрия ядер (гематоксилин есть и в H&E,
и в докраске IHC). Метрики ниже устроены так, чтобы мерить именно соответствие
структуры, а не общее сходство внешнего вида.

Три семейства:

  1. `ngf`   — Normalized Gradient Fields: квадрат косинуса угла между направлениями
              градиентов. Инвариантен к окраске и к инверсии контраста, чувствителен
              к рассогласованию границ. Классика мультимодальной регистрации.

  2. `tile_*` — те же попарные метрики (NCC / MI / NGF), но по сетке тайлов с
              агрегацией порядковыми статистиками. Человек называет пару плохой,
              если расхождение есть *хоть где-то*, поэтому минимум и худший
              квартиль информативнее среднего по кадру.

  3. `flow_residual` — остаточное смещение. Поверх уже свормленной пары ищем
              локальные сдвиги phase correlation'ом по тайлам. Если пара выровнена
              хорошо, сдвиги near-zero. Отдельно измеряется scatter — разброс
              относительно медианного сдвига, отличающий глобальный офсет
              (терпимый) от развала варпа (не терпимый).

  4. `nuclei_*` — ядерная геометрия. Сегментация ядер Cellpose по каналу
              гематоксилина, затем сопоставление центроидов взаимным ближайшим
              соседом, RANSAC на сматченных парах, Chamfer, Dice масок.

Все функции возвращают **similarity** (больше = лучше пара), как того требует
METRICS_MAP в nested_cv_metrics_opt.
"""

from __future__ import annotations

import hashlib
from typing import Optional, Tuple

import cv2
import numpy as np

from scripts.metrics import calc_mi, calc_ncc

# ---------------------------------------------------------------------------
# Общие утилиты
# ---------------------------------------------------------------------------

# Значение, возвращаемое когда метрику посчитать нельзя (нет ткани, нет ядер).
# Ранговые метрики (Spearman/AUC) устойчивы к константе, важно лишь чтобы она
# была худшей и одинаковой для всех вырожденных случаев.
_WORST = 0.0


def _to_gray01(img: np.ndarray) -> np.ndarray:
    """(H, W) или (H, W, C) float [0, 1] -> (H, W) float32."""
    a = np.asarray(img)
    if a.ndim == 3:
        a = a.mean(axis=2) if a.shape[2] == 3 else a[..., 0]
    return np.ascontiguousarray(a, dtype=np.float32)


def _gradients(g: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    return gx, gy


def _grad_magnitude(g: np.ndarray) -> np.ndarray:
    gx, gy = _gradients(g)
    return cv2.magnitude(gx, gy)


def _finite(val: float) -> float:
    v = float(val)
    return v if np.isfinite(v) else _WORST


# ---------------------------------------------------------------------------
# 1. Normalized Gradient Fields
# ---------------------------------------------------------------------------


def ngf_map(a: np.ndarray, b: np.ndarray, mask_np: Optional[np.ndarray] = None,
            eta: float = 0.5) -> np.ndarray:
    """
    Попиксельная карта NGF: (n_a · n_b)^2, где n = ∇I / sqrt(|∇I|² + ε²).

    Возведение в квадрат делает метрику нечувствительной к знаку градиента —
    ровно то, что нужно между H&E и IHC, где контраст одной и той же структуры
    может быть инвертирован.

    eta задаёт шумовой порог ε как долю от среднего модуля градиента: чем он
    больше, тем сильнее подавляется вклад плоских (фоновых) областей.
    """
    ga = _to_gray01(a)
    gb = _to_gray01(b)
    ax, ay = _gradients(ga)
    bx, by = _gradients(gb)
    ma = cv2.magnitude(ax, ay)
    mb = cv2.magnitude(bx, by)

    sel = _mask_or_all(mask_np, ga.shape)
    eps_a = eta * float(ma[sel].mean()) + 1e-8
    eps_b = eta * float(mb[sel].mean()) + 1e-8

    na = np.sqrt(ma * ma + eps_a * eps_a)
    nb = np.sqrt(mb * mb + eps_b * eps_b)
    dot = (ax * bx + ay * by) / (na * nb)
    return dot * dot


def _mask_or_all(mask_np: Optional[np.ndarray], shape: Tuple[int, int]) -> np.ndarray:
    if mask_np is None:
        return np.ones(shape, dtype=bool)
    sel = np.asarray(mask_np).astype(bool)
    if sel.shape != shape or sel.sum() < 16:
        return np.ones(shape, dtype=bool)
    return sel


def calc_ngf(src_np: np.ndarray, trg_np: np.ndarray,
             mask_np: Optional[np.ndarray] = None, eta: float = 0.5) -> float:
    """NGF-схожесть в [0, 1]: 1 = направления градиентов совпадают всюду."""
    m = ngf_map(src_np, trg_np, mask_np, eta=eta)
    sel = _mask_or_all(mask_np, m.shape)
    return _finite(m[sel].mean())


# ---------------------------------------------------------------------------
# 2. Тайловые метрики с агрегацией порядковыми статистиками
# ---------------------------------------------------------------------------


def _tile_origins(size: int, tile: int, stride: int):
    if size <= tile:
        return [0]
    return list(range(0, size - tile + 1, stride))


def _informative_tiles(a: np.ndarray, b: np.ndarray, tile: int, stride: int,
                       mask_np: Optional[np.ndarray], min_cover: float = 0.25,
                       min_std_frac: float = 0.1):
    """
    Список (y, x) тайлов, в которых есть что сравнивать.

    Плоский фон отбрасывается по локальному std (порог относительно глобального),
    что работает при любой полярности изображения — после flip_intensity/CLAHE
    «белый фон» уже не обязательно белый. Если передана маска ткани,
    дополнительно требуется её покрытие.
    """
    h, w = a.shape
    ref_std = max(float(a.std()), float(b.std()), 1e-6)
    thr = min_std_frac * ref_std

    out = []
    for y in _tile_origins(h, tile, stride):
        for x in _tile_origins(w, tile, stride):
            ta = a[y:y + tile, x:x + tile]
            tb = b[y:y + tile, x:x + tile]
            if max(float(ta.std()), float(tb.std())) < thr:
                continue
            if mask_np is not None:
                cover = float(mask_np[y:y + tile, x:x + tile].mean())
                if cover < min_cover:
                    continue
            out.append((y, x))
    return out


def tile_values(src_np: np.ndarray, trg_np: np.ndarray,
                mask_np: Optional[np.ndarray] = None, base: str = "ncc",
                tile: int = 64, stride: Optional[int] = None,
                eta: float = 0.5) -> np.ndarray:
    """Вектор попарных similarity по тайлам (только по информативным тайлам)."""
    a = _to_gray01(src_np)
    b = _to_gray01(trg_np)
    stride = tile if stride is None else stride

    origins = _informative_tiles(a, b, tile, stride, mask_np)
    if not origins:
        return np.empty(0, dtype=float)

    if base == "ngf":
        # NGF векторизуется на весь кадр сразу — считаем карту один раз
        # и усредняем по тайлам, а не пересчитываем на каждом окне.
        m = ngf_map(a, b, mask_np, eta=eta)
        vals = [float(m[y:y + tile, x:x + tile].mean()) for y, x in origins]
    elif base == "ncc":
        vals = [_finite(calc_ncc(a[y:y + tile, x:x + tile], b[y:y + tile, x:x + tile]))
                for y, x in origins]
    elif base == "mi":
        # 16 бинов: на 64x64 = 4096 пикселей 32 бина уже слишком разрежены
        vals = [_finite(calc_mi(a[y:y + tile, x:x + tile], b[y:y + tile, x:x + tile], bins=16))
                for y, x in origins]
    else:
        raise ValueError(f"Unknown tile base metric: {base}")

    arr = np.asarray(vals, dtype=float)
    return arr[np.isfinite(arr)]


def _aggregate(vals: np.ndarray, agg: str) -> float:
    if vals.size == 0:
        return _WORST
    if agg == "mean":
        return float(vals.mean())
    if agg == "min":
        return float(vals.min())
    if agg == "p10":
        return float(np.percentile(vals, 10))
    if agg == "q25":
        # среднее по худшему квартилю — устойчивее чистого минимума
        k = max(1, int(round(0.25 * vals.size)))
        return float(np.sort(vals)[:k].mean())
    if agg == "nstd":
        # пространственная неоднородность: чем сильнее разброс по кадру, тем хуже
        return -float(vals.std())
    raise ValueError(f"Unknown aggregation: {agg}")


def calc_tile_stat(src_np: np.ndarray, trg_np: np.ndarray,
                   mask_np: Optional[np.ndarray] = None, base: str = "ncc",
                   tile: int = 64, agg: str = "q25", eta: float = 0.5) -> float:
    vals = tile_values(src_np, trg_np, mask_np, base=base, tile=tile, eta=eta)
    return _finite(_aggregate(vals, agg))


# ---------------------------------------------------------------------------
# 3. Остаточное смещение (phase correlation по тайлам)
# ---------------------------------------------------------------------------

_HANN_CACHE = {}


def _hann(tile: int) -> np.ndarray:
    win = _HANN_CACHE.get(tile)
    if win is None:
        win = cv2.createHanningWindow((tile, tile), cv2.CV_32F)
        _HANN_CACHE[tile] = win
    return win


def tile_shifts(src_np: np.ndarray, trg_np: np.ndarray,
                mask_np: Optional[np.ndarray] = None, tile: int = 64,
                overlap: float = 0.5, on: str = "grad") -> Tuple[np.ndarray, np.ndarray]:
    """
    Локальные сдвиги между парой, найденные phase correlation'ом по тайлам.

    on="grad": корреляция считается по модулю градиента, а не по интенсивности.
    Phase correlation на сырых интенсивностях ломается при инверсии контраста
    (пик уходит в отрицательную часть), а модуль градиента к полярности
    нечувствителен — для H&E↔IHC это принципиально.

    Возвращает (shifts (N, 2) в пикселях, responses (N,)).
    """
    a = _to_gray01(src_np)
    b = _to_gray01(trg_np)
    if on == "grad":
        a = _grad_magnitude(a)
        b = _grad_magnitude(b)

    stride = max(1, int(round(tile * (1.0 - overlap))))
    origins = _informative_tiles(a, b, tile, stride, mask_np)
    if not origins:
        return np.empty((0, 2)), np.empty(0)

    win = _hann(tile)
    shifts, resps = [], []
    for y, x in origins:
        ta = a[y:y + tile, x:x + tile]
        tb = b[y:y + tile, x:x + tile]
        if ta.shape != (tile, tile):
            continue
        # снятие DC: постоянная составляющая даёт паразитный пик в нуле
        ta = np.ascontiguousarray(ta - ta.mean(), dtype=np.float32)
        tb = np.ascontiguousarray(tb - tb.mean(), dtype=np.float32)
        try:
            (dx, dy), resp = cv2.phaseCorrelate(ta, tb, win)
        except cv2.error:
            continue
        if not (np.isfinite(dx) and np.isfinite(dy)):
            continue
        shifts.append((dx, dy))
        resps.append(resp if np.isfinite(resp) else 0.0)

    return np.asarray(shifts, dtype=float).reshape(-1, 2), np.asarray(resps, dtype=float)


def calc_flow_residual(src_np: np.ndarray, trg_np: np.ndarray,
                       mask_np: Optional[np.ndarray] = None, tile: int = 64,
                       stat: str = "p90_mag", on: str = "grad",
                       min_resp: float = 0.0) -> float:
    """
    Качество регистрации через величину остаточного локального смещения.

    Все статистики по величине сдвига возвращаются со знаком минус (больше = лучше).
    `scatter` — разброс относительно медианного сдвига: отделяет ровный глобальный
    офсет от несогласованного (развалившегося) варпа.
    `mean_resp` — средняя высота пика корреляции, то есть уверенность локального
    соответствия; чем чётче совпадает структура, тем она выше.

    min_resp отбрасывает тайлы, где пик корреляции слишком низкий: между H&E и IHC
    у части тайлов честного соответствия просто нет, и их сдвиг — шум.
    """
    shifts, resps = tile_shifts(src_np, trg_np, mask_np, tile=tile, on=on)

    if min_resp > 0.0 and shifts.shape[0] > 0:
        keep = resps >= min_resp
        if keep.sum() >= 2:
            shifts, resps = shifts[keep], resps[keep]

    if shifts.shape[0] == 0:
        return _WORST if stat == "mean_resp" else -float(tile)

    # Сдвиг больше половины тайла окном не разрешается: субпиксельная подгонка
    # пика в таких случаях выдаёт мусор вплоть до тысяч пикселей. Обрезаем по
    # модулю, сохраняя направление — «не хуже чем на полтайла» это честная оценка.
    limit = tile / 2.0
    mag_raw = np.linalg.norm(shifts, axis=1)
    scale = np.minimum(1.0, limit / np.maximum(mag_raw, 1e-8))
    shifts = shifts * scale[:, None]

    mag = np.linalg.norm(shifts, axis=1)
    if stat == "mean_mag":
        return -_finite(mag.mean())
    if stat == "p90_mag":
        return -_finite(np.percentile(mag, 90))
    if stat == "frac_bad":
        # Доля тайлов с заметным остаточным сдвигом. Чистый максимум здесь не
        # работает: он упирается в потолок клипа почти у каждой пары, тогда как
        # доля «плохих» тайлов различает пары в полном диапазоне.
        return -_finite((mag > tile / 8.0).mean())
    if stat == "scatter":
        dev = np.linalg.norm(shifts - np.median(shifts, axis=0), axis=1)
        return -_finite(dev.mean())
    if stat == "mean_resp":
        return _finite(resps.mean())
    raise ValueError(f"Unknown flow stat: {stat}")


# ---------------------------------------------------------------------------
# 4. Ядерная геометрия
# ---------------------------------------------------------------------------

_SEG_MODELS = {}
# Сегментация не зависит от препроцессинга CV (Cellpose нормализует сам), поэтому
# кэшируется по содержимому патча и переиспользуется всеми конфигами и фолдами.
_SEG_CACHE = {}
_SEG_CACHE_LIMIT = 16384


def _get_seg_model(model_type: str = "cyto2", device="cuda"):
    model = _SEG_MODELS.get(model_type)
    if model is None:
        from cellpose import models as cp_models

        gpu = str(device).startswith("cuda")
        try:
            model = cp_models.CellposeModel(gpu=gpu, model_type=model_type, net_avg=False)
        except TypeError:
            model = cp_models.CellposeModel(gpu=gpu, model_type=model_type)
        _SEG_MODELS[model_type] = model
    return model


def _as_float_rgb(rgb: np.ndarray) -> np.ndarray:
    a = np.asarray(rgb)
    if a.dtype == np.uint8 or a.max() > 1.0:
        a = a.astype(np.float32) / 255.0
    else:
        a = a.astype(np.float32)
    if a.ndim == 2:
        a = np.repeat(a[:, :, None], 3, axis=2)
    return a[:, :, :3]


def hematoxylin_channel(rgb: np.ndarray) -> np.ndarray:
    """
    RGB -> канал гематоксилина в [0, 1], ядра = ярко.

    В отличие от `preprocessing.get_hematoxylin_ch` здесь НЕТ инверсии: Cellpose
    ожидает объекты светлее фона. Растяжка по 1/99 перцентилям выравнивает
    разброс интенсивности окраски между слайдами.
    """
    from skimage.color import rgb2hed

    h = rgb2hed(_as_float_rgb(rgb))[:, :, 0]
    lo, hi = np.percentile(h, (1, 99))
    return np.clip((h - lo) / (hi - lo + 1e-8), 0.0, 1.0).astype(np.float32)


def _inverted_gray(rgb: np.ndarray) -> np.ndarray:
    """RGB -> инвертированный grayscale: ткань ярко, белый фон темно."""
    g = _as_float_rgb(rgb).mean(axis=2)
    lo, hi = np.percentile(g, (1, 99))
    g = np.clip((g - lo) / (hi - lo + 1e-8), 0.0, 1.0)
    return (1.0 - g).astype(np.float32)


def nuclei_input_channel(rgb: np.ndarray, channel: str = "gray") -> np.ndarray:
    """
    Одноканальный вход для Cellpose.

    channel="hem" опирается на деконволюцию окрасок, но на IHC с насыщенным DAB
    она подмешивает в H-канал мембранное окрашивание вместо ядер, поэтому
    "gray" (инвертированная яркость) обычно надёжнее для кросс-модальной пары.
    """
    if channel == "hem":
        return hematoxylin_channel(rgb)
    if channel == "gray":
        return _inverted_gray(rgb)
    raise ValueError(f"Unknown nuclei input channel: {channel}")


def _seg_key(img: np.ndarray, diameter: float, model_type: str, channel: str) -> bytes:
    digest = hashlib.blake2b(np.ascontiguousarray(img).tobytes(), digest_size=16).digest()
    return digest + f"|{model_type}|{channel}|{float(diameter)}".encode()


def segment_nuclei(rgb: np.ndarray, diameter: float = 12.0, model_type: str = "cyto2",
                   channel: str = "gray", device="cuda") -> dict:
    """
    Сегментация ядер. Возвращает dict:
      centroids : (N, 2) центры масс инстансов в координатах (y, x)
      binary    : (H, W) bool — объединение всех инстансов
      cellprob  : (H, W) float — непрерывная карта «здесь ядро»

    cellprob полезнее инстансов, когда детекция в двух модальностях
    несимметрична: в IHC гематоксилиновая докраска слабее и часть ядер просто
    не набирает порог, из-за чего облака центроидов становятся несравнимыми,
    а непрерывная карта сохраняет структуру.

    Результат кэшируется по содержимому патча: сегментация не зависит от
    препроцессинга CV (Cellpose нормализует вход сам), поэтому одна и та же
    картинка сегментируется один раз на весь прогон.
    """
    key = _seg_key(rgb, diameter, model_type, channel)
    hit = _SEG_CACHE.get(key)
    if hit is not None:
        return hit

    from scipy import ndimage

    ch = nuclei_input_channel(rgb, channel)
    model = _get_seg_model(model_type, device)
    masks, flows, *_ = model.eval(ch, channels=[0, 0], diameter=diameter, normalize=True)
    labels = np.asarray(masks)

    n = int(labels.max())
    if n > 0:
        cm = ndimage.center_of_mass(labels > 0, labels, range(1, n + 1))
        centroids = np.asarray(cm, dtype=float).reshape(-1, 2)
    else:
        centroids = np.empty((0, 2), dtype=float)

    # flows = [dP_circ, dP, cellprob, p]; берём cellprob, если формат ожидаемый
    cellprob = None
    try:
        cand = np.asarray(flows[2], dtype=np.float32)
        if cand.shape == labels.shape:
            cellprob = cand
    except (IndexError, TypeError, ValueError):
        cellprob = None
    if cellprob is None:
        cellprob = (labels > 0).astype(np.float32)

    out = {"centroids": centroids, "binary": labels > 0, "cellprob": cellprob}
    if len(_SEG_CACHE) >= _SEG_CACHE_LIMIT:
        _SEG_CACHE.clear()
    _SEG_CACHE[key] = out
    return out


def _mutual_nn(ca: np.ndarray, cb: np.ndarray, radius: float):
    """
    Взаимные ближайшие соседи в пределах radius.
    Возвращает (idx_a, idx_b, dists, d_ab, d_ba) — последние два нужны для Chamfer.
    """
    from scipy.spatial import cKDTree

    ta, tb = cKDTree(ca), cKDTree(cb)
    d_ab, i_ab = tb.query(ca, k=1)
    d_ba, i_ba = ta.query(cb, k=1)

    ia, ib, dd = [], [], []
    for i, j in enumerate(i_ab):
        if i_ba[j] == i and d_ab[i] <= radius:
            ia.append(i)
            ib.append(int(j))
            dd.append(float(d_ab[i]))
    return (np.asarray(ia, dtype=int), np.asarray(ib, dtype=int),
            np.asarray(dd, dtype=float), d_ab, d_ba)


def calc_nuclei_geometry(raw_src: np.ndarray, raw_trg: np.ndarray,
                         stat: str = "ransac_inliers", radius: float = 8.0,
                         diameter: float = 12.0, model_type: str = "cyto2",
                         channel: str = "gray", device="cuda") -> float:
    """
    Качество регистрации по соответствию облаков центроидов ядер.

    Формализует то, что глазами делает аннотатор: «те же ядра лежат в тех же
    местах?». Инвариантно к окраске по построению, так как ядра видны в обеих
    модальностях через гематоксилин.

    stat:
      dice           — Dice бинарных масок ядер (после дилатации)
      match_rate     — доля ядер, у которых есть взаимный ближайший сосед в radius
      median_resid   — медианное остаточное расстояние сматченных ядер (со знаком минус)
      chamfer        — двусторонняя Chamfer-дистанция между облаками (со знаком минус)
      ransac_inliers — доля инлайеров жёсткого преобразования, найденного RANSAC'ом
      prob_ncc       — NCC между непрерывными картами вероятности ядра
      prob_ngf       — NGF между картами вероятности (согласованность границ ядер)
      prob_tile      — тайловый NCC по картам вероятности, худший квартиль
    """
    seg_kw = dict(diameter=diameter, model_type=model_type, channel=channel, device=device)
    sa = segment_nuclei(raw_src, **seg_kw)
    sb = segment_nuclei(raw_trg, **seg_kw)
    ca, ba = sa["centroids"], sa["binary"]
    cb, bb = sb["centroids"], sb["binary"]

    # Метрики по непрерывным картам не требуют сопоставления инстансов и потому
    # устойчивы к разной чувствительности детекции в H&E и IHC.
    if stat.startswith("prob_"):
        pa, pb = sa["cellprob"], sb["cellprob"]
        if stat == "prob_ncc":
            return _finite(calc_ncc(pa, pb))
        if stat == "prob_ngf":
            return calc_ngf(pa, pb, eta=0.5)
        if stat == "prob_tile":
            return calc_tile_stat(pa, pb, base="ncc", tile=64, agg="q25")
        raise ValueError(f"Unknown nuclei stat: {stat}")

    if stat == "dice":
        k = np.ones((3, 3), np.uint8)
        da = cv2.dilate(ba.astype(np.uint8), k)
        db = cv2.dilate(bb.astype(np.uint8), k)
        denom = float(da.sum() + db.sum())
        if denom == 0:
            return _WORST
        return _finite(2.0 * float((da & db).sum()) / denom)

    n_a, n_b = len(ca), len(cb)
    if n_a < 3 or n_b < 3:
        # ядер практически нет — сравнивать нечего, худшее значение
        return _WORST if stat in ("match_rate", "ransac_inliers") else -float(radius * 4)

    ia, ib, dd, d_ab, d_ba = _mutual_nn(ca, cb, radius)

    if stat == "chamfer":
        return -_finite(0.5 * (float(np.mean(d_ab)) + float(np.mean(d_ba))))

    if stat == "match_rate":
        return _finite(2.0 * len(ia) / (n_a + n_b))

    if stat == "median_resid":
        if len(dd) == 0:
            return -float(radius)
        # немного сматченных ядер при большом их числе — тоже плохой признак,
        # поэтому остаток штрафуется долей несматченных
        unmatched = 1.0 - 2.0 * len(dd) / (n_a + n_b)
        return -_finite(float(np.median(dd)) + unmatched * radius)

    if stat == "ransac_inliers":
        if len(ia) < 4:
            return _WORST
        from skimage.measure import ransac
        from skimage.transform import EuclideanTransform

        # (x, y) для skimage
        src_pts = ca[ia][:, ::-1]
        dst_pts = cb[ib][:, ::-1]
        kw = dict(min_samples=3, residual_threshold=max(2.0, radius / 2), max_trials=100)
        try:
            # skimage >= 0.22 переименовал random_state в rng
            try:
                _, inliers = ransac((src_pts, dst_pts), EuclideanTransform, rng=0, **kw)
            except TypeError:
                _, inliers = ransac((src_pts, dst_pts), EuclideanTransform, random_state=0, **kw)
        except Exception:
            return _WORST
        if inliers is None:
            return _WORST
        # знаменатель — все ядра, а не только сматченные: так штрафуются и
        # пропущенные соответствия, и геометрически несогласованные
        return _finite(float(inliers.sum()) / max(n_a, n_b))

    raise ValueError(f"Unknown nuclei stat: {stat}")
