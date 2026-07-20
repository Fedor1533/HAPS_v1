"""
Глубокие (feature-based) метрики схожести, портированные из репозитория
`vs-filtering` (https://github.com/VladKozlovskiy/vs-filtering) в протокол HAPS_v1.

Все функции принимают тензоры (1, 3, H, W), float [0, 1] на нужном device и
возвращают **similarity** (больше = лучше пара), как того требует METRICS_MAP.

Метрики:
  - lpips_cellpose : LPIPS-стиль поверх энкодера Cellpose cyto2 (ResUNet)
  - cellpose       : cosine / normalized-L2 по фичам энкодера Cellpose
  - transpath      : cosine / L2 по эмбеддингу CTransPath (foundation-модель)

Модели грузятся лениво (по первому обращению) и кэшируются в глобалах модуля,
поэтому корректно переинициализируются в каждом spawn-процессе ProcessPoolExecutor.
"""

import os
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Общие утилиты агрегации (single-pair, batch=1)
# ---------------------------------------------------------------------------


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.reshape(1, -1)
    b = b.reshape(1, -1)
    return F.cosine_similarity(a, b, dim=1).item()


def _normalized_dist(a: torch.Tensor, b: torch.Tensor) -> float:
    a = F.normalize(a.reshape(1, -1), dim=1)
    b = F.normalize(b.reshape(1, -1), dim=1)
    return torch.norm(a - b, dim=1).item()


def _to_minus_one_one(x01: torch.Tensor) -> torch.Tensor:
    """[0, 1] -> [-1, 1] (формат, который ждут портированные из vs-filtering сети)."""
    return x01 * 2.0 - 1.0


# ===========================================================================
# 1. Cellpose cyto2 ResUNet encoder
# ===========================================================================


class CellposeEncoder(nn.Module):
    """
    Обёртка над downsample-энкодером Cellpose CPnet (cyto2).
    Вход forward: (B, 3, H, W) float [0, 1]. Возвращает список из 4 feature-map.

    Cellpose обучался на percentile-normalized [0, 1] микроскопии и ждёт 2 канала,
    поэтому RGB усредняется в grayscale и дублируется в 2 канала.
    """

    def __init__(self, model_type: str = "cyto2", net_avg: bool = False, device="cuda"):
        super().__init__()
        from cellpose import models as cp_models

        self.device = torch.device(device) if isinstance(device, str) else device
        try:
            self.cp_model = cp_models.CellposeModel(
                gpu=(str(self.device).startswith("cuda")),
                model_type=model_type,
                net_avg=net_avg,
                device=self.device,
            )
        except TypeError:
            # cellpose >= 3.x убрал net_avg
            self.cp_model = cp_models.CellposeModel(
                gpu=(str(self.device).startswith("cuda")),
                model_type=model_type,
                device=self.device,
            )
        self.encoder = self.cp_model.net.downsample
        self.chns = [32, 64, 128, 256]
        self.L = len(self.chns)
        for p in self.parameters():
            p.requires_grad = False
        self.eval()

    @staticmethod
    def _to_2ch(x01: torch.Tensor) -> torch.Tensor:
        gray = x01.mean(dim=1, keepdim=True)
        return gray.expand(-1, 2, -1, -1)

    def forward(self, x01: torch.Tensor):
        xd = self.encoder(self._to_2ch(x01.clamp(0.0, 1.0)))
        return list(xd)


# ===========================================================================
# 2. LPIPS-Cellpose (LPIPS-стиль поверх энкодера Cellpose)
# ===========================================================================


class LPIPSCellpose(nn.Module):
    """
    LPIPS-стиль метрика с энкодером Cellpose cyto2.
    Формула как у LPIPS (lpips=False): L2-norm фич, diff^2, spatial-avg, sum по слоям.
    Без LPIPS ScalingLayer (Cellpose обучен не на ImageNet).
    Вход: (1, 3, H, W) float [0, 1].
    """

    def __init__(self, model_type: str = "cyto2", net_avg: bool = False, device="cuda"):
        super().__init__()
        import lpips as _lpips

        self._lpips = _lpips
        self.net = CellposeEncoder(model_type=model_type, net_avg=net_avg, device=device)
        self.L = self.net.L

    def forward(self, in0_01: torch.Tensor, in1_01: torch.Tensor) -> torch.Tensor:
        outs0 = self.net(in0_01)
        outs1 = self.net(in1_01)
        val = 0.0
        for kk in range(self.L):
            f0 = self._lpips.normalize_tensor(outs0[kk])
            f1 = self._lpips.normalize_tensor(outs1[kk])
            diff = (f0 - f1) ** 2
            val = val + diff.sum(dim=1, keepdim=True).mean([2, 3], keepdim=True)
        return val


# ===========================================================================
# 3. CTransPath (TransPath) foundation-эмбеддинг
# ===========================================================================


def _load_ctranspath(device, weights_path: Optional[str], repo_root: Optional[str]):
    """
    Грузит CTransPath. Требует:
      - склонированный репозиторий TransPath (модуль `ctran.py` + патченный timm 0.5.4),
      - веса `ctranspath.pth`.

    Пути настраиваются через аргументы либо переменные окружения:
      TRANSPATH_ROOT    — директория с ctran.py (по умолчанию ./TransPath)
      TRANSPATH_WEIGHTS — путь к ctranspath.pth (по умолчанию $TRANSPATH_ROOT/ctranspath.pth)

    См. scripts/setup_transpath.sh и AGENTS.md.
    """
    import sys
    from pathlib import Path

    root = Path(repo_root or os.environ.get("TRANSPATH_ROOT", "TransPath")).resolve()
    weights = Path(
        weights_path
        or os.environ.get("TRANSPATH_WEIGHTS", str(root / "ctranspath.pth"))
    ).resolve()

    if not root.exists():
        raise FileNotFoundError(
            f"TransPath repo not found: {root}. "
            "Клонируйте https://github.com/Xiyue-Wang/TransPath и/или задайте "
            "TRANSPATH_ROOT. См. scripts/setup_transpath.sh"
        )
    if not weights.exists():
        raise FileNotFoundError(
            f"CTransPath weights not found: {weights}. "
            "Скачайте ctranspath.pth (Google Drive из репо TransPath) и задайте "
            "TRANSPATH_WEIGHTS. См. scripts/setup_transpath.sh"
        )

    if str(root.parent) not in sys.path:
        sys.path.insert(0, str(root.parent))
    from TransPath.ctran import ctranspath  # noqa: PLC0415

    model = ctranspath()
    model.head = nn.Identity()
    td = torch.load(str(weights), map_location="cpu", weights_only=True)
    model.load_state_dict(td["model"], strict=True)
    return model.to(device).eval()


# ImageNet-нормализация для входа CTransPath.
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def _transpath_input(x01: torch.Tensor) -> torch.Tensor:
    """(1, 3, H, W) [0, 1] -> resize 224 + ImageNet-norm."""
    x = x01
    if x.shape[1] == 1:
        x = x.repeat(1, 3, 1, 1)
    if x.shape[-1] != 224 or x.shape[-2] != 224:
        x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
    mean = torch.tensor(_IMAGENET_MEAN, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(_IMAGENET_STD, device=x.device).view(1, 3, 1, 1)
    return (x - mean) / std


# ---------------------------------------------------------------------------
# Метрик-функции (возвращают similarity: больше = лучше)
# ---------------------------------------------------------------------------


def calc_lpips_cellpose(src_t, trg_t, model: "LPIPSCellpose") -> float:
    """LPIPS-Cellpose distance -> возвращаем similarity (= -distance)."""
    with torch.inference_mode():
        dist = model(src_t, trg_t).mean().item()
    return -dist


def calc_cellpose(
    src_t, trg_t, encoder: "CellposeEncoder", feature: str = "neck", agg: str = "cos"
) -> float:
    """
    Similarity по фичам энкодера Cellpose.
      feature: "neck" (последний слой) | "mean" (усреднение по всем 4 слоям)
      agg:     "cos" (cosine sim) | "dist" (normalized L2 -> возвращаем -dist)
    """
    with torch.inference_mode():
        f0 = encoder(src_t)
        f1 = encoder(trg_t)

    if feature == "neck":
        pairs = [(f0[-1], f1[-1])]
    elif feature == "mean":
        pairs = list(zip(f0, f1))
    else:
        raise ValueError(f"Unknown cellpose feature: {feature}")

    if agg == "cos":
        vals = [_cosine(a, b) for a, b in pairs]
        return float(sum(vals) / len(vals))
    elif agg == "dist":
        vals = [_normalized_dist(a, b) for a, b in pairs]
        return -float(sum(vals) / len(vals))
    raise ValueError(f"Unknown cellpose agg: {agg}")


def calc_transpath(src_t, trg_t, model, agg: str = "cos") -> float:
    """
    Similarity по эмбеддингу CTransPath.
      agg: "cos" (cosine sim) | "dist" (L2 -> возвращаем -dist)
    """
    with torch.inference_mode():
        e0 = model(_transpath_input(src_t))
        e1 = model(_transpath_input(trg_t))
    if agg == "cos":
        return _cosine(e0, e1)
    elif agg == "dist":
        return -_normalized_dist(e0, e1)
    raise ValueError(f"Unknown transpath agg: {agg}")
