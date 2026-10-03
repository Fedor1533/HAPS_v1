"""
Pathology foundation backbones as image-pair similarity metrics.

Каждая метрика: эмбеддинг пары → cosine / -L2 (similarity, больше = лучше).

Модели (HF):
  uni2h      MahmoodLab/UNI2-h           (gated)
  gigapath   prov-gigapath/prov-gigapath (gated)
  virchow2   paige-ai/Virchow2           (gated)
  hoptimus0  bioptimus/H-optimus-0       (gated)
  hibou_l    histai/hibou-L              (gated)
  phikon     owkin/phikon
  ctranspath 1aurent/swin_tiny_patch4_window7_224.CTransPath
             (HF-зеркало; локальный TransPath остаётся в deep_metrics.transpath)
  cpsam      mouseland/cellpose-sam (cpsam_v2) — Cellpose-SAM ViT-L
             Нужен cellpose>=4 в .venv-foundation (см. setup_foundation.sh).
             Локальный cyto2 ResUNet остаётся в deep_metrics.cellpose.

Требуется: `huggingface-cli login` + accept license на странице модели.
Большинство timm-моделей требуют `timm>=1.0` (SwiGLUPacked / reg tokens).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Ожидаемые глубины (для nested-CV search space). Реальный extract_* может
# вернуть меньше/больше — noval всегда берёт фактически извлечённые ключи.
FOUNDATION_FEATURE_DEPTHS: Dict[str, int] = {
    "uni2h": 24,          # loader явно depth=24
    "gigapath": 40,       # vit_giant_patch14_dinov2
    "virchow2": 32,       # vit_huge_patch14_224
    "hoptimus0": 40,      # vit_giant_patch14_reg4_dinov2
    "hibou_l": 24,
    "phikon": 12,
    "ctranspath_hf": 12,  # Swin-T depths (2,2,6,2)
    "cpsam": 32,          # SAM ViT-L upper bound; лишние обрежутся
}

# ---------------------------------------------------------------------------
# Shared utils
# ---------------------------------------------------------------------------

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)
# H-optimus-0 uses a histology-specific normalisation (from model card).
_HOPTIMUS_MEAN = (0.707223, 0.578729, 0.703617)
_HOPTIMUS_STD = (0.211883, 0.230117, 0.177517)
# hibou-L ships its own BitImageProcessor stats (preprocessor_config.json).
_HIBOU_MEAN = (0.7068, 0.5755, 0.722)
_HIBOU_STD = (0.195, 0.2316, 0.1816)


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.reshape(1, -1)
    b = b.reshape(1, -1)
    return F.cosine_similarity(a, b, dim=1).item()


def _normalized_dist(a: torch.Tensor, b: torch.Tensor) -> float:
    a = F.normalize(a.reshape(1, -1), dim=1)
    b = F.normalize(b.reshape(1, -1), dim=1)
    return torch.norm(a - b, dim=1).item()


def _ensure_rgb01(x: torch.Tensor) -> torch.Tensor:
    if x.shape[1] == 1:
        x = x.repeat(1, 3, 1, 1)
    return x.clamp(0.0, 1.0)


def _center_crop(x: torch.Tensor, size: int) -> torch.Tensor:
    h, w = x.shape[-2], x.shape[-1]
    top = max((h - size) // 2, 0)
    left = max((w - size) // 2, 0)
    return x[..., top : top + size, left : left + size]


def _resize_norm(
    x01: torch.Tensor,
    size: int,
    mean: Tuple[float, float, float],
    std: Tuple[float, float, float],
    crop_pct: float = 1.0,
    interpolation: str = "bilinear",
) -> torch.Tensor:
    """
    Повторяет timm-семантику inference-трансформа:
    resize короткой стороны до round(size / crop_pct) + center-crop до size.

    crop_pct=1.0 сводится к обычному resize в size.
    """
    x = _ensure_rgb01(x01)

    resize_to = int(round(size / crop_pct)) if crop_pct and crop_pct > 0 else size
    if x.shape[-1] != resize_to or x.shape[-2] != resize_to:
        kwargs = {"antialias": True} if interpolation in ("bilinear", "bicubic") else {}
        x = F.interpolate(x, size=(resize_to, resize_to), mode=interpolation,
                          align_corners=False, **kwargs)
    if resize_to != size:
        x = _center_crop(x, size)

    m = torch.tensor(mean, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    s = torch.tensor(std, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    return (x - m) / s


# ---------------------------------------------------------------------------
# Backbone registry
# ---------------------------------------------------------------------------


@dataclass
class BackboneSpec:
    name: str
    hf_id: str
    img_size: int
    mean: Tuple[float, float, float]
    std: Tuple[float, float, float]
    # loader(device) -> nn.Module in eval mode
    load: Callable[[str], nn.Module]
    # embed(model, x_preprocessed) -> (B, D)
    embed: Callable[[nn.Module, torch.Tensor], torch.Tensor]
    gated: bool = True
    # timm-семантика inference-трансформа: resize до size/crop_pct + center-crop до size.
    crop_pct: float = 1.0
    interpolation: str = "bilinear"
    # Переопределяет _resize_norm, если модель ждёт нестандартный препроцессинг
    # (например percentile-нормализацию Cellpose вместо mean/std).
    preprocess: Optional[Callable[["BackboneSpec", torch.Tensor], torch.Tensor]] = None


def _resolve_data_config(spec: BackboneSpec, model: nn.Module) -> BackboneSpec:
    """
    Берёт mean/std/input_size из самой модели (timm pretrained_cfg или HF
    image processor) вместо захардкоженных значений в реестре.

    Захардкоженные значения остаются фолбэком, если модель ничего не объявляет.
    """
    if spec.preprocess is not None:
        return spec

    mean = std = None
    size = None
    interp = None

    if hasattr(model, "pretrained_cfg"):
        try:
            from timm.data import resolve_model_data_config

            cfg = resolve_model_data_config(model)
            mean, std = cfg.get("mean"), cfg.get("std")
            size = cfg.get("input_size", (3, spec.img_size, spec.img_size))[-1]
            interp = cfg.get("interpolation")
        except Exception:  # noqa: BLE001
            mean = std = size = interp = None

    if mean is None:
        try:
            from transformers import AutoImageProcessor

            proc = AutoImageProcessor.from_pretrained(spec.hf_id, trust_remote_code=True)
            mean, std = tuple(proc.image_mean), tuple(proc.image_std)
            crop = getattr(proc, "crop_size", None) or getattr(proc, "size", None)
            if isinstance(crop, dict):
                size = crop.get("height") or crop.get("shortest_edge")
            elif isinstance(crop, int):
                size = crop
        except Exception:  # noqa: BLE001
            mean = std = size = None

    if mean is None or std is None:
        return spec

    # crop_pct/interpolation в реестре заданы по README авторов и приоритетнее
    # timm-конфига там, где они расходятся (например gigapath).
    resolved = replace(
        spec,
        mean=tuple(float(v) for v in mean),
        std=tuple(float(v) for v in std),
        img_size=int(size or spec.img_size),
        interpolation=spec.interpolation or interp or "bilinear",
    )
    print(
        f"[foundation:{spec.name}] input spec: size={resolved.img_size} "
        f"crop_pct={resolved.crop_pct} interp={resolved.interpolation} "
        f"mean={tuple(round(v, 4) for v in resolved.mean)} "
        f"std={tuple(round(v, 4) for v in resolved.std)}",
        flush=True,
    )
    return resolved


def _normalize_percentile(x01: torch.Tensor, lower: float = 1.0, upper: float = 99.0) -> torch.Tensor:
    """Per-channel percentile normalisation — то, что Cellpose делает со своим входом."""
    out = torch.empty_like(x01)
    for c in range(x01.shape[1]):
        ch = x01[:, c]
        lo = torch.quantile(ch.reshape(ch.shape[0], -1), lower / 100.0, dim=1)
        hi = torch.quantile(ch.reshape(ch.shape[0], -1), upper / 100.0, dim=1)
        lo = lo.view(-1, 1, 1)
        hi = hi.view(-1, 1, 1)
        out[:, c] = (ch - lo) / (hi - lo + 1e-8)
    return out


def _preprocess_cpsam(spec: BackboneSpec, x01: torch.Tensor) -> torch.Tensor:
    """Cellpose-SAM: resize до bsize + percentile-нормализация (как в cellpose.transforms)."""
    x = _ensure_rgb01(x01)
    if x.shape[-1] != spec.img_size or x.shape[-2] != spec.img_size:
        x = F.interpolate(x, size=(spec.img_size, spec.img_size), mode="bilinear", align_corners=False)
    return _normalize_percentile(x)


def prepare_input(spec: BackboneSpec, x01: torch.Tensor) -> torch.Tensor:
    """
    Единая точка препроцессинга: на вход приходит тензор ПОСЛЕ CV-преобразований
    (normalization / channel_mode / flip / hist-match / CLAHE / smoothing),
    (1, C, H, W) float в [0, 1]. Дальше — только то, что требует конкретный бэкбон.
    """
    if spec.preprocess is not None:
        return spec.preprocess(spec, x01)
    return _resize_norm(
        x01,
        spec.img_size,
        spec.mean,
        spec.std,
        crop_pct=spec.crop_pct,
        interpolation=spec.interpolation,
    )


def _embed_timm_cls(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Standard timm forward → CLS / pooled embedding (B, D)."""
    out = model(x)
    if out.ndim == 3:
        # tokens: take CLS
        return out[:, 0]
    return out


def _embed_virchow2(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """CLS || mean(patch tokens), skip 4 register tokens. → (B, 2560)."""
    # Важно: обычный forward() при num_classes=0 может сразу пулить CLS.
    if hasattr(model, "forward_features"):
        out = model.forward_features(x)
    else:
        out = model(x)
    if out.ndim != 3:
        return out
    cls = out[:, 0]
    patches = out[:, 5:]
    return torch.cat([cls, patches.mean(dim=1)], dim=-1)


def _embed_transformers_cls(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    out = model(pixel_values=x)
    hs = out.last_hidden_state
    return hs[:, 0]


def _load_timm_hf(hf_id: str, device: str, **kwargs) -> nn.Module:
    import timm

    model = timm.create_model(f"hf-hub:{hf_id}", pretrained=True, **kwargs)
    return model.to(device).eval()


def _load_uni2h(device: str) -> nn.Module:
    import timm
    from timm.layers import SwiGLUPacked

    kwargs = dict(
        img_size=224,
        patch_size=14,
        depth=24,
        num_heads=24,
        init_values=1e-5,
        embed_dim=1536,
        mlp_ratio=2.66667 * 2,
        num_classes=0,
        no_embed_class=True,
        mlp_layer=SwiGLUPacked,
        act_layer=torch.nn.SiLU,
        reg_tokens=8,
        dynamic_img_size=True,
    )
    return _load_timm_hf("MahmoodLab/UNI2-h", device, **kwargs)


def _load_gigapath(device: str) -> nn.Module:
    # tile encoder; num_classes=0 → embedding
    return _load_timm_hf("prov-gigapath/prov-gigapath", device, num_classes=0)


def _load_virchow2(device: str) -> nn.Module:
    from timm.layers import SwiGLUPacked

    return _load_timm_hf(
        "paige-ai/Virchow2",
        device,
        mlp_layer=SwiGLUPacked,
        act_layer=torch.nn.SiLU,
        num_classes=0,
    )


def _load_hoptimus0(device: str) -> nn.Module:
    return _load_timm_hf(
        "bioptimus/H-optimus-0",
        device,
        init_values=1e-5,
        dynamic_img_size=False,
    )


class _CTransPathConvStem(nn.Module):
    """
    Свёрточный patch-embed из репозитория CTransPath, совместимый с timm>=1.0 Swin.

    Чекпойнт 1aurent/...CTransPath везёт ключи patch_embed.proj.{0,1,3,4,6},
    т.е. Conv-BN-ReLU x2 + Conv1x1, а не одиночный Conv обычного timm PatchEmbed.
    Без этого stem загрузка state_dict падает по missing/unexpected keys.
    """

    def __init__(
        self,
        img_size=224,
        patch_size=4,
        in_chans=3,
        embed_dim=96,
        norm_layer=None,
        output_fmt="NHWC",
        strict_img_size=True,
        **kwargs,
    ):
        super().__init__()
        img_size = (img_size, img_size) if isinstance(img_size, int) else tuple(img_size)
        patch_size = (patch_size, patch_size) if isinstance(patch_size, int) else tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.output_fmt = output_fmt
        self.strict_img_size = strict_img_size

        stem = []
        in_dim, out_dim = in_chans, embed_dim // 8
        for _ in range(2):
            stem += [
                nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(out_dim),
                nn.ReLU(inplace=True),
            ]
            in_dim, out_dim = out_dim, out_dim * 2
        stem.append(nn.Conv2d(in_dim, embed_dim, kernel_size=1))
        self.proj = nn.Sequential(*stem)
        self.norm = norm_layer(embed_dim) if norm_layer is not None else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        # timm Swin ждёт NHWC, а norm_layer (LayerNorm) работает по последней оси.
        x = x.permute(0, 2, 3, 1)
        return self.norm(x)


def _load_ctranspath_hf(device: str) -> nn.Module:
    return _load_timm_hf(
        "1aurent/swin_tiny_patch4_window7_224.CTransPath",
        device,
        num_classes=0,
        embed_layer=_CTransPathConvStem,
    )


def _load_cpsam(device: str) -> nn.Module:
    """
    Cellpose-SAM (cpsam_v2): SAM ViT-L backbone, fine-tuned for cell segmentation.
    Требует cellpose>=4 (+ segment-anything). Веса: ~/.cellpose/models/cpsam_v2
    или HF mouseland/cellpose-sam.
    """
    from importlib.metadata import version

    ver = version("cellpose")
    if int(ver.split(".")[0]) < 4:
        raise RuntimeError(
            f"cpsam needs cellpose>=4 (found {ver}). "
            "Use .venv-foundation and run: bash scripts/setup_foundation.sh"
        )

    from cellpose import models as cp_models

    # Prefetch HF weights into CELLPOSE model dir if missing.
    model_dir = Path(
        os.environ.get("CELLPOSE_LOCAL_MODELS_PATH")
        or (Path.home() / ".cellpose" / "models")
    )
    model_dir.mkdir(parents=True, exist_ok=True)
    weights = model_dir / "cpsam_v2"
    if not weights.exists():
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(
            repo_id="mouseland/cellpose-sam",
            filename="cpsam_v2",
            local_dir=str(model_dir),
            local_dir_use_symlinks=False,
        )
        # hf may write into nested dirs; ensure expected filename.
        if Path(path).resolve() != weights.resolve() and not weights.exists():
            import shutil

            shutil.copy2(path, weights)

    dev = torch.device(device)
    cp = cp_models.CellposeModel(
        gpu=dev.type == "cuda",
        pretrained_model="cpsam_v2",
        device=dev,
        use_bfloat16=False,
    )
    net = cp.net.to(dev).eval()
    for p in net.parameters():
        p.requires_grad = False
    return net


def _embed_cpsam(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """
    Global-average-pool of SAM encoder neck features (before Cellpose readout).
    → (B, C), C≈256 for ViT-L neck.
    """
    # CPSAM.forward returns (flows, dummy); we want neck features instead.
    x = x.to(dtype=getattr(model, "dtype", x.dtype))
    ps = model.ps
    enc = model.encoder
    x = F.conv2d(
        x,
        enc.patch_embed.proj.weight.data[:, : x.shape[1]],
        bias=enc.patch_embed.proj.bias.data,
        stride=ps,
    )
    x = x.permute(0, 2, 3, 1)
    if enc.pos_embed is not None:
        x = x + enc.pos_embed
    for blk in enc.blocks:
        x = blk(x)
    feat = enc.neck(x.permute(0, 3, 1, 2))  # (B, C, H, W)
    return feat.mean(dim=(2, 3))


def _load_phikon(device: str) -> nn.Module:
    from transformers import ViTModel

    model = ViTModel.from_pretrained("owkin/phikon", add_pooling_layer=False)
    return model.to(device).eval()


def _load_hibou_l(device: str) -> nn.Module:
    from transformers import AutoModel

    model = AutoModel.from_pretrained("histai/hibou-L", trust_remote_code=True)
    return model.to(device).eval()


FOUNDATION_BACKBONES: Dict[str, BackboneSpec] = {
    "uni2h": BackboneSpec(
        name="uni2h",
        hf_id="MahmoodLab/UNI2-h",
        img_size=224,
        mean=_IMAGENET_MEAN,
        std=_IMAGENET_STD,
        load=_load_uni2h,
        embed=_embed_timm_cls,
        gated=True,
    ),
    # README авторов: Resize(256, bicubic) + CenterCrop(224) => crop_pct 224/256.
    "gigapath": BackboneSpec(
        name="gigapath",
        hf_id="prov-gigapath/prov-gigapath",
        img_size=224,
        mean=_IMAGENET_MEAN,
        std=_IMAGENET_STD,
        load=_load_gigapath,
        embed=_embed_timm_cls,
        gated=True,
        crop_pct=224 / 256,
        interpolation="bicubic",
    ),
    "virchow2": BackboneSpec(
        name="virchow2",
        hf_id="paige-ai/Virchow2",
        img_size=224,
        mean=_IMAGENET_MEAN,
        std=_IMAGENET_STD,
        load=_load_virchow2,
        embed=_embed_virchow2,
        gated=True,
        interpolation="bicubic",
    ),
    "hoptimus0": BackboneSpec(
        name="hoptimus0",
        hf_id="bioptimus/H-optimus-0",
        img_size=224,
        mean=_HOPTIMUS_MEAN,
        std=_HOPTIMUS_STD,
        load=_load_hoptimus0,
        embed=_embed_timm_cls,
        gated=True,
        # Модель ждёт ровно 224 @0.5 µm/px и не ресайзит вход => из 256 берём
        # центральный кроп, чтобы не менять эффективное увеличение.
        crop_pct=224 / 256,
        interpolation="bicubic",
    ),
    "hibou_l": BackboneSpec(
        name="hibou_l",
        hf_id="histai/hibou-L",
        img_size=224,
        mean=_HIBOU_MEAN,
        std=_HIBOU_STD,
        load=_load_hibou_l,
        embed=_embed_transformers_cls,
        gated=True,
        interpolation="bicubic",  # preprocessor_config: resample=3
    ),
    "phikon": BackboneSpec(
        name="phikon",
        hf_id="owkin/phikon",
        img_size=224,
        mean=_IMAGENET_MEAN,
        std=_IMAGENET_STD,
        load=_load_phikon,
        embed=_embed_transformers_cls,
        gated=False,
    ),
    "ctranspath_hf": BackboneSpec(
        name="ctranspath_hf",
        hf_id="1aurent/swin_tiny_patch4_window7_224.CTransPath",
        img_size=224,
        mean=_IMAGENET_MEAN,
        std=_IMAGENET_STD,
        load=_load_ctranspath_hf,
        embed=_embed_timm_cls,
        gated=False,
        crop_pct=0.9,
        interpolation="bicubic",
    ),
    # Cellpose-SAM: native bsize 256 + percentile-нормализация (не ImageNet).
    "cpsam": BackboneSpec(
        name="cpsam",
        hf_id="mouseland/cellpose-sam",
        img_size=256,
        mean=(0.0, 0.0, 0.0),
        std=(1.0, 1.0, 1.0),
        load=_load_cpsam,
        embed=_embed_cpsam,
        gated=False,
        preprocess=_preprocess_cpsam,
    ),
}


# ---------------------------------------------------------------------------
# Intermediate features (per-block / per-layer) — TransPath / vs-filtering style
# ---------------------------------------------------------------------------


def _pool_tokens(
    tokens: torch.Tensor,
    pool: str = "flat",
    n_skip: int = 0,
) -> torch.Tensor:
    """
    tokens: (B, N, D) или (B, H, W, C) / (B, C, H, W).
      pool="flat"     → flatten всех токенов/пикселей (как TransPath / vs-filtering)
      pool="cls"      → tokens[:, 0]
      pool="mean"     → mean(tokens[:, n_skip:])
      pool="cls_mean" → CLS || mean(patches)  (Virchow2 neck)
      pool="gap"      → mean over token/spatial dims
    """
    if pool == "flat":
        return tokens.reshape(tokens.shape[0], -1)
    if tokens.ndim == 4:
        if tokens.shape[1] <= 16 and tokens.shape[-1] > 16:
            return tokens.mean(dim=(1, 2))
        return tokens.mean(dim=(2, 3))
    if pool == "cls":
        return tokens[:, 0]
    if pool == "mean":
        return tokens[:, n_skip:].mean(dim=1)
    if pool == "cls_mean":
        return torch.cat([tokens[:, 0], tokens[:, n_skip:].mean(dim=1)], dim=-1)
    if pool == "gap":
        return tokens.mean(dim=1)
    raise ValueError(f"Unknown pool: {pool}")


def _neck_embedding(name: str, model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Глобальный neck — CLS / CLS‖mean / GAP (не flatten)."""
    if name == "virchow2":
        return _embed_virchow2(model, x)
    if name in ("hibou_l", "phikon"):
        return _embed_transformers_cls(model, x)
    if name == "cpsam":
        return _embed_cpsam(model, x)
    return _embed_timm_cls(model, x)


def _features_timm_vit(
    model: nn.Module,
    x: torch.Tensor,
    name: str = "",
) -> Dict[str, torch.Tensor]:
    """
    Per-block features в стиле TransPath:
      layer_k = flatten(token map после блока k)
      neck    = глобальный embed модели
    """
    depth = len(model.blocks) if hasattr(model, "blocks") else None

    if hasattr(model, "get_intermediate_layers") and depth:
        try:
            intermediates = list(
                model.get_intermediate_layers(
                    x,
                    n=depth,
                    reshape=False,
                    return_prefix_tokens=False,
                    norm=False,
                )
            )
            out = {
                f"layer_{i}": _pool_tokens(t, pool="flat")
                for i, t in enumerate(intermediates)
            }
            out["neck"] = _neck_embedding(name, model, x)
            return out
        except Exception as e:  # noqa: BLE001
            print(f"[features] get_intermediate_layers failed: {e}; trying forward_intermediates", flush=True)

    if hasattr(model, "forward_intermediates") and depth:
        try:
            feats = model.forward_intermediates(
                x,
                indices=depth,
                norm=False,
                stop_early=False,
                output_fmt="NLC",
                intermediates_only=True,
            )
            out = {
                f"layer_{i}": _pool_tokens(t, pool="flat")
                for i, t in enumerate(feats)
            }
            out["neck"] = _neck_embedding(name, model, x)
            return out
        except Exception as e:  # noqa: BLE001
            print(f"[features] forward_intermediates failed: {e}; falling back to block walk", flush=True)

    out: Dict[str, torch.Tensor] = {}
    if not hasattr(model, "patch_embed") or not hasattr(model, "blocks"):
        out["neck"] = _neck_embedding(name, model, x)
        return out

    z = model.patch_embed(x)
    if hasattr(model, "_pos_embed"):
        z = model._pos_embed(z)
    if hasattr(model, "patch_drop"):
        z = model.patch_drop(z)
    if hasattr(model, "norm_pre"):
        z = model.norm_pre(z)
    for i, blk in enumerate(model.blocks):
        z = blk(z)
        out[f"layer_{i}"] = _pool_tokens(z, pool="flat")
    out["neck"] = _neck_embedding(name, model, x)
    return out


def _features_transformers(model: nn.Module, x: torch.Tensor, name: str = "") -> Dict[str, torch.Tensor]:
    """HF ViT / Hibou: flatten hidden state на каждом слое + CLS neck."""
    out_hs = model(pixel_values=x, output_hidden_states=True)
    hidden = out_hs.hidden_states
    layer_states = hidden[1:]
    out = {
        f"layer_{i}": _pool_tokens(h, pool="flat")
        for i, h in enumerate(layer_states)
    }
    out["neck"] = _neck_embedding(name, model, x)
    return out


def _features_swin_blocks(model: nn.Module, x: torch.Tensor, name: str = "") -> Dict[str, torch.Tensor]:
    """Swin: per-block flatten + neck mean(norm) — как local TransPath."""
    out: Dict[str, torch.Tensor] = {}
    z = model.patch_embed(x)
    layer_idx = 0
    stages = getattr(model, "layers", None) or getattr(model, "layers_", None)
    if stages is None:
        out["neck"] = _neck_embedding(name, model, x)
        return out
    for stage in stages:
        blocks = getattr(stage, "blocks", None)
        if blocks is None:
            continue
        for block in blocks:
            z = block(z)
            out[f"layer_{layer_idx}"] = _pool_tokens(z, pool="flat")
            layer_idx += 1
        if getattr(stage, "downsample", None) is not None:
            z = stage.downsample(z)
    if hasattr(model, "norm"):
        z = model.norm(z)
    if z.ndim == 3:
        out["neck"] = z.mean(dim=1)
    elif z.ndim == 4:
        out["neck"] = z.mean(dim=(1, 2)) if z.shape[-1] > z.shape[1] else z.mean(dim=(2, 3))
    else:
        out["neck"] = _neck_embedding(name, model, x)
    return out


def _features_cpsam(model: nn.Module, x: torch.Tensor) -> Dict[str, torch.Tensor]:
    """SAM blocks → flatten spatial map; neck = GAP после enc.neck."""
    x = x.to(dtype=getattr(model, "dtype", x.dtype))
    ps = model.ps
    enc = model.encoder
    z = F.conv2d(
        x,
        enc.patch_embed.proj.weight.data[:, : x.shape[1]],
        bias=enc.patch_embed.proj.bias.data,
        stride=ps,
    )
    z = z.permute(0, 2, 3, 1)
    if enc.pos_embed is not None:
        z = z + enc.pos_embed
    out: Dict[str, torch.Tensor] = {}
    for i, blk in enumerate(enc.blocks):
        z = blk(z)
        out[f"layer_{i}"] = _pool_tokens(z, pool="flat")
    feat = enc.neck(z.permute(0, 3, 1, 2))
    out["neck"] = feat.mean(dim=(2, 3))
    return out


def extract_foundation_features(
    name: str,
    model: nn.Module,
    x: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """
    Диспетчер: name → dict[feature → (B, D)].

    Агрегация как у TransPath/vs-filtering:
      layer_* = cosine по flatten полной token/spatial map
      neck    = глобальный pooled embed (старое поведение)
    """
    if name in ("hibou_l", "phikon"):
        return _features_transformers(model, x, name=name)
    if name == "ctranspath_hf":
        return _features_swin_blocks(model, x, name=name)
    if name == "cpsam":
        return _features_cpsam(model, x)
    return _features_timm_vit(model, x, name=name)


def foundation_feature_names(name: str) -> List[str]:
    """Search-space feature list для nested CV (без загрузки модели)."""
    n = FOUNDATION_FEATURE_DEPTHS.get(name, 12)
    return [f"layer_{i}" for i in range(n)] + ["neck", "mean"]


# ---------------------------------------------------------------------------
# Lazy cache + metric function
# ---------------------------------------------------------------------------

_CACHE: Dict[str, nn.Module] = {}
_SPEC_CACHE: Dict[str, BackboneSpec] = {}


def get_foundation_model(name: str, device: Optional[str] = None) -> Tuple[BackboneSpec, nn.Module]:
    if name not in FOUNDATION_BACKBONES:
        raise KeyError(f"Unknown foundation backbone: {name}. Known: {list(FOUNDATION_BACKBONES)}")
    spec = FOUNDATION_BACKBONES[name]
    if name not in _CACHE:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        # HF token from env if present
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        if token:
            try:
                from huggingface_hub import login

                login(token=token, add_to_git_credential=False)
            except Exception:
                pass
        model = spec.load(device)
        _CACHE[name] = model
        _SPEC_CACHE[name] = _resolve_data_config(spec, model)
    return _SPEC_CACHE.get(name, spec), _CACHE[name]


def _score_feature_pair(
    f0: Dict[str, torch.Tensor],
    f1: Dict[str, torch.Tensor],
    feature: str,
    agg: str,
) -> float:
    keys = list(f0.keys())
    if feature == "mean":
        layer_keys = [k for k in keys if k.startswith("layer_")]
        if not layer_keys:
            layer_keys = ["neck"] if "neck" in f0 else keys
        pairs = [(f0[k], f1[k]) for k in layer_keys]
    elif feature in f0:
        pairs = [(f0[feature], f1[feature])]
    else:
        raise ValueError(f"Unknown feature '{feature}'. Known: {keys + ['mean']}")

    if agg == "cos":
        vals = [_cosine(a, b) for a, b in pairs]
        return float(sum(vals) / len(vals))
    if agg == "dist":
        vals = [_normalized_dist(a, b) for a, b in pairs]
        return -float(sum(vals) / len(vals))
    raise ValueError(f"Unknown agg: {agg}")


def calc_foundation(
    src_t: torch.Tensor,
    trg_t: torch.Tensor,
    name: str,
    agg: str = "cos",
    feature: str = "neck",
    device: Optional[str] = None,
) -> float:
    """
    Similarity между парой через foundation-эмбеддинг.
      feature: "neck" | "mean" | "layer_k"
      agg: "cos" | "dist"  (dist → возвращаем -L2)
    """
    spec, model = get_foundation_model(name, device=device)
    with torch.inference_mode():
        x0 = prepare_input(spec, src_t)
        x1 = prepare_input(spec, trg_t)
        if feature == "neck":
            # быстрый путь — старый embed без полного layer-walk
            e0 = spec.embed(model, x0)
            e1 = spec.embed(model, x1)
            if agg == "cos":
                return _cosine(e0, e1)
            if agg == "dist":
                return -_normalized_dist(e0, e1)
            raise ValueError(f"Unknown agg: {agg}")
        f0 = extract_foundation_features(name, model, x0)
        f1 = extract_foundation_features(name, model, x1)
    return _score_feature_pair(f0, f1, feature, agg)


def calc_foundation_all_features(
    src_t: torch.Tensor,
    trg_t: torch.Tensor,
    name: str,
    agg: str = "cos",
    device: Optional[str] = None,
) -> Dict[str, float]:
    """Один forward → scores для всех layer_k + neck + mean (для noval-sweep)."""
    spec, model = get_foundation_model(name, device=device)
    with torch.inference_mode():
        x0 = prepare_input(spec, src_t)
        x1 = prepare_input(spec, trg_t)
        f0 = extract_foundation_features(name, model, x0)
        f1 = extract_foundation_features(name, model, x1)
    names = list(f0.keys()) + ["mean"]
    return {feat: _score_feature_pair(f0, f1, feat, agg) for feat in names}
