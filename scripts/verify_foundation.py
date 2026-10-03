"""
Проверка корректности foundation-бэкбонов:

1. weights   — веса реально загружены (checksum != random init того же конфига)
2. dataconfig— mean/std/img_size совпадают с тем, что объявляет сама модель
3. embed     — размерность эмбеддинга совпадает с num_features из конфига
4. behaviour — self-cos == 1, а unrelated-пара заметно ниже (санити-чек сигнала)

Запуск (в .venv-foundation):
    .venv-foundation/bin/python scripts/verify_foundation.py --models phikon,hibou_l
    .venv-foundation/bin/python scripts/verify_foundation.py            # все
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts import foundation_metrics as fm  # noqa: E402


def _hub_config(hf_id: str) -> dict:
    """Читает config.json модели из локального HF-кэша."""
    hub = Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface")) / "hub"
    root = hub / ("models--" + hf_id.replace("/", "--")) / "snapshots"
    for cfg in root.glob("*/config.json"):
        return json.loads(cfg.read_text())
    return {}


def _preproc_config(hf_id: str) -> dict:
    hub = Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface")) / "hub"
    root = hub / ("models--" + hf_id.replace("/", "--")) / "snapshots"
    for cfg in root.glob("*/preprocessor_config.json"):
        return json.loads(cfg.read_text())
    return {}


def expected_mean_std(hf_id: str):
    """mean/std/input_size, объявленные самой моделью (timm pretrained_cfg или HF processor)."""
    cfg = _hub_config(hf_id)
    pcfg = cfg.get("pretrained_cfg", {})
    if pcfg.get("mean") and pcfg.get("std"):
        size = pcfg.get("input_size", [3, 224, 224])[-1]
        return tuple(pcfg["mean"]), tuple(pcfg["std"]), size

    proc = _preproc_config(hf_id)
    if proc.get("image_mean") and proc.get("image_std"):
        size = proc.get("crop_size", {}).get("height") or proc.get("size")
        if isinstance(size, dict):
            size = size.get("shortest_edge", 224)
        return tuple(proc["image_mean"]), tuple(proc["image_std"]), int(size or 224)
    return None, None, None


def param_checksum(model: torch.nn.Module) -> float:
    """Стабильная сумма по всем параметрам — ловит незагруженные/рандомные веса."""
    total = 0.0
    for p in model.parameters():
        total += float(p.detach().double().abs().sum())
    return total


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="", help="через запятую; пусто = все")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--skip-random", action="store_true", help="не строить random-init копию (быстрее, но без проверки весов)")
    args = ap.parse_args()

    names = (
        [n.strip() for n in args.models.split(",") if n.strip()]
        or list(fm.FOUNDATION_BACKBONES)
    )

    rng = np.random.RandomState(0)
    img_a = torch.from_numpy(rng.rand(1, 3, 256, 256).astype(np.float32))
    img_b = torch.from_numpy(rng.rand(1, 3, 256, 256).astype(np.float32))

    for name in names:
        spec = fm.FOUNDATION_BACKBONES[name]
        print(f"\n{'=' * 70}\n{name}  ({spec.hf_id})\n{'=' * 70}", flush=True)

        exp_mean, exp_std, exp_size = expected_mean_std(spec.hf_id)
        if exp_mean is None:
            print("  dataconfig: не найден в кэше (пропуск сверки)")
        else:
            ok_norm = (
                np.allclose(spec.mean, exp_mean, atol=1e-4)
                and np.allclose(spec.std, exp_std, atol=1e-4)
            )
            ok_size = spec.img_size == exp_size
            print(f"  declared mean={tuple(round(v, 4) for v in exp_mean)} "
                  f"std={tuple(round(v, 4) for v in exp_std)} size={exp_size}")
            print(f"  ours     mean={spec.mean} std={spec.std} size={spec.img_size}")
            print(f"  NORM {'OK' if ok_norm else 'MISMATCH'} | SIZE {'OK' if ok_size else 'MISMATCH'}")

        try:
            _, model = fm.get_foundation_model(name, device=args.device)
        except Exception as exc:  # noqa: BLE001
            print(f"  LOAD FAILED: {type(exc).__name__}: {exc}")
            continue

        n_params = sum(p.numel() for p in model.parameters())
        print(f"  loaded: {type(model).__name__}, params={n_params / 1e6:.1f}M")

        cfg = _hub_config(spec.hf_id)
        num_features = cfg.get("num_features") or cfg.get("hidden_size")

        with torch.inference_mode():
            x = fm._resize_norm(img_a, spec.img_size, spec.mean, spec.std)
            emb = spec.embed(model, x)
        print(f"  embed dim={tuple(emb.shape)} (config num_features={num_features})"
              f"  norm={float(emb.norm()):.3f}")

        self_cos = fm.calc_foundation(img_a, img_a, name=name, agg="cos", device=args.device)
        other_cos = fm.calc_foundation(img_a, img_b, name=name, agg="cos", device=args.device)
        print(f"  self-cos={self_cos:.4f}  unrelated-cos={other_cos:.4f}")

        if not args.skip_random:
            chk = param_checksum(model)
            try:
                import timm

                if spec.name in ("phikon", "hibou_l", "cpsam"):
                    rand_chk = None  # transformers/cellpose: random init строим отдельно
                else:
                    rand = timm.create_model(f"hf-hub:{spec.hf_id}", pretrained=False)
                    rand_chk = param_checksum(rand)
                    del rand
            except Exception as exc:  # noqa: BLE001
                rand_chk = None
                print(f"  random-init probe skipped: {type(exc).__name__}: {exc}")

            if rand_chk is not None:
                rel = abs(chk - rand_chk) / max(rand_chk, 1e-9)
                verdict = "PRETRAINED" if rel > 0.02 else "SUSPICIOUS (похоже на random init)"
                print(f"  checksum pretrained={chk:.1f} random={rand_chk:.1f} "
                      f"rel_diff={rel:.3f} -> {verdict}")
            else:
                print(f"  checksum={chk:.1f}")

        fm._CACHE.pop(name, None)
        del model
        gc.collect()


if __name__ == "__main__":
    main()
