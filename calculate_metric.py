import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch

from PIL import Image
from tqdm import tqdm


from scripts.metrics import (
    calc_ncc,
    calc_lpips,
)

from scripts.preprocessing import (
    Preprocessor,
)

from scripts.spatial_preprocessing import (
    preprocess_pair_spatial,
)


def parse_args():

    parser = argparse.ArgumentParser(
        description="Calculate image similarity metric"
    )

    parser.add_argument(
        "--metric",
        type=str,
        required=True,
        choices=["ncc", "lpips"],
        help="Metric to calculate",
    )

    parser.add_argument(
        "--csv",
        type=str,
        required=True,
        help="Input CSV",
    )

    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output CSV",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="cpu or cuda",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="generic",
        choices=["generic", "mist"],
        help="Dataset type",
    )

    parser.add_argument(
        "--spatial-mode",
        type=str,
        default="full",
        choices=[
            "full",
            "downsample",
            "patches",
        ],
        help="Spatial preprocessing mode",
    )

    parser.add_argument(
        "--source-mpp",
        type=float,
        default=0.4661,
        help="Source image MPP (for MIST dataset)",
    )

    parser.add_argument(
        "--target-mpp",
        type=float,
        default=1.0112,
        help="Target image MPP (for MIST dataset)",
    )

    parser.add_argument(
        "--lpips-net",
        type=str,
        default="vgg",
        choices=[
            "alex",
            "vgg",
            "squeeze",
        ],
        help="LPIPS backbone",
    )

    return parser.parse_args()


NCC_CONFIG = {
    "normalization": True,
    "channel_mode": "hed",
    "flip_intensity": False,
    "match_histogram": False,
    "clahe": False,
    "smoothing": True,
    "metric": "ncc",
}

LPIPS_CONFIG = {
    "normalization": True,
    "channel_mode": "rgb",
    "flip_intensity": True,
    "match_histogram": True,
    "clahe": True,
    "smoothing": True,
    "metric": "lpips",
}


def get_config(metric_name):
    """
    Return preprocessing config for the given metric.
    """
    if metric_name == "ncc":
        return NCC_CONFIG.copy()
    elif metric_name == "lpips":
        return LPIPS_CONFIG.copy()
    else:
        raise ValueError(f"Unknown metric: {metric_name}")


def load_image(path):
    """
    Load image as RGB numpy array (uint8).
    """
    image = Image.open(path).convert("RGB")
    return np.array(image)


def create_lpips_model(
    net,
    device,
):
    import lpips

    loss_fn = lpips.LPIPS(
        net=net,
    ).to(device)

    loss_fn.eval()

    return loss_fn


def calculate_single_metric(
    src_np,
    trg_np,
    metric,
    preprocessor,
    config,
    device,
    loss_fn=None,
):
    """
    Calculate metric for one pair using the Preprocessor pipeline.

    Parameters
    ----------
    src_np : np.ndarray
        Source image (H, W, C) uint8.

    trg_np : np.ndarray
        Target image (H, W, C) uint8.

    metric : str
        Metric name ("ncc" or "lpips").

    preprocessor : Preprocessor
        Preprocessor instance for pixel-level preprocessing.

    config : dict
        Preprocessing config.

    device : torch.device

    loss_fn : lpips.LPIPS or None

    Returns
    -------
    float
        Metric score.
    """

    inp = preprocessor.process(src_np, trg_np, config)

    if metric == "ncc":
        score = calc_ncc(
            inp.src_np,
            inp.trg_np,
            mask_np=inp.mask_np,
        )

    elif metric == "lpips":
        score = calc_lpips(
            src_t=inp.src_t,
            trg_t=inp.trg_t,
            mask_t=inp.mask_t,
            bg_val=inp.bg_val,
            loss_fn=loss_fn,
        )

    else:
        raise ValueError(f"Unknown metric: {metric}")

    return score


def main():

    args = parse_args()

    print(f"Metric:       {args.metric}")
    print(f"Dataset:      {args.dataset}")
    print(f"Spatial mode: {args.spatial_mode}")
    print(f"Device:       {args.device}")

    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    config = get_config(args.metric)
    print(f"Config: {config}")

    preprocessor = Preprocessor(device=str(device))

    df = pd.read_csv(args.csv)

    required_columns = ["fixed_path", "warped_path"]
    for column in required_columns:
        if column not in df.columns:
            raise ValueError(f"Missing column: {column}")

    print(f"Loaded {len(df)} samples")

    loss_fn = None
    if args.metric == "lpips":
        loss_fn = create_lpips_model(
            net=args.lpips_net,
            device=device,
        )

    scores = []

    for _, row in tqdm(
        df.iterrows(),
        total=len(df),
        desc=f"Calculating {args.metric}",
    ):

        fixed_path = row["fixed_path"]
        warped_path = row["warped_path"]

        src = load_image(warped_path)
        trg = load_image(fixed_path)

        src_proc, trg_proc = preprocess_pair_spatial(
            src,
            trg,
            spatial_mode=args.spatial_mode,
            dataset=args.dataset,
            source_mpp=args.source_mpp,
            target_mpp=args.target_mpp,
        )

        if args.spatial_mode == "patches":

            patch_scores = []
            for src_patch, trg_patch in zip(src_proc, trg_proc):
                score = calculate_single_metric(
                    src_patch,
                    trg_patch,
                    metric=args.metric,
                    preprocessor=preprocessor,
                    config=config,
                    device=device,
                    loss_fn=loss_fn,
                )
                patch_scores.append(score)

            final_score = float(np.mean(patch_scores))

        else:

            final_score = calculate_single_metric(
                src_proc,
                trg_proc,
                metric=args.metric,
                preprocessor=preprocessor,
                config=config,
                device=device,
                loss_fn=loss_fn,
            )

        scores.append(final_score)

    # Build a descriptive column name
    metric_column = f"{args.metric}"
    if args.dataset == "mist":
        metric_column += f"_mist_mpp{args.target_mpp}"
    metric_column += f"_{args.spatial_mode}"

    df[metric_column] = scores

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    df.to_csv(args.output, index=False)

    print(f"\nSaved results to: {args.output}")


if __name__ == "__main__":
    main()
