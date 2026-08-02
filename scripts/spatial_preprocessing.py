from PIL import Image
import numpy as np


def downsample_by_scale(
    img,
    scale,
    resample=Image.BICUBIC,
):
    """
    Downsample image by a given scale factor.

    Parameters
    ----------
    img : np.ndarray
        Image with shape (H, W) or (H, W, C).

    scale : float
        Downsampling factor.

        Example:
            scale = 2.0
            1024x1024 -> 512x512

    resample : PIL.Image.Resampling
        Resampling method.

    Returns
    -------
    np.ndarray
        Resized image.
    """

    if img.ndim not in (2, 3):
        raise ValueError(
            f"Expected image with 2 or 3 dimensions, "
            f"got {img.ndim}"
        )

    if scale <= 0:
        raise ValueError(
            f"Scale must be positive, got {scale}"
        )

    h, w = img.shape[:2]

    new_w = max(
        1,
        int(round(w / scale))
    )

    new_h = max(
        1,
        int(round(h / scale))
    )

    pil_img = Image.fromarray(img)

    pil_resized = pil_img.resize(
        (new_w, new_h),
        resample=resample
    )

    return np.array(pil_resized)


def downsample_to_mpp(
    img,
    source_mpp,
    target_mpp,
    resample=Image.BICUBIC,
):
    """
    Rescale image from source MPP to target MPP.

    Example
    -------
    MIST:
        source_mpp = 0.4661
        target_mpp = 1.0112

    The resulting image is downsampled by:

        target_mpp / source_mpp

    Parameters
    ----------
    img : np.ndarray

    source_mpp : float
        Original resolution of the image.

    target_mpp : float
        Target resolution.

    Returns
    -------
    np.ndarray
    """

    if source_mpp <= 0:
        raise ValueError(
            f"source_mpp must be positive, got {source_mpp}"
        )

    if target_mpp <= 0:
        raise ValueError(
            f"target_mpp must be positive, got {target_mpp}"
        )

    scale = target_mpp / source_mpp

    return downsample_by_scale(
        img,
        scale=scale,
        resample=resample,
    )


def center_crop(
    img,
    crop_size=256,
):
    """
    Center crop.

    Parameters
    ----------
    img : np.ndarray
        Image (H, W) or (H, W, C).

    crop_size : int or tuple
        Crop size.

    Returns
    -------
    np.ndarray
    """

    if isinstance(crop_size, int):
        crop_h = crop_w = crop_size
    else:
        crop_h, crop_w = crop_size

    h, w = img.shape[:2]

    if h < crop_h or w < crop_w:
        raise ValueError(
            f"Image is smaller than crop size: "
            f"image=({h}, {w}), "
            f"crop=({crop_h}, {crop_w})"
        )

    top = (h - crop_h) // 2
    left = (w - crop_w) // 2

    return img[
        top:top + crop_h,
        left:left + crop_w
    ]


def split_into_patches(
    img,
    patch_size=256,
):
    """
    Split image into non-overlapping patches.

    Example:

        512x512
            ↓
        4 patches of 256x256

    Returns
    -------
    list[np.ndarray]
    """

    h, w = img.shape[:2]

    if h % patch_size != 0:
        raise ValueError(
            f"Height {h} is not divisible by "
            f"patch_size={patch_size}"
        )

    if w % patch_size != 0:
        raise ValueError(
            f"Width {w} is not divisible by "
            f"patch_size={patch_size}"
        )

    patches = []

    for y in range(0, h, patch_size):
        for x in range(0, w, patch_size):

            patch = img[
                y:y + patch_size,
                x:x + patch_size
            ]

            patches.append(patch)

    return patches


def preprocess_pair_spatial(
    src,
    trg,
    spatial_mode="full",
    dataset="generic",
    source_mpp=None,
    target_mpp=None,
    output_size=1024,
    patch_size=256,
):
    """
    Spatial preprocessing of a paired image.

    Modes
    -----

    full
        Original image. Optionally resized to output_size.

    downsample
        Image resized to output_size.

    patches
        Image resized to 512x512 and split into
        4 patches of 256x256.

    dataset="mist"
        Applies MPP-based scale correction before
        the final spatial processing.

    Returns
    -------
    tuple
        (src_processed, trg_processed)

    For:
        full / downsample

    returns:
        np.ndarray, np.ndarray

    For:
        patches

    returns:
        list[np.ndarray], list[np.ndarray]
    """

    src_proc = src
    trg_proc = trg

    # --------------------------------------------------
    # 1. Dataset-specific scale correction
    # --------------------------------------------------

    if dataset.lower() == "mist":

        if source_mpp is None:
            raise ValueError(
                "source_mpp is required for MIST dataset"
            )

        if target_mpp is None:
            raise ValueError(
                "target_mpp is required for MIST dataset"
            )

        src_proc = downsample_to_mpp(
            src_proc,
            source_mpp=source_mpp,
            target_mpp=target_mpp,
        )

        trg_proc = downsample_to_mpp(
            trg_proc,
            source_mpp=source_mpp,
            target_mpp=target_mpp,
        )

    elif dataset.lower() != "generic":

        raise ValueError(
            f"Unknown dataset: {dataset}"
        )

    # --------------------------------------------------
    # 2. Spatial processing
    # --------------------------------------------------

    if spatial_mode == "full":

        # Full-resolution image.
        # If output_size is specified, resize to it.

        if output_size is not None:

            src_proc = resize_to_size(
                src_proc,
                output_size,
            )

            trg_proc = resize_to_size(
                trg_proc,
                output_size,
            )

        return src_proc, trg_proc

    elif spatial_mode == "downsample":

        src_proc = resize_to_size(
            src_proc,
            output_size,
        )

        trg_proc = resize_to_size(
            trg_proc,
            output_size,
        )

        return src_proc, trg_proc

    elif spatial_mode == "patches":

        # First resize to 512x512
        src_proc = resize_to_size(
            src_proc,
            512,
        )

        trg_proc = resize_to_size(
            trg_proc,
            512,
        )

        src_patches = split_into_patches(
            src_proc,
            patch_size=patch_size,
        )

        trg_patches = split_into_patches(
            trg_proc,
            patch_size=patch_size,
        )

        return src_patches, trg_patches

    else:

        raise ValueError(
            f"Unknown spatial_mode: {spatial_mode}. "
            f"Available: full, downsample, patches"
        )


def resize_to_size(
    img,
    size,
    resample=Image.BICUBIC,
):
    """
    Resize image to square size x size.
    """

    if isinstance(size, int):
        new_h = new_w = size
    else:
        new_h, new_w = size

    pil_img = Image.fromarray(img)

    pil_resized = pil_img.resize(
        (new_w, new_h),
        resample=resample,
    )

    return np.array(pil_resized)