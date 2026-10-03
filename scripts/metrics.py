import math
import numpy as np
import torch
import cv2
import piq
from skimage.metrics import structural_similarity

# --- Pixel-wise metrics ---

def calc_ssim(src_np, trg_np, mask_np=None, win_size=7):
    """
    Ожидает: Numpy (H, W), float [0, 1].
    """
    # data_range=1.0 критически важно для float данных!
    _, simg = structural_similarity(src_np, trg_np, win_size=win_size, data_range=1.0, full=True)
    
    if mask_np is not None:
        ssim_val = np.mean(simg[mask_np == 1])
    else:
        ssim_val = np.mean(simg)
    return ssim_val

def psnr_float(original, compressed):
    # Формула для max_pixel=1.0: 20 * log10(1.0 / sqrt(mse)) = -10 * log10(mse)
    mse = np.mean((original - compressed) ** 2)
    if mse == 0:
        return 33.0 # Идеальное совпадение (в оригинале 100.0)
    return -10 * math.log10(mse)

def calc_psnr(src_np, trg_np, mask_np=None):
    """
    Ожидает: Numpy (H, W), float [0, 1].
    """
    if mask_np is not None:
        src_flat = src_np[mask_np == 1]
        trg_flat = trg_np[mask_np == 1]
    else:
        src_flat = src_np.ravel()
        trg_flat = trg_np.ravel()
        
    psnr_val = psnr_float(src_flat, trg_flat)
    return psnr_val

def norm_data(data):
    """
    Normalize data to have mean = 0 and std = 1
    """
    mean_data = np.mean(data)
    std_data = np.std(data, ddof=1)
    # Добавляем epsilon к std, чтобы избежать деления на ноль
    return (data - mean_data) / (std_data + 1e-8)

def ncc(data0, data1):
    """
    Normalized Cross-Correlation (Pearson)
    """
    if data0.size <= 1: return 0.0
    return (1.0 / (data0.size - 1)) * np.sum(norm_data(data0) * norm_data(data1))

def calc_ncc(src_np, trg_np, mask_np=None):
    """
    Ожидает: Numpy (H, W), float [0, 1].
    """
    if mask_np is not None:
        src_flat = src_np[mask_np == 1]
        trg_flat = trg_np[mask_np == 1]
    else:
        src_flat = src_np.ravel()
        trg_flat = trg_np.ravel()
    
    ncc_val = ncc(src_flat.ravel(), trg_flat.ravel())
    return ncc_val

def mutual_information(hgram):
    """Mutual information for joint histogram"""
    # Convert bins counts to probability values
    pxy = hgram / (float(np.sum(hgram)) + 1e-9)
    px = np.sum(pxy, axis=1)  # marginal for x over y
    py = np.sum(pxy, axis=0)  # marginal for y over x
    px_py = px[:, None] * py[None, :] # Broadcast
    
    # Only non-zero pxy values contribute to the sum
    nzs = pxy > 0 
    return np.sum(pxy[nzs] * np.log(pxy[nzs] / px_py[nzs]))

def calc_mi(src_np, trg_np, mask_np=None, bins=32):
    """Вход: Numpy (H, W) float [0, 1]"""

    # Apply mask before calculating MI
    if mask_np is not None:
        src_flat = src_np[mask_np == 1]
        trg_flat = trg_np[mask_np == 1]
    else:
        src_flat = src_np.ravel()
        trg_flat = trg_np.ravel()

    # bins=32 автоматически квантует float [0, 1]
    hgram, _, _ =  np.histogram2d(src_flat, trg_flat, bins=bins)
    mi_val = mutual_information(hgram)

    return mi_val

# def calc_ms_ssim(src_np, trg_np, mask_np=None, fill_value=0.0):
#     """
#     Вход: Tensor (1, C, H, W) float [0, 1].
#     fill_value: Значение для фона (0.0 или 1.0).
#     """
#     src, trg = src_np.copy(), trg_np.copy()
#     # 1. Применение маски (Smart Fill)
#     if mask_np is not None:
#         # Заливаем фон правильным цветом, чтобы не было резких границ
#         src[mask_np == 0] = fill_value
#         trg[mask_np == 0] = fill_value
#     # 2. Конвертация в Tensor (1, 1, H, W)
#     src_t = torch.from_numpy(src).float().unsqueeze(0).unsqueeze(0)
#     trg_t = torch.from_numpy(trg).float().unsqueeze(0).unsqueeze(0)
#     # 3. MS-SSIM
#     ms_ssim = piq.multi_scale_ssim(src_t, trg_t, data_range=1.0) # data_range=255. для uint8
#     return ms_ssim.item()

def calc_ms_ssim(src_t, trg_t, mask_t=None, fill_value=0.0):
    """
    Вход: Tensor (1, C, H, W) float [0, 1].
    fill_value: Значение для фона (0.0 или 1.0).
    """
    # 1. Применение маски (Smart Fill)
    if mask_t is not None:
        src_t = src_t * mask_t + fill_value * (1.0 - mask_t)
        trg_t = trg_t * mask_t + fill_value * (1.0 - mask_t)
    
    # 3. MS-SSIM
    ms_ssim = piq.multi_scale_ssim(src_t, trg_t, data_range=1.0)

    return ms_ssim.item()

# --- Perceptual metrics ---

def calc_fsim(src_t, trg_t, chromatic=False):
    """
    FSIM / FSIMc через библиотеку piq.
    Вход: Tensor (1, C, H, W) на GPU/CPU.
    """
    if chromatic: # FSIMc - RGB
        # FSIMc требует 3 канала. Если пришел 1, дублируем.
        if src_t.shape[1] == 1:
            src_t = src_t.repeat(1, 3, 1, 1)
            trg_t = trg_t.repeat(1, 3, 1, 1)
        fsim_val = piq.fsim(src_t, trg_t, chromatic=True)
    else: 
        # FSIM (grayscale) - обычно работает с 1 каналом (яркость)
        fsim_val = piq.fsim(src_t, trg_t, chromatic=False)

    return fsim_val.item() # Возвращаем float

def calc_lpips(src_t, trg_t, mask_t, bg_val, loss_fn):
    """
    Вход: Tensor [0, 1] + Маска, Значение фона.
    """
    assert src_t.shape[1] == 3 and trg_t.shape[1] == 3, "LPIPS требует 3 канала"
    
    # 1. Применяем маску (Smart Fill)
    if mask_t is not None:
        # In-place операции опасны для градиентов, но здесь мы в no_grad
        src_t = src_t * mask_t + bg_val * (1.0 - mask_t)
        trg_t = trg_t * mask_t + bg_val * (1.0 - mask_t)
    
    # 2. Расчет LPIPS
    with torch.no_grad():
        # normalize=True - LPIPS сам сделает (x*2 - 1)
        lpips_val = loss_fn(src_t, trg_t, normalize=True).item()

    return lpips_val

def calc_dists(src_t, trg_t, mask_t, bg_val, loss_fn):
    """
    Вход: Tensor [0, 1] + Маска, Значение фона.
    """
    assert src_t.shape[1] == 3 and trg_t.shape[1] == 3, "DISTS требует 3 канала"
    
    # 1. Применяем маску (Smart Fill)
    if mask_t is not None:
        src_t = src_t * mask_t + bg_val * (1.0 - mask_t)
        trg_t = trg_t * mask_t + bg_val * (1.0 - mask_t)
    
    # 2. Расчет DISTS
    with torch.no_grad():
        dists_val = loss_fn(src_t, trg_t).item()

    return dists_val