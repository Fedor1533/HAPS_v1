import numpy as np
import torch
import cv2
from dataclasses import dataclass
from scipy.ndimage import median_filter, gaussian_filter
from skimage.exposure import match_histograms, rescale_intensity
from skimage.color import rgb2hed

# --- Low-level Utils ---

def create_non_white_mask(image, threshold=None, percentile=80, buffer=5):
    """
    Create a binary mask where non-white pixels are 1, and white pixels are 0.
    """
    if len(image.shape) == 3:  # RGB image or RGBA
        # Convert to grayscale
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    if threshold is None:
        # Calculate intensity threshold
        threshold = np.percentile(image, percentile) - buffer

    # Create mask (0=background, 1=foreground)
    mask = (image < threshold)

    return mask.astype(np.uint8)

def smooth_image_float(image, method='gaussian', kernel_size=5):
    """
    Apply smoothing to an image (float or uint8).
    Handles both (H, W) and (H, W, C).
    """
    has_channels = (image.ndim == 3)
    if method == 'gaussian':
        # 1. Подбираем sigma как в OpenCV(https://docs.opencv.org/2.4/modules/imgproc/doc/filtering.html#getgaussiankernel)
        sigma = 0.3*((kernel_size-1)*0.5 - 1) + 0.8
        # 2. Вычисляем Truncate, чтобы ограничить ядро размером kernel_size
        # OpenCV radius = (kernel_size - 1) // 2 -> Следовательно: truncate = radius / sigma
        radius = (kernel_size - 1) // 2
        truncate_val = radius / sigma
        if has_channels:
            sigma = (sigma, sigma, 0) # (H, W, C) -> фильтруем H и W, не фильтруем C
            
        return gaussian_filter(image, sigma=sigma, truncate=truncate_val, mode='nearest')

    elif method == 'median':
        # (H, W, C) -> ядро (k, k, 1), HW -> k
        size = kernel_size
        if has_channels:
            size = (size, size, 1)  
        # mode='nearest' имитирует cv2.BORDER_REPLICATE
        return median_filter(image, size=size, mode='nearest')
    else:
        raise ValueError("Invalid method. Choose 'gaussian' or 'median'.")

def normalize(tensor):
    """Channel-wise Min-Max normalization [0, 1]"""
    if isinstance(tensor, torch.Tensor):
        dims = (-2, -1) 
        mins = tensor.amin(dim=(-2, -1), keepdim=True)
        maxs = tensor.amax(dim=(-2, -1), keepdim=True)
        return (tensor - mins) / (maxs - mins + 1e-8)
        
    elif isinstance(tensor, np.ndarray):
        if tensor.ndim == 3: # HWC
            mins = tensor.min(axis=(0, 1), keepdims=True)
            maxs = tensor.max(axis=(0, 1), keepdims=True)
        else:
            mins = tensor.min()
            maxs = tensor.max()
            
        return (tensor - mins) / (maxs - mins + 1e-8)
    else:
        raise ValueError("Unsupported array library.")

def apply_clahe_np(img_np):
    """Apply CLAHE to numpy (H, W)/(H, W, C) float [0, 1].""" 
    # 1. Конвертация в uint8 [0, 255] для работы OpenCV
    img_uint8 = (np.clip(img_np, 0, 1) * 255).astype(np.uint8)
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    
    if img_uint8.ndim == 2:
        # Grayscale (H, W)
        res_uint8 = clahe.apply(img_uint8)     
    elif img_uint8.ndim == 3 and img_uint8.shape[2] == 3:
        # Конвертируем в LAB -> CLAHE на L -> RGB
        lab = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2LAB)
        # 0 - 'L' channel
        lab[:,:,0] = clahe.apply(lab[:,:,0])
        res_uint8 = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    else:
        raise ValueError(f"Unsupported shape for CLAHE: {img_np.shape}")
        
    # 3. Возврат в float32 [0, 1]
    return res_uint8.astype(np.float32) / 255.0

def get_hematoxylin_ch(img_rgb):
    """RGB -> HED space -> Hematoxylin""" 
    img_hed = rgb2hed(img_rgb)
    # Rescale to use full range based on 99th percentile
    h = rescale_intensity(
        img_hed[:, :, 0],
        out_range=(0, 1),
        in_range=(0, np.percentile(img_hed[:, :, 0], 99)),
    )
    return 1 - h

def np_to_tensor(img, device='cpu'):
    # (H, W) -> (1, 1, H, W)
    if img.ndim == 2:
        t = torch.from_numpy(img).unsqueeze(0).unsqueeze(0)
    # (H, W, C) -> (1, C, H, W)
    else:
        t = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
    return t.float().to(device)

# --- High-level Preprocessing ---

def preprocess_pair_numpy(src, trg, params):
    """
    Единая функция предобработки на Numpy.
    Вход: (H, W, C) uint8 или float.
    Выход: (H, W, C) или (H, W) float32 [0, 1].
    """
    # 0. Гарантия float [0, 1]
    if src.dtype == np.uint8 or src.max() > 1.0:
        src = src.astype(np.float32) / 255.0
    if trg.dtype == np.uint8 or trg.max() > 1.0:
        trg = trg.astype(np.float32) / 255.0
        
    # 1. Normalization
    if params['normalization']:
        src, trg = normalize(src), normalize(trg) 
        
    mode = params.get('channel_mode', 'rgb')

    # 2.1. Grayscale
    if mode == 'gray' and src.ndim == 3 and src.shape[2] == 3:
        src = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY)
        trg = cv2.cvtColor(trg, cv2.COLOR_RGB2GRAY)
    # 2.2. HED
    elif mode == 'hed' and src.ndim == 3 and src.shape[2] == 3:
        src = get_hematoxylin_ch(src)
        trg = get_hematoxylin_ch(trg)
            
    # 3. Flip Intensity
    if params['flip_intensity']:
        src = 1 - src
        trg = 1 - trg

    # 4. Операции (Hist / CLAHE) -> CPU
    match_hist = params['match_histogram']
    clahe = params['clahe']

    if match_hist:
        # channel_axis=-1 указывает, что каналы последние. 
        chan_axis = -1 if src.ndim == 3 else None
        src = match_histograms(src, trg, channel_axis=chan_axis)
        # trg = match_histograms(trg, src, channel_axis=chan_axis) # TRY H&E -> IHC matching

    # CLAHE (LAB strategy): apply_clahe_np - сделает float->uint8->[LAB->RGB]->float  
    if clahe:
        src = apply_clahe_np(src)
        trg = apply_clahe_np(trg)

    return src.astype(np.float32), trg.astype(np.float32)

@dataclass
class MetricInput:
    """Контейнер для подготовленных данных."""
    # Для классических метрик (SSIM, PSNR) -> Numpy (H, W)
    src_np: np.ndarray 
    trg_np: np.ndarray
    mask_np: np.ndarray # (H, W) uint8, 0 или 1
    
    # Для Deep Learning метрик (LPIPS, DISTS) -> Tensor (1, 3, H, W)
    src_t: torch.Tensor 
    trg_t: torch.Tensor
    mask_t: torch.Tensor # (1, 1, H, W) float {0., 1.} или None
    
    # Метаданные
    bg_val: float # Значение фона (0.0 или 1.0) для "умной" заливки в метриках

class Preprocessor:
    """
    Класс для предобработки примеров(paired imgs).
    Вход: Paired imgs
    Выход: object of MetricInput (предобработанные изоб-ия в виде torch.Tensor, numpy.ndarray)
    """
    def __init__(self, device="cuda"):
        self.device = device

    def process(self, f_patch, w_patch, config):
        """
        Принимает сырые данные(ndarray (H, W, C)) и конфиг.
        Возвращает MetricInput.
        """
        # 1. Базовая предобработка (float [0, 1])
        pre_src, pre_trg = preprocess_pair_numpy(w_patch, f_patch, config)
        
        # 2. Вычисление маски (закомментировано — пока не используется)
        mask_np = None
        # if config.get('binary_mask', False):
        #     f_mask = create_non_white_mask(f_patch, threshold=230)
        #     w_mask = create_non_white_mask(w_patch, threshold=230)
        #     mask_np = np.logical_or(f_mask, w_mask).astype(np.uint8)
        #     
        #     # Проверка на пустую маску ткани
        #     mask_mean = np.mean(mask_np)
        #     if mask_mean < 0.1:
        #         print(f"WARNING: Tissue mask is too small ({mask_mean:.3f})!")
        
        # 3. Определяем цвет фона: flip_intensity=True -> фон черный (0.0). Иначе белый (1.0).
        bg_val = 0.0 if config['flip_intensity'] else 1.0
        
        # 4. Сглаживание
        if config['smoothing']:
            pre_src = smooth_image_float(pre_src, method='median', kernel_size=5)
            pre_trg = smooth_image_float(pre_trg, method='median', kernel_size=5)

        # --- ВЕТКА 1: Numpy ---
        src_np, trg_np = pre_src.copy(), pre_trg.copy()

        # --- ВЕТКА 2: Данные для Torch метрик (LPIPS, DISTS) ---
        # Конвертация Numpy -> Tensor GPU
        src_t = np_to_tensor(pre_src, self.device)
        trg_t = np_to_tensor(pre_trg, self.device)
        
        mask_t = None
        if mask_np is not None:
            mask_t = torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(0).float().to(self.device)
        
        # Repeat channels if gray
        if src_t.shape[1] == 1:
            src_t = src_t.repeat(1, 3, 1, 1)
            trg_t = trg_t.repeat(1, 3, 1, 1)

        return MetricInput(src_np, trg_np, mask_np, src_t, trg_t, mask_t, bg_val)