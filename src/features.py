"""
Module trích xuất đặc trưng từ ảnh đã preprocessing (YCrCb 256×256).

6 nhóm features:
  1. FFT (tần số)         — phổ tần số từ kênh Y
  2. GLCM (texture)       — ma trận đồng hiện mức xám
  3. Residual             — high-pass / Laplacian residual
  4. Color correlation    — tương quan giữa kênh Y, Cr, Cb
  5. Statistical          — thống kê cơ bản pixel
  6. Patch aggregation    — gộp features từ 16 patches
"""

import cv2
import numpy as np
from scipy import stats as sp_stats
from scipy.fft import fft2, fftshift
from skimage.feature import graycomatrix, graycoprops


# ============================================================
# 1. FREQUENCY FEATURES (FFT)
# ============================================================

def extract_fft_features(patch_y):
    """
    Trích đặc trưng tần số từ kênh Y của 1 patch.
    
    - FFT 2D → magnitude spectrum
    - Chia thành 3 vùng: low / mid / high frequency
    - Return: dict với 4 features
    """
    h, w = patch_y.shape
    
    # FFT 2D
    f_transform = fft2(patch_y.astype(np.float64))
    f_shift = fftshift(f_transform)
    magnitude = np.abs(f_shift) + 1e-10  # tránh log(0)
    
    # Tạo mask vùng tần số dựa trên khoảng cách từ tâm
    cy, cx = h // 2, w // 2
    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
    max_dist = np.sqrt(cx ** 2 + cy ** 2)
    
    # 3 vùng: low (0-33%), mid (33-66%), high (66-100%)
    low_mask = dist <= max_dist * 0.33
    mid_mask = (dist > max_dist * 0.33) & (dist <= max_dist * 0.66)
    high_mask = dist > max_dist * 0.66
    
    low_energy = np.mean(np.log1p(magnitude[low_mask]))
    mid_energy = np.mean(np.log1p(magnitude[mid_mask]))
    high_energy = np.mean(np.log1p(magnitude[high_mask]))
    
    ratio = high_energy / (low_energy + 1e-10)
    
    return {
        "fft_low_energy": low_energy,
        "fft_mid_energy": mid_energy,
        "fft_high_energy": high_energy,
        "fft_high_low_ratio": ratio
    }


# ============================================================
# 2. TEXTURE FEATURES (GLCM)
# ============================================================

def extract_glcm_features(patch_y, levels=32):
    """
    Trích đặc trưng texture bằng GLCM (Gray-Level Co-occurrence Matrix).
    
    - Lượng tử hóa kênh Y về `levels` mức
    - Tính GLCM với 4 hướng (0°, 45°, 90°, 135°)
    - Return: dict với 4 features (mean over angles)
    """
    # Lượng tử hóa
    patch_q = (patch_y / 256.0 * levels).astype(np.uint8)
    patch_q = np.clip(patch_q, 0, levels - 1)
    
    # GLCM: distance=1, 4 angles
    glcm = graycomatrix(
        patch_q, 
        distances=[1], 
        angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
        levels=levels, 
        symmetric=True, 
        normed=True
    )
    
    contrast = graycoprops(glcm, 'contrast').mean()
    homogeneity = graycoprops(glcm, 'homogeneity').mean()
    energy = graycoprops(glcm, 'energy').mean()
    correlation = graycoprops(glcm, 'correlation').mean()
    
    return {
        "glcm_contrast": contrast,
        "glcm_homogeneity": homogeneity,
        "glcm_energy": energy,
        "glcm_correlation": correlation
    }


# ============================================================
# 3. RESIDUAL FEATURES
# ============================================================

def extract_residual_features(patch_y):
    """
    Trích đặc trưng từ residual (phần dư) — nhấn mạnh cạnh, nhiễu, chi tiết mảnh.
    
    - Áp dụng Laplacian filter lên kênh Y
    - Tính thống kê trên ảnh residual
    - Return: dict với 4 features
    """
    # Laplacian filter → nhấn mạnh cạnh và nhiễu
    residual = cv2.Laplacian(patch_y.astype(np.float64), cv2.CV_64F)
    
    res_std = np.std(residual)
    res_mean = np.mean(np.abs(residual))
    res_kurtosis = float(sp_stats.kurtosis(residual.flatten()))
    
    # Entropy của histogram residual
    hist, _ = np.histogram(residual.flatten(), bins=64, density=True)
    hist = hist[hist > 0]
    res_entropy = float(sp_stats.entropy(hist))
    
    return {
        "residual_std": res_std,
        "residual_mean": res_mean,
        "residual_kurtosis": res_kurtosis,
        "residual_entropy": res_entropy
    }


# ============================================================
# 4. COLOR CORRELATION FEATURES
# ============================================================

def extract_color_features(patch_ycrcb):
    """
    Trích đặc trưng tương quan màu giữa 3 kênh Y, Cr, Cb.
    
    - Correlation coefficient giữa từng cặp kênh
    - Return: dict với 3 features
    """
    y_ch = patch_ycrcb[:, :, 0].flatten().astype(np.float64)
    cr_ch = patch_ycrcb[:, :, 1].flatten().astype(np.float64)
    cb_ch = patch_ycrcb[:, :, 2].flatten().astype(np.float64)
    
    # Tránh division by zero nếu std = 0
    def safe_corr(a, b):
        if np.std(a) < 1e-10 or np.std(b) < 1e-10:
            return 0.0
        return float(np.corrcoef(a, b)[0, 1])
    
    return {
        "corr_y_cr": safe_corr(y_ch, cr_ch),
        "corr_y_cb": safe_corr(y_ch, cb_ch),
        "corr_cr_cb": safe_corr(cr_ch, cb_ch)
    }


# ============================================================
# 5. STATISTICAL FEATURES
# ============================================================

def extract_stat_features(patch_y):
    """
    Thống kê cơ bản trên kênh Y.
    
    Return: dict với 4 features
    """
    pixels = patch_y.flatten().astype(np.float64)
    
    return {
        "stat_mean": np.mean(pixels),
        "stat_std": np.std(pixels),
        "stat_skew": float(sp_stats.skew(pixels)),
        "stat_kurtosis": float(sp_stats.kurtosis(pixels))
    }


# ============================================================
# 6. PATCH EXTRACTION & AGGREGATION
# ============================================================

def extract_patch_features(patch_ycrcb):
    """Trích tất cả features từ 1 patch YCrCb."""
    patch_y = patch_ycrcb[:, :, 0]
    
    features = {}
    features.update(extract_fft_features(patch_y))
    features.update(extract_glcm_features(patch_y))
    features.update(extract_residual_features(patch_y))
    features.update(extract_color_features(patch_ycrcb))
    features.update(extract_stat_features(patch_y))
    
    return features


def extract_image_features(img_ycrcb, patch_size=64):
    """
    Trích đặc trưng cho toàn bộ 1 ảnh YCrCb 256×256.
    
    Pipeline:
      1. Chia ảnh thành grid patches (256/64 = 4×4 = 16 patches)
      2. Trích features mỗi patch
      3. Gộp 16 patches: mean, std, percentile 90
    
    Returns: dict (feature_name → value), khoảng 57 features
    """
    h, w = img_ycrcb.shape[:2]
    
    # Chia patches
    all_patch_features = []
    for i in range(0, h, patch_size):
        for j in range(0, w, patch_size):
            patch = img_ycrcb[i:i + patch_size, j:j + patch_size]
            if patch.shape[0] == patch_size and patch.shape[1] == patch_size:
                pf = extract_patch_features(patch)
                all_patch_features.append(pf)
    
    if not all_patch_features:
        return {}
    
    # Gộp: mean, std, p90 cho mỗi feature
    feature_names = list(all_patch_features[0].keys())
    aggregated = {}
    
    for fname in feature_names:
        values = [pf[fname] for pf in all_patch_features]
        aggregated[f"{fname}_mean"] = np.mean(values)
        aggregated[f"{fname}_std"] = np.std(values)
        aggregated[f"{fname}_p90"] = np.percentile(values, 90)
    
    return aggregated


def get_feature_names(patch_size=64):
    """Trả về danh sách tên features (để lưu vào file)."""
    # Tạo 1 ảnh giả để lấy tên features
    dummy = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
    features = extract_image_features(dummy, patch_size)
    return list(features.keys())
