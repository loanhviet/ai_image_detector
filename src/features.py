"""
Module trích xuất đặc trưng từ ảnh đã preprocessing (YCrCb 256×256).

9 nhóm features:
  1. FFT (tần số)         — phổ tần số từ kênh Y
  2. GLCM (texture)       — ma trận đồng hiện mức xám
  3. Residual             — high-pass / Laplacian + SRM + noise residual
  4. Color correlation    — tương quan + thống kê kênh Cr, Cb
  5. Statistical          — thống kê cơ bản pixel
  6. DCT features         — DCT 2D coefficients
  7. LBP features         — Local Binary Pattern
  8. Patch aggregation    — gộp features từ 16 patches
"""

import cv2
import numpy as np
from scipy import stats as sp_stats
from scipy.fft import fft2, fftshift, dctn
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern


# ============================================================
# 1. FREQUENCY FEATURES (FFT)
# ============================================================

def extract_fft_features(patch_y):
    """
    Trích đặc trưng tần số từ kênh Y của 1 patch.

    - FFT 2D → magnitude + phase spectrum
    - 3 vùng cũ (backward compat): low / mid / high frequency
    - 5 vùng mới: chia đều 0-20%, 20-40%, 40-60%, 60-80%, 80-100%
    - Phase features: std, entropy
    - Peak detection: fft_peak_ratio (GAN grid artifact)
    - Return: dict với ~17 features
    """
    h, w = patch_y.shape

    f_transform = fft2(patch_y.astype(np.float64))
    f_shift = fftshift(f_transform)
    magnitude = np.abs(f_shift) + 1e-10

    cy, cx = h // 2, w // 2
    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
    max_dist = np.sqrt(cx ** 2 + cy ** 2)

    # Old 3-zone bands (backward compat)
    low_mask = dist <= max_dist * 0.33
    mid_mask = (dist > max_dist * 0.33) & (dist <= max_dist * 0.66)
    high_mask = dist > max_dist * 0.66

    low_energy = float(np.mean(np.log1p(magnitude[low_mask])))
    mid_energy = float(np.mean(np.log1p(magnitude[mid_mask])))
    high_energy = float(np.mean(np.log1p(magnitude[high_mask])))
    ratio = float(high_energy / (low_energy + 1e-10))

    # Phase features
    phase = np.angle(f_shift)
    phase_std = float(np.std(phase))
    phase_hist, _ = np.histogram(phase.flatten(), bins=64, density=True)
    phase_hist = phase_hist + 1e-10
    phase_entropy = float(-np.sum(phase_hist * np.log(phase_hist)))

    # Peak ratio — detects GAN grid artifacts
    mag_norm = magnitude / (magnitude.max() + 1e-10)
    fft_peak_ratio = float(np.sum(mag_norm > 0.5) / magnitude.size)

    # 5 equal frequency bands: energy mean + std
    band_edges = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    band_features = {}
    for i in range(5):
        if i == 0:
            mask = dist <= max_dist * band_edges[i + 1]
        else:
            mask = (dist > max_dist * band_edges[i]) & (dist <= max_dist * band_edges[i + 1])
        vals = np.log1p(magnitude[mask]) if mask.any() else np.array([0.0])
        band_features[f"fft_b{i}_energy"] = float(np.mean(vals))
        band_features[f"fft_b{i}_std"] = float(np.std(vals))

    result = {
        "fft_low_energy": low_energy,
        "fft_mid_energy": mid_energy,
        "fft_high_energy": high_energy,
        "fft_high_low_ratio": ratio,
        "phase_std": phase_std,
        "phase_entropy": phase_entropy,
        "fft_peak_ratio": fft_peak_ratio,
    }
    result.update(band_features)
    return result


# ============================================================
# 2. TEXTURE FEATURES (GLCM)
# ============================================================

def extract_glcm_features(patch_y, levels=64):
    """
    Trích đặc trưng texture bằng GLCM (Gray-Level Co-occurrence Matrix).

    - Lượng tử hóa kênh Y về `levels` mức (default 64)
    - GLCM với distances=[1,2,4] và 4 hướng
    - Mỗi property: mean và std over angles (std = anisotropy indicator)
    - Properties: contrast, homogeneity, energy, correlation, dissimilarity
    - Return: dict với 30 features (5 props × 3 distances × 2 stats)
    """
    patch_q = (patch_y / 256.0 * levels).astype(np.uint8)
    patch_q = np.clip(patch_q, 0, levels - 1)

    distances = [1, 2, 4]
    angles = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
    properties = ["contrast", "homogeneity", "energy", "correlation", "dissimilarity"]

    glcm = graycomatrix(
        patch_q,
        distances=distances,
        angles=angles,
        levels=levels,
        symmetric=True,
        normed=True,
    )

    result = {}
    for prop in properties:
        vals = graycoprops(glcm, prop)  # shape (len(distances), len(angles))
        for d_idx, d in enumerate(distances):
            result[f"glcm_{prop}_d{d}_mean"] = float(np.mean(vals[d_idx, :]))
            result[f"glcm_{prop}_d{d}_std"] = float(np.std(vals[d_idx, :]))
    return result


# ============================================================
# 3. RESIDUAL FEATURES
# ============================================================

def extract_residual_features(patch_y):
    """
    Trích đặc trưng từ residual — nhấn mạnh cạnh, nhiễu, chi tiết mảnh.

    - Laplacian residual (4 features, backward compat)
    - SRM (Steganalysis Rich Model) residual (3 features)
    - Gaussian noise residual + autocorrelation (3 features)
    - Return: dict với 10 features
    """
    # Laplacian
    residual = cv2.Laplacian(patch_y.astype(np.float64), cv2.CV_64F)
    res_std = float(np.std(residual))
    res_mean = float(np.mean(np.abs(residual)))
    res_kurtosis = float(sp_stats.kurtosis(residual.flatten()))
    hist, _ = np.histogram(residual.flatten(), bins=64, density=True)
    hist = hist[hist > 0]
    res_entropy = float(sp_stats.entropy(hist))

    # SRM residual
    kernel_srm = np.array(
        [
            [0, 0, -1, 0, 0],
            [0, -1, 2, -1, 0],
            [-1, 2, -4, 2, -1],
            [0, -1, 2, -1, 0],
            [0, 0, -1, 0, 0],
        ],
        dtype=np.float64,
    ) / 4.0
    srm_residual = cv2.filter2D(patch_y.astype(np.float64), -1, kernel_srm)
    srm_std = float(np.std(srm_residual))
    srm_mean = float(np.mean(np.abs(srm_residual)))
    srm_kurtosis = float(sp_stats.kurtosis(srm_residual.flatten()))

    # Gaussian noise residual
    blurred = cv2.GaussianBlur(patch_y.astype(np.float64), (5, 5), 1.0)
    noise = patch_y.astype(np.float64) - blurred
    noise_std = float(np.std(noise))
    noise_mean = float(np.mean(np.abs(noise)))

    # Vertical autocorrelation of noise
    try:
        n1 = noise[:-1, :].flatten()
        n2 = noise[1:, :].flatten()
        if np.std(n1) < 1e-10 or np.std(n2) < 1e-10:
            noise_autocorr = 0.0
        else:
            noise_autocorr = float(np.corrcoef(n1, n2)[0, 1])
    except Exception:
        noise_autocorr = 0.0
    noise_autocorr = float(np.nan_to_num(noise_autocorr, nan=0.0, posinf=0.0, neginf=0.0))

    return {
        "residual_std": res_std,
        "residual_mean": res_mean,
        "residual_kurtosis": res_kurtosis,
        "residual_entropy": res_entropy,
        "srm_std": srm_std,
        "srm_mean": srm_mean,
        "srm_kurtosis": srm_kurtosis,
        "noise_std": noise_std,
        "noise_mean": noise_mean,
        "noise_autocorr": noise_autocorr,
    }


# ============================================================
# 4. COLOR CORRELATION FEATURES
# ============================================================

def extract_color_features(patch_ycrcb):
    """
    Trích đặc trưng màu từ 3 kênh Y, Cr, Cb.

    - 3 correlation coefficients (backward compat)
    - Thống kê riêng cho kênh Cr và Cb
    - Return: dict với 11 features
    """
    y_ch = patch_ycrcb[:, :, 0].flatten().astype(np.float64)
    cr_ch = patch_ycrcb[:, :, 1].flatten().astype(np.float64)
    cb_ch = patch_ycrcb[:, :, 2].flatten().astype(np.float64)

    def safe_corr(a, b):
        if np.std(a) < 1e-10 or np.std(b) < 1e-10:
            return 0.0
        return float(np.nan_to_num(np.corrcoef(a, b)[0, 1], nan=0.0))

    return {
        "corr_y_cr": safe_corr(y_ch, cr_ch),
        "corr_y_cb": safe_corr(y_ch, cb_ch),
        "corr_cr_cb": safe_corr(cr_ch, cb_ch),
        "cr_mean": float(np.mean(cr_ch)),
        "cr_std": float(np.std(cr_ch)),
        "cr_skew": float(sp_stats.skew(cr_ch)),
        "cr_kurtosis": float(sp_stats.kurtosis(cr_ch)),
        "cb_mean": float(np.mean(cb_ch)),
        "cb_std": float(np.std(cb_ch)),
        "cb_skew": float(sp_stats.skew(cb_ch)),
        "cb_kurtosis": float(sp_stats.kurtosis(cb_ch)),
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
        "stat_mean": float(np.mean(pixels)),
        "stat_std": float(np.std(pixels)),
        "stat_skew": float(sp_stats.skew(pixels)),
        "stat_kurtosis": float(sp_stats.kurtosis(pixels)),
    }


# ============================================================
# 6. DCT FEATURES
# ============================================================

def extract_dct_features(patch_y):
    """
    Trích đặc trưng từ DCT 2D của kênh Y.

    - Tính DCT 2D, tách DC và AC coefficients
    - dc_energy: năng lượng DC (index 0)
    - ac: tất cả coefficients sau khi zero-out DC
    - Return: dict với 4 features: dct_kurtosis, dct_sparsity, dct_energy_ratio, dct_std
    """
    dct_coeffs = dctn(patch_y.astype(np.float64), norm="ortho")
    ac = dct_coeffs.flatten().copy()
    dc_energy = float(ac[0] ** 2)
    ac[0] = 0.0  # zero-out DC component, ac now holds only AC coefficients

    ac_energy = float(np.sum(ac ** 2))
    dct_kurtosis = float(np.nan_to_num(sp_stats.kurtosis(ac), nan=0.0))
    dct_sparsity = float(np.sum(np.abs(ac) < 1.0) / len(ac))
    dct_energy_ratio = float(ac_energy / (dc_energy + 1e-10))
    dct_std = float(np.std(ac))

    return {
        "dct_kurtosis": dct_kurtosis,
        "dct_sparsity": dct_sparsity,
        "dct_energy_ratio": dct_energy_ratio,
        "dct_std": dct_std,
    }


# ============================================================
# 7. LBP FEATURES
# ============================================================

def extract_lbp_features(patch_y):
    """
    Trích đặc trưng Local Binary Pattern từ kênh Y.

    - 3 cặp (P, R): (8,1), (16,2), (24,3), method='uniform'
    - Mỗi cặp: entropy và uniformity của histogram LBP
    - Return: dict với 6 features
    """
    result = {}
    for r_idx, (P, R) in enumerate([(8, 1), (16, 2), (24, 3)], 1):
        lbp = local_binary_pattern(patch_y, P=P, R=R, method="uniform")
        n_bins = P + 2
        hist, _ = np.histogram(lbp.flatten(), bins=n_bins, range=(0, n_bins), density=True)
        hist = hist + 1e-10
        entropy = float(-np.sum(hist * np.log(hist)))
        uniformity = float(np.sum(hist ** 2))
        result[f"lbp_r{r_idx}_entropy"] = entropy
        result[f"lbp_r{r_idx}_uniformity"] = uniformity
    return result


# ============================================================
# 8. PATCH EXTRACTION & AGGREGATION
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
    features.update(extract_dct_features(patch_y))
    features.update(extract_lbp_features(patch_y))

    return features


def extract_image_features(img_ycrcb, patch_size=64):
    """
    Trích đặc trưng cho toàn bộ 1 ảnh YCrCb 256×256.

    Pipeline:
      1. Chia ảnh thành grid patches (256/64 = 4×4 = 16 patches)
      2. Trích features mỗi patch
      3. Gộp 16 patches: mean, std, p90, min, max, p10
      4. Thêm 4 spatial comparison features (center vs corner patches)

    Returns: dict (feature_name → float)
    """
    h, w = img_ycrcb.shape[:2]

    all_patch_features = []
    for i in range(0, h, patch_size):
        for j in range(0, w, patch_size):
            patch = img_ycrcb[i:i + patch_size, j:j + patch_size]
            if patch.shape[0] == patch_size and patch.shape[1] == patch_size:
                pf = extract_patch_features(patch)
                all_patch_features.append(pf)

    if not all_patch_features:
        return {}

    feature_names = list(all_patch_features[0].keys())
    aggregated = {}

    for fname in feature_names:
        values = np.array([pf[fname] for pf in all_patch_features], dtype=np.float64)
        values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
        aggregated[f"{fname}_mean"] = float(np.mean(values))
        aggregated[f"{fname}_std"] = float(np.std(values))
        aggregated[f"{fname}_p90"] = float(np.percentile(values, 90))
        aggregated[f"{fname}_min"] = float(np.min(values))
        aggregated[f"{fname}_max"] = float(np.max(values))
        aggregated[f"{fname}_p10"] = float(np.percentile(values, 10))

    # Spatial comparison: center patches vs corner patches (4×4 grid)
    # Center: indices 5,6,9,10 — Corner: indices 0,3,12,15
    spatial_features = [
        "fft_high_low_ratio",
        "residual_std",
        "glcm_contrast_d1_mean",
        "noise_std",
    ]
    center_idx = [5, 6, 9, 10]
    corner_idx = [0, 3, 12, 15]

    for fname in spatial_features:
        if len(all_patch_features) >= 16 and fname in all_patch_features[0]:
            center_val = float(np.mean([all_patch_features[i][fname] for i in center_idx]))
            corner_val = float(np.mean([all_patch_features[i][fname] for i in corner_idx]))
            diff = float(np.nan_to_num(center_val - corner_val, nan=0.0, posinf=0.0, neginf=0.0))
        else:
            diff = 0.0
        aggregated[f"spatial_center_vs_corner_{fname}_diff"] = diff

    return aggregated


def get_feature_names(patch_size=64):
    """Trả về danh sách tên features (để lưu vào file)."""
    dummy = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
    features = extract_image_features(dummy, patch_size)
    return list(features.keys())
