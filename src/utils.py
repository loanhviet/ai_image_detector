import cv2
import numpy as np
import pandas as pd


def center_crop(img, size=256):
    """Crop vùng trung tâm của ảnh về kích thước size×size."""
    h, w = img.shape[:2]
    cy, cx = h // 2, w // 2
    half = size // 2
    return img[cy - half:cy + half, cx - half:cx + half]


def split_patches(img, patch_size=64):
    """Chia ảnh thành grid các patch vuông.

    Returns:
        list of numpy arrays, mỗi patch có shape (patch_size, patch_size, C)
    """
    h, w = img.shape[:2]
    patches = []
    for i in range(0, h, patch_size):
        for j in range(0, w, patch_size):
            patch = img[i:i + patch_size, j:j + patch_size]
            if patch.shape[0] == patch_size and patch.shape[1] == patch_size:
                patches.append(patch)
    return patches


def load_manifest(csv_path):
    """Đọc manifest CSV trả về DataFrame."""
    return pd.read_csv(csv_path)
