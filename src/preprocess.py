import cv2
import numpy as np
from PIL import Image, ImageOps
import os


def process_image(src_path, dst_path, img_size=256, min_size=64):
    """
    Xử lý 1 ảnh: đọc → kiểm tra → fix orientation → xóa metadata
    → chuyển YCrCb → center crop → lưu PNG.

    Returns:
        (success: bool, reason: str)
    """
    try:
        # B1: Đọc bằng PIL để xử lý EXIF
        pil_img = Image.open(src_path)

        # B2: Fix EXIF orientation (ảnh chụp xoay theo EXIF)
        pil_img = ImageOps.exif_transpose(pil_img)

        # B3: Convert sang RGB (bỏ alpha channel nếu có, xử lý grayscale)
        pil_img = pil_img.convert("RGB")

        # B4: Chuyển sang numpy array — lúc này đã strip metadata
        img_rgb = np.array(pil_img)

        # B5: Kiểm tra kích thước tối thiểu
        h, w = img_rgb.shape[:2]
        if h < min_size or w < min_size:
            return False, "quá nhỏ"

        if h < img_size or w < img_size:
            return False, f"nhỏ hơn {img_size}"

        # B6: Chuyển RGB → BGR (OpenCV format) → YCrCb
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        img_ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)

        # B7: Center crop
        cy, cx = h // 2, w // 2
        half = img_size // 2
        cropped = img_ycrcb[cy - half:cy + half, cx - half:cx + half]

        # B8: Lưu PNG (lossless)
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        cv2.imwrite(dst_path, cropped)
        return True, "ok"

    except Exception as e:
        return False, str(e)


def process_image_multicrop(src_path, dst_dir, img_size=256, min_size=64):
    """
    Xử lý 1 ảnh và tạo 5 crops (center, top-left, top-right, bottom-left, bottom-right).

    Mỗi crop được chuyển sang YCrCb và lưu thành file PNG riêng trong dst_dir:
      center.png, tl.png, tr.png, bl.png, br.png

    Returns:
        (success: bool, reason: str)
    """
    try:
        pil_img = Image.open(src_path)
        pil_img = ImageOps.exif_transpose(pil_img)
        pil_img = pil_img.convert("RGB")
        img_rgb = np.array(pil_img)

        h, w = img_rgb.shape[:2]
        if h < min_size or w < min_size:
            return False, "quá nhỏ"

        if h < img_size or w < img_size:
            return False, f"nhỏ hơn {img_size}"

        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        img_ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)

        cy, cx = h // 2, w // 2
        half = img_size // 2

        crops = {
            "center": img_ycrcb[cy - half:cy + half, cx - half:cx + half],
            "tl": img_ycrcb[0:img_size, 0:img_size],
            "tr": img_ycrcb[0:img_size, w - img_size:w],
            "bl": img_ycrcb[h - img_size:h, 0:img_size],
            "br": img_ycrcb[h - img_size:h, w - img_size:w],
        }

        os.makedirs(dst_dir, exist_ok=True)
        for name, crop in crops.items():
            cv2.imwrite(os.path.join(dst_dir, f"{name}.png"), crop)

        return True, "ok"

    except Exception as e:
        return False, str(e)


def load_image_ycrcb(path):
    """
    Đọc file PNG YCrCb đã lưu từ disk.

    Returns:
        numpy array shape (H, W, 3) dtype uint8
    """
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Không thể đọc ảnh: {path}")
    return img
