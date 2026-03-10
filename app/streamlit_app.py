"""
Streamlit App — AI Image Detector Demo
Upload ảnh → Preprocess → Extract features → Predict Real/Fake
"""

import os
import sys
import cv2
import numpy as np
import joblib
import streamlit as st
from PIL import Image, ImageOps

# Import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.features import extract_image_features

# ============ CONFIG ============
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'model.pkl')
FEATURE_NAMES_PATH = os.path.join(os.path.dirname(__file__), '..', 'features', 'feature_names.txt')
IMG_SIZE = 256
PATCH_SIZE = 64


@st.cache_resource
def load_model():
    """Load trained model."""
    return joblib.load(MODEL_PATH)


def preprocess_image(pil_img):
    """
    Preprocess ảnh upload: fix orientation → strip metadata → YCrCb → center crop.
    Returns: numpy array (256, 256, 3) YCrCb hoặc None nếu lỗi.
    """
    # Fix EXIF orientation
    pil_img = ImageOps.exif_transpose(pil_img)
    pil_img = pil_img.convert("RGB")
    img_rgb = np.array(pil_img)

    h, w = img_rgb.shape[:2]
    if h < IMG_SIZE or w < IMG_SIZE:
        return None, f"Ảnh quá nhỏ ({w}×{h}). Cần ít nhất {IMG_SIZE}×{IMG_SIZE}."

    # RGB → BGR → YCrCb
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    img_ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)

    # Center crop
    cy, cx = h // 2, w // 2
    half = IMG_SIZE // 2
    cropped = img_ycrcb[cy - half:cy + half, cx - half:cx + half]

    return cropped, None


def predict_image(img_ycrcb, model):
    """Trích features → predict."""
    features = extract_image_features(img_ycrcb, patch_size=PATCH_SIZE)
    
    # Load feature names để đảm bảo đúng thứ tự
    with open(FEATURE_NAMES_PATH) as f:
        feature_names = f.read().strip().split("\n")
    
    # Tạo vector theo đúng thứ tự
    X = np.array([[features.get(fn, 0.0) for fn in feature_names]])
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    pred = model.predict(X)[0]
    proba = model.predict_proba(X)[0]
    
    return pred, proba


# ============ APP UI ============
st.set_page_config(page_title="AI Image Detector", page_icon="🔍", layout="centered")

st.title("🔍 AI Image Detector")
st.markdown("Upload ảnh để kiểm tra: **Real** (ảnh thật) hay **Fake** (AI tạo ra)?")

uploaded = st.file_uploader("Chọn ảnh (JPG, PNG)", type=["jpg", "jpeg", "png"])

if uploaded is not None:
    # Hiển thị ảnh
    pil_img = Image.open(uploaded)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.image(pil_img, caption="Ảnh upload", use_container_width=True)
        st.caption(f"Kích thước: {pil_img.size[0]}×{pil_img.size[1]}")
    
    with col2:
        with st.spinner("Đang phân tích..."):
            # Preprocess
            img_ycrcb, error = preprocess_image(pil_img)
            
            if error:
                st.error(error)
            else:
                # Load model & predict
                model = load_model()
                pred, proba = predict_image(img_ycrcb, model)
                
                label = "🟢 REAL (Ảnh thật)" if pred == 0 else "🔴 FAKE (AI tạo)"
                confidence = proba[pred] * 100
                
                st.markdown(f"### {label}")
                st.metric("Confidence", f"{confidence:.1f}%")
                
                # Progress bars
                st.markdown("**Chi tiết xác suất:**")
                st.progress(float(proba[0]), text=f"Real: {proba[0]*100:.1f}%")
                st.progress(float(proba[1]), text=f"Fake: {proba[1]*100:.1f}%")

st.markdown("---")
st.markdown("*AI Image Detector — Dự án môn Tiền xử lí dữ liệu*")
