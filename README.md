# 🔍 AI Image Detector

Phát hiện ảnh do AI tạo ra (AI-Generated Image Detection) dựa trên đặc trưng thống kê.

## 📁 Cấu trúc dự án

```
ai_image_detector/
├── README.md
├── requirements.txt
│
├── data/
│   ├── raw/
│   │   ├── real/          # Ảnh gốc (7 generators)
│   │   └── fake/          # Ảnh AI-generated (7 generators)
│   └── processed/
│       ├── real/           # Ảnh đã tiền xử lí (256×256 YCrCb PNG)
│       ├── fake/
│       ├── manifest.csv
│       ├── manifest_train.csv
│       └── manifest_test.csv
│
├── notebooks/
│   ├── 01_preprocessing.ipynb      # EDA + Tiền xử lí dữ liệu ★
│   ├── 02_feature_extraction.ipynb # Trích xuất đặc trưng
│   └── 03_training_eval.ipynb      # Training & Evaluation
│
├── src/
│   ├── __init__.py
│   ├── utils.py        # Hàm tiện ích (center_crop, split_patches)
│   ├── preprocess.py   # Pipeline tiền xử lí ảnh
│   ├── features.py     # 6 nhóm đặc trưng (FFT, GLCM, Residual, Color, Stat, Aggregation)
│   └── train.py        # Training & Evaluation functions
│
├── features/
│   ├── X_train.npy, X_test.npy    # Feature matrices
│   ├── y_train.npy, y_test.npy    # Labels
│   └── feature_names.txt          # Tên features
│
├── models/
│   └── model.pkl       # Best trained model
│
└── app/
    └── streamlit_app.py  # Demo web app
```

## 🚀 Pipeline

### 1. Tiền xử lí dữ liệu (Notebook 01) ★

```
Ảnh gốc → Fix EXIF orientation → Strip metadata → BGR→YCrCb → Center crop 256×256 → Lưu PNG
```

- **EDA**: thống kê số lượng, kiểm tra kích thước, trùng lặp, pixel statistics
- **Cleaning**: fix orientation, strip metadata, chuyển color space, center crop
- **Validation**: kiểm tra output, class balance, stratified train/test split

### 2. Trích xuất đặc trưng (Notebook 02)

Chia ảnh thành **16 patches** (64×64), trích **5 nhóm features**:

| Nhóm        | Features                                   | Ý nghĩa               |
| ----------- | ------------------------------------------ | --------------------- |
| FFT         | low/mid/high energy, ratio                 | Phân tích phổ tần số  |
| GLCM        | contrast, homogeneity, energy, correlation | Kết cấu bề mặt        |
| Residual    | std, mean, kurtosis, entropy               | Phần dư (cạnh, nhiễu) |
| Color       | corr(Y,Cr), corr(Y,Cb), corr(Cr,Cb)        | Tương quan màu        |
| Statistical | mean, std, skew, kurtosis                  | Thống kê pixel        |

Gộp 16 patches bằng mean/std/p90 → **57 features** mỗi ảnh.

### 3. Training (Notebook 03)

- **Random Forest** + **XGBoost** với GridSearchCV + 5-fold CV
- So sánh: classification report, confusion matrix, ROC curve
- Feature importance analysis
- Per-generator accuracy breakdown

## ⚙️ Cài đặt

```bash
pip install -r requirements.txt
```

## 🎯 Chạy

### Chạy notebooks theo thứ tự:

1. `notebooks/01_preprocessing.ipynb` — Tiền xử lí dữ liệu
2. `notebooks/02_feature_extraction.ipynb` — Trích xuất đặc trưng
3. `notebooks/03_training_eval.ipynb` — Training & Evaluation

### Chạy demo app:

```bash
streamlit run app/streamlit_app.py
```

## 📊 Dữ liệu

- **7 AI generators**: ADM, GLIDE, Midjourney, SDv14, SDv15, VQDM, Wukong
- Mỗi generator có ảnh **real** và **fake** tương ứng
- Phân loại **binary**: Real (0) vs Fake (1)

## 🛠️ Công nghệ

- Python 3.11+
- OpenCV, NumPy, Pandas, PIL
- scikit-image (GLCM), SciPy (FFT, statistics)
- scikit-learn (Random Forest), XGBoost
- Streamlit (demo app)
- Matplotlib, Seaborn (visualization)
