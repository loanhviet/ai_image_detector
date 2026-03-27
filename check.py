# check_exif_report.py
import os, random
from PIL import Image
from PIL.ExifTags import TAGS
from collections import defaultdict
import pandas as pd

def get_exif_fields(img_path):
    try:
        img = Image.open(img_path)
        exif_data = img._getexif()
        if exif_data is None:
            return {}
        return {TAGS.get(tag, tag): str(val)[:80] for tag, val in exif_data.items()}
    except Exception:
        return {}

def sample_and_analyze(root_dir, sample_per_folder=300):
    """
    Cấu trúc: root/real/generator/*.jpg  và  root/fake/generator/*.jpg
    """
    results = []
    field_counter = defaultdict(int)

    for label in ["real", "fake"]:
        label_path = os.path.join(root_dir, label)
        if not os.path.isdir(label_path):
            print(f"Không tìm thấy thư mục: {label_path}")
            continue

        for generator in sorted(os.listdir(label_path)):
            gen_path = os.path.join(label_path, generator)
            if not os.path.isdir(gen_path):
                continue

            files = [f for f in os.listdir(gen_path)
                     if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            if not files:
                continue

            sample = random.sample(files, min(sample_per_folder, len(files)))
            has_exif_count = 0

            for fname in sample:
                fpath = os.path.join(gen_path, fname)
                fields = get_exif_fields(fpath)
                has_any = len(fields) > 0
                if has_any:
                    has_exif_count += 1
                    for k in fields:
                        field_counter[k] += 1

                results.append({
                    "label": label,
                    "generator": generator,
                    "file": fname,
                    "has_exif": has_any,
                    "num_fields": len(fields),
                    "Make": fields.get("Make", ""),
                    "Model": fields.get("Model", ""),
                    "Software": fields.get("Software", ""),
                    "DateTime": fields.get("DateTime", ""),
                    "GPSInfo": "yes" if "GPSInfo" in fields else "",
                })

            pct = has_exif_count / len(sample) * 100
            print(f"  [{label}] {generator}: {has_exif_count}/{len(sample)} có EXIF ({pct:.1f}%)")

    return pd.DataFrame(results), field_counter

# ==============================
ROOT = r"E:\ai_image_detector\data\raw"  # <-- giữ nguyên path của bạn
# ==============================

df, field_counter = sample_and_analyze(ROOT, sample_per_folder=300)

# --- Tổng quan ---
print("\n========== EXIF SUMMARY ==========")
print(f"Tổng ảnh đã sample : {len(df)}")
print(f"Có EXIF            : {df['has_exif'].sum()} ({df['has_exif'].mean()*100:.1f}%)")
print(f"Không có EXIF      : {(~df['has_exif']).sum()} ({(~df['has_exif']).mean()*100:.1f}%)")

print("\n--- Theo label (real vs fake) ---")
print(df.groupby('label')['has_exif']
        .agg(có_exif='sum', total='count')
        .assign(pct=lambda x: (x['có_exif']/x['total']*100).round(1)))

print("\n--- Theo generator ---")
print(df.groupby(['label','generator'])['has_exif']
        .agg(có_exif='sum', total='count')
        .assign(pct=lambda x: (x['có_exif']/x['total']*100).round(1))
        .to_string())

print("\n--- Top 15 trường EXIF phổ biến nhất ---")
for field, count in sorted(field_counter.items(), key=lambda x: -x[1])[:15]:
    print(f"  {field:<25}: {count}")

# --- Export ---
df.to_csv("exif_report.csv", index=False)
print("\n✓ Đã lưu: exif_report.csv")