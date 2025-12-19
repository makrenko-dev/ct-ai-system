from pathlib import Path
import json
import cv2
import random
import csv
import shutil
from tqdm import tqdm

# =====================
# PATHS
# =====================

BASE = Path.home() / "Documents" / "uni" / "ct-ai-system" / "datasets"

GLOBAL_IMAGES = BASE / "CMMD_global_roi" / "images"
GLOBAL_LABELS = BASE / "CMMD_global_roi" / "labels"
SEGMENTATIONS = BASE / "TOMPEI" / "segmentations" / "TOMPEI-CMMD_v01_20250123"

OUT = BASE / "CMMD_lesion_cls"
OUT_IMAGES = OUT / "images"
OUT_IMAGES_TRAIN = OUT_IMAGES / "train"
OUT_IMAGES_VAL = OUT_IMAGES / "val"

TRAIN_SPLIT = 0.8
IMG_SIZE = 224

LABEL_MAP = {
    "calc": 0,
    "calcification": 0,
    "mass": 1,
}

# =====================
# UTILS
# =====================

def yolo_to_bbox(txt: str, w: int, h: int):
    _, xc, yc, bw, bh = map(float, txt.split())
    x1 = int((xc - bw / 2) * w)
    y1 = int((yc - bh / 2) * h)
    x2 = int((xc + bw / 2) * w)
    y2 = int((yc + bh / 2) * h)
    return x1, y1, x2, y2


def polygon_to_bbox(points):
    xs = [p["x"] for p in points]
    ys = [p["y"] for p in points]
    return int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))


# =====================
# MAIN
# =====================

def main():
    # ✅ clean rebuild
    if OUT.exists():
        shutil.rmtree(OUT)

    OUT_IMAGES_TRAIN.mkdir(parents=True, exist_ok=True)
    OUT_IMAGES_VAL.mkdir(parents=True, exist_ok=True)

    rows = []
    idx = 0

    files = list(SEGMENTATIONS.glob("*.json"))
    random.shuffle(files)
    split = int(len(files) * TRAIN_SPLIT)

    for i, js in enumerate(tqdm(files, desc="Lesion crops")):
        data = json.loads(js.read_text())
        patient_id = js.stem.split("_")[0]

        img_path = (
            next((GLOBAL_IMAGES / "train").glob(f"{patient_id}.png"), None)
            or next((GLOBAL_IMAGES / "val").glob(f"{patient_id}.png"), None)
        )
        if img_path is None:
            continue

        img = cv2.imread(str(img_path))
        H, W = img.shape[:2]

        lab_path = (
            next((GLOBAL_LABELS / "train").glob(f"{patient_id}.txt"), None)
            or next((GLOBAL_LABELS / "val").glob(f"{patient_id}.txt"), None)
        )
        if lab_path is None:
            continue

        gx1, gy1, gx2, gy2 = yolo_to_bbox(lab_path.read_text(), W, H)
        breast_crop = img[gy1:gy2, gx1:gx2]
        bh, bw = breast_crop.shape[:2]

        for ann in data:
            label_raw = ann.get("label", "").strip().lower()
            if label_raw not in LABEL_MAP:
                continue

            lx1, ly1, lx2, ly2 = polygon_to_bbox(ann["cgPoints"])
            lx1 -= gx1
            ly1 -= gy1
            lx2 -= gx1
            ly2 -= gy1

            if lx2 <= 0 or ly2 <= 0 or lx1 >= bw or ly1 >= bh:
                continue

            lx1 = max(0, lx1)
            ly1 = max(0, ly1)
            lx2 = min(bw, lx2)
            ly2 = min(bh, ly2)

            crop = breast_crop[ly1:ly2, lx1:lx2]
            if crop.size < 500:
                continue

            crop = cv2.resize(crop, (IMG_SIZE, IMG_SIZE))
            subset = "train" if i < split else "val"

            out_img = (OUT_IMAGES_TRAIN if subset == "train" else OUT_IMAGES_VAL) / f"{idx:06d}.png"
            cv2.imwrite(str(out_img), crop)

            rows.append([
                f"images/{subset}/{out_img.name}",
                LABEL_MAP[label_raw]
            ])
            idx += 1

    with open(OUT / "labels.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "label"])
        writer.writerows(rows)

    print(f"✅ DONE. Samples: {len(rows)}")


if __name__ == "__main__":
    main()
