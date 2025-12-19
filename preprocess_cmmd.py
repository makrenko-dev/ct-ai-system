import os
from pathlib import Path
import random
import cv2
import numpy as np
import pydicom
from tqdm import tqdm

# =====================
# PATHS
# =====================

BASE = Path.home() / "Documents" / "uni" / "ct-ai-system" / "datasets"

DICOM_ROOT = BASE / "CMMD" / "dicom" / "manifest-1616439774456" / "CMMD"
OUT_ROOT = BASE / "CMMD_breast_roi"

IMAGES_OUT = OUT_ROOT / "images"
LABELS_OUT = OUT_ROOT / "labels"

TRAIN_SPLIT = 0.8

# =====================
# DICOM → PNG
# =====================

def read_dicom(path: Path) -> np.ndarray:
    ds = pydicom.dcmread(str(path))
    img = ds.pixel_array.astype(np.float32)

    img -= img.min()
    img /= (img.max() + 1e-6)
    img *= 255

    return img.astype(np.uint8)

# =====================
# BREAST MASK
# =====================

def get_breast_mask(img: np.ndarray) -> np.ndarray:
    """
    Простая, но устойчивая морфологическая маска груди
    """
    _, mask = cv2.threshold(img, 10, 255, cv2.THRESH_BINARY)

    kernel = np.ones((25, 25), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    return mask

def mask_to_bbox(mask: np.ndarray):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    return xs.min(), ys.min(), xs.max(), ys.max()

# =====================
# YOLO LABEL
# =====================

def save_yolo_label(path, bbox, w, h):
    x1, y1, x2, y2 = bbox
    xc = ((x1 + x2) / 2) / w
    yc = ((y1 + y2) / 2) / h
    bw = (x2 - x1) / w
    bh = (y2 - y1) / h
    path.write_text(f"0 {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")

# =====================
# MAIN
# =====================

def main():
    IMAGES_OUT.mkdir(parents=True, exist_ok=True)
    LABELS_OUT.mkdir(parents=True, exist_ok=True)

    patients = sorted(DICOM_ROOT.glob("D*"))
    random.shuffle(patients)

    split = int(len(patients) * TRAIN_SPLIT)

    for i, patient_dir in enumerate(tqdm(patients)):
        # берем только MLO
        dicoms = list(patient_dir.rglob("*.dcm"))
        dicom = None

        for d in dicoms:
            try:
                ds = pydicom.dcmread(d, stop_before_pixels=True)
                if hasattr(ds, "ViewCodeSequence"):
                    meaning = ds.ViewCodeSequence[0].CodeMeaning.lower()
                    if "medio" in meaning and "oblique" in meaning:
                        dicom = d
                        break
            except:
                continue

        if dicom is None:
            continue

        img = read_dicom(dicom)
        h, w = img.shape

        mask = get_breast_mask(img)
        bbox = mask_to_bbox(mask)

        if bbox is None:
            continue

        subset = "train" if i < split else "val"
        (IMAGES_OUT / subset).mkdir(parents=True, exist_ok=True)
        (LABELS_OUT / subset).mkdir(parents=True, exist_ok=True)

        name = patient_dir.name
        cv2.imwrite(str(IMAGES_OUT / subset / f"{name}.png"), img)
        save_yolo_label(LABELS_OUT / subset / f"{name}.txt", bbox, w, h)

    print("✅ Breast ROI dataset ready:", OUT_ROOT)

if __name__ == "__main__":
    main()
