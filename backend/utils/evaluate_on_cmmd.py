import csv
from pathlib import Path
import cv2
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix

from backend.infer import YOLOONNX
from backend.models.lesion_classifier import LesionClassifierONNX

# =====================
# PATHS
# =====================
BASE_DIR = Path(__file__).resolve().parents[2]
DATASETS = BASE_DIR / "datasets"

# =====================
# MODELS
# =====================
model_breast = YOLOONNX(
    BASE_DIR / "backend/models/global_breast_ir9.onnx",
    imgsz=1024,
    conf_thres=0.5,
)

model_lesion = YOLOONNX(
    BASE_DIR / "backend/models/small_tumor_best_ir9.onnx",
    imgsz=1024,
    conf_thres=0.25,
)

classifier = LesionClassifierONNX(
    BASE_DIR / "backend/models/lesion_effb0_cls.onnx"
)

# =====================
# HELPERS
# =====================
def load_image(path: Path):
    return cv2.imread(str(path), cv2.IMREAD_COLOR)


# =====================
# EXPERIMENT 1 — BREAST ROI
# =====================
def eval_breast_roi():
    images_dir = DATASETS / "CMMD_breast_roi/images/val"
    labels_dir = DATASETS / "CMMD_breast_roi/labels/val"

    total, detected = 0, 0

    for img_path in images_dir.glob("*.png"):
        label_path = labels_dir / f"{img_path.stem}.txt"
        if not label_path.exists():
            continue

        img = load_image(img_path)
        if img is None:
            continue

        total += 1
        preds = model_breast.predict(img)
        if len(preds) > 0:
            detected += 1

    recall = detected / total if total else 0

    print("\n=== EXPERIMENT 1: BREAST ROI ===")
    print(f"Samples: {total}")
    print(f"Recall: {recall:.3f}")

    return recall


# =====================
# EXPERIMENT 2 — LESION ROI
# =====================
def eval_lesion_roi():
    images_dir = DATASETS / "CMMD_small_roi/images/val"
    labels_dir = DATASETS / "CMMD_small_roi/labels/val"

    total, detected = 0, 0

    for img_path in images_dir.glob("*.png"):
        label_path = labels_dir / f"{img_path.stem}.txt"
        if not label_path.exists():
            continue

        img = load_image(img_path)
        if img is None:
            continue

        total += 1
        preds = model_lesion.predict(img)
        if len(preds) > 0:
            detected += 1

    recall = detected / total if total else 0

    print("\n=== EXPERIMENT 2: LESION ROI ===")
    print(f"Samples: {total}")
    print(f"Recall: {recall:.3f}")

    return recall


# =====================
# EXPERIMENT 3 — CLASSIFICATION
# =====================
def eval_classification():
    dataset = DATASETS / "CMMD_lesion_cls"
    images_root = dataset / "images"
    labels_csv = dataset / "labels.csv"

    y_true, y_pred = [], []

    with open(labels_csv) as f:
        reader = csv.DictReader(f)
        rows = [r for r in reader if "/val/" in r["path"]]

    for r in rows:
        img_path = dataset / r["path"]
        if not img_path.exists():
            continue

        img = load_image(img_path)
        if img is None:
            continue

        gt = int(r["label"])
        pred = classifier.predict(img)
        pred_label = 1 if pred["label"] == "mass" else 0

        y_true.append(gt)
        y_pred.append(pred_label)

    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)

    print("\n=== EXPERIMENT 3: CLASSIFICATION ===")
    print(f"Samples: {len(y_true)}")
    print(f"Accuracy: {acc:.3f}")
    print("Confusion matrix:")
    print(cm)

    return acc, cm


# =====================
# RUN ALL
# =====================
if __name__ == "__main__":
    eval_breast_roi()
    eval_lesion_roi()
    eval_classification()
