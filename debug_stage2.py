import cv2
from pathlib import Path

from backend.infer import YOLOONNX

# -------- PATHS --------

BASE = Path.home() / "Documents" / "uni" / "ct-ai-system" / "datasets"

IMG = BASE / "CMMD_small_roi/images/train/D1-0001.png"
LBL = BASE / "CMMD_small_roi/labels/train/D1-0001.txt"

MODEL = Path("backend/models/small_tumor_best_ir9.onnx")

# -------- LOAD --------

img = cv2.imread(str(IMG))
h, w = img.shape[:2]

print("IMG shape:", img.shape)

# GT bbox (YOLO → pixel)
cls, xc, yc, bw, bh = map(float, LBL.read_text().split())

gt_x1 = int((xc - bw / 2) * w)
gt_y1 = int((yc - bh / 2) * h)
gt_x2 = int((xc + bw / 2) * w)
gt_y2 = int((yc + bh / 2) * h)

print("GT box:", gt_x1, gt_y1, gt_x2, gt_y2)

# -------- MODEL --------

model = YOLOONNX(
    MODEL,
    imgsz=1024,
    conf_thres=0.25,
    iou_thres=0.45)

preds = model.predict(img)

print("Predictions:", preds)

# -------- DRAW --------

out = img.copy()

# GT = green
cv2.rectangle(out, (gt_x1, gt_y1), (gt_x2, gt_y2), (0, 255, 0), 2)

# Pred = red
for p in preds:
    x1, y1, x2, y2 = p["bbox"]
    cv2.rectangle(out, (x1, y1), (x2, y2), (0, 0, 255), 2)

cv2.imwrite("debug_stage2_result.png", out)
print("Saved debug_stage2_result.png")
