import cv2
import random
from pathlib import Path
import matplotlib.pyplot as plt

# =====================
# PATHS
# =====================

BASE = Path.home() / "Documents" / "uni" / "ct-ai-system" / "datasets"
IMG_DIR = BASE / "CMMD_small_roi/images/train"
LAB_DIR = BASE / "CMMD_small_roi/labels/train"

# =====================
# PICK RANDOM SAMPLE
# =====================

img_path = random.choice(list(IMG_DIR.glob("*.png")))
lab_path = LAB_DIR / f"{img_path.stem}.txt"

print("Image:", img_path.name)
print("Label:", lab_path.name)

# =====================
# LOAD IMAGE
# =====================

img = cv2.imread(str(img_path))
if img is None:
    raise RuntimeError("❌ Image not found")

h, w = img.shape[:2]

# =====================
# LOAD YOLO LABEL
# =====================

with open(lab_path) as f:
    class_id, xc, yc, bw, bh = map(float, f.readline().split())

# YOLO → pixels
x1 = int((xc - bw / 2) * w)
y1 = int((yc - bh / 2) * h)
x2 = int((xc + bw / 2) * w)
y2 = int((yc + bh / 2) * h)

# =====================
# DRAW
# =====================

cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

plt.figure(figsize=(5, 5))
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.title("Small ROI — tumor bbox sanity check")
plt.show()
