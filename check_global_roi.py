import cv2
import random
from pathlib import Path
import matplotlib.pyplot as plt

img_dir = Path("/Users/mariamakrenko/Documents/uni/ct-ai-system/datasets/CMMD_breast_roi/images/train")
lab_dir = Path("/Users/mariamakrenko/Documents/uni/ct-ai-system/datasets/CMMD_breast_roi/labels/train")

img_path = random.choice(list(img_dir.glob("*.png")))
lab_path = lab_dir / f"{img_path.stem}.txt"

img = cv2.imread(str(img_path))
h, w, _ = img.shape

with open(lab_path) as f:
    _, xc, yc, bw, bh = map(float, f.readline().split())

x1 = int((xc - bw / 2) * w)
y1 = int((yc - bh / 2) * h)
x2 = int((xc + bw / 2) * w)
y2 = int((yc + bh / 2) * h)

cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)

plt.figure(figsize=(6, 6))
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.title("Breast ROI sanity check")
plt.show()
