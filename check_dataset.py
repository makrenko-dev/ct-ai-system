"""
CMMD lesion classification dataset checker (mass / calcification)

Проверяет:
1) структуру датасета
2) labels.csv
3) существование файлов
4) битые изображения
5) размер изображений
6) распределение классов
7) train / val split
8) визуальный sanity-check
"""

from pathlib import Path
import csv
import cv2
import random
import collections
import matplotlib.pyplot as plt

# =====================
# CONFIG
# =====================

ROOT = Path.home() / "Documents" / "uni" / "ct-ai-system" / "datasets" / "CMMD_lesion_cls"
CSV_PATH = ROOT / "labels.csv"

EXPECTED_SIZE = (224, 224)

LABEL_INV = {
    0: "calcification",
    1: "mass",
}

# =====================
# HELPERS
# =====================

def full_path(rel_path: str) -> Path:
    """Преобразует относительный путь из CSV в абсолютный"""
    return ROOT / rel_path

# =====================
# 1. STRUCTURE CHECK
# =====================

print("\n[1] Checking folder structure...")

assert ROOT.exists(), "❌ Dataset root not found"
assert (ROOT / "images/train").exists(), "❌ images/train missing"
assert (ROOT / "images/val").exists(), "❌ images/val missing"
assert CSV_PATH.exists(), "❌ labels.csv missing"

print("✅ Structure OK")

# =====================
# 2. LOAD CSV
# =====================

print("\n[2] Loading labels.csv...")

rows = []
with open(CSV_PATH, newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        rows.append(row)

assert len(rows) > 0, "❌ labels.csv is empty"
assert "path" in rows[0], "❌ CSV missing 'path'"
assert "label" in rows[0], "❌ CSV missing 'label'"

print(f"✅ Loaded {len(rows)} samples")

# =====================
# 3. CHECK FILE EXISTENCE
# =====================

print("\n[3] Checking file existence...")

missing = []
for r in rows:
    p = full_path(r["path"])
    if not p.exists():
        missing.append(p)

if missing:
    print(f"❌ Missing files: {len(missing)}")
    for p in missing[:5]:
        print("   ", p)
    raise SystemExit
else:
    print("✅ All image files exist")

# =====================
# 4. CHECK BROKEN IMAGES
# =====================

print("\n[4] Checking for broken images...")

broken = []
for r in rows:
    img = cv2.imread(str(full_path(r["path"])))
    if img is None or img.size == 0:
        broken.append(r["path"])

if broken:
    print(f"❌ Broken images: {len(broken)}")
    for p in broken[:5]:
        print("   ", p)
    raise SystemExit
else:
    print("✅ No broken images")

# =====================
# 5. CHECK IMAGE SIZES
# =====================

print("\n[5] Checking image sizes...")

size_counter = collections.Counter()

sampled = random.sample(rows, min(300, len(rows)))
for r in sampled:
    img = cv2.imread(str(full_path(r["path"])))
    h, w = img.shape[:2]
    size_counter[(h, w)] += 1

print("Sizes found:")
for k, v in size_counter.items():
    print(f"  {k}: {v}")

if list(size_counter.keys()) != [EXPECTED_SIZE]:
    print("❌ Unexpected image sizes!")
    raise SystemExit
else:
    print("✅ Image sizes OK")

# =====================
# 6. CLASS DISTRIBUTION
# =====================

print("\n[6] Class distribution:")

label_counter = collections.Counter(int(r["label"]) for r in rows)

for k, v in label_counter.items():
    name = LABEL_INV.get(k, f"unknown({k})")
    print(f"  {name:16s}: {v}")

# =====================
# 7. TRAIN / VAL SPLIT
# =====================

print("\n[7] Train / Val split:")

train = [r for r in rows if "/train/" in r["path"]]
val = [r for r in rows if "/val/" in r["path"]]

total = len(train) + len(val)
ratio = len(train) / max(1, total)

print(f"  Train: {len(train)}")
print(f"  Val  : {len(val)}")
print(f"  Ratio: {ratio:.2f}")

# =====================
# 8. VISUAL SANITY CHECK
# =====================

print("\n[8] Visual sanity check (random samples)...")

samples = random.sample(rows, min(9, len(rows)))

plt.figure(figsize=(9, 9))

for i, r in enumerate(samples):
    img = cv2.imread(str(full_path(r["path"])))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    plt.subplot(3, 3, i + 1)
    plt.imshow(img)
    plt.title(LABEL_INV.get(int(r["label"]), r["label"]))
    plt.axis("off")

plt.tight_layout()
plt.show()

print("\n✅ DATASET CHECK FINISHED SUCCESSFULLY")
