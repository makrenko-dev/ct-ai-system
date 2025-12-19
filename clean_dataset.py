from pathlib import Path
import pandas as pd
import shutil

# ===== CONFIG =====
DATASET_ROOT = Path("datasets/CMMD_lesion_cls")
LABELS_CSV = DATASET_ROOT / "labels.csv"
DROP_LABEL_IDS = {5}  # lipoma
SPLITS = ["train", "val"]
# ==================

def main():
    print("[1] Loading labels.csv")
    df = pd.read_csv(LABELS_CSV)

    print("Original class distribution:")
    print(df["label"].value_counts(), "\n")

    to_drop = df[df["label"].isin(DROP_LABEL_IDS)]
    print(f"[2] Samples to remove: {len(to_drop)}")

    removed = 0

    for _, row in to_drop.iterrows():
        filename = row["filename"]
        class_name = row.get("class_name", None)

        for split in SPLITS:
            if class_name:
                img_path = DATASET_ROOT / split / class_name / filename
                if img_path.exists():
                    img_path.unlink()
                    removed += 1
            else:
                # если class_name нет — ищем во всех папках
                for cls_dir in (DATASET_ROOT / split).iterdir():
                    img_path = cls_dir / filename
                    if img_path.exists():
                        img_path.unlink()
                        removed += 1

    # удаляем строки из CSV
    df = df[~df["label"].isin(DROP_LABEL_IDS)]
    df.to_csv(LABELS_CSV, index=False)

    # удаляем папку lipoma если осталась
    for split in SPLITS:
        cls_dir = DATASET_ROOT / split / "lipoma"
        if cls_dir.exists():
            shutil.rmtree(cls_dir)

    print(f"🗑 Removed images: {removed}")
    print("\n[3] Final class distribution:")
    print(df["label"].value_counts())

    print("\n✅ Dataset cleaned successfully")

if __name__ == "__main__":
    main()
