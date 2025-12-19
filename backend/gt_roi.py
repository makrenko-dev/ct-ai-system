from pathlib import Path

def load_gt_breast_bbox(patient_id: str, base_dir: Path):
    """
    Returns bbox: [x1, y1, x2, y2] in GLOBAL coords
    """
    label_paths = [
        base_dir / "datasets/CMMD_global_roi/labels/train",
        base_dir / "datasets/CMMD_global_roi/labels/val",
    ]

    for lp in label_paths:
        txt = lp / f"{patient_id}.txt"
        if txt.exists():
            c, xc, yc, bw, bh = map(float, txt.read_text().split())
            return xc, yc, bw, bh

    return None
