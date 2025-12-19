import numpy as np
from sklearn.cluster import DBSCAN


def cluster_boxes(boxes, eps=30, min_samples=2):
    """
    boxes: list[{bbox:[x1,y1,x2,y2], conf}]
    eps: расстояние между центрами боксов (px)
    min_samples: минимум боксов для кластера
    """

    if not boxes:
        return []

    # центры боксов
    centers = []
    for b in boxes:
        x1, y1, x2, y2 = b["bbox"]
        centers.append([
            (x1 + x2) / 2,
            (y1 + y2) / 2,
        ])

    X = np.array(centers)

    clustering = DBSCAN(
        eps=eps,
        min_samples=min_samples
    ).fit(X)

    labels = clustering.labels_

    clusters = {}
    results = []

    # группируем только реальные кластеры
    for idx, label in enumerate(labels):
        if label == -1:
            # ✅ одиночный бокс — оставляем
            b = boxes[idx]
            results.append({
                "bbox": b["bbox"],
                "conf": float(b["conf"]),
                "size": 1,
            })
        else:
            clusters.setdefault(label, []).append(boxes[idx])

    # агрегируем кластеры
    for group in clusters.values():
        xs, ys, xe, ye, confs = [], [], [], [], []

        for b in group:
            x1, y1, x2, y2 = b["bbox"]
            xs.append(x1)
            ys.append(y1)
            xe.append(x2)
            ye.append(y2)
            confs.append(b["conf"])

        results.append({
            "bbox": [
                int(min(xs)),
                int(min(ys)),
                int(max(xe)),
                int(max(ye)),
            ],
            "conf": float(max(confs)),
            "size": len(group),
        })

    return results
