# backend/utils/heatmap.py
import numpy as np
import cv2


def build_heatmap(img_shape, boxes, blur_ksize=41):
    """
    img_shape: (H, W, C) или (H, W)
    boxes: список dict { "bbox": [x1,y1,x2,y2], "conf": float }
    return: heatmap float32 [0..1] размера (H, W)
    """
    if not boxes:
        h, w = img_shape[:2]
        return np.zeros((h, w), dtype=np.float32)

    h, w = img_shape[:2]
    heat = np.zeros((h, w), dtype=np.float32)

    for b in boxes:
        x1, y1, x2, y2 = map(int, b["bbox"])
        x1 = max(0, min(w, x1))
        x2 = max(0, min(w, x2))
        y1 = max(0, min(h, y1))
        y2 = max(0, min(h, y2))
        if x2 <= x1 or y2 <= y1:
            continue

        # добавляем confidence во всю область бокса
        heat[y1:y2, x1:x2] += float(b["conf"])

    # сглаживание, чтобы не было "ступенек"
    if blur_ksize and blur_ksize > 1:
        heat = cv2.GaussianBlur(heat, (blur_ksize, blur_ksize), 0)

    # нормализация 0..1
    mx = heat.max()
    if mx > 0:
        heat = heat / mx
    return heat.astype(np.float32)


def downsample_heatmap(heatmap, target_size=128):
    """
    Уменьшаем размер heatmap для передачи на фронт.
    Возвращаем dict с width/height/values (flattened list).
    """
    h, w = heatmap.shape[:2]

    # приводим к квадрату target_size x target_size, сохраняя пропорции
    if max(h, w) > target_size:
        if h >= w:
            nh = target_size
            nw = int(w * target_size / h)
        else:
            nw = target_size
            nh = int(h * target_size / w)
        hm_resized = cv2.resize(heatmap, (nw, nh), interpolation=cv2.INTER_LINEAR)
    else:
        hm_resized = heatmap.copy()
        nh, nw = hm_resized.shape[:2]

    # в uint8 0..255
    hm_uint8 = np.clip(hm_resized * 255.0, 0, 255).astype(np.uint8)

    return {
        "width": int(nw),
        "height": int(nh),
        "values": hm_uint8.flatten().tolist(),
    }


def heatmap_stats_to_birads(heatmap, thr_low=0.25, thr_high=0.6):
    """
    Очень простой хэндмейд BI-RADS по статистике heatmap.
    Это НЕ мед. рекомендация, а демонстрация фичи для диплома.
    """
    h, w = heatmap.shape[:2]
    area = float(h * w)

    # доля “горячих” пикселей
    frac_mid = float((heatmap >= thr_low).sum()) / area
    frac_high = float((heatmap >= thr_high).sum()) / area
    max_val = float(heatmap.max())

    # примитивная эвристика
    if max_val < 0.2 or frac_mid < 0.01:
        score = 2  # скорее доброкачественно
    elif frac_high < 0.01:
        score = 3  # сомнительно
    elif frac_high < 0.03:
        score = 4  # подозрительно
    else:
        score = 5  # высокая вероятность

    return {
        "birads": score,
        "max_val": max_val,
        "frac_mid": frac_mid,
        "frac_high": frac_high,
    }
