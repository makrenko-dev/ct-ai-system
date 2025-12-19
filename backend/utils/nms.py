# backend/utils/nms.py

import numpy as np

def global_nms(boxes, iou_thresh=0.4):
    if not boxes:
        return []

    b = np.array([x["bbox"] for x in boxes], dtype=np.float32)
    s = np.array([x["conf"] for x in boxes], dtype=np.float32)

    x1, y1, x2, y2 = b.T
    areas = (x2 - x1) * (y2 - y1)
    order = s.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h

        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        order = order[np.where(iou <= iou_thresh)[0] + 1]

    return [boxes[i] for i in keep]
