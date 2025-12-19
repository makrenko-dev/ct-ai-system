import cv2
import numpy as np

def draw_nested_boxes(img, stage1, stage2):
    """
    img: np.ndarray (BGR)
    stage1: [{bbox:[x1,y1,x2,y2], conf:float}]
    stage2: [{bbox:[x1,y1,x2,y2], conf:float}]
    """

    out = img.copy()
    h, w = img.shape[:2]

    for b1 in stage1:
        x1, y1, x2, y2 = b1["bbox"]

        # clamp stage1 bbox
        x1 = max(0, min(w, x1))
        y1 = max(0, min(h, y1))
        x2 = max(0, min(w, x2))
        y2 = max(0, min(h, y2))

        # --- draw stage1 (green) ---
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # --- draw nested stage2 only ---
        for b2 in stage2:
            bx1, by1, bx2, by2 = b2["bbox"]

            # check that stage2 is REALLY inside stage1
            if (
                bx1 >= x1 and by1 >= y1 and
                bx2 <= x2 and by2 <= y2
            ):
                bx1 = max(0, min(w, bx1))
                by1 = max(0, min(h, by1))
                bx2 = max(0, min(w, bx2))
                by2 = max(0, min(h, by2))

                cv2.rectangle(
                    out,
                    (bx1, by1),
                    (bx2, by2),
                    (0, 0, 255),
                    2
                )

    return out
