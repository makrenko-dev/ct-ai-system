from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import numpy as np
import cv2
import base64

from backend.infer import YOLOONNX
from backend.utils.nms import global_nms
from backend.utils.draw import draw_nested_boxes
from backend.utils.heatmap import (
    build_heatmap,
    downsample_heatmap,
    heatmap_stats_to_birads,
)

BASE_DIR = Path(__file__).resolve().parent

# ================= MODELS =================

model_stage1 = YOLOONNX(
    BASE_DIR / "models/global_breast_ir9.onnx",
    imgsz=1024,
    conf_thres=0.5,
)

model_stage2 = YOLOONNX(
    BASE_DIR / "models/small_tumor_best_ir9.onnx",
    imgsz=1024,
    conf_thres=0.3,   # как в colab
)

# ================= APP =================

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def encode_img(img: np.ndarray):
    ok, buf = cv2.imencode(".jpg", img)
    return base64.b64encode(buf).decode() if ok else None


# ================= STAGE 2 (CORRECT) =================

def run_stage2_with_heatmap(img: np.ndarray, stage1_boxes: list[dict]):
    """
    SECOND MODEL INFERENCE EXACTLY LIKE CMMD_small_roi TRAINING
    """

    final_boxes = []
    crop_debug_images = []

    for b in stage1_boxes:
        gx1, gy1, gx2, gy2 = map(int, b["bbox"])
        roi = img[gy1:gy2, gx1:gx2]

        if roi.size == 0:
            continue

        # ---------- DEBUG: what we REALLY send to model ----------
        crop_debug_images.append(roi.copy())

        # ---------- INFERENCE ----------
        detections = model_stage2.predict(roi)

        for d in detections:
            x1, y1, x2, y2 = map(int, d["bbox"])
            conf = float(d["conf"])

            if conf < 0.3:
                continue

            if (x2 - x1) < 20 or (y2 - y1) < 20:
                continue

            final_boxes.append({
                "bbox": [
                    gx1 + x1,
                    gy1 + y1,
                    gx1 + x2,
                    gy1 + y2,
                ],
                "conf": conf,
            })

    # ---------- OPTIONAL: keep only best ----------
    final_boxes = global_nms(final_boxes, iou_thresh=0.4)
    final_boxes = sorted(final_boxes, key=lambda x: x["conf"], reverse=True)[:1]

    # ---------- HEATMAP ----------
    heat_full = build_heatmap(img.shape, final_boxes)
    heat_small = downsample_heatmap(heat_full, target_size=128)
    heat_stats = heatmap_stats_to_birads(heat_full)

    return final_boxes, heat_small, heat_stats, crop_debug_images


# ================= ENDPOINT =================

@app.post("/pipeline_visual")
async def pipeline_visual(file: UploadFile = File(...)):
    img_bytes = await file.read()
    img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)

    # -------- STAGE 1 --------
    stage1 = model_stage1.predict(img)

    # -------- STAGE 2 --------
    stage2_boxes, heatmap_small, heat_stats, crops = run_stage2_with_heatmap(
        img, stage1
    )

    # -------- VISUALS --------

    # 1) full image with boxes
    vis_full = draw_nested_boxes(img.copy(), stage1, stage2_boxes)

    # 2) crop actually sent to stage2 (like CMMD_small_roi val)
    crop_vis = None
    if crops:
        crop_vis = crops[0]

        # draw bbox ON CROP (relative coords)
        for b in stage2_boxes:
            x1 = b["bbox"][0] - stage1[0]["bbox"][0]
            y1 = b["bbox"][1] - stage1[0]["bbox"][1]
            x2 = b["bbox"][2] - stage1[0]["bbox"][0]
            y2 = b["bbox"][3] - stage1[0]["bbox"][1]
            cv2.rectangle(crop_vis, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # -------- RESPONSE --------
    return {
        "images": {
            "full_with_boxes": encode_img(vis_full),
            "stage2_input_crop": encode_img(crop_vis) if crop_vis is not None else None,
        },
        "boxes": {
            "stage1": stage1,
            "stage2": stage2_boxes,
        },
        "heatmap": heatmap_small,
        "assessment": {
            "birads": heat_stats["birads"],
            "max_val": heat_stats["max_val"],
            "frac_mid": heat_stats["frac_mid"],
            "frac_high": heat_stats["frac_high"],
        },
    }
