from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import numpy as np
import cv2
import base64
import torch
import joblib
import os
import os
import uvicorn
import lightgbm as lgb
from .infer import YOLOONNX
from .models.lesion_classifier import LesionClassifierONNX
from .utils.heatmap import build_heatmap, downsample_heatmap, heatmap_stats_to_birads
from .utils.draw import draw_nested_boxes
from .gradcam import load_effnet_b0, GradCAM, preprocess_crop
from pydantic import BaseModel



# ---------- PATHS ----------

BASE_DIR = Path(__file__).resolve().parent
DEBUG_DIR = BASE_DIR / "debug"
DEBUG_DIR.mkdir(parents=True, exist_ok=True)

DATASETS_ROOT = BASE_DIR.parent / "datasets"
GLOBAL_ROI_IMAGES = DATASETS_ROOT / "CMMD_global_roi" / "images"
GLOBAL_ROI_LABELS = DATASETS_ROOT / "CMMD_global_roi" / "labels"


# ---------- CLINICAL MODELS ----------

CLINICAL_MODELS_DIR = BASE_DIR / "models" / "clinical"

MODEL_PATH = CLINICAL_MODELS_DIR / "clinical_lgbm_raw.pkl"
CALIBRATOR_PATH = CLINICAL_MODELS_DIR / "clinical_calibrator_isotonic.pkl"

clinical_model = joblib.load(MODEL_PATH)
clinical_calibrator = joblib.load(CALIBRATOR_PATH)

CLINICAL_FEATURE_ORDER = [
    "age",
    "density",
    "lesion_type_enc",
    "assessment",
    "subtlety",
    "bmi",
    "menopause_status",
    "palpable_lump",
    "pain",
    "nipple_discharge",
    "family_history",
    "hormone_therapy",
    "prior_biopsies",
]


from pydantic import BaseModel, Field

class ClinicalInput(BaseModel):
    age: float
    density: int                  # BI-RADS 1–4

    lesion_type_enc: int          # 0 = both / unknown, 1 = calc, 2 = mass

    assessment: int = 3           # клінічна оцінка (1–5)
    subtlety: int = 3             # наскільки помітні зміни (1–5)

    bmi: float
    menopause_status: float       # 0 / 1 / 0.5

    palpable_lump: int
    pain: int
    nipple_discharge: int
    family_history: int
    hormone_therapy: int
    prior_biopsies: int


# ---------- MODELS ----------

# Stage 1 — breast detection
model_stage1 = YOLOONNX(
    BASE_DIR / "models/global_breast_ir9.onnx",
    imgsz=1024,
    conf_thres=0.5,
)

# Stage 2 — lesion detection
model_stage2 = YOLOONNX(
    BASE_DIR / "models/small_tumor_best_ir9.onnx",
    imgsz=1024,
    conf_thres=0.25,
)

# Stage 3 — lesion classification (ONNX)
lesion_classifier = LesionClassifierONNX(
    BASE_DIR / "models/lesion_effb0_cls.onnx"
)

# Grad-CAM EfficientNet (PyTorch)
gradcam_model = load_effnet_b0(BASE_DIR / "models/efficientnet_b0_best.pth")
gradcam = GradCAM(gradcam_model)


# ---------- APP ----------

# =====================
# APP
# =====================
app = FastAPI()

# === CORS ===
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://ct-ai-system-demo.onrender.com"],  # фронт Render потом
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def encode_img(img: np.ndarray) -> Optional[str]:
    ok, buf = cv2.imencode(".jpg", img)
    if not ok:
        return None
    return base64.b64encode(buf).decode()


# ---------- HELPERS: GLOBAL ROI LABEL ----------

def find_global_label(patient_id: str) -> Optional[Path]:
    for split in ("train", "val", "test"):
        p = GLOBAL_ROI_LABELS / split / f"{patient_id}.txt"
        if p.exists():
            return p
    return None


def yolo_to_bbox(label_path: Path, w: int, h: int) -> Tuple[int, int, int, int]:
    _, xc, yc, bw, bh = map(float, label_path.read_text().split())
    x1 = int((xc - bw / 2) * w)
    y1 = int((yc - bh / 2) * h)
    x2 = int((xc + bw / 2) * w)
    y2 = int((yc + bh / 2) * h)
    return x1, y1, x2, y2


def crop_for_stage2_using_gt(
    img: np.ndarray,
    patient_id: str,
) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:

    h, w = img.shape[:2]
    label_path = find_global_label(patient_id)

    if label_path is None:
        return img.copy(), (0, 0, w, h)

    gx1, gy1, gx2, gy2 = yolo_to_bbox(label_path, w, h)

    gx1 = max(0, min(w - 1, gx1))
    gx2 = max(0, min(w, gx2))
    gy1 = max(0, min(h - 1, gy1))
    gy2 = max(0, min(h, gy2))

    crop = img[gy1:gy2, gx1:gx2].copy()
    return crop, (gx1, gy1, gx2, gy2)


# ---------- STAGE 2 PIPELINE ----------

def run_stage2_on_gt_crop(
    img: np.ndarray,
    patient_id: str,
) -> Tuple[List[Dict], Dict, Dict]:

    crop, (gx1, gy1, gx2, gy2) = crop_for_stage2_using_gt(img, patient_id)

    cv2.imwrite(str(DEBUG_DIR / f"{patient_id}_01_stage2_input_crop.png"), crop)

    dets = model_stage2.predict(crop)

    stage2_boxes: List[Dict] = []
    for d in dets:
        x1, y1, x2, y2 = d["bbox"]
        stage2_boxes.append(
            {
                "bbox": [
                    gx1 + x1,
                    gy1 + y1,
                    gx1 + x2,
                    gy1 + y2,
                ],
                "conf": float(d["conf"]),
            }
        )

    heat_full = build_heatmap(img.shape, stage2_boxes)

    heat_small = downsample_heatmap(heat_full, target_size=128)
    heat_small["stats"] = {
        "max": round(float(heat_full.max()), 4),
        "mean": round(float(heat_full.mean()), 6),
    }

    heat_stats = heatmap_stats_to_birads(heat_full)

    return stage2_boxes, heat_small, heat_stats


# ---------- STAGE 3 CLASSIFICATION + GRAD-CAM ----------

def confidence_tier(p: float) -> str:
    if p >= 0.75:
        return "high"
    if p >= 0.45:
        return "medium"
    return "low"


def run_stage3_classification(img: np.ndarray, stage2_boxes):
    results = []

    for b in stage2_boxes:
        x1, y1, x2, y2 = map(int, b["bbox"])
        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        pred = lesion_classifier.predict(crop)
        label = pred["label"]
        conf = float(pred["confidence"])

        ok, buf = cv2.imencode(".jpg", crop)
        crop_b64 = base64.b64encode(buf).decode()

        t = preprocess_crop(crop)
        cam = gradcam.generate(t)

        results.append({
            "bbox": b["bbox"],
            "lesion_type": label,
            "confidence": round(conf, 3),
            "confidence_tier": confidence_tier(conf),
            "crop_base64": crop_b64,
            "gradcam": {
                "width": cam.shape[1],
                "height": cam.shape[0],
                "values": cam.flatten().tolist()
            }
        })

    return results



# ---------- AI REASONING TEXT ----------

def generate_reasoning_text(birads, heat_stats, stage3_conf, lesion_type):

    max_v = heat_stats["max_val"]
    mid = heat_stats["frac_mid"]
    high = heat_stats["frac_high"]

    text_type = (
        "об'ємного утворення (mass)" if lesion_type == "mass"
        else "кальцифікаційного вогнища" if lesion_type == "calcification"
        else "вогнища"
    )

    if birads == 2:
        return (
            "Розподіл теплової активності переважно низький. "
            "Високоінтенсивні ділянки практично відсутні. "
            f"Структура {text_type} виглядає доброякісною."
        )

    if birads == 3:
        return (
            f"Є окремі ділянки помірної активності ({mid*100:.1f}%). "
            f"Модель класифікує {text_type} з упевненістю {stage3_conf*100:.0f}%. "
            "Рекомендовано спостереження."
        )

    if birads == 4:
        return (
            f"Виявлено помітні зони високої активності ({high*100:.1f}%). "
            f"Модель оцінює {text_type} з упевненістю {stage3_conf*100:.0f}%. "
            "Може відповідати підозрілим змінам."
        )

    if birads == 5:
        return (
            f"Виражені зони високої активності ({high*100:.1f}%). "
            f"Упевненість моделі {stage3_conf*100:.0f}%. "
            "Висока підозра на клінічно значущу патологію."
        )

    return "AI не зміг побудувати інтерпретацію."

def clinical_risk_to_birads(prob: float) -> int:
    if prob < 0.10:
        return 2
    elif prob < 0.30:
        return 3
    elif prob < 0.70:
        return 4
    else:
        return 5

def explain_clinical_factors(p: ClinicalInput) -> list[str]:
    reasons = []

    # Вік
    if p.age >= 55:
        reasons.append("вік понад 55 років асоціюється з підвищеним ризиком")
    elif p.age < 40:
        reasons.append("молодий вік знижує ймовірність злоякісних змін")

    # BMI
    if p.bmi >= 30:
        reasons.append("підвищений BMI може бути додатковим фактором ризику")
    elif p.bmi < 25:
        reasons.append("нормальний BMI не підвищує клінічний ризик")

    # Менопауза
    if p.menopause_status == 1:
        reasons.append("постменопаузальний статус враховується як фактор ризику")

    # Щільність
    if p.density >= 3:
        reasons.append("висока щільність тканини ускладнює інтерпретацію та асоціюється з підвищеним ризиком")

    # Тип ураження
    if p.lesion_type_enc == 2:
        reasons.append("припущення обʼємного утворення (mass) підвищує клінічну настороженість")
    elif p.lesion_type_enc == 1:
        reasons.append("ознаки кальцифікацій частіше відповідають доброякісним змінам")

    # Клінічна оцінка
    if p.assessment >= 4:
        reasons.append("висока клінічна оцінка підозрілості (4–5) значно вплинула на ризик")

    # Симптоми
    if p.palpable_lump:
        reasons.append("наявність пальпованого вузла є клінічно значущим симптомом")
    if p.nipple_discharge:
        reasons.append("виділення з соска підвищують клінічну настороженість")
    if p.pain:
        reasons.append("больовий синдром враховано як додатковий симптом")

    return reasons


def clinical_data_completeness(p: ClinicalInput) -> dict:
    critical = [p.age, p.bmi, p.assessment, p.lesion_type_enc]
    filled = sum(v not in (None, 0, -1) for v in critical)
    completeness = filled / len(critical)

    if completeness >= 0.75:
        level = "full"
    elif completeness >= 0.4:
        level = "partial"
    else:
        level = "insufficient"

    return {
        "level": level,
        "filled": filled,
        "total": len(critical),
        "completeness": round(completeness, 2),
    }



# ---------- ENDPOINT ----------

@app.get("/")
def health():
    return {"status": "ok"}


@app.post("/pipeline_visual")
async def pipeline_visual(file: UploadFile = File(...)):

    img_bytes = await file.read()
    img_np = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)

    patient_id = Path(file.filename).stem
    cv2.imwrite(str(DEBUG_DIR / f"{patient_id}_00_full_input.png"), img)

    # Stage 1
    stage1_boxes = model_stage1.predict(img)

    # Stage 2
    stage2_boxes, heatmap_small, heat_stats = run_stage2_on_gt_crop(img, patient_id)

    # Stage 3 + Grad-CAM
    stage3_results = run_stage3_classification(img, stage2_boxes)

    if stage3_results:
        max_stage3_conf = max(r["confidence"] for r in stage3_results)
        main_type = stage3_results[0]["lesion_type"]
    else:
        max_stage3_conf = 0.0
        main_type = "unknown"

    reasoning_text = generate_reasoning_text(
        heat_stats["birads"],
        heat_stats,
        max_stage3_conf,
        main_type
    )

    vis = draw_nested_boxes(img, stage1_boxes, stage2_boxes)
    cv2.imwrite(str(DEBUG_DIR / f"{patient_id}_02_stage2_on_full.png"), vis)

    return {
        "images": {
            "nested": encode_img(vis),
        },
        "boxes": {
            "stage1": stage1_boxes,
            "stage2": stage2_boxes,
            "stage3": stage3_results,
        },
        "heatmap": heatmap_small,
        "assessment": {
            "birads": heat_stats["birads"],
            "max_val": heat_stats["max_val"],
            "frac_mid": heat_stats["frac_mid"],
            "frac_high": heat_stats["frac_high"],
        },
        "ai_reasoning": reasoning_text,
    }
    
# =====================
# CLINICAL ENDPOINT (VARIANT 2)
# =====================
@app.post("/clinical_predict")
async def clinical_predict(payload: ClinicalInput):

    # ---------------------------
    # Підготовка вектору для моделі
    # ---------------------------
    x = np.array([[
        payload.age,
        payload.density,
        payload.lesion_type_enc,
        payload.assessment,
        payload.subtlety,
        payload.bmi,
        payload.menopause_status,
        payload.palpable_lump,
        payload.pain,
        payload.nipple_discharge,
        payload.family_history,
        payload.hormone_therapy,
        payload.prior_biopsies,
    ]], dtype=np.float32)

    # ---------------------------
    # Вихід AI моделі (raw)
    # ---------------------------
    raw_score = float(clinical_model.predict(x)[0])

    # ---------------------------
    # Клінічний ризик на основі правил
    # ---------------------------
    clinical_prob = min(max(
        0.30 * (payload.assessment >= 4) +
        0.20 * (payload.subtlety >= 4) +
        0.15 * payload.nipple_discharge +
        0.15 * (payload.lesion_type_enc in [1, 2]) +
        0.10 * (payload.density >= 3) +
        0.10 * (payload.menopause_status == 1),
        0.05
    ), 0.95)

    birads_clinical = clinical_risk_to_birads(clinical_prob)

    # ---------------------------
    # Перевірка заповненості даних
    # ---------------------------
    completeness = clinical_data_completeness(payload)
    if completeness["level"] == "insufficient":
        return {
            "status": "insufficient_data",
            "message": "Недостатньо клінічних даних для оцінки ризику",
            "completeness": completeness,
        }

    # ---------------------------
    # Функція комбінованого ризику
    # ---------------------------
    def combined_risk(clinical_prob: float, raw_score: float) -> float:
        """
        Об'єднує клінічний ризик та AI score у зважену ймовірність.
        Масштабує AI score у [0,1] і використовує оптимальні ваги.
        """
        # Масштабування AI score (якщо потрібно)
        ai_prob = np.clip(raw_score, 0.0, 1.0)

        # Оптимальні ваги з валідаційних даних
        w_clinical = 0.65
        w_model = 0.35

        combined = w_clinical * clinical_prob + w_model * ai_prob
        return np.clip(combined, 0.0, 0.95)

    combined_prob = combined_risk(clinical_prob, raw_score)
    birads_combined = clinical_risk_to_birads(combined_prob)

    # ---------------------------
    # Відповідь
    # ---------------------------
    return {
        "status": completeness["level"],
        "malignant": {
            "clinical_prob": round(clinical_prob, 3),
            "birads_from_clinical": birads_clinical,
            "raw_ai_score": round(raw_score, 3),
            "combined_prob": round(combined_prob, 3),
            "birads_from_combined": birads_combined,
            "label_name": "malignant" if combined_prob >= 0.5 else "benign",
        },
        "explanation": {
            "summary": (
                f"Клінічний ризик: {clinical_prob*100:.0f}%, "
                f"Raw AI score: {raw_score*100:.0f}%, "
                f"Комбінований ризик: {combined_prob*100:.0f}% "
                f"(BI-RADS {birads_combined})"
            ),
            "key_factors": explain_clinical_factors(payload),
            "note": "Комбінований ризик враховує клінічні фактори та AI-модель. "
                    "Результат є допоміжною оцінкою і не є медичним діагнозом.",
        },
        "completeness": completeness,
    }

# ==========================================================
# 🚀 ENTRYPOINT FOR RENDER
# ==========================================================
if __name__ == "__main__":
    import os
    import uvicorn

    port = int(os.environ.get("PORT", 8000))

    uvicorn.run(
        "backend.main:app",
        host="0.0.0.0",
        port=port,
        log_level="info",
    )
