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
from .infer import YOLOONNX
from .models.lesion_classifier import LesionClassifierONNX
from .utils.heatmap import build_heatmap, downsample_heatmap, heatmap_stats_to_birads
from .utils.draw import draw_nested_boxes
from .gradcam import load_effnet_b0, GradCAM, preprocess_crop

from pydantic import BaseModel


# =====================
# PATHS
# =====================
BASE_DIR = Path(__file__).resolve().parent
DEBUG_DIR = BASE_DIR / "debug"
DEBUG_DIR.mkdir(parents=True, exist_ok=True)

DATASETS_ROOT = BASE_DIR.parent / "datasets"
GLOBAL_ROI_LABELS = DATASETS_ROOT / "CMMD_global_roi" / "labels"

# =====================
# CLINICAL MODEL (RAW ONLY)
# =====================
CLINICAL_MODELS_DIR = BASE_DIR / "models" / "clinical"
MODEL_PATH = CLINICAL_MODELS_DIR / "clinical_lgbm_raw.pkl"

clinical_model = joblib.load(MODEL_PATH)


# =====================
# Pydantic schema
# =====================
class ClinicalInput(BaseModel):
    age: float
    density: int
    lesion_type_enc: int
    assessment: int
    subtlety: int
    bmi: float
    menopause_status: float
    palpable_lump: int
    pain: int
    nipple_discharge: int
    family_history: int
    hormone_therapy: int
    prior_biopsies: int


# =====================
# MODELS
# =====================
model_stage1 = YOLOONNX(
    BASE_DIR / "models/global_breast_ir9.onnx",
    imgsz=1024,
    conf_thres=0.5,
)

model_stage2 = YOLOONNX(
    BASE_DIR / "models/small_tumor_best_ir9.onnx",
    imgsz=1024,
    conf_thres=0.25,
)

lesion_classifier = LesionClassifierONNX(
    BASE_DIR / "models/lesion_effb0_cls.onnx"
)

gradcam_model = load_effnet_b0(BASE_DIR / "models/efficientnet_b0_best.pth")
gradcam = GradCAM(gradcam_model)


# =====================
# APP
# =====================
app = FastAPI()

# === CORS ===
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # фронт Render потом
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



# =====================
# HELPERS
# =====================
def encode_img(img: np.ndarray) -> Optional[str]:
    ok, buf = cv2.imencode(".jpg", img)
    if not ok:
        return None
    return base64.b64encode(buf).decode()


def clinical_risk_to_birads(prob: float) -> int:
    if prob < 0.10:
        return 2
    elif prob < 0.30:
        return 3
    elif prob < 0.70:
        return 4
    else:
        return 5


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


def explain_clinical_factors(p: ClinicalInput) -> List[str]:
    reasons = []

    if p.age >= 55:
        reasons.append("вік понад 55 років асоціюється з підвищеним ризиком")
    if p.bmi >= 30:
        reasons.append("підвищений BMI є додатковим фактором ризику")
    if p.menopause_status == 1:
        reasons.append("постменопаузальний статус підвищує клінічну настороженість")
    if p.density >= 3:
        reasons.append("висока щільність тканини ускладнює діагностику")
    if p.palpable_lump:
        reasons.append("наявність пальпованого вузла є клінічно значущою")
    if p.nipple_discharge:
        reasons.append("виділення з соска підвищують клінічний ризик")
    if p.assessment >= 4:
        reasons.append("висока клінічна оцінка (BI-RADS 4–5)")

    return reasons


# =====================
# CLINICAL ENDPOINT (VARIANT 2)
# =====================
@app.post("/clinical_predict")
async def clinical_predict(payload: ClinicalInput):

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

    # === RAW MODEL OUTPUT ===
    raw_score = float(clinical_model.predict(x)[0])

    # === CLINICAL PROBABILITY (LOGIC-BASED) ===
    # ВАЖЛИВО: тут raw_score НЕ є імовірністю
    clinical_prob = min(max(
        0.15 * (payload.assessment / 5) +
        0.15 * payload.palpable_lump +
        0.15 * payload.nipple_discharge +
        0.15 * (payload.lesion_type_enc == 2) +
        0.10 * (payload.density >= 3) +
        0.10 * (payload.age >= 55),
        0.01
    ), 0.99)

    birads = clinical_risk_to_birads(clinical_prob)
    completeness = clinical_data_completeness(payload)

    if completeness["level"] == "insufficient":
        return {
            "status": "insufficient_data",
            "message": "Недостатньо клінічних даних для оцінки ризику",
            "completeness": completeness,
        }

    return {
        "status": completeness["level"],
        "malignant": {
            "prob": round(clinical_prob, 3),
            "label_name": "malignant" if clinical_prob >= 0.5 else "benign",
            "birads_from_symptoms": birads,
        },
        "model_output": {
            "model_score": round(raw_score, 3),
            "note": "Raw AI score (не є клінічною ймовірністю)",
        },
        "explanation": {
            "summary": (
                f"Клінічний ризик оцінено як {clinical_prob*100:.0f}%, "
                f"що відповідає BI-RADS {birads}."
            ),
            "key_factors": explain_clinical_factors(payload),
            "note": (
                "Результат є допоміжною AI-оцінкою та не є медичним діагнозом."
            ),
        },
        "completeness": completeness,
    }
