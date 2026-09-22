"""
MedAI Backend — Hardened (v2)
==============================
Changes vs purana version:
  1. MODALITY GATE  — random photo pe model tak nahi jata (modality.py)
  2. CONFIDENCE GATE — temperature-scaled, honest confidence (calibration.py)
  3. INPUT VALIDATION — size/type limits, structured errors (HTTPException)
  4. CORS LOCK      — sirf allowed origins (env se)
  5. RATE LIMITING  — simple in-process limiter (koi extra dependency nahi)
  6. /metrics/{model} — retraining metrics serve karta hai (dashboard ke liye)
  7. /ai-doctor REMOVED — assistant ab sirf frontend (Next.js API route) me hai.
     Yahan 410 Gone milta hai purane callers ko.
  8. DUAL ENGINE    — agar ONNX models mojood hon to torch-free ONNX runtime
     use hota hai (512MB RAM free tier ke liye — onnx_engine.py), warna
     PyTorch .pth wala original engine.

Env vars:
  ALLOWED_ORIGINS   comma-separated, e.g. "https://final-year-project-medai.vercel.app,http://localhost:3000"
  MAX_UPLOAD_MB     default 10
  PRELOAD_MODELS    "1" => startup pe teeno models load (recommended)
  ENGINE            "auto" (default) | "onnx" | "torch"
"""

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
import os
import io
import json
import time
import tempfile
from datetime import datetime, timedelta, timezone
from collections import defaultdict, deque

from PIL import Image, ImageFile

import numpy as np

from modality import check_image
from scan_type_classifier import (
    classify_scan_type,
    looks_like_color_photo,
    MIN_CONFIDENCE as MIN_CLASSIFIER_CONFIDENCE,
)

# model_type -> CNN modality label (scan_type_classifier.CLASSES)
MODEL_MODALITY = {"fracture": "xray", "brain": "mri", "kidney": "ct"}
from calibration import get_temperature, calibrated_softmax, evaluate_confidence

# ── Auth + database (v3: personalization layer — models untouched) ──
import base64 as _b64
from database import (
    get_db, init_db, utcnow, User, Analysis, ChatSession, ChatMessage, UserMemory,
    ActivityLog,
)
from auth import (
    hash_password, verify_password, create_token, decode_token,
    get_current_user, require_admin, get_client_ip,
    register_login_fail, record_login_fail, clear_login_fails,
)
from sqlalchemy.orm import Session
from fastapi import Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

# ── Engine selection: ONNX (torch-free) ya PyTorch ──
ENGINE_MODE = os.getenv("ENGINE", "auto")  # auto | onnx | torch


def _has_onnx(model_type: str) -> bool:
    base = os.path.dirname(os.path.abspath(__file__))
    return os.path.exists(os.path.join(base, f"{model_type}_model_int8.onnx")) or \
        os.path.exists(os.path.join(base, f"{model_type}_model.onnx"))


USE_ONNX = (
    ENGINE_MODE == "onnx"
    or (ENGINE_MODE == "auto" and all(_has_onnx(k) for k in ("fracture", "brain", "kidney")))
)

if USE_ONNX:
    import onnx_engine
    # cv2 free-tier images me nahi — overlay PIL se hota hai (onnx_engine.overlay_heatmap)
else:
    import torch
    import torchvision.models as models
    from torchvision import transforms
    import cv2
    import base64
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

ImageFile.LOAD_TRUNCATED_IMAGES = True

ImageFile.LOAD_TRUNCATED_IMAGES = True

# ─────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Inconclusive band: is se neeche confidence pe backend "uncertain" flag deta hai.
# OOD (bahar ki) images pe model overconfident hota hai — ye honest uncertainty
# ke liye zaroori hai (MURA val eval: normals ab sahi, lekin confidence high hoti
# hai — demo me "99% sure galat jawab" se bachne ke liye band rakha gaya hai).
INCONCLUSIVE_BELOW = float(os.getenv("INCONCLUSIVE_BELOW", "60"))
if not USE_ONNX:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    DEVICE = None

ALLOWED_ORIGINS = [
    o.strip() for o in os.getenv(
        "ALLOWED_ORIGINS",
        # default: local dev + tumhari Vercel site (naya HF URL baad me add hoga)
        "http://localhost:3000,https://final-year-project-medai.vercel.app",
    ).split(",") if o.strip()
]

MAX_UPLOAD_MB = float(os.getenv("MAX_UPLOAD_MB", "10"))
MAX_UPLOAD_BYTES = int(MAX_UPLOAD_MB * 1024 * 1024)

RATE_LIMIT_REQUESTS = int(os.getenv("RATE_LIMIT_REQUESTS", "30"))   # requests
RATE_LIMIT_WINDOW_S  = int(os.getenv("RATE_LIMIT_WINDOW_S", "60"))  # per window

MODELS_DIR = os.path.join(BASE_DIR, "metrics")  # metrics/*.json

if USE_ONNX:
    # ONNX engine ka apna registry use karo (classes wahi, file names .onnx)
    MODELS = {
        k: {
            "classes": onnx_engine.MODELS[k]["classes"],
            "scan": onnx_engine.MODELS[k]["scan"],
            "num_classes": onnx_engine.MODELS[k]["num_classes"],
            "modality": MODEL_MODALITY.get(k),
            "model": None,
        }
        for k in onnx_engine.MODELS
    }


# ─────────────────────────────────────────────────────────────
# App + middleware
# ─────────────────────────────────────────────────────────────
app = FastAPI(
    title="MedAI Screening API",
    version="2.0.0",
    description="Hardened AI screening backend: modality gate, calibrated confidence, Grad-CAM, PDF reports.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,          # lock: koi wildcard nahi
    allow_credentials=False,
    allow_methods=["GET", "POST", "DELETE", "PATCH", "OPTIONS"],
    allow_headers=["*"],
)


class RateLimitMiddleware:
    """Halka in-process rate limiter (single-worker uvicorn ke liye kaafi)."""

    def __init__(self, app, max_requests: int, window_s: int):
        self.app = app
        self.max_requests = max_requests
        self.window_s = window_s
        self.hits: dict[str, deque] = defaultdict(deque)

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            client = scope.get("client")
            ip = client[0] if client else "unknown"
            now = time.time()
            q = self.hits[ip]
            while q and now - q[0] > self.window_s:
                q.popleft()
            if len(q) >= self.max_requests:
                payload = JSONResponse(
                    {"error": "Rate limit exceeded. Please wait a minute and try again."},
                    status_code=429,
                )
                await payload(scope, receive, send)
                return
            q.append(now)
        await self.app(scope, receive, send)


app.add_middleware(RateLimitMiddleware, max_requests=RATE_LIMIT_REQUESTS, window_s=RATE_LIMIT_WINDOW_S)


# ─────────────────────────────────────────────────────────────
# Models (PyTorch engine — sirf tab jab ONNX mojood na ho)
# ─────────────────────────────────────────────────────────────
if not USE_ONNX:
    MODELS = {
        "fracture": {
            "path": "fracture_model.pth",
            "classes": ["Fractured", "Not Fractured"],
            "scan": "X-ray",
            "num_classes": 2,
            "modality": "xray",
            "model": None,
        },
        "brain": {
            "path": "brain_model.pth",
            "classes": ["Glioma", "Meningioma", "No Tumor", "Pituitary"],
            "scan": "Brain MRI",
            "num_classes": 4,
            "modality": "mri",
            "model": None,
        },
        "kidney": {
            "path": "kidney_model.pth",
            "classes": ["Cyst", "Normal", "Stone", "Tumor"],
            "scan": "CT Scan",
            "num_classes": 4,
            "modality": "ct",
            "model": None,
        },
    }

    # IMPORTANT: yeh order RETRAINING NOTEBOOK ke classes se match hona chahiye.
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                            [0.229, 0.224, 0.225]),
    ])


if not USE_ONNX:
    def load_model(path: str, num_classes: int):
        m = models.resnet50(weights=None)
        m.fc = torch.nn.Linear(m.fc.in_features, num_classes)
        m.load_state_dict(torch.load(path, map_location=DEVICE))
        return m.eval().to(DEVICE)

    def get_model(model_type: str):
        entry = MODELS[model_type]
        if entry["model"] is None:
            print(f"[startup] Loading {model_type} model...")
            entry["model"] = load_model(
                os.path.join(BASE_DIR, entry["path"]), entry["num_classes"]
            )
            print(f"[startup] {model_type} model loaded!")
        return entry["model"]


@app.on_event("startup")
def preload():
    if init_db():
        print("[startup] Database ready (Neon Postgres).")
    else:
        print("[startup] WARNING: DATABASE_URL not set — user features disabled.")
    if os.getenv("PRELOAD_MODELS", "0") == "1":
        if USE_ONNX:
            onnx_engine.preload_all()
        else:
            for k in MODELS:
                get_model(k)


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────
def read_upload_image(file: UploadFile) -> Image.Image:
    """Upload validate karo: size limit + PIL verification."""
    contents = file.file.read(MAX_UPLOAD_BYTES + 1)
    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"File too large (max {MAX_UPLOAD_MB:.0f} MB).")
    if not contents:
        raise HTTPException(400, "Empty file.")
    try:
        img = Image.open(io.BytesIO(contents))
        img.load()
    except Exception:
        raise HTTPException(422, "Invalid image file. Please upload a JPG/PNG scan image.")
    return img.convert("RGB")


def run_inference(model, image: Image.Image, model_type: str):
    """Temperature-calibrated inference + Grad-CAM (PyTorch engine)."""
    img_resized = image.resize((224, 224))
    img_float = np.float32(np.array(img_resized)) / 255.0
    tensor = transform(image).unsqueeze(0).to(DEVICE)

    temperature = get_temperature(model_type)

    with torch.no_grad():
        logits = model(tensor).cpu().numpy()[0]

    probs = calibrated_softmax(logits, temperature)
    order = np.argsort(probs)[::-1]
    predicted = int(order[0])
    confidence = round(float(probs[predicted]) * 100, 2)
    alternatives = [
        {"class": MODELS[model_type]["classes"][int(i)],
         "confidence": round(float(probs[int(i)]) * 100, 2)}
        for i in order[1:4]
    ]

    gradcam_b64 = None
    try:
        cam = GradCAM(model=model, target_layers=[model.layer4[-1]])
        targets = [ClassifierOutputTarget(predicted)]
        grayscale_cam = cam(input_tensor=tensor, targets=targets)[0]
        visualization = show_cam_on_image(img_float, grayscale_cam, use_rgb=True)
        _, buffer = cv2.imencode('.jpg', cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))
        gradcam_b64 = base64.b64encode(buffer).decode('utf-8')
    except Exception as e:  # Grad-CAM fail ho to result band na ho
        print(f"[warn] Grad-CAM failed: {e}")

    reliability = evaluate_confidence(confidence)
    return predicted, confidence, gradcam_b64, alternatives, reliability, temperature


# ─────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────
@app.get("/")
def home():
    return {
        "message": "MedAI Screening API v2 ✅",
        "features": ["modality gate", "calibrated confidence", "grad-cam", "pdf reports", "metrics"],
        "models": {k: v["scan"] for k, v in MODELS.items()},
    }


@app.get("/health")
def health():
    return {"status": "ok", "device": "onnx-cpu" if USE_ONNX else str(DEVICE),
            "engine": "onnx" if USE_ONNX else "torch", "time": datetime.utcnow().isoformat() + "Z"}


def _save_analysis(db, user_id, model_type, entry, predicted, confidence,
                   reliability, scan_type_warning, image,
                   alternatives=None, calibration=None, gate=None) -> int | None:
    """Analysis ko Neon me save karo — returns analysis id (ya None)."""
    if db is None:
        return None
    try:
        thumb = image.copy()
        thumb.thumbnail((256, 256))
        buf = io.BytesIO()
        thumb.save(buf, format="JPEG", quality=70)
        thumbnail_b64 = _b64.b64encode(buf.getvalue()).decode("ascii")
    except Exception:
        thumbnail_b64 = ""
    result_json = json.dumps({
        "alternatives": alternatives or [],
        "calibration": calibration or {},
        "modality_gate": gate or {},
    })
    row = Analysis(
        user_id=user_id,
        model_type=model_type,
        scan_type=entry["scan"],
        result=entry["classes"][predicted],
        confidence=float(confidence),
        inconclusive=bool(reliability.get("inconclusive")),
        scan_type_warning=scan_type_warning or "",
        thumbnail_b64=thumbnail_b64,
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return row.id


@app.post("/predict/{model_type}")
async def predict(
    model_type: str,
    file: UploadFile = File(...),
    user: User = Depends(get_current_user),
    db: Session | None = Depends(get_db),
):
    if model_type not in MODELS:
        raise HTTPException(404, f"Invalid model type '{model_type}'. Use: fracture | brain | kidney")

    entry = MODELS[model_type]
    image = read_upload_image(file)

    # ── GATE 1: kya ye sach me medical scan hai? (CNN classifier, fallback: heuristics) ──
    scan_cls = classify_scan_type(image)
    scan_type_warning = None  # GATE 1b set karega (agar CNN available ho)
    # Color/noise guard (CNN se pehle):
    # 1) Colored image (camera photo / random noise) — scan nahi. EXCEPTION:
    #    colored medical overlays (glioma MRI) gray_ratio se pehchane jate hain —
    #    unka colored area chhota hota hai. Colored photo me colored area spread hota hai.
    # 2) High-frequency speckle (random noise) — laplacian energy scan se 10x zyada.
    from scan_type_classifier import saturation_stats
    s_stats = saturation_stats(image)
    a_small = np.asarray(image.convert("L").resize((256, 256)), dtype=np.float32) / 255.0
    lap_energy = float(np.abs(
        4 * a_small[1:-1, 1:-1] - a_small[:-2, 1:-1] - a_small[2:, 1:-1]
        - a_small[1:-1, :-2] - a_small[1:-1, 2:]
    ).mean())

    colored_noise = False
    if s_stats["mean_sat"] >= 0.15 and s_stats["gray_ratio"] < 0.20 and lap_energy > 0.05:
        # REAL photos me colored pixels BRIGHT hote hain; X-rays ka blue/dark tint
        # dark hota hai. Saturated pixels ki brightness check karo.
        rgb = np.asarray(image.convert("RGB").resize((256, 256)), dtype=np.float32) / 255.0
        sat = (rgb.max(-1) - rgb.min(-1)) / (rgb.max(-1) + 1e-6)
        mask = sat >= 0.15
        bright_colored = float(rgb[..., 0][mask].mean() + rgb[..., 1][mask].mean() + rgb[..., 2][mask].mean()) / 3 if mask.any() else 0.0
        if bright_colored > 0.30:
            colored_noise = True
    speckle_noise = lap_energy > 0.20       # scans ~0.02-0.11, noise ~0.43
    if colored_noise or speckle_noise:
        reason = (
            "This looks like a regular color photo, not a medical scan. "
            "Please upload a grayscale X-ray, MRI, or CT image."
            if colored_noise else
            "This image looks like noise, not a medical scan. "
            "Please upload a clear X-ray, MRI, or CT image."
        )
        raise HTTPException(
            422,
            detail={
                "error": "not_a_scan",
                "message": reason,
                "gate": {"score": 0.0, "source": "color_noise_guard"},
            },
        )
    if scan_cls.get("available"):
        label = scan_cls["label"]
        # Sirf CONFIDENT photo/other reject karo (>=0.65). Unsure cases (conf ~0.5)
        # pass karne do — warna real X-rays/MRIs false-reject hote the (MURA eval).
        # Garbage jo pass ho jaye wo inconclusive-band me pakda jayega.
        # EXCEPTION: fully-grayscale images kabhi "photo" nahi — MURA jaisi
        # white-background X-rays CNN ko confuse karti thin (0.66-0.80 photo).
        is_grayscale_img = s_stats["mean_sat"] < 0.05
        if (
            label in ("photo", "other")
            and scan_cls["confidence"] >= 0.65
            and not is_grayscale_img
        ):
            raise HTTPException(
                422,
                detail={
                    "error": "not_a_scan",
                    "message": (
                        "This image does not look like a medical scan. "
                        "Please upload an X-ray, MRI, or CT scan image."
                    ),
                    "gate": {"score": scan_cls["confidence"], "source": "cnn_classifier"},
                },
            )
        # GATE 1b: scan-type ka BIG mismatch (>=0.90) ab HARD BLOCK nahi —
        # warning flag bhejo. Internet X-rays ko MRI/CT confusion se block ho
        # rahi thin (user screenshots); result + warning dono return karte hain.
        expected = entry.get("modality")  # fracture->xray, brain->mri, kidney->ct
        scan_type_warning = None
        if expected and label != expected and scan_cls["confidence"] >= 0.90:
            scan_type_warning = (
                f"Heads-up: this image looks like a {label.upper()}, but the "
                f"{entry['scan']} analyzer was selected. Result may be unreliable "
                f"— consider switching to the correct analyzer."
            )
        gate = {"passed": True, "score": round(scan_cls["confidence"], 4), "source": "cnn_classifier"}
    else:
        # fallback: purana heuristic gate (jab CNN weights mojood na hon)
        gate = check_image(image)
        if not gate["passed"]:
            raise HTTPException(
                422,
                detail={
                    "error": "not_a_scan",
                    "message": gate["reason"],
                    "gate": {"score": gate["score"], "source": gate["source"]},
                },
            )

    # ── GATE 2: calibrated confidence ──
    temperature = get_temperature(model_type)

    if USE_ONNX:
        predicted, confidence, heat, logits = onnx_engine.run_inference(
            model_type, image, temperature
        )
        gradcam = onnx_engine.overlay_heatmap(image, heat) if heat is not None else None

        z = logits / max(temperature, 1e-6)
        z -= z.max()
        e = np.exp(z)
        probs = e / e.sum()
        order = np.argsort(probs)[::-1]
        alternatives = [
            {"class": entry["classes"][int(i)],
             "confidence": round(float(probs[int(i)]) * 100, 2)}
            for i in order[1:4]
        ]
        reliability = evaluate_confidence(confidence)
        # env-override (INCONCLUSIVE_BELOW) calibration ke 60 default ko replace kare
        if INCONCLUSIVE_BELOW > 60 and confidence < INCONCLUSIVE_BELOW and not reliability.get("inconclusive"):
            reliability = {
                "status": "inconclusive",
                "inconclusive": True,
                "message": (
                    "Model is not confident enough for a reliable screening result. "
                    "Please try a clearer, properly-exposed scan image — "
                    "and confirm with a qualified doctor regardless."
                ),
            }
    else:
        m = get_model(model_type)
        predicted, confidence, gradcam, alternatives, reliability, temperature = \
            run_inference(m, image, model_type)

    # ── History save (Neon) — thumbnail + full result JSON ──
    analysis_id = None
    try:
        analysis_id = _save_analysis(
            db, user.id, model_type, entry, predicted, confidence,
            reliability, scan_type_warning, image,
            alternatives=alternatives,
            calibration={"temperature": temperature},
            gate={"score": gate["score"], "source": gate["source"]},
        )
    except Exception as e:
        print(f"[warn] analysis save failed: {e}")

    return {
        "analysis_id": analysis_id,
        "result": entry["classes"][predicted],
        "confidence": confidence,
        "gradcam_image": gradcam,
        "alternatives": alternatives,
        "reliability": reliability,
        "calibration": {"temperature": temperature},
        "modality_gate": {"score": gate["score"], "source": gate["source"]},
        "scan_type_warning": scan_type_warning,
        "scan_type": entry["scan"],
    }


@app.get("/metrics/{model_type}")
async def get_metrics(model_type: str):
    """Retraining notebook ka output: metrics/{model_type}.json"""
    if model_type not in MODELS:
        raise HTTPException(404, f"Invalid model type '{model_type}'.")
    path = os.path.join(MODELS_DIR, f"{model_type}.json")
    if not os.path.exists(path):
        return {"available": False,
                "note": "Metrics file not found — retraining notebook ka output abhi deploy nahi hua."}
    with open(path, "r") as f:
        data = json.load(f)
    data["available"] = True
    return data


@app.post("/generate-report")
async def generate_report(
    file: UploadFile = File(...),
    name: str = Form(...),
    age: str = Form(...),
    gender: str = Form(...),
    phone: str = Form(""),
    result: str = Form(...),
    confidence: str = Form(...),
    gradcam_image: str = Form(""),
    scan_type: str = Form("X-ray"),
    inconclusive: str = Form("false"),
    analysis_id: int = Form(0),
    user: User = Depends(get_current_user),
    db: Session | None = Depends(get_db),
):
    # History record update (patient details) PDF banne ke baad hoga.
    resp = await _build_pdf_response(read_upload_image(file), {
        "name": name, "age": age, "gender": gender, "phone": phone,
        "result": result, "confidence": confidence,
        "gradcam_image": gradcam_image, "scan_type": scan_type,
        "inconclusive": inconclusive,
    })
    try:
        if db is not None and analysis_id:
            row = db.query(Analysis).filter(Analysis.id == analysis_id, Analysis.user_id == user.id).first()
            if row:
                row.patient_name = name
                row.patient_age = age
                row.patient_gender = gender
                db.commit()
    except Exception as e:
        print(f"[warn] report record update failed: {e}")
    return resp


async def _build_pdf_response(image: Image.Image, p: dict):
    """Reportlab PDF builder — /generate-report aur profile-history PDF dono use karte hain."""
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Image as RLImage,
                                    Table, TableStyle, HRFlowable)
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib.enums import TA_CENTER

    name = p["name"]; age = p["age"]; gender = p["gender"]; phone = p.get("phone", "")
    result = p["result"]; confidence = p["confidence"]
    gradcam_image = p.get("gradcam_image", "")
    scan_type = p.get("scan_type", "X-ray"); inconclusive = p.get("inconclusive", "false")
    is_inconclusive = inconclusive.lower() == "true"

    orig_path = tempfile.mktemp(suffix=".jpg")
    image.save(orig_path)

    gradcam_path = None
    if gradcam_image:
        try:
            gradcam_path = tempfile.mktemp(suffix=".jpg")
            with open(gradcam_path, 'wb') as f:
                f.write(base64.b64decode(gradcam_image))
        except Exception:
            gradcam_path = None

    pdf_path = tempfile.mktemp(suffix=".pdf")
    doc = SimpleDocTemplate(
        pdf_path, pagesize=A4,
        rightMargin=0.75*inch, leftMargin=0.75*inch,
        topMargin=0.75*inch, bottomMargin=0.75*inch,
    )

    story = []
    W = 7 * inch

    header_data = [[Paragraph(
        "<font size=20><b>Medical AI Screening Report</b></font><br/>"
        "<font size=10 color='#C9A84C'>AI-Powered Medical Image Analysis System — Calibrated v2</font>",
        ParagraphStyle('hdr', alignment=TA_CENTER, textColor=colors.white, leading=26),
    )]]
    header_table = Table(header_data, colWidths=[W])
    header_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#1a0000')),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('TOPPADDING', (0,0), (-1,-1), 18),
        ('BOTTOMPADDING', (0,0), (-1,-1), 18),
    ]))
    story.append(header_table)
    story.append(Spacer(1, 16))

    label_style = ParagraphStyle('lbl', fontSize=9, textColor=colors.white, fontName='Helvetica-Bold')
    value_style = ParagraphStyle('val', fontSize=10, textColor=colors.HexColor('#1e293b'))

    patient_data = [
        [Paragraph("PATIENT NAME", label_style), Paragraph(name, value_style),
         Paragraph("DATE", label_style), Paragraph(datetime.now().strftime("%d %B %Y"), value_style)],
        [Paragraph("AGE", label_style), Paragraph(age + " years", value_style),
         Paragraph("GENDER", label_style), Paragraph(gender, value_style)],
        [Paragraph("PHONE", label_style), Paragraph(phone if phone else "N/A", value_style),
         Paragraph("SCAN TYPE", label_style), Paragraph(scan_type, value_style)],
    ]
    pt = Table(patient_data, colWidths=[1.4*inch, 2.1*inch, 1.4*inch, 2.1*inch])
    pt.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (0,-1), colors.HexColor('#1a0000')),
        ('BACKGROUND', (2,0), (2,-1), colors.HexColor('#1a0000')),
        ('BACKGROUND', (1,0), (1,-1), colors.HexColor('#fdf8ee')),
        ('BACKGROUND', (3,0), (3,-1), colors.HexColor('#fdf8ee')),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#C9A84C')),
        ('TOPPADDING', (0,0), (-1,-1), 8),
        ('BOTTOMPADDING', (0,0), (-1,-1), 8),
        ('LEFTPADDING', (0,0), (-1,-1), 10),
        ('RIGHTPADDING', (0,0), (-1,-1), 10),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ]))
    story.append(pt)
    story.append(Spacer(1, 16))

    if is_inconclusive:
        banner_color = colors.HexColor('#92400e')  # amber — honest uncertainty
        icon = "〜"
    else:
        normal_results = ["normal", "not fractured", "no tumor"]
        is_normal = result.lower() in normal_results
        banner_color = colors.HexColor('#14532d') if is_normal else colors.HexColor('#7f1d1d')
        icon = "✓" if is_normal else "⚠"

    banner_data = [[Paragraph(
        f"<font size=18><b>{icon}  {result.upper()}</b></font><br/>"
        f"<font size=11>Confidence Score: {confidence}%"
        + ("  ·  INCONCLUSIVE — low model confidence" if is_inconclusive else "")
        + "</font>",
        ParagraphStyle('banner', alignment=TA_CENTER, textColor=colors.white, leading=24),
    )]]
    banner = Table(banner_data, colWidths=[W])
    banner.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), banner_color),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('TOPPADDING', (0,0), (-1,-1), 14),
        ('BOTTOMPADDING', (0,0), (-1,-1), 14),
    ]))
    story.append(banner)
    story.append(Spacer(1, 16))

    if gradcam_path:
        img_w = 2.8*inch
        images_data = [[
            RLImage(orig_path, width=img_w, height=img_w),
            Spacer(0.3*inch, 1),
            RLImage(gradcam_path, width=img_w, height=img_w),
        ], [
            Paragraph("Original Scan", ParagraphStyle('cap', fontSize=9, alignment=TA_CENTER, textColor=colors.grey)),
            Spacer(1, 1),
            Paragraph("AI Focus Area (Grad-CAM)", ParagraphStyle('cap', fontSize=9, alignment=TA_CENTER, textColor=colors.grey)),
        ]]
        img_table = Table(images_data, colWidths=[img_w+0.2*inch, 0.3*inch, img_w+0.2*inch])
        img_table.setStyle(TableStyle([
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('PADDING', (0,0), (-1,-1), 6),
        ]))
        story.append(img_table)
        story.append(Spacer(1, 16))
    else:
        img_w = 2.8*inch
        story.append(RLImage(orig_path, width=img_w, height=img_w))
        story.append(Spacer(1, 16))

    story.append(HRFlowable(width=W, thickness=0.5, color=colors.HexColor('#C9A84C')))
    story.append(Spacer(1, 6))
    disclaimer = ("DISCLAIMER: This report is AI-generated (screening aid, calibrated confidence) "
                  "and must be reviewed by a qualified medical professional.")
    if is_inconclusive:
        disclaimer += " Note: The model reported LOW confidence for this scan — treat this result as inconclusive."
    story.append(Paragraph(
        disclaimer,
        ParagraphStyle('disc', fontSize=7.5, textColor=colors.HexColor('#94a3b8'), alignment=TA_CENTER),
    ))

    doc.build(story)
    os.remove(orig_path)
    if gradcam_path:
        os.remove(gradcam_path)

    return FileResponse(pdf_path, media_type="application/pdf", filename=f"Report_{name}.pdf")


# ── Purana /ai-doctor REMOVE — ek hi source of truth (frontend) ──
@app.post("/ai-doctor")
async def ai_doctor_gone():
    raise HTTPException(
        410,
        detail="The AI assistant has moved to the frontend (MedAI site). "
               "Please use the website's AI Health Assistant instead.",
    )


# ═════════════════════════════════════════════════════════════
# v3 — PERSONALIZATION: auth, profile history, assistant, admin
# (model inference code upar wala hi hai — untouched)
# ═════════════════════════════════════════════════════════════

# ── Schemas ──
class RegisterBody(BaseModel):
    email: str
    password: str
    full_name: str = ""
    age: str = ""
    gender: str = ""


class LoginBody(BaseModel):
    email: str
    password: str


class ProfileBody(BaseModel):
    full_name: str | None = None
    age: str | None = None
    gender: str | None = None


class ChatSendBody(BaseModel):
    session_id: int | None = None
    message: str


class MemoryUpsertBody(BaseModel):
    key: str
    value: str


class ActivityBody(BaseModel):
    kind: str
    detail: str = ""


class AdminLoginBody(BaseModel):
    username: str
    password: str


def _db_ready(db) -> bool:
    return db is not None


# ── AUTH ──
@app.post("/auth/register")
async def auth_register(body: RegisterBody, request: Request, db: Session | None = Depends(get_db)):
    if not _db_ready(db):
        raise HTTPException(503, "Database not configured.")
    email = body.email.strip().lower()
    if not email or "@" not in email or len(email) > 255:
        raise HTTPException(422, "Please enter a valid email address.")
    if len(body.password) < 6:
        raise HTTPException(422, "Password must be at least 6 characters.")

    ip = get_client_ip(request)
    if not register_login_fail(ip, "reg:" + email):
        raise HTTPException(429, "Too many attempts — please wait 15 minutes.")

    exists = db.query(User).filter(User.email == email).first()
    if exists:
        raise HTTPException(409, "An account with this email already exists — please log in.")

    user = User(
        email=email,
        password_hash=hash_password(body.password),
        full_name=body.full_name.strip()[:120],
        age=body.age.strip()[:10],
        gender=body.gender.strip()[:20],
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    clear_login_fails(ip, "reg:" + email)
    return {
        "token": create_token(user.id, user.email),
        "user": {"id": user.id, "email": user.email, "full_name": user.full_name,
                 "age": user.age, "gender": user.gender},
    }


@app.post("/auth/login")
async def auth_login(body: LoginBody, request: Request, db: Session | None = Depends(get_db)):
    if not _db_ready(db):
        raise HTTPException(503, "Database not configured.")
    email = body.email.strip().lower()
    ip = get_client_ip(request)
    if not register_login_fail(ip, email):
        raise HTTPException(429, "Too many failed attempts — please wait 15 minutes.")

    user = db.query(User).filter(User.email == email).first()
    if not user or not verify_password(body.password, user.password_hash):
        record_login_fail(ip, email)
        raise HTTPException(401, "Incorrect email or password.")
    if not user.is_active:
        raise HTTPException(403, "This account has been deactivated.")

    clear_login_fails(ip, email)
    return {
        "token": create_token(user.id, user.email),
        "user": {"id": user.id, "email": user.email, "full_name": user.full_name,
                 "age": user.age, "gender": user.gender},
    }


@app.get("/auth/me")
async def auth_me(user: User = Depends(get_current_user)):
    return {"id": user.id, "email": user.email, "full_name": user.full_name,
            "age": user.age, "gender": user.gender, "created_at": user.created_at.isoformat()}


@app.patch("/auth/profile")
async def auth_update_profile(body: ProfileBody, user: User = Depends(get_current_user),
                              db: Session | None = Depends(get_db)):
    if body.full_name is not None:
        user.full_name = body.full_name.strip()[:120]
    if body.age is not None:
        user.age = body.age.strip()[:10]
    if body.gender is not None:
        user.gender = body.gender.strip()[:20]
    db.commit()
    return {"id": user.id, "email": user.email, "full_name": user.full_name,
            "age": user.age, "gender": user.gender}


# ── PROFILE: analysis history ──
def _analysis_dict(a: Analysis, include_json: bool = False) -> dict:
    d = {
        "id": a.id, "model_type": a.model_type, "scan_type": a.scan_type,
        "result": a.result, "confidence": a.confidence, "inconclusive": a.inconclusive,
        "scan_type_warning": a.scan_type_warning,
        "has_thumbnail": bool(a.thumbnail_b64),
        "patient_name": a.patient_name, "patient_age": a.patient_age,
        "patient_gender": a.patient_gender,
        "created_at": a.created_at.isoformat() if a.created_at else None,
    }
    if include_json:
        try:
            d["result_json"] = json.loads(a.result_json or "{}")
        except Exception:
            d["result_json"] = {}
    return d


@app.get("/profile/analyses")
async def profile_analyses(
    model_type: str = "",
    limit: int = 60,
    offset: int = 0,
    user: User = Depends(get_current_user),
    db: Session | None = Depends(get_db),
):
    q = db.query(Analysis).filter(Analysis.user_id == user.id)
    if model_type in ("fracture", "brain", "kidney"):
        q = q.filter(Analysis.model_type == model_type)
    rows = q.order_by(Analysis.created_at.desc()).offset(max(0, offset)).limit(min(120, max(1, limit))).all()
    total = q.count()
    return {"total": total, "analyses": [_analysis_dict(a) for a in rows]}


@app.get("/profile/analyses/{analysis_id}")
async def profile_analysis_detail(
    analysis_id: int,
    user: User = Depends(get_current_user),
    db: Session | None = Depends(get_db),
):
    a = db.query(Analysis).filter(Analysis.id == analysis_id, Analysis.user_id == user.id).first()
    if not a:
        raise HTTPException(404, "Analysis not found.")
    d = _analysis_dict(a, include_json=True)
    d["thumbnail_b64"] = a.thumbnail_b64 or ""
    return d


@app.get("/profile/analyses/{analysis_id}/thumbnail")
async def profile_analysis_thumb(
    analysis_id: int,
    user: User = Depends(get_current_user),
    db: Session | None = Depends(get_db),
):
    a = db.query(Analysis).filter(Analysis.id == analysis_id, Analysis.user_id == user.id).first()
    if not a or not a.thumbnail_b64:
        raise HTTPException(404, "Thumbnail not found.")
    return {"thumbnail": a.thumbnail_b64}


@app.post("/profile/analyses/{analysis_id}/pdf")
async def profile_analysis_pdf(
    analysis_id: int,
    user: User = Depends(get_current_user),
    db: Session | None = Depends(get_db),
):
    """Purani analysis ka PDF dobara generate karo (stored data se)."""
    a = db.query(Analysis).filter(Analysis.id == analysis_id, Analysis.user_id == user.id).first()
    if not a:
        raise HTTPException(404, "Analysis not found.")
    # Thumbnail se scan image reconstruct karo taake PDF me original scan dikhe.
    img = None
    try:
        img = Image.open(io.BytesIO(_b64.b64decode(a.thumbnail_b64)))
        img.load()
    except Exception:
        pass
    if img is None:
        raise HTTPException(422, "Original scan image not available for this record.")

    return await _build_pdf_response(img, {
        "name": a.patient_name or user.full_name or user.email.split("@")[0],
        "age": a.patient_age or "N/A",
        "gender": a.patient_gender or "N/A",
        "phone": "",
        "result": a.result,
        "confidence": f"{a.confidence}",
        "gradcam_image": "",
        "scan_type": a.scan_type or "X-ray",
        "inconclusive": "true" if a.inconclusive else "false",
    })


# ── CHAT: sessions (ChatGPT-style sidebar) ──
@app.get("/chat/sessions")
async def chat_sessions(user: User = Depends(get_current_user), db: Session | None = Depends(get_db)):
    rows = db.query(ChatSession).filter(ChatSession.user_id == user.id)\
        .order_by(ChatSession.updated_at.desc()).limit(60).all()
    return {"sessions": [{"id": s.id, "title": s.title,
                          "updated_at": s.updated_at.isoformat()} for s in rows]}


@app.post("/chat/sessions")
async def chat_create_session(user: User = Depends(get_current_user), db: Session | None = Depends(get_db)):
    s = ChatSession(user_id=user.id, title="New chat")
    db.add(s)
    db.commit()
    db.refresh(s)
    return {"id": s.id, "title": s.title, "updated_at": s.updated_at.isoformat(), "messages": []}


@app.get("/chat/sessions/{session_id}")
async def chat_get_session(
    session_id: int,
    user: User = Depends(get_current_user),
    db: Session | None = Depends(get_db),
):
    s = db.query(ChatSession).filter(ChatSession.id == session_id, ChatSession.user_id == user.id).first()
    if not s:
        raise HTTPException(404, "Chat session not found.")
    return {"id": s.id, "title": s.title, "updated_at": s.updated_at.isoformat(),
            "messages": [{"role": m.role, "content": m.content} for m in s.messages]}


@app.delete("/chat/sessions/{session_id}")
async def chat_delete_session(
    session_id: int,
    user: User = Depends(get_current_user),
    db: Session | None = Depends(get_db),
):
    s = db.query(ChatSession).filter(ChatSession.id == session_id, ChatSession.user_id == user.id).first()
    if not s:
        raise HTTPException(404, "Chat session not found.")
    db.delete(s)
    db.commit()
    return {"deleted": True}


# ── CHAT: assistant engine (Groq, server-side key) ──
GROQ_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")


def _load_skills() -> str:
    p = os.path.join(BASE_DIR, "skills.md")
    try:
        with open(p, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return "You are the MedAI Health Assistant — a careful, warm AI screening companion."


def _memory_block(db, user_id: int) -> str:
    rows = db.query(UserMemory).filter(UserMemory.user_id == user_id)\
        .order_by(UserMemory.updated_at.desc()).limit(40).all()
    if not rows:
        return "(No stored facts yet — this may be a first conversation. If the user seems new, briefly introduce yourself and start the intake.)"
    lines = [f"- {m.key}: {m.value}" for m in rows]
    return "\n".join(lines)


def _parse_markers(text: str):
    """###MEMORY### / ###HANDOFF### / ###DOCTORFIND### lines nikalo aur clean text do."""
    memory_pairs, handoff, doctorfind = [], None, None
    out_lines = []
    for line in text.split("\n"):
        s = line.strip()
        if s.startswith("###MEMORY###"):
            body = s[len("###MEMORY###"):].strip().lstrip("-").strip()
            if "=" in body:
                k, v = body.split("=", 1)
                k = k.strip()[:80]
                v = v.strip()[:300]
                if k and v:
                    memory_pairs.append((k, v))
            continue
        if s.startswith("###HANDOFF###"):
            try:
                j = s[len("###HANDOFF###"):]
                handoff = json.loads(j[j.index("{"): j.rindex("}") + 1])
            except Exception:
                handoff = None
            continue
        if s.startswith("###DOCTORFIND###"):
            try:
                j = s[len("###DOCTORFIND###"):]
                doctorfind = json.loads(j[j.index("{"): j.rindex("}") + 1])
            except Exception:
                doctorfind = None
            continue
        out_lines.append(line)
    return "\n".join(out_lines).rstrip(), memory_pairs, handoff, doctorfind


def _save_memory(db, user_id: int, pairs):
    for k, v in pairs:
        existing = db.query(UserMemory).filter(UserMemory.user_id == user_id, UserMemory.key == k).first()
        if existing:
            if existing.value.strip().lower() != v.strip().lower():
                existing.value = v
                existing.updated_at = utcnow()
        else:
            db.add(UserMemory(user_id=user_id, key=k, value=v, source="chat"))
    db.commit()


import re as _re

_PAIN_PARTS = ("wrist", "knee", "head", "back", "neck", "shoulder", "elbow",
               "ankle", "hip", "stomach", "chest", "throat", "leg", "arm",
               "foot", "hand", "eye", "ear", "tooth", "jaw")


def _regex_doctorfind_extract(text: str):
    """Deterministic doctor-finder fallback — user ke message me find/doctor/hospital
    intent + city detect karo. LLM marker miss ho tab bhi map khul jaye."""
    t = (text or "")[:1000]
    if not _re.search(r"\b(find|search|nearby|near me|around me|where|kahan|dhundo|dhundna|looking for)\b", t, _re.I):
        return None
    if not _re.search(r"\b(doctor|doctors|hospital|hospitals|clinic|clinics|dentist|orthopedic|cardiologist|pediatrician|eye specialist|ophthalmologist|urologist|nephrologist|neurologist|neurosurgeon|gp|pharmacy|emergency)\b", t, _re.I):
        return None
    spec_map = [
        (r"dentist|dental|tooth|teeth", "dentist"),
        (r"orthopedic|ortho|bone", "orthopedic"),
        (r"cardiolog|heart", "cardiologist"),
        (r"pediatric|child doctor", "pediatrician"),
        (r"eye|ophthal", "eye specialist"),
        (r"urolog|kidney", "urologist"),
        (r"nephrol", "nephrologist"),
        (r"neurosurgeon|brain surgeon", "neurosurgeon"),
        (r"neurolog", "neurologist"),
        (r"pharmacy|medicine shop", "pharmacy"),
        (r"emergency|ambulance", "emergency"),
    ]
    specialty = "general"
    for pat, sp in spec_map:
        if _re.search(pat, t, _re.I):
            specialty = sp
            break
    m = _re.search(r"\b(?:in|near|around|from|at|paas|pass)\s+([A-Z][a-zA-Z]{2,20}(?:\s[A-Z][a-zA-Z]{2,20})?)", t)
    if not m:  # Roman-Urdu: "Karachi me doctor dhundo" — city pehle, baad me 'me'
        m = _re.search(r"\b([A-Z][a-zA-Z]{2,20})\s+(?:me|mein|may|main)\b", t)
    city = m.group(1) if m else ""
    return {"specialty": specialty, "query": city}


def _regex_memory_extract(text: str):
    """Fallback extractor — jab LLM ###MEMORY### marker emit na kare tab basic
    durable facts (age / city / body-part pain) user ke message se pakarta hai."""
    pairs = []
    t = (text or "")[:2000]
    m = _re.search(r"\b(?:i am|i'm|im|age|aged)\s+(\d{1,3})\b", t, _re.I)
    if m and 1 <= int(m.group(1)) <= 120:
        pairs.append(("age", m.group(1)))
    m = _re.search(
        r"\b(?:i live in|i am based in|i'm based in|based in|live in|city of|city)\s+([A-Za-z][A-Za-z ]{2,40})",
        t, _re.I)
    if m:
        city = m.group(1).strip().split("\n")[0].strip(" .,!?.")
        city = " ".join(city.split()[:3]).title()
        if city:
            pairs.append(("city", city))
    for part in _PAIN_PARTS:
        m = _re.search(r"\b" + part + r"\b", t, _re.I)
        if m:
            lo, hi = max(0, m.start() - 40), min(len(t), m.end() + 40)
            if _re.search(r"\b(hurts?|pain|aching|ache|sore)\b", t[lo:hi], _re.I):
                pairs.append((f"{part}_pain", "reported"))
                break
    return pairs


@app.post("/chat/send")
async def chat_send(
    body: ChatSendBody,
    user: User = Depends(get_current_user),
    db: Session | None = Depends(get_db),
):
    if not _db_ready(db):
        raise HTTPException(503, "Database not configured.")
    if not GROQ_KEY:
        raise HTTPException(503, "AI service not configured (GROQ_API_KEY missing).")
    text = body.message.strip()[:4000]
    if not text:
        raise HTTPException(422, "Message is empty.")

    # Session (existing ya new)
    if body.session_id:
        s = db.query(ChatSession).filter(ChatSession.id == body.session_id, ChatSession.user_id == user.id).first()
        if not s:
            raise HTTPException(404, "Chat session not found.")
    else:
        s = ChatSession(user_id=user.id, title=text[:60] + ("…" if len(text) > 60 else ""))
        db.add(s)
        db.commit()
        db.refresh(s)

    # History load (last 16)
    prev = db.query(ChatMessage).filter(ChatMessage.session_id == s.id).order_by(ChatMessage.id).all()
    history = [{"role": m.role, "content": m.content} for m in prev][-16:]

    user_row_msg = ChatMessage(session_id=s.id, role="user", content=text)
    db.add(user_row_msg)

    # System prompt: skills + memory
    system = (
        _load_skills()
        + "\n\n## What we already know about the patient\n"
        + _memory_block(db, user.id)
        + f"\n\nPatient profile: name={user.full_name or 'unknown'}, age={user.age or 'unknown'}, gender={user.gender or 'unknown'}."
    )

    # Groq (non-streaming — markers parse karne ke liye poora text chahiye)
    try:
        resp = await _groq_chat(system, history + [{"role": "user", "content": text}])
    except Exception as e:
        print(f"[warn] groq error: {e}")
        raise HTTPException(502, "AI service is busy — please try again in a moment.")

    clean, mem_pairs, handoff, doctorfind = _parse_markers(resp)
    assistant_msg = ChatMessage(session_id=s.id, role="assistant", content=clean)
    db.add(assistant_msg)
    s.title = s.title if s.title != "New chat" else (text[:60] + ("…" if len(text) > 60 else ""))
    s.updated_at = utcnow()
    db.commit()

    if not mem_pairs:
        mem_pairs = _regex_memory_extract(text)  # fallback — LLM marker miss case
    if mem_pairs:
        _save_memory(db, user.id, mem_pairs)

    # Deterministic doctor-finder fallback — LLM marker miss kare to intent+city
    # user ke message se hi pakdo (find/doctor/hospital/dentist... keywords).
    if not doctorfind:
        doctorfind = _regex_doctorfind_extract(text)

    return {
        "session_id": s.id,
        "reply": clean,
        "handoff": handoff,
        "doctorfind": doctorfind,
        "memory_saved": [k for k, _ in mem_pairs],
    }


async def _groq_chat(system: str, messages: list) -> str:
    import httpx
    async with httpx.AsyncClient(timeout=45) as client:
        r = await client.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {GROQ_KEY}", "Content-Type": "application/json"},
            json={
                "model": GROQ_MODEL,
                "messages": [{"role": "system", "content": system}, *messages],
                "max_tokens": 1400,
                "temperature": 0.4,
            },
        )
        if r.status_code != 200:
            raise RuntimeError(f"groq {r.status_code}: {r.text[:200]}")
        return r.json()["choices"][0]["message"]["content"] or ""


# ── MEMORY: user-visible, editable ──
@app.get("/profile/memory")
async def profile_memory(user: User = Depends(get_current_user), db: Session | None = Depends(get_db)):
    rows = db.query(UserMemory).filter(UserMemory.user_id == user.id)\
        .order_by(UserMemory.updated_at.desc()).all()
    return {"memory": [{"id": m.id, "key": m.key, "value": m.value,
                        "source": m.source, "updated_at": m.updated_at.isoformat()} for m in rows]}


@app.post("/profile/memory")
async def profile_memory_add(body: MemoryUpsertBody, user: User = Depends(get_current_user),
                             db: Session | None = Depends(get_db)):
    k = body.key.strip()[:80].lower().replace(" ", "_")
    if not k:
        raise HTTPException(422, "Key is required.")
    existing = db.query(UserMemory).filter(UserMemory.user_id == user.id, UserMemory.key == k).first()
    if existing:
        existing.value = body.value.strip()[:300]
        existing.updated_at = utcnow()
    else:
        db.add(UserMemory(user_id=user.id, key=k, value=body.value.strip()[:300], source="profile"))
    db.commit()
    return {"ok": True}


@app.delete("/profile/memory/{memory_id}")
async def profile_memory_del(memory_id: int, user: User = Depends(get_current_user),
                             db: Session | None = Depends(get_db)):
    m = db.query(UserMemory).filter(UserMemory.id == memory_id, UserMemory.user_id == user.id).first()
    if not m:
        raise HTTPException(404, "Memory entry not found.")
    db.delete(m)
    db.commit()
    return {"deleted": True}


# ── DOCTOR FINDER — OpenStreetMap (free, no API key) ──
@app.get("/doctors/nearby")
async def doctors_nearby(
    lat: float,
    lon: float,
    specialty: str = "",
    user: User = Depends(get_current_user),
):
    import httpx
    amenity = "hospital"
    tag = "doctors"
    radius = 8000
    overpass_q = (
        f"[out:json][timeout:25];"
        f"nwr[~'^(amenity|healthcare)$'~'^(hospital|clinic|doctors)$'](around:{radius},{lat},{lon});"
        f"out center 60;"
    )
    data = None
    last_err = None
    # Primary + mirror endpoints — Overpass busy hone pe 504 deta hai
    for endpoint in ("https://overpass-api.de/api/interpreter", "https://overpass.kumi.systems/api/interpreter"):
        try:
            async with httpx.AsyncClient(timeout=35, headers={"User-Agent": "MedAI-FYP/1.0"}) as client:
                r = await client.post(endpoint, data={"data": overpass_q})
                r.raise_for_status()
                data = r.json()
                break
        except Exception as e:
            last_err = e
            continue
    if data is None:
        print(f"[warn] overpass error: {last_err}")
        return {"facilities": [], "note": "Location service temporarily unavailable — please try again."}

    import math

    def haversine(la1, lo1, la2, lo2):
        p = math.pi / 180
        return int(6371 * 2 * math.asin(math.sqrt(
            0.5 - math.cos((la2 - la1) * p) / 2 + math.cos(la1 * p) * math.cos(la2 * p) * (1 - math.cos((lo2 - lo1) * p)) / 2)))

    specs = specialty.lower()  # e.g. "orthopedic"
    facilities = []
    for el in data.get("elements", []):
        tags = el.get("tags", {}) or {}
        name = tags.get("name") or "Unnamed clinic"
        la = el.get("lat") or (el.get("center", {}) or {}).get("lat")
        lo = el.get("lon") or (el.get("center", {}) or {}).get("lon")
        if la is None or lo is None:
            continue
        d = haversine(lat, lon, la, lo)
        kind = tags.get("amenity") or tags.get("healthcare") or "clinic"
        specialities = (tags.get("healthcare:speciality") or "").lower()
        score = d
        if specs and specs[:5] in specialities:
            score -= 5  # specialty match ko upar rakho
        facilities.append({
            "name": name, "kind": kind, "distance_km": d,
            "specialities": tags.get("healthcare:speciality", ""),
            "phone": tags.get("phone") or tags.get("contact:phone") or "",
            "address": tags.get("addr:street", "") + (", " + tags.get("addr:city", "") if tags.get("addr:city") else ""),
            "lat": la, "lon": lo,
            "maps": f"https://www.google.com/maps/search/?api=1&query={la},{lo}",
            "_score": score,
        })
    facilities.sort(key=lambda x: x["_score"])
    for f_ in facilities:
        f_.pop("_score", None)
    return {"facilities": facilities[:20]}


@app.get("/doctors/geocode")
async def doctors_geocode(q: str, user: User = Depends(get_current_user)):
    """City name -> lat/lon (Nominatim)."""
    import httpx
    try:
        async with httpx.AsyncClient(timeout=20, headers={"User-Agent": "MedAI-FYP/1.0"}) as client:
            r = await client.get("https://nominatim.openstreetmap.org/search",
                                 params={"q": q, "format": "json", "limit": 1})
            r.raise_for_status()
            data = r.json()
    except Exception:
        raise HTTPException(502, "Location lookup failed — try again or use precise location.")
    if not data:
        raise HTTPException(404, "Location not found — try a bigger nearby city.")
    return {"lat": float(data[0]["lat"]), "lon": float(data[0]["lon"]), "name": data[0].get("display_name", q)}


@app.get("/doctors/directions")
async def doctors_directions(
    lat1: float, lon1: float, lat2: float, lon2: float,
    user: User = Depends(get_current_user),
):
    """OSRM proxy — driving route (lat1,lon1)->(lat2,lon2). Free OSRM servers.
    Primary + mirror endpoints; browser se direct OSRM blocked ho to yehi chalta hai."""
    import httpx
    urls = [
        f"https://router.project-osrm.org/route/v1/driving/{lon1},{lat1};{lon2},{lat2}?overview=full&geometries=geojson",
        f"https://routing.openstreetmap.de/routed-car/route/v1/driving/{lon1},{lat1};{lon2},{lat2}?overview=full&geometries=geojson",
    ]
    last_err = None
    for url in urls:
        try:
            async with httpx.AsyncClient(timeout=25, headers={"User-Agent": "MedAI-FYP/1.0"}) as client:
                r = await client.get(url)
                r.raise_for_status()
                d = r.json()
                rt = (d.get("routes") or [None])[0]
                if not rt:
                    last_err = "empty route"
                    continue
                coords = [[c[1], c[0]] for c in rt["geometry"]["coordinates"]]
                return {"coords": coords, "km": round(rt["distance"] / 1000, 1), "min": int(rt["duration"] // 60)}
        except Exception as e:
            last_err = e
            continue
    print(f"[warn] osrm error: {last_err}")
    raise HTTPException(502, "Routing service unavailable right now — try again shortly.")


# ── ADMIN PANEL API (hardcoded creds: admin / admin123, env-overridable) ──
@app.post("/admin/login")
async def admin_login(body: AdminLoginBody):
    if not require_admin(body.username, body.password):
        raise HTTPException(401, "Invalid admin credentials.")
    # Simple signed token (1-hour)
    payload = {"role": "admin", "exp": datetime.now(timezone.utc) + timedelta(hours=1)}
    from auth import SECRET_KEY, ALGORITHM
    import jwt as _jwt
    return {"token": _jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)}


def _admin_guard(authorization: str = Header(default="")):
    from auth import SECRET_KEY, ALGORITHM
    import jwt as _jwt
    if not authorization.startswith("Bearer "):
        raise HTTPException(401, "Admin login required.")
    try:
        payload = _jwt.decode(authorization[7:], SECRET_KEY, algorithms=[ALGORITHM])
        if payload.get("role") != "admin":
            raise HTTPException(401, "Admin login required.")
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(401, "Admin session expired — log in again.")


@app.get("/admin/overview")
async def admin_overview(authorization: str = Header(default=""), db: Session | None = Depends(get_db)):
    _admin_guard(authorization)
    if not _db_ready(db):
        raise HTTPException(503, "Database not configured.")
    users_n = db.query(User).count()
    analyses_n = db.query(Analysis).count()
    chats_n = db.query(ChatSession).count()
    msgs_n = db.query(ChatMessage).count()
    mem_n = db.query(UserMemory).count()
    act_n = db.query(ActivityLog).count()

    by_model = {"fracture": 0, "brain": 0, "kidney": 0}
    by_result: dict[str, int] = {}
    avg_conf = 0.0
    inconclusive_n = 0
    rows = db.query(Analysis.model_type, Analysis.result, Analysis.confidence, Analysis.inconclusive).all()
    if rows:
        for mt, res, conf, inc in rows:
            if mt in by_model:
                by_model[mt] += 1
            key = f"{mt}:{res}"
            by_result[key] = by_result.get(key, 0) + 1
            avg_conf += conf or 0
            if inc:
                inconclusive_n += 1
        avg_conf = round(avg_conf / len(rows), 2)

    # Daily activity (last 14 days)
    daily: dict[str, int] = {}
    for a in db.query(Analysis.created_at).all():
        if a.created_at:
            day = a.created_at.date().isoformat()
            daily[day] = daily.get(day, 0) + 1

    return {
        "totals": {"users": users_n, "analyses": analyses_n, "chats": chats_n,
                   "messages": msgs_n, "memories": mem_n, "activities": act_n},
        "by_model": by_model,
        "by_result": by_result,
        "avg_confidence": avg_conf,
        "inconclusive": inconclusive_n,
        "daily": sorted(daily.items())[-14:],
    }


@app.get("/admin/users")
async def admin_users(authorization: str = Header(default=""), db: Session | None = Depends(get_db)):
    _admin_guard(authorization)
    if not _db_ready(db):
        raise HTTPException(503, "Database not configured.")
    users = db.query(User).order_by(User.created_at.desc()).limit(200).all()
    out = []
    for u in users:
        out.append({
            "id": u.id, "email": u.email, "full_name": u.full_name,
            "age": u.age, "gender": u.gender, "created_at": u.created_at.isoformat(),
            "analysis_count": db.query(Analysis).filter(Analysis.user_id == u.id).count(),
            "chat_count": db.query(ChatSession).filter(ChatSession.user_id == u.id).count(),
            "memory_count": db.query(UserMemory).filter(UserMemory.user_id == u.id).count(),
            "activity_count": db.query(ActivityLog).filter(ActivityLog.user_id == u.id).count(),
        })
    return {"users": out}


@app.get("/admin/users/{user_id}")
async def admin_user_detail(
    user_id: int,
    authorization: str = Header(default=""),
    db: Session | None = Depends(get_db),
):
    _admin_guard(authorization)
    if not _db_ready(db):
        raise HTTPException(503, "Database not configured.")
    u = db.query(User).filter(User.id == user_id).first()
    if not u:
        raise HTTPException(404, "User not found.")
    analyses = db.query(Analysis).filter(Analysis.user_id == u.id).order_by(Analysis.created_at.desc()).limit(100).all()
    sessions = db.query(ChatSession).filter(ChatSession.user_id == u.id).order_by(ChatSession.updated_at.desc()).limit(50).all()
    session_list = []
    for s in sessions:
        msgs = db.query(ChatMessage).filter(ChatMessage.session_id == s.id).order_by(ChatMessage.id).all()
        session_list.append({
            "id": s.id, "title": s.title, "updated_at": s.updated_at.isoformat(),
            "messages": [{"role": m.role, "content": m.content} for m in msgs],
        })
    memories = db.query(UserMemory).filter(UserMemory.user_id == u.id).all()
    activities = db.query(ActivityLog).filter(ActivityLog.user_id == u.id).order_by(ActivityLog.created_at.desc()).limit(100).all()
    return {
        "user": {"id": u.id, "email": u.email, "full_name": u.full_name,
                 "age": u.age, "gender": u.gender, "created_at": u.created_at.isoformat()},
        "analyses": [_analysis_dict(a, include_json=False) | {"thumbnail_b64": a.thumbnail_b64 or ""} for a in analyses],
        "chats": session_list,
        "memory": [{"key": m.key, "value": m.value, "updated_at": m.updated_at.isoformat()} for m in memories],
        "activities": [{"id": a.id, "kind": a.kind, "detail": a.detail or "", "created_at": a.created_at.isoformat()} for a in activities],
    }


# ── ACTIVITY LOG (user actions — ambulance, find-care, directions) ──
@app.post("/activity")
async def activity_log(body: ActivityBody, user: User = Depends(get_current_user), db: Session | None = Depends(get_db)):
    """Frontend se har chhoti action log karo: {kind: 'ambulance_call', detail: 'Pakistan 1122'}"""
    if not _db_ready(db):
        raise HTTPException(503, "Database not configured.")
    kind = body.kind[:40].strip()
    detail = (body.detail or "")[:300].strip()
    if not kind:
        raise HTTPException(422, "kind required")
    db.add(ActivityLog(user_id=user.id, kind=kind, detail=detail))
    db.commit()
    return {"ok": True}


@app.get("/admin/activity")
async def admin_activity(
    limit: int = 200,
    authorization: str = Header(default=""),
    db: Session | None = Depends(get_db),
):
    """All users' recent activity — admin Activity tab (with user info joined)."""
    _admin_guard(authorization)
    if not _db_ready(db):
        raise HTTPException(503, "Database not configured.")
    rows = (
        db.query(ActivityLog, User.email, User.full_name)
        .join(User, ActivityLog.user_id == User.id)
        .order_by(ActivityLog.created_at.desc())
        .limit(min(limit, 500))
        .all()
    )
    return {
        "activities": [
            {
                "id": a.id, "kind": a.kind, "detail": a.detail or "",
                "created_at": a.created_at.isoformat(),
                "user_id": a.user_id, "user_email": email, "user_name": name,
            }
            for a, email, name in rows
        ]
    }


@app.get("/activity/me")
async def activity_me(limit: int = 50, user: User = Depends(get_current_user), db: Session | None = Depends(get_db)):
    if not _db_ready(db):
        raise HTTPException(503, "Database not configured.")
    rows = db.query(ActivityLog).filter(ActivityLog.user_id == user.id).order_by(ActivityLog.created_at.desc()).limit(min(limit, 100)).all()
    return {"activities": [{"id": a.id, "kind": a.kind, "detail": a.detail or "", "created_at": a.created_at.isoformat()} for a in rows]}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
