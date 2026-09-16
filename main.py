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

Env vars:
  ALLOWED_ORIGINS   comma-separated, e.g. "https://final-year-project-medai.vercel.app,http://localhost:3000"
  MAX_UPLOAD_MB     default 10
  PRELOAD_MODELS    "1" => startup pe teeno models load (HF Space ke liye recommended)
"""

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
import torch
import torchvision.models as models
from torchvision import transforms
from PIL import Image, ImageFile
import io
import base64
import json
import os
import time
import tempfile
from datetime import datetime
from collections import defaultdict, deque

import numpy as np
import cv2
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from modality import check_image
from calibration import get_temperature, calibrated_softmax, evaluate_confidence

ImageFile.LOAD_TRUNCATED_IMAGES = True

# ─────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    allow_methods=["GET", "POST"],
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
# Models
# ─────────────────────────────────────────────────────────────
MODELS = {
    "fracture": {
        "path": "fracture_model.pth",
        "classes": ["Fractured", "Not Fractured"],
        "scan": "X-ray",
        "num_classes": 2,
        "model": None,
    },
    "brain": {
        "path": "brain_model.pth",
        "classes": ["Glioma", "Meningioma", "No Tumor", "Pituitary"],
        "scan": "Brain MRI",
        "num_classes": 4,
        "model": None,
    },
    "kidney": {
        "path": "kidney_model.pth",
        "classes": ["Cyst", "Normal", "Stone", "Tumor"],
        "scan": "CT Scan",
        "num_classes": 4,
        "model": None,
    },
}

# IMPORTANT: yeh order RETRAINING NOTEBOOK ke classes se match hona chahiye.
# Notebook ye order hi use karta hai, isliye drop-in compatible hai.

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])


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
    if os.getenv("PRELOAD_MODELS", "0") == "1":
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
    """Temperature-calibrated inference + Grad-CAM."""
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
    return {"status": "ok", "device": str(DEVICE), "time": datetime.utcnow().isoformat() + "Z"}


@app.post("/predict/{model_type}")
async def predict(model_type: str, file: UploadFile = File(...)):
    if model_type not in MODELS:
        raise HTTPException(404, f"Invalid model type '{model_type}'. Use: fracture | brain | kidney")

    entry = MODELS[model_type]
    image = read_upload_image(file)

    # ── GATE 1: kya ye sach me medical scan hai? ──
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
    m = get_model(model_type)
    predicted, confidence, gradcam, alternatives, reliability, temperature = \
        run_inference(m, image, model_type)

    return {
        "result": entry["classes"][predicted],
        "confidence": confidence,
        "gradcam_image": gradcam,
        "alternatives": alternatives,
        "reliability": reliability,
        "calibration": {"temperature": temperature},
        "modality_gate": {"score": gate["score"], "source": gate["source"]},
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
):
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Image as RLImage,
                                    Table, TableStyle, HRFlowable)
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib.enums import TA_CENTER

    image = read_upload_image(file)
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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
