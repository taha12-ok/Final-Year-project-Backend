from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import torch
import torchvision.models as models
from torchvision import transforms
from PIL import Image, ImageFile
import io
import base64
import numpy as np
import cv2
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle, HRFlowable
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER
import tempfile
import os
from datetime import datetime
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

ImageFile.LOAD_TRUNCATED_IMAGES = True

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def load_model(path, num_classes):
    model_path = os.path.join(BASE_DIR, path)
    m = models.resnet50(weights=None)
    m.fc = torch.nn.Linear(m.fc.in_features, num_classes)
    m.load_state_dict(torch.load(model_path, map_location=device))
    m.eval().to(device)
    return m

MODELS = {
    "fracture": {
        "path": "fracture_model.pth",
        "classes": ["Fractured", "Not Fractured"],
        "scan": "X-ray",
        "num_classes": 2,
        "model": None
    },
    "chest": {
        "path": "chest_model.pth",
        "classes": ["COVID-19", "Normal", "Pneumonia"],
        "scan": "Chest X-ray",
        "num_classes": 3,
        "model": None
    },
    "brain": {
        "path": "brain_model.pth",
        "classes": ["Glioma", "Meningioma", "No Tumor", "Pituitary"],
        "scan": "Brain MRI",
        "num_classes": 4,
        "model": None
    },
    "skin": {
        "path": "skin_model.pth",
        "classes": ["Actinic Keratosis", "Basal Cell Carcinoma", "Dermatofibroma",
                    "Melanoma", "Nevus", "Pigmented Benign Keratosis", "Seborrheic Keratosis"],
        "scan": "Skin Image",
        "num_classes": 7,
        "model": None
    },
    "breast": {
        "path": "breast_model.pth",
        "classes": ["Benign", "Normal", "Malignant"],
        "scan": "Ultrasound",
        "num_classes": 3,
        "model": None
    },
    "eye": {
        "path": "eye_model.pth",
        "classes": ["Cataract", "Diabetic Retinopathy", "Glaucoma", "Normal"],
        "scan": "Retina Image",
        "num_classes": 4,
        "model": None
    },
    "kidney": {
        "path": "kidney_model.pth",
        "classes": ["Cyst", "Normal", "Stone", "Tumor"],
        "scan": "CT Scan",
        "num_classes": 4,
        "model": None
    },
}

def get_model(model_type):
    if MODELS[model_type]["model"] is None:
        print(f"Loading {model_type} model...")
        MODELS[model_type]["model"] = load_model(
            MODELS[model_type]["path"],
            MODELS[model_type]["num_classes"]
        )
        print(f"{model_type} model loaded!")
    return MODELS[model_type]["model"]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

def run_inference(model, image):
    img_resized = image.resize((224, 224))
    img_array   = np.array(img_resized) / 255.0
    img_float   = np.float32(img_array)
    tensor      = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs               = model(tensor)
        probs                 = torch.softmax(outputs, dim=1)
        confidence, predicted = probs.max(1)

    cam           = GradCAM(model=model, target_layers=[model.layer4[-1]])
    grayscale_cam = cam(input_tensor=tensor)[0]
    visualization = show_cam_on_image(img_float, grayscale_cam, use_rgb=True)

    _, buffer  = cv2.imencode('.jpg', cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))
    img_base64 = base64.b64encode(buffer).decode('utf-8')

    return predicted.item(), round(confidence.item() * 100, 2), img_base64

@app.get("/")
def home():
    return {"message": "Medical AI Platform Ready! 7 Models + AI Doctor Loaded ✅"}

@app.post("/predict/{model_type}")
async def predict(model_type: str, file: UploadFile = File(...)):
    if model_type not in MODELS:
        return {"error": "Invalid model type"}

    contents = await file.read()
    image    = Image.open(io.BytesIO(contents)).convert("RGB")
    m        = get_model(model_type)
    classes  = MODELS[model_type]["classes"]
    predicted, confidence, gradcam = run_inference(m, image)

    return {
        "result":        classes[predicted],
        "confidence":    confidence,
        "gradcam_image": gradcam
    }

@app.post("/ai-doctor")
async def ai_doctor(data: dict):
    name     = data.get("name", "Patient")
    age      = data.get("age", "")
    gender   = data.get("gender", "")
    history  = data.get("history", "")
    symptoms = data.get("symptoms", "")
    duration = data.get("duration", "")
    severity = data.get("severity", "")

    prompt = f"""You are an expert AI medical assistant with years of clinical experience. 
Analyze the following patient information and provide a comprehensive medical assessment.

Patient Information:
- Name: {name}
- Age: {age} years
- Gender: {gender}
- Medical History: {history}
- Current Symptoms: {symptoms}
- Duration of Symptoms: {duration}
- Severity: {severity}

Please provide your assessment in the following structured format:

## 🔍 Initial Assessment
Brief overview of the patient's condition based on symptoms.

## ⚠️ Possible Conditions
List 2-4 possible conditions with brief explanations.

## 🧪 Recommended Tests & Scans
List specific tests needed (blood tests, X-ray, MRI, CT scan, etc.)

## 🏥 Which AI Model to Use
Based on the symptoms, suggest which of our AI detection models to use:
- 🦴 Fracture Detection (X-ray)
- 🫁 Chest Disease (Chest X-ray)
- 🧠 Brain Tumor (Brain MRI)
- 🔬 Skin Cancer (Skin Image)
- 🎗️ Breast Cancer (Ultrasound)
- 👁️ Eye Disease (Retina Image)
- 🫘 Kidney Disease (CT Scan)

## 💊 Immediate Recommendations
Lifestyle changes and immediate steps to take.

## 🚨 Urgency Level
State: LOW / MEDIUM / HIGH and why.

## 👨‍⚕️ Specialist to Consult
Which type of doctor to see.

---
⚠️ DISCLAIMER: This is an AI assessment for educational purposes only. Always consult a qualified medical professional for actual diagnosis and treatment."""

    response = groq_client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1500,
        temperature=0.3,
    )

    return {"response": response.choices[0].message.content}

@app.post("/generate-report")
async def generate_report(
    file: UploadFile = File(...),
    name: str = Form(...),
    age: str = Form(...),
    gender: str = Form(...),
    phone: str = Form(""),
    result: str = Form(...),
    confidence: str = Form(...),
    gradcam_image: str = Form(...),
    scan_type: str = Form("X-ray"),
):
    contents  = await file.read()
    orig_img  = Image.open(io.BytesIO(contents)).convert("RGB")
    orig_path = tempfile.mktemp(suffix=".jpg")
    orig_img.save(orig_path)

    gradcam_bytes = base64.b64decode(gradcam_image)
    gradcam_path  = tempfile.mktemp(suffix=".jpg")
    with open(gradcam_path, 'wb') as f:
        f.write(gradcam_bytes)

    pdf_path = tempfile.mktemp(suffix=".pdf")
    doc = SimpleDocTemplate(
        pdf_path, pagesize=A4,
        rightMargin=0.75*inch, leftMargin=0.75*inch,
        topMargin=0.75*inch,   bottomMargin=0.75*inch
    )

    story = []
    W = 7 * inch

    header_data = [[Paragraph(
        "<font size=20><b>Medical AI Diagnosis Report</b></font><br/>"
        "<font size=10 color='#C9A84C'>AI-Powered Medical Image Analysis System</font>",
        ParagraphStyle('hdr', alignment=TA_CENTER, textColor=colors.white, leading=26)
    )]]
    header_table = Table(header_data, colWidths=[W])
    header_table.setStyle(TableStyle([
        ('BACKGROUND',    (0,0), (-1,-1), colors.HexColor('#1a0000')),
        ('ALIGN',         (0,0), (-1,-1), 'CENTER'),
        ('TOPPADDING',    (0,0), (-1,-1), 18),
        ('BOTTOMPADDING', (0,0), (-1,-1), 18),
    ]))
    story.append(header_table)
    story.append(Spacer(1, 16))

    label_style = ParagraphStyle('lbl', fontSize=9,  textColor=colors.white, fontName='Helvetica-Bold')
    value_style = ParagraphStyle('val', fontSize=10, textColor=colors.HexColor('#1e293b'))

    patient_data = [
        [Paragraph("PATIENT NAME", label_style), Paragraph(name, value_style),
         Paragraph("DATE",         label_style), Paragraph(datetime.now().strftime("%d %B %Y"), value_style)],
        [Paragraph("AGE",          label_style), Paragraph(age + " years", value_style),
         Paragraph("GENDER",       label_style), Paragraph(gender, value_style)],
        [Paragraph("PHONE",        label_style), Paragraph(phone if phone else "N/A", value_style),
         Paragraph("SCAN TYPE",    label_style), Paragraph(scan_type, value_style)],
    ]
    pt = Table(patient_data, colWidths=[1.4*inch, 2.1*inch, 1.4*inch, 2.1*inch])
    pt.setStyle(TableStyle([
        ('BACKGROUND',    (0,0), (0,-1), colors.HexColor('#1a0000')),
        ('BACKGROUND',    (2,0), (2,-1), colors.HexColor('#1a0000')),
        ('BACKGROUND',    (1,0), (1,-1), colors.HexColor('#fdf8ee')),
        ('BACKGROUND',    (3,0), (3,-1), colors.HexColor('#fdf8ee')),
        ('GRID',          (0,0), (-1,-1), 0.5, colors.HexColor('#C9A84C')),
        ('TOPPADDING',    (0,0), (-1,-1), 8),
        ('BOTTOMPADDING', (0,0), (-1,-1), 8),
        ('LEFTPADDING',   (0,0), (-1,-1), 10),
        ('RIGHTPADDING',  (0,0), (-1,-1), 10),
        ('VALIGN',        (0,0), (-1,-1), 'MIDDLE'),
    ]))
    story.append(pt)
    story.append(Spacer(1, 16))

    normal_results = ["normal", "not fractured", "no tumor", "nevus", "benign"]
    is_normal    = result.lower() in normal_results
    banner_color = colors.HexColor('#14532d') if is_normal else colors.HexColor('#7f1d1d')
    icon         = "✓" if is_normal else "⚠"

    banner_data = [[Paragraph(
        f"<font size=18><b>{icon}  {result.upper()}</b></font><br/>"
        f"<font size=11>Confidence Score: {confidence}%</font>",
        ParagraphStyle('banner', alignment=TA_CENTER, textColor=colors.white, leading=24)
    )]]
    banner = Table(banner_data, colWidths=[W])
    banner.setStyle(TableStyle([
        ('BACKGROUND',    (0,0), (-1,-1), banner_color),
        ('ALIGN',         (0,0), (-1,-1), 'CENTER'),
        ('TOPPADDING',    (0,0), (-1,-1), 14),
        ('BOTTOMPADDING', (0,0), (-1,-1), 14),
    ]))
    story.append(banner)
    story.append(Spacer(1, 16))

    img_w = 2.8*inch
    images_data = [[
        RLImage(orig_path,    width=img_w, height=img_w),
        Spacer(0.3*inch, 1),
        RLImage(gradcam_path, width=img_w, height=img_w),
    ],[
        Paragraph("Original Scan", ParagraphStyle('cap', fontSize=9, alignment=TA_CENTER, textColor=colors.grey)),
        Spacer(1,1),
        Paragraph("AI Focus Area (Grad-CAM)", ParagraphStyle('cap', fontSize=9, alignment=TA_CENTER, textColor=colors.grey)),
    ]]
    img_table = Table(images_data, colWidths=[img_w+0.2*inch, 0.3*inch, img_w+0.2*inch])
    img_table.setStyle(TableStyle([
        ('ALIGN',  (0,0), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('PADDING',(0,0), (-1,-1), 6),
    ]))
    story.append(img_table)
    story.append(Spacer(1, 16))

    story.append(HRFlowable(width=W, thickness=0.5, color=colors.HexColor('#C9A84C')))
    story.append(Spacer(1, 6))
    story.append(Paragraph(
        "DISCLAIMER: This report is AI-generated and must be reviewed by a qualified medical professional.",
        ParagraphStyle('disc', fontSize=7.5, textColor=colors.HexColor('#94a3b8'), alignment=TA_CENTER)
    ))

    doc.build(story)
    os.remove(orig_path)
    os.remove(gradcam_path)

    return FileResponse(pdf_path, media_type="application/pdf", filename=f"Report_{name}.pdf")
