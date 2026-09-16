---
title: MedAI Screening Backend
emoji: 🏥
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 8000
pinned: false
---

# MedAI Screening Backend (FastAPI)

AI medical screening API — fracture (X-ray), brain tumor (MRI), kidney disease (CT).

- Modality gate: rejects non-scan images before inference
- Temperature-calibrated confidence + inconclusive verdicts
- Grad-CAM heatmaps + PDF reports
- `/metrics/{model}` for the Model Lab dashboard

Frontend: https://github.com/taha12-ok/Final-Year-project-Frontend
Live site: https://final-year-project-medai.vercel.app

> Screening aid only — NOT a medical diagnosis.
