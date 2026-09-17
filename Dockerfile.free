# Free-tier Dockerfile — ONNX engine, cv2-free, Back4App Containers (256MB) me fit
FROM python:3.11-slim

WORKDIR /app

# libgl1/libglib2.0 opencv ke liye the — ab cv2 hat gaya, sirf minimal libs
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-free.txt .
RUN pip install --no-cache-dir -r requirements-free.txt

COPY . .

# LRU memory cap: 256MB RAM me 2 models loaded rehte hain, teesra demand pe swap hota hai
ENV PRELOAD_MODELS=1 \
    MAX_LOADED_MODELS=2 \
    ORT_THREADS=1 \
    ORT_MEM_ARENA=0

# Back4App/Render dynamic PORT dete hain; local pe 8000 default
ENV PORT=8000
EXPOSE 8000
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT}"]
