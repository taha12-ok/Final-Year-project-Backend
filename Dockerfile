FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libxcb1 \
    libx11-6 \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Models startup pe load honge (PRELOAD_MODELS=1) — pehli request slow na ho
ENV PRELOAD_MODELS=1

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
