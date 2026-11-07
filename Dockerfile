FROM python:3.11-slim

LABEL maintainer="Livio Marzol <https://github.com/livio-marzol/edgebuds>"
LABEL description="EdgeBuds Cloud Backend — Acoustic Respiratory Biomarker (FastAPI + DSP + PyTorch/ONNX + Grad-CAM)"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000

# Instala bibliotecas nativas de sistema para processamento de áudio (libsndfile1, ffmpeg)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    ffmpeg \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Instala dependências Python
COPY backend/requirements.txt /app/backend/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /app/backend/requirements.txt

# Copia código do backend e scripts de treinamento
COPY backend /app/backend
COPY ml_training /app/ml_training

RUN mkdir -p /app/backend/static/gradcam

EXPOSE 8000

CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
