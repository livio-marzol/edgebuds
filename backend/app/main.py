"""
Cloud Backend API (FastAPI) — EdgeBuds Acoustic Respiratory Biomarker.

Implementa o contrato de interface da Seção 6 do PRD/Spec:
- POST /api/v1/analyze-cough (multipart/form-data: audio_file, device_type, patient_metadata)
- POST /api/v1/simulate-cough (simulação de perfis acústicos clínicos para demonstração)
- GET /health (status de saúde do serviço)
- GET / (Dashboard Interativo de Demonstração & Visualização de Grad-CAM)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

from .dsp.dsp_pipeline import generate_synthetic_cough_wav
from .service import AcousticAnalysisOrchestrator

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"
GRADCAM_DIR = STATIC_DIR / "gradcam"
GRADCAM_DIR.mkdir(parents=True, exist_ok=True)

PUBLIC_API_BASE_URL = os.getenv("PUBLIC_API_BASE_URL", "")

orchestrator = AcousticAnalysisOrchestrator(
    static_gradcam_dir=GRADCAM_DIR,
    weights_path=os.getenv("MODEL_WEIGHTS_PATH"),
    base_url=PUBLIC_API_BASE_URL or "https://api.domain.com",
)

try:
    from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import HTMLResponse, JSONResponse
    from fastapi.staticfiles import StaticFiles

    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    FastAPI = None  # type: ignore


DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="pt-BR">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>EdgeBuds — Acoustic Respiratory Biomarker Dashboard</title>
  <style>
    :root {
      --bg: #0b0f19;
      --card: #131b2e;
      --border: #1e293b;
      --accent: #38bdf8;
      --text: #f1f5f9;
      --muted: #94a3b8;
    }
    * { box-sizing: border-box; font-family: system-ui, -apple-system, sans-serif; }
    body { margin: 0; background: var(--bg); color: var(--text); padding: 24px; }
    .container { max-width: 1080px; margin: 0 auto; }
    header { border-bottom: 1px solid var(--border); padding-bottom: 16px; margin-bottom: 24px; display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 12px; }
    .badge { background: rgba(56, 189, 248, 0.15); color: var(--accent); border: 1px solid var(--accent); padding: 4px 10px; border-radius: 999px; font-size: 12px; font-weight: 600; }
    .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 20px; }
    .card { background: var(--card); border: 1px solid var(--border); border-radius: 12px; padding: 20px; }
    h1 { margin: 0; font-size: 22px; }
    h2 { margin-top: 0; font-size: 16px; color: var(--accent); }
    button { background: var(--accent); color: #0f172a; border: none; border-radius: 8px; padding: 10px 16px; font-weight: 700; cursor: pointer; margin-right: 8px; margin-bottom: 8px; }
    button.secondary { background: #1e293b; color: var(--text); border: 1px solid #334155; }
    pre { background: #070a12; padding: 12px; border-radius: 8px; overflow-x: auto; font-size: 12px; color: #a5f3fc; border: 1px solid var(--border); }
    img { max-width: 100%; border-radius: 8px; border: 1px solid var(--border); margin-top: 10px; }
    .disclaimer { margin-top: 24px; font-size: 12px; color: var(--muted); border-top: 1px solid var(--border); padding-top: 12px; }
  </style>
</head>
<body>
  <div class="container">
    <header>
      <div>
        <h1>EdgeBuds — Biomarcador Acústico Respiratório (Hearable MVP)</h1>
        <p style="margin:4px 0 0; color:var(--muted); font-size:14px;">
          Samsung Galaxy Buds (Bluetooth SCO 16 kHz Mono) + DSP Pipeline + CNN/YAMNet + Grad-CAM
        </p>
      </div>
      <span class="badge">SaMD Research MVP • FastAPI v1</span>
    </header>
    <div class="grid">
      <div class="card">
        <h2>1. Simular Check-in Ativo (Amostras Acústicas 16 kHz)</h2>
        <p style="color:var(--muted); font-size:13px;">
          Teste o pipeline completo (Filtro Passa-Faixa 100-7500 Hz, VAD RMS 50ms, Mel-Spectrogram 128 bandas e Grad-CAM):
        </p>
        <button onclick="runSimulation('tosse_produtiva')">Tosse Produtiva (Úmida)</button>
        <button onclick="runSimulation('tosse_seca')">Tosse Seca (Irritativa)</button>
        <button class="secondary" onclick="runSimulation('saudavel')">Saudável / Limpo</button>
        <hr style="border-color:var(--border); margin:16px 0;" />
        <h2>2. Upload de Arquivo WAV (16 kHz Mono)</h2>
        <input type="file" id="wavInput" accept=".wav" style="margin-bottom:12px; display:block;" />
        <button onclick="uploadWav()">Analisar WAV via /api/v1/analyze-cough</button>
      </div>
      <div class="card">
        <h2>Laudo Explicativo (Grad-CAM & JSON)</h2>
        <div id="heatmapContainer"></div>
        <pre id="jsonOutput">Selecione uma simulação ou envie um arquivo .wav para visualizar o resultado.</pre>
      </div>
    </div>
    <div class="disclaimer">
      <strong>Aviso Ético e Regulatório (SaMD / LGPD):</strong> Este sistema é um protótipo de pesquisa em Engenharia Biomédica e Processamento Digital de Sinais. Os resultados probabilísticos não constituem diagnóstico médico definitivo e não substituem avaliação clínica presencial.
    </div>
  </div>
  <script>
    async function runSimulation(profile) {
      document.getElementById('jsonOutput').textContent = 'Processando pipeline DSP + Grad-CAM...';
      const form = new FormData();
      form.append('profile', profile);
      form.append('device_type', 'samsung_buds_2');
      const res = await fetch('/api/v1/simulate-cough', { method: 'POST', body: form });
      const data = await res.json();
      renderResult(data);
    }
    async function uploadWav() {
      const fileInput = document.getElementById('wavInput');
      if (!fileInput.files.length) return alert('Selecione um arquivo .wav primeiro.');
      const form = new FormData();
      form.append('audio_file', fileInput.files[0]);
      form.append('device_type', 'samsung_buds_2');
      form.append('patient_metadata', JSON.stringify({ age: 34, has_fever: false, smoker: false }));
      const res = await fetch('/api/v1/analyze-cough', { method: 'POST', body: form });
      const data = await res.json();
      renderResult(data);
    }
    function renderResult(data) {
      document.getElementById('jsonOutput').textContent = JSON.stringify(data, null, 2);
      if (data.explainability && data.explainability.spectrogram_heatmap_url) {
        document.getElementById('heatmapContainer').innerHTML =
          `<img src="${data.explainability.spectrogram_heatmap_url}" alt="Grad-CAM Spectrogram" />`;
      }
    }
  </script>
</body>
</html>
"""


if FASTAPI_AVAILABLE:
    app = FastAPI(
        title="EdgeBuds Acoustic Respiratory Biomarker API",
        description=(
            "API assíncrona para triagem acústica de tosse e distúrbios respiratórios "
            "via fones Samsung Galaxy Buds (Bluetooth SCO 16 kHz Mono), DSP e Grad-CAM."
        ),
        version="1.0.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.get("/", response_class=HTMLResponse)
    async def root_dashboard() -> HTMLResponse:
        """Renderiza o Dashboard Interativo de Demonstração do EdgeBuds."""
        return HTMLResponse(content=DASHBOARD_HTML)

    @app.get("/health")
    async def healthcheck() -> Dict[str, Any]:
        """Verifica integridade operacional do backend e do pipeline DSP."""
        return {
            "status": "healthy",
            "service": "edgebuds-acoustic-biomarker-api",
            "dsp_config": {
                "sample_rate_hz": 16000,
                "channels": 1,
                "bandpass_hz": [100.0, 7500.0],
                "vad_window_ms": 50.0,
                "stft": {"n_fft": 2048, "hop_length": 512, "n_mels": 128},
            },
        }

    @app.post("/api/v1/analyze-cough")
    async def analyze_cough(
        request: Request,
        audio_file: UploadFile = File(..., description="Arquivo .wav (PCM 16-bit, 16kHz mono)"),
        device_type: str = Form("samsung_buds_2", description="Identificador do hearable"),
        patient_metadata: Optional[str] = Form(
            None, description="JSON string opcional (idade, febre, tabagismo)"
        ),
    ) -> JSONResponse:
        """
        Endpoint oficial de inferência clínica (Seção 6 do PRD/Spec).
        Recebe um arquivo WAV capturado via Bluetooth SCO, executa o pipeline DSP,
        classifica em [saudavel, tosse_seca, tosse_produtiva] e gera o mapa Grad-CAM.
        """
        filename = (audio_file.filename or "cough.wav").lower()
        if not filename.endswith(".wav"):
            raise HTTPException(
                status_code=400,
                detail="Formato de arquivo inválido. Envie um arquivo de áudio '.wav'.",
            )

        audio_bytes = await audio_file.read()
        base_url = PUBLIC_API_BASE_URL or str(request.base_url).rstrip("/")

        try:
            payload = orchestrator.analyze_cough_wav(
                audio_bytes=audio_bytes,
                device_type=device_type,
                patient_metadata_json=patient_metadata,
                request_base_url=base_url,
            )
            return JSONResponse(status_code=200, content=payload)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/api/v1/simulate-cough")
    async def simulate_cough(
        request: Request,
        profile: str = Form("tosse_produtiva"),
        device_type: str = Form("samsung_buds_2"),
    ) -> JSONResponse:
        """
        Gera uma amostra sintética de áudio 16 kHz Mono para o perfil selecionado
        ('tosse_produtiva', 'tosse_seca', 'saudavel', 'silencio') e executa o pipeline completo.
        """
        if profile not in ("tosse_produtiva", "tosse_seca", "saudavel", "silencio"):
            raise HTTPException(status_code=400, detail="Perfil de simulação inválido.")

        wav_bytes = generate_synthetic_cough_wav(profile=profile)
        base_url = PUBLIC_API_BASE_URL or str(request.base_url).rstrip("/")
        payload = orchestrator.analyze_cough_wav(
            audio_bytes=wav_bytes,
            device_type=device_type,
            request_base_url=base_url,
        )
        return JSONResponse(status_code=200, content=payload)
