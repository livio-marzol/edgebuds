"""
Cloud Backend API (FastAPI) — EdgeBuds Acoustic Respiratory Biomarker.

Implementa o contrato de interface da Seção 6 do PRD/Spec:
- POST /api/v1/analyze-cough (multipart/form-data: audio_file, device_type, patient_metadata)
- POST /api/v1/simulate-cough (simulação de perfis acústicos clínicos para demonstração)
- GET /health (status de saúde do serviço)
- GET / (Web App Mobile-First para Check-in Ativo no Celular + Visualização Grad-CAM)
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
  <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0" />
  <title>EdgeBuds — Mobile Check-in & Grad-CAM</title>
  <style>
    :root {
      --bg: #0b0f19;
      --card: #131b2e;
      --border: #1e293b;
      --accent: #38bdf8;
      --green: #10b981;
      --amber: #f59e0b;
      --text: #f1f5f9;
      --muted: #94a3b8;
    }
    * { box-sizing: border-box; font-family: system-ui, -apple-system, sans-serif; }
    body { margin: 0; background: var(--bg); color: var(--text); padding: 16px; }
    .container { max-width: 960px; margin: 0 auto; }
    header { border-bottom: 1px solid var(--border); padding-bottom: 14px; margin-bottom: 18px; display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px; }
    .badge { background: rgba(56, 189, 248, 0.15); color: var(--accent); border: 1px solid var(--accent); padding: 4px 10px; border-radius: 999px; font-size: 12px; font-weight: 600; }
    .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 16px; }
    .card { background: var(--card); border: 1px solid var(--border); border-radius: 14px; padding: 18px; }
    h1 { margin: 0; font-size: 20px; }
    h2 { margin-top: 0; font-size: 15px; color: var(--accent); text-transform: uppercase; letter-spacing: 0.5px; }
    button, .file-btn { display: inline-block; width: 100%; text-align: center; background: var(--accent); color: #0f172a; border: none; border-radius: 10px; padding: 13px 16px; font-weight: 700; font-size: 14px; cursor: pointer; margin-bottom: 10px; text-decoration: none; }
    button.green { background: var(--green); color: #052e16; }
    button.secondary { background: #1e293b; color: var(--text); border: 1px solid #334155; }
    .countdown-box { background: #070a12; border: 1px solid var(--amber); color: var(--amber); border-radius: 10px; padding: 14px; text-align: center; font-size: 20px; font-weight: 800; margin-bottom: 12px; display: none; }
    .metrics-grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 8px; margin-bottom: 12px; }
    .metric-pill { background: #070a12; border: 1px solid var(--border); border-radius: 8px; padding: 10px; }
    .metric-pill span { display: block; font-size: 11px; color: var(--muted); }
    .metric-pill strong { font-size: 15px; color: var(--text); }
    .prob-bar-wrap { margin-bottom: 8px; font-size: 13px; }
    .prob-bar { height: 8px; background: #1e293b; border-radius: 99px; overflow: hidden; margin-top: 4px; }
    .prob-fill { height: 100%; background: var(--accent); }
    pre { background: #070a12; padding: 12px; border-radius: 8px; overflow-x: auto; font-size: 11px; color: #a5f3fc; border: 1px solid var(--border); max-height: 260px; }
    img { width: 100%; border-radius: 8px; border: 1px solid var(--border); margin-top: 8px; }
    .disclaimer { margin-top: 20px; font-size: 12px; color: var(--muted); border-top: 1px solid var(--border); padding-top: 12px; }
  </style>
</head>
<body>
  <div class="container">
    <header>
      <div>
        <h1>EdgeBuds — Check-in Acústico Mobile</h1>
        <p style="margin:4px 0 0; color:var(--muted); font-size:13px;">
          Conecte seu Samsung Galaxy Buds via Bluetooth e execute o teste guiado de 3 tosses
        </p>
      </div>
      <span class="badge">16 kHz Mono • DSP + Grad-CAM</span>
    </header>
    <div class="grid">
      <div class="card">
        <h2>1. Captura no Celular (Galaxy Buds)</h2>
        <div id="countdownBox" class="countdown-box">Gravando... Tussa 3 vezes! (6s)</div>

        <button class="green" id="btnLiveRecord" onclick="startLiveGuidedCheckIn()">
          🎙️ Iniciar Teste Guiado ao Vivo (6s • 3 Tosses)
        </button>

        <label class="file-btn" style="background:#38bdf8; color:#0f172a;">
          📱 Gravar Áudio / Escolher do Celular (Auto-converte p/ 16kHz WAV)
          <input type="file" id="mobileAudioInput" accept="audio/*,.wav,.m4a,.mp3,.ogg,.aac" capture="user" style="display:none;" onchange="handleMobileAudioFile(this)" />
        </label>

        <p style="color:var(--muted); font-size:12px; margin:6px 0 14px;">
          Dica: Se abrir via HTTP local no celular, o botão azul acima abre o gravador nativo com os Galaxy Buds e converte automaticamente para WAV 16 kHz Mono!
        </p>

        <hr style="border-color:var(--border); margin:14px 0;" />
        <h2>2. Simulador Rápido de Perfis Clínicos</h2>
        <button class="secondary" onclick="runSimulation('tosse_produtiva')">Simular Tosse Produtiva (Úmida)</button>
        <button class="secondary" onclick="runSimulation('tosse_seca')">Simular Tosse Seca (Irritativa)</button>
        <button class="secondary" onclick="runSimulation('saudavel')">Simular Saudável / Limpo</button>
      </div>

      <div class="card">
        <h2>3. Dashboard Clínico & Grad-CAM</h2>
        <div id="clinicalSummary"></div>
        <div id="heatmapContainer"></div>
        <pre id="jsonOutput">Aguardando gravação ou simulação...</pre>
      </div>
    </div>
    <div class="disclaimer">
      <strong>Aviso Ético e Regulatório (SaMD / LGPD):</strong> Protótipo de pesquisa em Engenharia Biomédica e Processamento Digital de Sinais. Não substitui diagnóstico médico presencial.
    </div>
  </div>

  <script>
    function encodeWAV16kHzMono(samplesFloat32, sampleRate) {
      const buffer = new ArrayBuffer(44 + samplesFloat32.length * 2);
      const view = new DataView(buffer);
      const writeString = (offset, str) => {
        for (let i = 0; i < str.length; i++) view.setUint8(offset + i, str.charCodeAt(i));
      };
      writeString(0, 'RIFF');
      view.setUint32(4, 36 + samplesFloat32.length * 2, true);
      writeString(8, 'WAVE');
      writeString(12, 'fmt ');
      view.setUint32(16, 16, true);
      view.setUint16(20, 1, true);
      view.setUint16(22, 1, true);
      view.setUint32(24, sampleRate, true);
      view.setUint32(28, sampleRate * 2, true);
      view.setUint16(32, 2, true);
      view.setUint16(34, 16, true);
      writeString(36, 'data');
      view.setUint32(40, samplesFloat32.length * 2, true);
      let offset = 44;
      for (let i = 0; i < samplesFloat32.length; i++, offset += 2) {
        let s = Math.max(-1, Math.min(1, samplesFloat32[i]));
        view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
      }
      return new Blob([buffer], { type: 'audio/wav' });
    }

    async function convertAudioBufferTo16kHzMonoWav(audioBuffer) {
      const targetSr = 16000;
      const targetLen = Math.max(1, Math.ceil(audioBuffer.duration * targetSr));
      const offlineCtx = new OfflineAudioContext(1, targetLen, targetSr);
      const source = offlineCtx.createBufferSource();
      source.buffer = audioBuffer;
      source.connect(offlineCtx.destination);
      source.start(0);
      const rendered = await offlineCtx.startRendering();
      return encodeWAV16kHzMono(rendered.getChannelData(0), targetSr);
    }

    async function startLiveGuidedCheckIn() {
      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        alert('No HTTP local sem HTTPS, use o botão azul "Gravar Áudio / Escolher do Celular" logo abaixo!');
        return;
      }
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          audio: { channelCount: 1, sampleRate: 16000, echoCancellation: false, noiseSuppression: false }
        });
        const box = document.getElementById('countdownBox');
        box.style.display = 'block';
        let sec = 6;
        box.textContent = `Tussa 3 vezes nos Galaxy Buds! (${sec}s)`;
        const timer = setInterval(() => {
          sec--;
          if (sec > 0) box.textContent = `Captando 16 kHz Mono... (${sec}s)`;
        }, 1000);

        const mediaRecorder = new MediaRecorder(stream);
        const chunks = [];
        mediaRecorder.ondataavailable = e => { if (e.data.size > 0) chunks.push(e.data); };
        mediaRecorder.onstop = async () => {
          clearInterval(timer);
          box.style.display = 'none';
          stream.getTracks().forEach(t => t.stop());
          document.getElementById('jsonOutput').textContent = 'Convertendo para PCM 16-bit 16kHz Mono e analisando...';
          const rawBlob = new Blob(chunks);
          const arrayBuf = await rawBlob.arrayBuffer();
          const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
          const decoded = await audioCtx.decodeAudioData(arrayBuf);
          const wavBlob = await convertAudioBufferTo16kHzMonoWav(decoded);
          await sendWavBlobToApi(wavBlob);
        };
        mediaRecorder.start();
        setTimeout(() => { if (mediaRecorder.state !== 'inactive') mediaRecorder.stop(); }, 6000);
      } catch (err) {
        alert('Permissão de microfone bloqueada em HTTP ou indisponível. Use o botão azul abaixo para gravar!');
      }
    }

    async function handleMobileAudioFile(input) {
      if (!input.files || !input.files.length) return;
      const file = input.files[0];
      document.getElementById('jsonOutput').textContent = 'Padronizando áudio para 16 kHz Mono e executando DSP + Grad-CAM...';
      try {
        if (file.name.toLowerCase().endsWith('.wav')) {
          await sendWavBlobToApi(file);
        } else {
          const arrayBuf = await file.arrayBuffer();
          const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
          const decoded = await audioCtx.decodeAudioData(arrayBuf);
          const wavBlob = await convertAudioBufferTo16kHzMonoWav(decoded);
          await sendWavBlobToApi(wavBlob);
        }
      } catch (e) {
        document.getElementById('jsonOutput').textContent = 'Erro ao processar áudio: ' + e;
      }
    }

    async function sendWavBlobToApi(wavBlob) {
      const form = new FormData();
      form.append('audio_file', wavBlob, 'galaxy_buds_checkin.wav');
      form.append('device_type', 'samsung_buds_2');
      form.append('patient_metadata', JSON.stringify({ age: 30, has_fever: false, smoker: false }));
      const res = await fetch('/api/v1/analyze-cough', { method: 'POST', body: form });
      const data = await res.json();
      renderResult(data);
    }

    async function runSimulation(profile) {
      document.getElementById('jsonOutput').textContent = 'Processando pipeline DSP + Grad-CAM...';
      const form = new FormData();
      form.append('profile', profile);
      form.append('device_type', 'samsung_buds_2');
      const res = await fetch('/api/v1/simulate-cough', { method: 'POST', body: form });
      const data = await res.json();
      renderResult(data);
    }

    function renderResult(data) {
      document.getElementById('jsonOutput').textContent = JSON.stringify(data, null, 2);
      if (!data.metrics) return;
      const m = data.metrics;
      const c = m.classification;
      const a = m.acoustic_properties;
      const p = c.probabilities;

      document.getElementById('clinicalSummary').innerHTML = `
        <div class="metrics-grid">
          <div class="metric-pill"><span>Classificação</span><strong style="color:#38bdf8">${c.label.toUpperCase()} (${(c.confidence*100).toFixed(1)}%)</strong></div>
          <div class="metric-pill"><span>Eventos de Tosse</span><strong>${m.cough_events_count} (${m.is_cough_detected ? 'Detectada' : 'Silêncio'})</strong></div>
          <div class="metric-pill"><span>Centroide Espectral</span><strong>${a.mean_spectral_centroid_hz} Hz</strong></div>
          <div class="metric-pill"><span>Duração / Energia</span><strong>${a.cough_duration_seconds}s / ${a.estimated_energy_db} dB</strong></div>
        </div>
        <div class="prob-bar-wrap">Tosse Produtiva: ${(p.tosse_produtiva*100).toFixed(1)}%
          <div class="prob-bar"><div class="prob-fill" style="width:${p.tosse_produtiva*100}%"></div></div>
        </div>
        <div class="prob-bar-wrap">Tosse Seca: ${(p.tosse_seca*100).toFixed(1)}%
          <div class="prob-bar"><div class="prob-fill" style="width:${p.tosse_seca*100}%"></div></div>
        </div>
        <div class="prob-bar-wrap">Saudável: ${(p.saudavel*100).toFixed(1)}%
          <div class="prob-bar"><div class="prob-fill" style="width:${p.saudavel*100}%"></div></div>
        </div>
      `;
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
        Gera uma amostra sintética de áudio 16 kHz Mono para o perfil selecionado.
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
