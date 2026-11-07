"""
Servidor de Execução Rápida — EdgeBuds Acoustic Respiratory Biomarker.

Permite testar o projeto imediatamente no celular (via Wi-Fi na mesma rede local)
ou no computador com um único comando:
    python run_server.py

Se FastAPI/Uvicorn estiverem instalados, utiliza o servidor ASGI Uvicorn.
Caso contrário, inicia um servidor HTTP multithread nativo da biblioteca padrão Python
expondo os mesmos endpoints REST (/api/v1/analyze-cough, /api/v1/simulate-cough, /health, /static/*).
"""

from __future__ import annotations

import email.parser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import socket
from typing import Dict, Optional, Tuple

from backend.app.dsp.dsp_pipeline import generate_synthetic_cough_wav
from backend.app.main import DASHBOARD_HTML, FASTAPI_AVAILABLE, GRADCAM_DIR, orchestrator


def get_local_lan_ip() -> str:
    """Descobre o endereço IPv4 da máquina na rede local (Wi-Fi / Ethernet)."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"
    finally:
        s.close()


def _parse_multipart_form_data(
    content_type: str, body: bytes
) -> Tuple[Dict[str, str], Dict[str, bytes]]:
    """Parser de multipart/form-data compatível com Python 3.13+ (sem módulo cgi)."""
    raw_headers = f"Content-Type: {content_type}\r\nMIME-Version: 1.0\r\n\r\n".encode("utf-8")
    msg = email.parser.BytesParser().parsebytes(raw_headers + body)

    fields: Dict[str, str] = {}
    files: Dict[str, bytes] = {}

    if msg.is_multipart():
        for part in msg.iter_parts():
            name = part.get_param("name", header="content-disposition")
            filename = part.get_filename()
            payload = part.get_payload(decode=True) or b""
            if not name:
                continue
            if filename is not None:
                files[name] = payload
            else:
                fields[name] = payload.decode("utf-8", errors="replace")
    return fields, files


class EdgeBudsHTTPHandler(BaseHTTPRequestHandler):
    """Handler HTTP nativo para servir o Web App Mobile e a API REST do EdgeBuds."""

    def _send_json(self, status_code: int, data: dict) -> None:
        raw = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self) -> None:
        path = self.path.split("?", 1)[0]
        if path == "/" or path == "/index.html":
            raw = DASHBOARD_HTML.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)
            return

        if path == "/health":
            self._send_json(
                200,
                {
                    "status": "healthy",
                    "service": "edgebuds-acoustic-biomarker-api",
                    "dsp_config": {
                        "sample_rate_hz": 16000,
                        "channels": 1,
                        "bandpass_hz": [100.0, 7500.0],
                        "vad_window_ms": 50.0,
                        "stft": {"n_fft": 2048, "hop_length": 512, "n_mels": 128},
                    },
                },
            )
            return

        if path.startswith("/static/gradcam/"):
            fname = path.split("/static/gradcam/", 1)[1]
            target_file = (GRADCAM_DIR / fname).resolve()
            if target_file.parent == GRADCAM_DIR.resolve() and target_file.is_file():
                img_bytes = target_file.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "image/png")
                self.send_header("Content-Length", str(len(img_bytes)))
                self.end_headers()
                self.wfile.write(img_bytes)
                return

        self._send_json(404, {"detail": "Rota não encontrada."})

    def do_POST(self) -> None:
        path = self.path.split("?", 1)[0]
        content_len = int(self.headers.get("Content-Length", "0"))
        content_type = self.headers.get("Content-Type", "")
        body = self.rfile.read(content_len) if content_len > 0 else b""
        host_header = self.headers.get("Host", f"{get_local_lan_ip()}:8000")
        base_url = f"http://{host_header}"

        fields, files = _parse_multipart_form_data(content_type, body)

        if path == "/api/v1/analyze-cough":
            audio_bytes = files.get("audio_file")
            if not audio_bytes:
                self._send_json(400, {"detail": "Campo 'audio_file' (.wav) obrigatório."})
                return
            device_type = fields.get("device_type", "samsung_buds_2")
            patient_meta = fields.get("patient_metadata")
            try:
                result = orchestrator.analyze_cough_wav(
                    audio_bytes=audio_bytes,
                    device_type=device_type,
                    patient_metadata_json=patient_meta,
                    request_base_url=base_url,
                )
                self._send_json(200, result)
            except Exception as exc:
                self._send_json(422, {"detail": str(exc)})
            return

        if path == "/api/v1/simulate-cough":
            profile = fields.get("profile", "tosse_produtiva")
            device_type = fields.get("device_type", "samsung_buds_2")
            wav_bytes = generate_synthetic_cough_wav(profile=profile)
            result = orchestrator.analyze_cough_wav(
                audio_bytes=wav_bytes,
                device_type=device_type,
                request_base_url=base_url,
            )
            self._send_json(200, result)
            return

        self._send_json(404, {"detail": "Endpoint POST não encontrado."})


def main(port: int = 8000) -> None:
    lan_ip = get_local_lan_ip()
    print("=" * 68)
    print(" EdgeBuds — Servidor Mobile & Cloud Backend Ativo!")
    print(f" -> No Computador: http://localhost:{port}")
    print(f" -> No Celular (mesmo Wi-Fi): http://{lan_ip}:{port}")
    print("=" * 68)

    if FASTAPI_AVAILABLE:
        try:
            import uvicorn  # type: ignore

            uvicorn.run("backend.app.main:app", host="0.0.0.0", port=port)
            return
        except ImportError:
            pass

    server = ThreadingHTTPServer(("0.0.0.0", port), EdgeBudsHTTPHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServidor encerrado.")
        server.server_close()


if __name__ == "__main__":
    main(port=8000)
