"""
Orquestrador de Análise Acústica Respiratória (DSP + Classificador + Grad-CAM).
Implementa o fluxo completo de inferência em nuvem e formatação do contrato JSON (Seção 6).
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional
import uuid

from .dsp.dsp_pipeline import DSPPipeline, DSPConfig
from .ml.explainability import GradCAMExplainabilityGenerator
from .ml.model import CoughClassifierService
from .schemas import (
    AcousticProperties,
    AnalysisMetrics,
    AnalyzeCoughResponse,
    ExplainabilityOutput,
    PatientMetadata,
)


class AcousticAnalysisOrchestrator:
    """
    Coordena o pipeline ponta a ponta:
    1. Padronização de Áudio (16 kHz Mono), Filtro Passa-Faixa (100-7500 Hz), VAD RMS (50ms) e Mel-Spectrogram (1, 128, T).
    2. Classificação [saudavel, tosse_seca, tosse_produtiva].
    3. Geração de mapa de calor Grad-CAM sobre o espectrograma.
    4. Montagem do JSON de resposta conforme contrato da API.
    """

    def __init__(
        self,
        static_gradcam_dir: Optional[str | Path] = None,
        weights_path: Optional[str] = None,
        base_url: str = "https://api.domain.com",
    ) -> None:
        if static_gradcam_dir is None:
            static_gradcam_dir = Path(__file__).resolve().parent.parent / "static" / "gradcam"
        self.static_gradcam_dir = Path(static_gradcam_dir)
        self.base_url = base_url.rstrip("/")

        self.dsp_pipeline = DSPPipeline(DSPConfig())
        self.classifier = CoughClassifierService(weights_path=weights_path)
        self.explainability = GradCAMExplainabilityGenerator(self.static_gradcam_dir)

    def analyze_cough_wav(
        self,
        audio_bytes: bytes,
        device_type: str = "samsung_buds_2",
        patient_metadata_json: Optional[str] = None,
        request_base_url: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Processa os bytes de um arquivo WAV e retorna o dicionário exato do contrato da API.
        """
        if not audio_bytes or len(audio_bytes) < 44:
            raise ValueError("Arquivo WAV inválido ou truncado (< 44 bytes).")

        patient_meta = PatientMetadata.from_json_str(patient_metadata_json)
        analysis_id = str(uuid.uuid4())
        timestamp_iso = datetime.now(timezone.utc).isoformat()

        # 1. Pipeline DSP
        dsp_result = self.dsp_pipeline.process_audio_bytes(audio_bytes)

        # 2. Inferência do Modelo + Pesos de Ativação Grad-CAM
        classification, cam_weights = self.classifier.predict(
            dsp_result=dsp_result,
            patient_metadata=patient_meta,
        )

        # 3. Renderização do Mapa de Calor Grad-CAM (.png)
        self.explainability.render_gradcam_overlay(
            analysis_id=analysis_id,
            mel_spectrogram_db=dsp_result.mel_spectrogram_db,
            cam_weights=cam_weights,
            predicted_label=classification.label,
            confidence=classification.confidence,
            mean_centroid_hz=dsp_result.mean_spectral_centroid_hz,
        )

        effective_base = (request_base_url or self.base_url).rstrip("/")
        heatmap_url = f"{effective_base}/static/gradcam/{analysis_id}.png"

        # 4. Montagem do Contrato de Saída (Seção 6)
        response = AnalyzeCoughResponse(
            status="success",
            analysis_id=analysis_id,
            timestamp=timestamp_iso,
            metrics=AnalysisMetrics(
                is_cough_detected=dsp_result.is_cough_detected,
                cough_events_count=dsp_result.cough_events_count,
                classification=classification,
                acoustic_properties=AcousticProperties(
                    mean_spectral_centroid_hz=dsp_result.mean_spectral_centroid_hz,
                    cough_duration_seconds=dsp_result.cough_duration_seconds,
                    estimated_energy_db=dsp_result.estimated_energy_db,
                ),
            ),
            explainability=ExplainabilityOutput(
                spectrogram_heatmap_url=heatmap_url,
            ),
        )
        return response.to_dict()
