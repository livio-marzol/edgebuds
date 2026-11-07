"""
Esquemas de Dados e Contratos de Interface da API (FastAPI / Pydantic).
Segue estritamente a Seção 6 do PRD/Spec para POST /api/v1/analyze-cough.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Literal, Optional
import uuid

try:
    from pydantic import BaseModel, Field

    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False
    BaseModel = object  # type: ignore

    def Field(default=None, **kwargs):  # type: ignore
        return default


CoughLabel = Literal["saudavel", "tosse_seca", "tosse_produtiva"]


@dataclass
class PatientMetadata:
    """Metadados clínicos opcionais enviados pelo cliente mobile."""

    age: Optional[int] = None
    has_fever: Optional[bool] = None
    smoker: Optional[bool] = None
    reported_symptom_days: Optional[int] = None
    sex: Optional[str] = None

    @classmethod
    def from_json_str(cls, raw_json: Optional[str]) -> Optional["PatientMetadata"]:
        if not raw_json or not raw_json.strip():
            return None
        try:
            data = json.loads(raw_json)
            if not isinstance(data, dict):
                raise ValueError("patient_metadata deve ser um objeto JSON.")
            return cls(
                age=data.get("age") or data.get("idade"),
                has_fever=data.get("has_fever") if "has_fever" in data else data.get("febre"),
                smoker=data.get("smoker") if "smoker" in data else data.get("tabagismo"),
                reported_symptom_days=data.get("reported_symptom_days"),
                sex=data.get("sex") or data.get("sexo"),
            )
        except json.JSONDecodeError as exc:
            raise ValueError(f"JSON inválido em patient_metadata: {exc}") from exc


@dataclass
class ClassificationProbabilities:
    saudavel: float
    tosse_seca: float
    tosse_produtiva: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "saudavel": round(float(self.saudavel), 4),
            "tosse_seca": round(float(self.tosse_seca), 4),
            "tosse_produtiva": round(float(self.tosse_produtiva), 4),
        }


@dataclass
class ClassificationResult:
    label: CoughLabel
    confidence: float
    probabilities: ClassificationProbabilities

    def to_dict(self) -> Dict[str, Any]:
        return {
            "label": self.label,
            "confidence": round(float(self.confidence), 4),
            "probabilities": self.probabilities.to_dict(),
        }


@dataclass
class AcousticProperties:
    mean_spectral_centroid_hz: float
    cough_duration_seconds: float
    estimated_energy_db: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "mean_spectral_centroid_hz": round(float(self.mean_spectral_centroid_hz), 1),
            "cough_duration_seconds": round(float(self.cough_duration_seconds), 2),
            "estimated_energy_db": round(float(self.estimated_energy_db), 1),
        }


@dataclass
class AnalysisMetrics:
    is_cough_detected: bool
    cough_events_count: int
    classification: ClassificationResult
    acoustic_properties: AcousticProperties

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_cough_detected": bool(self.is_cough_detected),
            "cough_events_count": int(self.cough_events_count),
            "classification": self.classification.to_dict(),
            "acoustic_properties": self.acoustic_properties.to_dict(),
        }


@dataclass
class ExplainabilityOutput:
    spectrogram_heatmap_url: str

    def to_dict(self) -> Dict[str, str]:
        return {
            "spectrogram_heatmap_url": self.spectrogram_heatmap_url,
        }


@dataclass
class AnalyzeCoughResponse:
    """Contrato de saída oficial do endpoint POST /api/v1/analyze-cough."""

    metrics: AnalysisMetrics
    explainability: ExplainabilityOutput
    status: str = "success"
    analysis_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "analysis_id": self.analysis_id,
            "timestamp": self.timestamp,
            "metrics": self.metrics.to_dict(),
            "explainability": self.explainability.to_dict(),
        }


if PYDANTIC_AVAILABLE:

    class ProbabilitiesSchema(BaseModel):
        saudavel: float = Field(..., ge=0.0, le=1.0, example=0.041)
        tosse_seca: float = Field(..., ge=0.0, le=1.0, example=0.067)
        tosse_produtiva: float = Field(..., ge=0.0, le=1.0, example=0.892)

    class ClassificationSchema(BaseModel):
        label: CoughLabel = Field(..., example="tosse_produtiva")
        confidence: float = Field(..., ge=0.0, le=1.0, example=0.892)
        probabilities: ProbabilitiesSchema

    class AcousticPropertiesSchema(BaseModel):
        mean_spectral_centroid_hz: float = Field(..., example=2341.2)
        cough_duration_seconds: float = Field(..., example=0.68)
        estimated_energy_db: float = Field(..., example=-14.2)

    class MetricsSchema(BaseModel):
        is_cough_detected: bool = Field(..., example=True)
        cough_events_count: int = Field(..., ge=0, example=3)
        classification: ClassificationSchema
        acoustic_properties: AcousticPropertiesSchema

    class ExplainabilitySchema(BaseModel):
        spectrogram_heatmap_url: str = Field(
            ..., example="https://api.domain.com/static/gradcam/uuid4.png"
        )

    class AnalyzeCoughResponseSchema(BaseModel):
        status: str = Field(default="success", example="success")
        analysis_id: str = Field(..., example="550e8400-e29b-41d4-a716-446655440000")
        timestamp: str = Field(..., example="2026-11-07T09:30:00+00:00")
        metrics: MetricsSchema
        explainability: ExplainabilitySchema
