"""Módulo de Processamento Digital de Sinais (DSP) para Biomarcadores Acústicos."""
from .dsp_pipeline import (
    DSPPipeline,
    DSPConfig,
    DSPResult,
    generate_synthetic_cough_wav,
)

__all__ = [
    "DSPPipeline",
    "DSPConfig",
    "DSPResult",
    "generate_synthetic_cough_wav",
]
