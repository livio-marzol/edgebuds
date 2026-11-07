"""Módulo de Inteligência Artificial Médica e Explicabilidade (Grad-CAM)."""
from .model import CoughClassifierService, CLASS_LABELS
from .explainability import GradCAMExplainabilityGenerator

__all__ = [
    "CoughClassifierService",
    "CLASS_LABELS",
    "GradCAMExplainabilityGenerator",
]
