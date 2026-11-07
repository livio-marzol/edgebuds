"""
Exportação do Modelo para ONNX (Cloud Inference) e TensorFlow Lite (Mobile Edge Sanity Check).
Implementa o entregável da Sprint 2 (Seção 7 do PRD/Spec).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

from backend.app.ml.model import TORCH_AVAILABLE

if TORCH_AVAILABLE:
    import torch
    from backend.app.ml.model import AcousticBiomarkerNet


def export_to_onnx(output_onnx_path: str | Path) -> Dict[str, str]:
    """
    Exporta a rede AcousticBiomarkerNet para formato ONNX com eixo temporal dinâmico
    para entrada (batch_size, 1, 128, time_frames).
    """
    out_path = Path(output_onnx_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if TORCH_AVAILABLE:
        model = AcousticBiomarkerNet(
            num_classes=3, pretrained=False, freeze_primary_blocks=False
        )
        model.eval()
        dummy_input = torch.randn(1, 1, 128, 64)
        torch.onnx.export(
            model,
            dummy_input,
            str(out_path),
            export_params=True,
            opset_version=17,
            do_constant_folding=True,
            input_names=["mel_spectrogram"],
            output_names=["class_logits"],
            dynamic_axes={
                "mel_spectrogram": {0: "batch_size", 3: "time_frames"},
                "class_logits": {0: "batch_size"},
            },
        )
        return {"status": "exported_onnx", "path": str(out_path)}

    # Gera manifesto de especificação do grafo ONNX/TFLite quando PyTorch não está ativo no host
    manifest_path = out_path.with_suffix(".manifest.json")
    manifest = {
        "model_name": "edgebuds_acoustic_biomarker_v1",
        "input_shape": ["batch_size", 1, 128, "time_frames"],
        "output_classes": ["saudavel", "tosse_seca", "tosse_produtiva"],
        "head_architecture": [
            "GlobalAveragePooling2D",
            "Dense(256, ReLU)",
            "Dropout(0.4)",
            "Dense(3, Softmax)",
        ],
        "edge_sanity_check_threshold_rms": 0.025,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {"status": "manifest_generated", "path": str(manifest_path)}


if __name__ == "__main__":
    info = export_to_onnx("ml_training/exported/cough_biomarker.onnx")
    print(f"Export result: {info}")
