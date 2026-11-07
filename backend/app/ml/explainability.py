"""
Módulo de Explicabilidade Visual (Grad-CAM) sobre Mel-Espectrogramas.

Implementa a Seção 2.3 e Seção 3.A do PRD/Spec:
- Gera mapa de calor de atenção (Grad-CAM) sobreposto ao Mel-Espectrograma (128 bandas Mel)
  destacando as faixas de frequência patológicas que motivaram a decisão do classificador.
- Renderiza laudo visual em alta definição (.png) com barra de cores, eixos de frequência
  (100 Hz – 7.500 Hz) e marcação da região de pico de ativação neural.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Tuple

from PIL import Image, ImageDraw, ImageFilter


def _jet_colormap(val: float) -> Tuple[int, int, int]:
    """Mapeia escalar normalizado [0.0, 1.0] para paleta térmica clínica (Jet/Turbo)."""
    v = max(0.0, min(1.0, val))
    if v < 0.25:
        t = v / 0.25
        r, g, b = 18, int(40 + 180 * t), int(140 + 115 * t)
    elif v < 0.50:
        t = (v - 0.25) / 0.25
        r, g, b = int(20 + 210 * t), int(220 + 35 * t), int(255 - 190 * t)
    elif v < 0.75:
        t = (v - 0.50) / 0.25
        r, g, b = int(230 + 25 * t), int(255 - 125 * t), int(65 - 50 * t)
    else:
        t = (v - 0.75) / 0.25
        r, g, b = 255, int(130 - 115 * t), int(15 + 35 * t)
    return (max(0, min(255, r)), max(0, min(255, g)), max(0, min(255, b)))


def _magma_spectrogram_color(norm_db: float) -> Tuple[int, int, int]:
    """Mapeia energia do Mel-Espectrograma [0.0, 1.0] para base escura clínica."""
    v = max(0.0, min(1.0, norm_db))
    r = int(12 + 165 * (v ** 1.3))
    g = int(14 + 110 * (v ** 1.6))
    b = int(32 + 150 * (v ** 0.9))
    return (r, g, b)


class GradCAMExplainabilityGenerator:
    """
    Gerador de mapas de atenção Grad-CAM sobre Mel-Espectrogramas de tosse.
    """

    def __init__(self, output_dir: str | Path) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def render_gradcam_overlay(
        self,
        analysis_id: str,
        mel_spectrogram_db: List[List[float]],
        cam_weights: List[List[float]],
        predicted_label: str,
        confidence: float,
        mean_centroid_hz: float,
    ) -> str:
        """
        Renderiza o Mel-Espectrograma (128 bandas Mel) sobreposto pelo mapa de calor
        Grad-CAM e salva em `{output_dir}/{analysis_id}.png`.
        Retorna o caminho absoluto do arquivo gerado.
        """
        n_mels = len(mel_spectrogram_db)
        n_frames = len(mel_spectrogram_db[0]) if n_mels > 0 else 1

        # Cria imagem base do espectrograma + overlay Grad-CAM (dimensão base: n_frames x n_mels)
        spec_w = max(16, n_frames)
        spec_h = max(16, n_mels)
        raw_canvas = Image.new("RGB", (spec_w, spec_h), (10, 14, 24))
        pixels = raw_canvas.load()

        peak_cam_val = -1.0
        peak_coord = (spec_w // 2, spec_h // 2)

        for m in range(n_mels):
            # Inverter eixo Y para que baixas frequências (100 Hz) fiquem na base
            y_img = (n_mels - 1) - m
            for t in range(n_frames):
                db_val = mel_spectrogram_db[m][t]
                norm_db = max(0.0, min(1.0, (db_val + 80.0) / 80.0))
                cam_val = cam_weights[m][t] if (m < len(cam_weights) and t < len(cam_weights[m])) else 0.0

                base_rgb = _magma_spectrogram_color(norm_db)
                heat_rgb = _jet_colormap(cam_val)

                # Alpha blending ponderado pela intensidade de ativação Grad-CAM
                alpha = min(0.82, max(0.08, cam_val * 0.85))
                blended = (
                    int(base_rgb[0] * (1.0 - alpha) + heat_rgb[0] * alpha),
                    int(base_rgb[1] * (1.0 - alpha) + heat_rgb[1] * alpha),
                    int(base_rgb[2] * (1.0 - alpha) + heat_rgb[2] * alpha),
                )
                pixels[t, y_img] = blended

                if cam_val > peak_cam_val:
                    peak_cam_val = cam_val
                    peak_coord = (t, y_img)

        # Redimensiona com interpolação bicúbica para layout de laudo clínico (760 x 360)
        plot_w, plot_h = 620, 256
        upscaled_spec = raw_canvas.resize((plot_w, plot_h), resample=Image.Resampling.BICUBIC)
        upscaled_spec = upscaled_spec.filter(ImageFilter.SMOOTH)

        canvas_w, canvas_h = 760, 350
        report_img = Image.new("RGB", (canvas_w, canvas_h), (13, 17, 26))
        draw = ImageDraw.Draw(report_img)

        # Moldura e cabeçalho do laudo espectral
        margin_left, margin_top = 72, 48
        draw.rectangle(
            [margin_left - 2, margin_top - 2, margin_left + plot_w + 2, margin_top + plot_h + 2],
            outline=(56, 189, 248),
            width=2,
        )
        report_img.paste(upscaled_spec, (margin_left, margin_top))

        # Destaca região de maior ativação do Grad-CAM (bounding box de foco patológico)
        px = margin_left + int((peak_coord[0] / float(max(1, spec_w - 1))) * plot_w)
        py = margin_top + int((peak_coord[1] / float(max(1, spec_h - 1))) * plot_h)
        box_rx, box_ry = 48, 28
        draw.rectangle(
            [
                max(margin_left, px - box_rx),
                max(margin_top, py - box_ry),
                min(margin_left + plot_w, px + box_rx),
                min(margin_top + plot_h, py + box_ry),
            ],
            outline=(250, 204, 21),
            width=2,
        )

        # Barra de escala Grad-CAM à direita
        cbar_x = margin_left + plot_w + 20
        cbar_w = 18
        for y in range(plot_h):
            norm_v = 1.0 - (y / float(max(1, plot_h - 1)))
            c_rgb = _jet_colormap(norm_v)
            draw.line([(cbar_x, margin_top + y), (cbar_x + cbar_w, margin_top + y)], fill=c_rgb)
        draw.rectangle(
            [cbar_x, margin_top, cbar_x + cbar_w, margin_top + plot_h],
            outline=(148, 163, 184),
            width=1,
        )

        # Textos e anotações de eixo (Frequência Mel Hz, Tempo e Diagnóstico)
        title_text = (
            f"EDGEBUDS GRAD-CAM | CLASS: {predicted_label.upper()} "
            f"({confidence * 100:.1f}%) | CENTROID: {mean_centroid_hz:.0f} Hz"
        )
        draw.text((margin_left, 18), title_text, fill=(241, 245, 249))

        # Marcações do eixo Y (128 bandas Mel: 100 Hz a 7500 Hz)
        freq_labels = [
            (0, "7.5kHz"),
            (int(plot_h * 0.30), "3.2kHz"),
            (int(plot_h * 0.60), "1.2kHz"),
            (int(plot_h * 0.85), "400Hz"),
            (plot_h - 10, "100Hz"),
        ]
        for y_offset, label in freq_labels:
            draw.text((12, margin_top + y_offset), label, fill=(148, 163, 184))

        # Marcações da Colorbar
        draw.text((cbar_x + 24, margin_top), "1.0", fill=(248, 113, 113))
        draw.text((cbar_x + 24, margin_top + plot_h // 2 - 6), "0.5", fill=(250, 204, 21))
        draw.text((cbar_x + 24, margin_top + plot_h - 12), "0.0", fill=(56, 189, 248))

        footer_text = (
            f"ID: {analysis_id} | STFT: n_fft=2048, hop=512, n_mels=128 | "
            "Focus Box (Yellow): Peak Grad-CAM Activation"
        )
        draw.text((margin_left, margin_top + plot_h + 14), footer_text, fill=(148, 163, 184))

        out_file = self.output_dir / f"{analysis_id}.png"
        report_img.save(out_file, format="PNG", optimize=True)
        return str(out_file)
