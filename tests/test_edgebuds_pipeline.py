"""
Suíte de Testes Automatizados — EdgeBuds Acoustic Respiratory Biomarker.

Valida:
1. Pipeline DSP (16 kHz Mono, Filtro Passa-Faixa 100-7500 Hz, Normalização [-1, 1],
   VAD RMS 50ms e Mel-Spectrogram shape (1, 128, T)).
2. Classificador de Tosse ([saudavel, tosse_seca, tosse_produtiva]).
3. Geração de Explicabilidade Visual (Grad-CAM .png).
4. Contrato estrito de saída JSON da Seção 6 (POST /api/v1/analyze-cough).
5. Técnicas de Data Augmentation (SpecAugment, White Noise SNR 15-30dB, Pitch Shift ±1.5 semitons).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import unittest

from backend.app.dsp.dsp_pipeline import (
    DSPConfig,
    DSPPipeline,
    generate_synthetic_cough_wav,
)
from backend.app.service import AcousticAnalysisOrchestrator
from ml_training.augmentation import AudioMedicalAugmenter
from ml_training.train import run_baseline_validation_benchmark


class TestEdgeBudsPipeline(unittest.TestCase):
    def setUp(self) -> None:
        self.dsp = DSPPipeline(DSPConfig())
        self.gradcam_dir = Path("backend/static/gradcam")
        self.orchestrator = AcousticAnalysisOrchestrator(
            static_gradcam_dir=self.gradcam_dir,
            base_url="https://api.domain.com",
        )

    def test_dsp_standardization_and_mel_shape(self) -> None:
        wav_bytes = generate_synthetic_cough_wav(profile="tosse_produtiva")
        result = self.dsp.process_audio_bytes(wav_bytes)

        self.assertEqual(result.sample_rate, 16000)
        self.assertTrue(result.is_cough_detected)
        self.assertGreaterEqual(result.cough_events_count, 2)
        self.assertEqual(result.tensor_shape[0], 1)
        self.assertEqual(result.tensor_shape[1], 128)
        self.assertGreater(result.tensor_shape[2], 1)
        self.assertTrue(all(-1.0 <= s <= 1.0 for s in result.raw_samples))

    def test_silence_rejected_by_vad(self) -> None:
        silence_wav = generate_synthetic_cough_wav(profile="silencio")
        result = self.dsp.process_audio_bytes(silence_wav)
        self.assertFalse(result.is_cough_detected)
        self.assertEqual(result.cough_events_count, 0)

    def test_api_contract_and_gradcam_generation(self) -> None:
        for expected_label in ("tosse_produtiva", "tosse_seca", "saudavel"):
            wav_bytes = generate_synthetic_cough_wav(profile=expected_label)
            meta_json = json.dumps({"age": 32, "has_fever": False, "smoker": False})

            payload = self.orchestrator.analyze_cough_wav(
                audio_bytes=wav_bytes,
                device_type="samsung_buds_2",
                patient_metadata_json=meta_json,
            )

            self.assertEqual(payload["status"], "success")
            self.assertIn("analysis_id", payload)
            self.assertIn("timestamp", payload)

            metrics = payload["metrics"]
            self.assertTrue(metrics["is_cough_detected"])
            self.assertGreaterEqual(metrics["cough_events_count"], 1)

            classification = metrics["classification"]
            self.assertEqual(classification["label"], expected_label)
            self.assertGreater(classification["confidence"], 0.45)

            probs = classification["probabilities"]
            self.assertAlmostEqual(
                probs["saudavel"] + probs["tosse_seca"] + probs["tosse_produtiva"],
                1.0,
                places=2,
            )

            acoustic = metrics["acoustic_properties"]
            self.assertGreater(acoustic["mean_spectral_centroid_hz"], 100.0)
            self.assertGreater(acoustic["cough_duration_seconds"], 0.1)
            self.assertLess(acoustic["estimated_energy_db"], 0.0)

            heatmap_url = payload["explainability"]["spectrogram_heatmap_url"]
            self.assertTrue(
                heatmap_url.startswith("https://api.domain.com/static/gradcam/")
            )
            analysis_id = payload["analysis_id"]
            png_file = self.gradcam_dir / f"{analysis_id}.png"
            self.assertTrue(png_file.is_file())
            self.assertGreater(png_file.stat().st_size, 1024)

            # Salva uma imagem Grad-CAM de referência para documentação no GitHub
            if expected_label == "tosse_produtiva":
                sample_png = self.gradcam_dir / "sample_gradcam.png"
                sample_png.write_bytes(png_file.read_bytes())

            if png_file.name != "sample_gradcam.png":
                png_file.unlink(missing_ok=True)

    def test_medical_audio_augmentation_and_validation_metrics(self) -> None:
        augmenter = AudioMedicalAugmenter(seed=123)
        wav_bytes = generate_synthetic_cough_wav(profile="tosse_seca")
        samples, _ = self.dsp.load_and_standardize_wav(wav_bytes)

        noisy = augmenter.inject_white_noise_snr(samples, target_snr_db=20.0)
        shifted = augmenter.pitch_shift_semitones(samples, semitones=1.0)
        self.assertEqual(len(noisy), len(samples))
        self.assertEqual(len(shifted), len(samples))

        dsp_res = self.dsp.process_audio_bytes(wav_bytes)
        masked_spec = augmenter.spec_augment(dsp_res.mel_spectrogram_db)
        self.assertEqual(len(masked_spec), 128)

        benchmark = run_baseline_validation_benchmark(num_samples_per_class=6)
        self.assertGreaterEqual(benchmark["estimated_roc_auc"], 0.85)
        self.assertGreaterEqual(benchmark["pathology_recall_sensitivity"], 0.85)


if __name__ == "__main__":
    unittest.main(verbosity=2)
