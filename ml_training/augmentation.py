"""
Técnicas de Data Augmentation para Áudio Médico (Seção 5.3 do PRD/Spec).

Implementa:
1. SpecAugment (Frequency Masking e Time Masking sobre Mel-Espectrogramas).
2. Injeção de Ruído Branco Gaussiano com relação Sinal-Ruído (SNR) entre 15 dB e 30 dB.
3. Pitch Shifting de até ±1.5 semitons (preservando duração temporal).
"""

from __future__ import annotations

import copy
import math
import random
from typing import List, Optional


class AudioMedicalAugmenter:
    """
    Pipeline de Data Augmentation acústico para prevenção de overfitting em bases clínicas
    (COUGHVID / Coswara).
    """

    def __init__(
        self,
        snr_min_db: float = 15.0,
        snr_max_db: float = 30.0,
        max_pitch_semitones: float = 1.5,
        freq_mask_max_width: int = 16,
        time_mask_max_width: int = 8,
        seed: Optional[int] = None,
    ) -> None:
        self.snr_min_db = snr_min_db
        self.snr_max_db = snr_max_db
        self.max_pitch_semitones = max_pitch_semitones
        self.freq_mask_max_width = freq_mask_max_width
        self.time_mask_max_width = time_mask_max_width
        self.rng = random.Random(seed)

    def inject_white_noise_snr(
        self, samples: List[float], target_snr_db: Optional[float] = None
    ) -> List[float]:
        """
        Injeta ruído branco de baixa intensidade com SNR controlado entre 15 dB e 30 dB.
        SNR(dB) = 10 * log10(P_signal / P_noise)
        """
        if not samples:
            return []

        snr_db = (
            target_snr_db
            if target_snr_db is not None
            else self.rng.uniform(self.snr_min_db, self.snr_max_db)
        )
        p_signal = sum(s * s for s in samples) / float(len(samples))
        if p_signal < 1e-12:
            return list(samples)

        p_noise = p_signal / (10.0 ** (snr_db / 10.0))
        noise_std = math.sqrt(p_noise)

        augmented = [
            max(-1.0, min(1.0, s + self.rng.gauss(0.0, noise_std)))
            for s in samples
        ]
        return augmented

    def pitch_shift_semitones(
        self, samples: List[float], semitones: Optional[float] = None
    ) -> List[float]:
        """
        Aplica desvio de tom (Pitch Shifting) de até ±1.5 semitons preservando o número de amostras.
        Fator de escala harmônica: factor = 2 ** (semitones / 12.0).
        """
        if len(samples) < 4:
            return list(samples)

        n_semi = (
            semitones
            if semitones is not None
            else self.rng.uniform(-self.max_pitch_semitones, self.max_pitch_semitones)
        )
        factor = 2.0 ** (n_semi / 12.0)
        n = len(samples)

        # Interpolação fracionária com janelamento cruzado para manter comprimento constante
        shifted: List[float] = []
        for i in range(n):
            src_pos = (i * factor) % (n - 1)
            idx = int(math.floor(src_pos))
            frac = src_pos - idx
            val = samples[idx] * (1.0 - frac) + samples[idx + 1] * frac
            shifted.append(val)
        return shifted

    def spec_augment(
        self,
        mel_spectrogram_db: List[List[float]],
        num_freq_masks: int = 2,
        num_time_masks: int = 2,
        fill_value_db: float = -80.0,
    ) -> List[List[float]]:
        """
        Aplica SpecAugment (Park et al.) sobre o Mel-Espectrograma (128, T):
        - Frequency Masking: mascara faixas contíguas de bandas Mel [f0, f0 + f).
        - Time Masking: mascara janelas contíguas de tempo [t0, t0 + t).
        """
        if not mel_spectrogram_db or not mel_spectrogram_db[0]:
            return []

        spec = copy.deepcopy(mel_spectrogram_db)
        n_mels = len(spec)
        n_frames = len(spec[0])

        # 1. Frequency Masking
        for _ in range(num_freq_masks):
            f_width = self.rng.randint(1, min(self.freq_mask_max_width, max(1, n_mels // 4)))
            f0 = self.rng.randint(0, max(0, n_mels - f_width))
            for m in range(f0, f0 + f_width):
                for t in range(n_frames):
                    spec[m][t] = fill_value_db

        # 2. Time Masking
        if n_frames > 2:
            for _ in range(num_time_masks):
                t_width = self.rng.randint(
                    1, min(self.time_mask_max_width, max(1, n_frames // 3))
                )
                t0 = self.rng.randint(0, max(0, n_frames - t_width))
                for m in range(n_mels):
                    for t in range(t0, t0 + t_width):
                        spec[m][t] = fill_value_db

        return spec
