"""
Pipeline de Processamento Digital de Sinais (DSP) — EdgeBuds Hearable MVP.

Implementa estritamente a Seção 4 do PRD/Spec:
1. Amostragem & Canais: Forçado a 16.000 Hz (16 kHz) e 1 canal (Mono) - Bluetooth SCO mSBC.
2. Filtro Passa-Faixa: 100 Hz a 7.500 Hz (Butterworth / Biquad IIR).
3. Normalização de Amplitude: Escala dinâmica de pico entre [-1.0, 1.0].
4. Corte e Segmentação de Energia (VAD de Tosse): Janela deslizante de 50 ms (RMS),
   descarte de silêncio abaixo de limiar dinâmico e contagem de eventos de tosse.
5. Conversão Time -> Frequency: STFT (n_fft=2048, hop_length=512) + Mel-Spectrogram
   (n_mels=128) convertido para dB (power_to_db), formatado como tensor (1, 128, T).
"""

from __future__ import annotations

import io
import math
import random
import struct
import wave
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

try:
    import numpy as np

    NUMPY_AVAILABLE = True
except ImportError:
    np = None  # type: ignore
    NUMPY_AVAILABLE = False

try:
    import librosa  # type: ignore
    import scipy.signal as spsig  # type: ignore
    import soundfile as sf  # type: ignore

    LIBROSA_AVAILABLE = True
except ImportError:
    librosa = None  # type: ignore
    spsig = None  # type: ignore
    sf = None  # type: ignore
    LIBROSA_AVAILABLE = False


@dataclass(frozen=True)
class DSPConfig:
    """Parâmetros oficiais de DSP definidos na Seção 4 da especificação."""

    target_sr: int = 16000
    channels: int = 1
    bandpass_low_hz: float = 100.0
    bandpass_high_hz: float = 7500.0
    vad_window_ms: float = 50.0
    vad_hop_ms: float = 25.0
    vad_min_rms_floor: float = 0.025
    vad_dynamic_ratio: float = 0.22
    n_fft: int = 2048
    hop_length: int = 512
    n_mels: int = 128
    top_db: float = 80.0


@dataclass
class DSPResult:
    """Resultado completo do processamento de sinais de uma amostra de áudio."""

    raw_samples: List[float]
    active_samples: List[float]
    sample_rate: int
    is_cough_detected: bool
    cough_events_count: int
    cough_duration_seconds: float
    mean_spectral_centroid_hz: float
    estimated_energy_db: float
    mel_spectrogram_db: List[List[float]]  # Shape: (128, T)
    tensor_shape: Tuple[int, int, int]     # Shape: (1, 128, T)
    band_energy_ratios: dict               # Sub-band energy ratios for acoustic analysis


def _hz_to_mel(hz: float) -> float:
    return 2595.0 * math.log10(1.0 + hz / 700.0)


def _mel_to_hz(mel: float) -> float:
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def _fft_radix2(x_real: List[float]) -> List[complex]:
    """
    Cooley-Tukey Radix-2 FFT iterativa para n_fft = 2048 (potência de 2).
    Utilizada quando NumPy/Librosa não estão presentes no ambiente host.
    """
    n = len(x_real)
    # Bit-reversal permutation
    j = 0
    a = [complex(v, 0.0) for v in x_real]
    for i in range(1, n):
        bit = n >> 1
        while j & bit:
            j ^= bit
            bit >>= 1
        j ^= bit
        if i < j:
            a[i], a[j] = a[j], a[i]

    length = 2
    while length <= n:
        half = length >> 1
        angle = -2.0 * math.pi / length
        wlen = complex(math.cos(angle), math.sin(angle))
        for i in range(0, n, length):
            w = 1.0 + 0.0j
            for k in range(half):
                u = a[i + k]
                v = a[i + k + half] * w
                a[i + k] = u + v
                a[i + k + half] = u - v
                w *= wlen
        length <<= 1
    return a


class DSPPipeline:
    """
    Motor de Processamento Digital de Sinais (DSP) para triagem acústica respiratória.
    """

    def __init__(self, config: Optional[DSPConfig] = None) -> None:
        self.config = config or DSPConfig()
        self._hann_window = [
            0.5 * (1.0 - math.cos(2.0 * math.pi * i / (self.config.n_fft - 1)))
            for i in range(self.config.n_fft)
        ]
        self._mel_filterbank = self._build_mel_filterbank()

    def _build_mel_filterbank(self) -> List[List[Tuple[int, float]]]:
        """
        Constrói banco de 128 filtros triangulares Mel entre bandpass_low_hz e bandpass_high_hz.
        Retorna lista esparsa de (bin_idx, weight) por banda Mel para máxima performance.
        """
        n_mels = self.config.n_mels
        n_fft = self.config.n_fft
        sr = self.config.target_sr
        low_mel = _hz_to_mel(self.config.bandpass_low_hz)
        high_mel = _hz_to_mel(min(self.config.bandpass_high_hz, sr / 2.0))

        mel_points = [
            low_mel + i * (high_mel - low_mel) / (n_mels + 1)
            for i in range(n_mels + 2)
        ]
        hz_points = [_mel_to_hz(m) for m in mel_points]
        bin_points = [int(math.floor((n_fft + 1) * h / sr)) for h in hz_points]

        filterbank: List[List[Tuple[int, float]]] = []
        num_bins = n_fft // 2 + 1
        for m in range(1, n_mels + 1):
            f_m_minus = bin_points[m - 1]
            f_m = bin_points[m]
            f_m_plus = bin_points[m + 1]

            if f_m == f_m_minus:
                f_m = min(f_m + 1, num_bins - 1)
            if f_m_plus <= f_m:
                f_m_plus = min(f_m + 1, num_bins - 1)

            sparse_weights: List[Tuple[int, float]] = []
            for k in range(max(0, f_m_minus), min(num_bins, f_m_plus + 1)):
                weight = 0.0
                if f_m_minus <= k < f_m and f_m > f_m_minus:
                    weight = (k - f_m_minus) / float(f_m - f_m_minus)
                elif f_m <= k <= f_m_plus and f_m_plus > f_m:
                    weight = (f_m_plus - k) / float(f_m_plus - f_m)
                if weight > 0.0:
                    sparse_weights.append((k, weight))

            if not sparse_weights:
                center_bin = min(max(0, f_m), num_bins - 1)
                sparse_weights.append((center_bin, 1.0))

            filterbank.append(sparse_weights)
        return filterbank

    def load_and_standardize_wav(self, audio_bytes: bytes) -> Tuple[List[float], int]:
        """
        1. Lê arquivo WAV (PCM 8/16/24/32-bit ou float), converte para Mono (1 canal)
           e reamostra para 16.000 Hz (16 kHz).
        """
        if LIBROSA_AVAILABLE and sf is not None and np is not None:
            data, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32", always_2d=True)
            mono = np.mean(data, axis=1)
            if sr != self.config.target_sr:
                mono = librosa.resample(mono, orig_sr=sr, target_sr=self.config.target_sr)
            return mono.astype(float).tolist(), self.config.target_sr

        with wave.open(io.BytesIO(audio_bytes), "rb") as wf:
            n_channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            orig_sr = wf.getframerate()
            n_frames = wf.getnframes()
            raw_frames = wf.readframes(n_frames)

        if n_frames == 0 or len(raw_frames) == 0:
            raise ValueError("Arquivo de áudio vazio ou sem quadros PCM válidos.")

        samples: List[float] = []
        if sampwidth == 2:
            total_samples = len(raw_frames) // 2
            unpacked = struct.unpack(f"<{total_samples}h", raw_frames[: total_samples * 2])
            if n_channels == 1:
                samples = [s / 32768.0 for s in unpacked]
            else:
                for i in range(0, total_samples, n_channels):
                    chunk = unpacked[i : i + n_channels]
                    samples.append(sum(chunk) / (len(chunk) * 32768.0))
        elif sampwidth == 1:
            for i in range(0, len(raw_frames), n_channels):
                chunk = raw_frames[i : i + n_channels]
                avg = sum((b - 128) / 128.0 for b in chunk) / len(chunk)
                samples.append(avg)
        elif sampwidth == 4:
            total_samples = len(raw_frames) // 4
            unpacked = struct.unpack(f"<{total_samples}i", raw_frames[: total_samples * 4])
            for i in range(0, total_samples, n_channels):
                chunk = unpacked[i : i + n_channels]
                samples.append(sum(chunk) / (len(chunk) * 2147483648.0))
        else:
            raise ValueError(f"Largura de amostra PCM não suportada: {sampwidth * 8} bits.")

        # Reamostragem linear para 16.000 Hz se necessário
        if orig_sr != self.config.target_sr and len(samples) > 1:
            duration = len(samples) / float(orig_sr)
            target_len = max(1, int(round(duration * self.config.target_sr)))
            resampled: List[float] = []
            ratio = (len(samples) - 1) / float(max(1, target_len - 1))
            for i in range(target_len):
                pos = i * ratio
                idx = int(math.floor(pos))
                frac = pos - idx
                if idx + 1 < len(samples):
                    val = samples[idx] * (1.0 - frac) + samples[idx + 1] * frac
                else:
                    val = samples[idx]
                resampled.append(val)
            samples = resampled

        return samples, self.config.target_sr

    def apply_bandpass_filter(self, samples: List[float], sr: int) -> List[float]:
        """
        2. Filtro Passa-Faixa (100 Hz - 7500 Hz).
        Remove ruído de baixa frequência (< 100 Hz, manuseio/vento) e aliasing perto de Nyquist.
        """
        if not samples:
            return []

        if LIBROSA_AVAILABLE and spsig is not None and np is not None:
            nyq = 0.5 * sr
            low = max(0.001, self.config.bandpass_low_hz / nyq)
            high = min(0.995, self.config.bandpass_high_hz / nyq)
            sos = spsig.butter(4, [low, high], btype="bandpass", output="sos")
            filtered = spsig.sosfiltfilt(sos, np.asarray(samples, dtype=np.float64))
            return filtered.astype(float).tolist()

        # Cascata IIR Passa-Alta (100 Hz) + Passa-Baixa (7500 Hz)
        rc_hp = 1.0 / (2.0 * math.pi * self.config.bandpass_low_hz)
        dt = 1.0 / float(sr)
        alpha_hp = rc_hp / (rc_hp + dt)

        hp_out: List[float] = [0.0] * len(samples)
        prev_x = samples[0]
        prev_y = 0.0
        for i, x in enumerate(samples):
            y = alpha_hp * (prev_y + x - prev_x)
            hp_out[i] = y
            prev_x = x
            prev_y = y

        rc_lp = 1.0 / (2.0 * math.pi * self.config.bandpass_high_hz)
        alpha_lp = dt / (rc_lp + dt)
        lp_out: List[float] = [0.0] * len(samples)
        prev_y_lp = hp_out[0]
        for i, x in enumerate(hp_out):
            prev_y_lp = prev_y_lp + alpha_lp * (x - prev_y_lp)
            lp_out[i] = prev_y_lp

        return lp_out

    @staticmethod
    def normalize_peak(samples: List[float]) -> Tuple[List[float], float]:
        """
        3. Normalização de Amplitude de Pico para [-1.0, 1.0].
        Retorna também a energia estimada em dBFS antes da normalização.
        """
        if not samples:
            return [], -96.0

        sum_sq = sum(s * s for s in samples)
        rms_orig = math.sqrt(sum_sq / len(samples))
        energy_db = 20.0 * math.log10(max(rms_orig, 1e-6))

        peak = max(abs(s) for s in samples)
        if peak < 0.015:
            # Evita amplificar ruído térmico de silêncio (< -36 dBFS de pico)
            return list(samples), energy_db

        normalized = [max(-1.0, min(1.0, s / peak)) for s in samples]
        return normalized, energy_db

    def segment_cough_vad(
        self, samples: List[float], sr: int
    ) -> Tuple[List[float], bool, int, float]:
        """
        4. Corte e Segmentação de Energia (VAD de Tosse):
        - Janela deslizante de 50 ms calculando Root Mean Square (RMS).
        - Amostras com energia abaixo de limiar dinâmico são removidas.
        - Contabiliza eventos discretos de tosse explosiva.
        """
        win_len = max(1, int(round((self.config.vad_window_ms / 1000.0) * sr)))
        hop_len = max(1, int(round((self.config.vad_hop_ms / 1000.0) * sr)))

        if len(samples) < win_len:
            rms = math.sqrt(sum(s * s for s in samples) / max(1, len(samples)))
            detected = rms >= self.config.vad_min_rms_floor
            dur = len(samples) / float(sr) if detected else 0.0
            return list(samples), detected, (1 if detected else 0), dur

        rms_frames: List[float] = []
        frame_bounds: List[Tuple[int, int]] = []
        for start in range(0, len(samples) - win_len + 1, hop_len):
            end = start + win_len
            window = samples[start:end]
            rms_val = math.sqrt(sum(w * w for w in window) / win_len)
            rms_frames.append(rms_val)
            frame_bounds.append((start, end))

        max_rms = max(rms_frames) if rms_frames else 0.0
        if max_rms < self.config.vad_min_rms_floor:
            return list(samples[: min(len(samples), sr)]), False, 0, 0.0

        dynamic_threshold = max(
            self.config.vad_min_rms_floor,
            max_rms * self.config.vad_dynamic_ratio,
        )

        active_mask = [r >= dynamic_threshold for r in rms_frames]

        # Contagem de eventos de tosse (agrupando bursts com gap < 120 ms)
        min_gap_frames = max(1, int(round(0.120 / (self.config.vad_hop_ms / 1000.0))))
        cough_events = 0
        in_burst = False
        silence_run = 0
        burst_frames = 0

        for is_active in active_mask:
            if is_active:
                if not in_burst:
                    cough_events += 1
                    in_burst = True
                silence_run = 0
                burst_frames += 1
            else:
                if in_burst:
                    silence_run += 1
                    if silence_run >= min_gap_frames:
                        in_burst = False

        # Extrair amostras ativas (descartando silêncio ambiente)
        sample_keep = [False] * len(samples)
        for is_active, (start, end) in zip(active_mask, frame_bounds):
            if is_active:
                for idx in range(start, end):
                    sample_keep[idx] = True

        active_samples = [s for s, keep in zip(samples, sample_keep) if keep]
        cough_duration_seconds = len(active_samples) / float(sr)
        is_cough_detected = cough_events >= 1 and cough_duration_seconds >= 0.08

        if not active_samples:
            active_samples = list(samples)

        return active_samples, is_cough_detected, cough_events, cough_duration_seconds

    def extract_mel_spectrogram_and_features(
        self, active_samples: List[float], sr: int
    ) -> Tuple[List[List[float]], float, dict]:
        """
        5. Conversão Time -> Frequency:
        - STFT (n_fft=2048, hop_length=512).
        - Mel-Spectrogram (n_mels=128) convertido para dB (power_to_db).
        - Calcula Centroide Espectral Médio (Hz) e distribuição de energia por sub-bandas.
        """
        n_fft = self.config.n_fft
        hop_length = self.config.hop_length
        n_mels = self.config.n_mels

        # Garante comprimento mínimo de pelo menos 1 janela STFT
        padded = list(active_samples)
        if len(padded) < n_fft:
            padded.extend([0.0] * (n_fft - len(padded)))

        if LIBROSA_AVAILABLE and librosa is not None and np is not None:
            y_np = np.asarray(padded, dtype=np.float32)
            mel_power = librosa.feature.melspectrogram(
                y=y_np,
                sr=sr,
                n_fft=n_fft,
                hop_length=hop_length,
                n_mels=n_mels,
                fmin=self.config.bandpass_low_hz,
                fmax=self.config.bandpass_high_hz,
            )
            mel_db = librosa.power_to_db(mel_power, ref=np.max, top_db=self.config.top_db)
            centroid = librosa.feature.spectral_centroid(
                y=y_np, sr=sr, n_fft=n_fft, hop_length=hop_length
            )
            mean_centroid = float(np.mean(centroid))

            # Sub-band ratios via FFT power
            stft_mag = np.abs(librosa.stft(y_np, n_fft=n_fft, hop_length=hop_length)) ** 2
            freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
            total_power = float(np.sum(stft_mag)) + 1e-12
            low_power = float(np.sum(stft_mag[(freqs >= 100) & (freqs < 650), :]))
            mid_power = float(np.sum(stft_mag[(freqs >= 650) & (freqs < 2000), :]))
            high_power = float(np.sum(stft_mag[freqs >= 2000, :]))
            band_ratios = {
                "low_band_ratio": low_power / total_power,
                "mid_band_ratio": mid_power / total_power,
                "high_band_ratio": high_power / total_power,
            }
            return mel_db.astype(float).tolist(), mean_centroid, band_ratios

        # Implementação STFT + Mel-Spectrogram determinística
        num_frames = max(1, 1 + (len(padded) - n_fft) // hop_length)
        num_bins = n_fft // 2 + 1
        bin_freqs = [k * sr / float(n_fft) for k in range(num_bins)]

        mel_power_matrix: List[List[float]] = [[] for _ in range(n_mels)]
        frame_centroids: List[float] = []
        low_power_sum = 0.0
        mid_power_sum = 0.0
        high_power_sum = 0.0

        for frame_idx in range(num_frames):
            start = frame_idx * hop_length
            frame = [
                padded[start + i] * self._hann_window[i] for i in range(n_fft)
            ]
            spectrum = _fft_radix2(frame)
            power_spec = [
                (spectrum[k].real ** 2 + spectrum[k].imag ** 2) / float(n_fft)
                for k in range(num_bins)
            ]
            mag_spec = [math.sqrt(p) for p in power_spec]

            mag_sum = sum(mag_spec)
            if mag_sum > 1e-9:
                centroid_hz = sum(f * m for f, m in zip(bin_freqs, mag_spec)) / mag_sum
                frame_centroids.append(centroid_hz)

            for k, p in enumerate(power_spec):
                f_hz = bin_freqs[k]
                if 100.0 <= f_hz < 650.0:
                    low_power_sum += p
                elif 650.0 <= f_hz < 2000.0:
                    mid_power_sum += p
                elif f_hz >= 2000.0:
                    high_power_sum += p

            for m_idx, sparse_filter in enumerate(self._mel_filterbank):
                mel_val = sum(power_spec[k] * w for k, w in sparse_filter)
                mel_power_matrix[m_idx].append(max(mel_val, 1e-10))

        # power_to_db(mel_spectrogram, ref=max)
        max_power = max(max(row) for row in mel_power_matrix) if mel_power_matrix else 1.0
        max_power = max(max_power, 1e-10)
        mel_db_matrix: List[List[float]] = []
        for row in mel_power_matrix:
            db_row = [
                max(-self.config.top_db, 10.0 * math.log10(val / max_power))
                for val in row
            ]
            mel_db_matrix.append(db_row)

        mean_centroid_hz = (
            sum(frame_centroids) / len(frame_centroids) if frame_centroids else 0.0
        )
        total_p = low_power_sum + mid_power_sum + high_power_sum + 1e-12
        band_ratios = {
            "low_band_ratio": low_power_sum / total_p,
            "mid_band_ratio": mid_power_sum / total_p,
            "high_band_ratio": high_power_sum / total_p,
        }

        return mel_db_matrix, mean_centroid_hz, band_ratios

    def process_audio_bytes(self, audio_bytes: bytes) -> DSPResult:
        """Executa o pipeline DSP ponta a ponta sobre os bytes de um arquivo WAV."""
        raw_samples, sr = self.load_and_standardize_wav(audio_bytes)
        filtered = self.apply_bandpass_filter(raw_samples, sr)
        normalized, estimated_energy_db = self.normalize_peak(filtered)

        active_samples, is_cough_detected, cough_events_count, cough_duration = (
            self.segment_cough_vad(normalized, sr)
        )

        if not is_cough_detected:
            # Se o áudio for apenas silêncio/ruído de fundo muito baixo, zera o centroide
            mel_db, mean_centroid, band_ratios = self.extract_mel_spectrogram_and_features(
                active_samples, sr
            )
            if estimated_energy_db < -45.0:
                mean_centroid = 0.0
        else:
            mel_db, mean_centroid, band_ratios = self.extract_mel_spectrogram_and_features(
                active_samples, sr
            )

        time_frames = len(mel_db[0]) if mel_db else 0
        tensor_shape = (1, self.config.n_mels, time_frames)

        return DSPResult(
            raw_samples=normalized,
            active_samples=active_samples,
            sample_rate=sr,
            is_cough_detected=is_cough_detected,
            cough_events_count=cough_events_count,
            cough_duration_seconds=cough_duration,
            mean_spectral_centroid_hz=mean_centroid,
            estimated_energy_db=estimated_energy_db,
            mel_spectrogram_db=mel_db,
            tensor_shape=tensor_shape,
            band_energy_ratios=band_ratios,
        )


def generate_synthetic_cough_wav(
    profile: str = "tosse_produtiva",
    duration_seconds: float = 2.4,
    sample_rate: int = 16000,
    seed: int = 42,
) -> bytes:
    """
    Gera um arquivo WAV PCM 16-bit, 16 kHz Mono sintético para validação de pipeline,
    testes unitários e simulação interativa.

    Perfis suportados:
    - 'tosse_produtiva': 3 eventos de tosse com forte energia em baixas/médias frequências
      (250-950 Hz) + modulação de crepitação mucosa (rattles/crackles).
    - 'tosse_seca': 3 eventos de tosse curta e irritativa com centroide espectral alto
      (1900-3600 Hz) e ataque explosivo seco.
    - 'saudavel': Pigarro leve / expiração limpa de baixa turbulência patológica.
    - 'silencio': Ruído térmico de fundo abaixo do limiar de VAD (-55 dBFS).
    """
    rng = random.Random(seed)
    total_samples = int(round(duration_seconds * sample_rate))
    signal = [0.0] * total_samples

    if profile == "silencio":
        for i in range(total_samples):
            signal[i] = rng.uniform(-0.002, 0.002)
    else:
        burst_starts = [0.25, 0.95, 1.65]
        burst_len = 0.26 if profile == "tosse_produtiva" else 0.19

        for b_start in burst_starts:
            start_idx = int(b_start * sample_rate)
            end_idx = min(total_samples, int((b_start + burst_len) * sample_rate))
            for i in range(start_idx, end_idx):
                t_local = (i - start_idx) / float(sample_rate)
                env = math.exp(-t_local * (9.5 if profile == "tosse_produtiva" else 14.0))
                attack = min(1.0, t_local / 0.015)
                envelope = attack * env

                if profile == "tosse_produtiva":
                    # Ressonância brônquica grave/média + crepitação de secreção mucosa
                    mucus_mod = 1.0 + 0.55 * math.sin(2.0 * math.pi * 28.0 * t_local)
                    c1 = 0.55 * math.sin(2.0 * math.pi * 340.0 * t_local)
                    c2 = 0.40 * math.sin(2.0 * math.pi * 720.0 * t_local)
                    c3 = 0.25 * math.sin(2.0 * math.pi * 1280.0 * t_local)
                    crackle = 0.30 * rng.uniform(-1.0, 1.0) * mucus_mod
                    sample_val = envelope * (c1 + c2 + c3 + crackle)
                elif profile == "tosse_seca":
                    # Harmônicos agudos irritativos de laringe/traqueia superior (> 2000 Hz)
                    c1 = 0.45 * math.sin(2.0 * math.pi * 2250.0 * t_local)
                    c2 = 0.40 * math.sin(2.0 * math.pi * 3150.0 * t_local)
                    c3 = 0.25 * math.sin(2.0 * math.pi * 4100.0 * t_local)
                    hiss = 0.25 * rng.uniform(-1.0, 1.0)
                    sample_val = envelope * (c1 + c2 + c3 + hiss)
                else:  # saudavel (expiração voluntária limpa / equilibrada)
                    c1 = 0.45 * math.sin(2.0 * math.pi * 1150.0 * t_local)
                    c2 = 0.35 * math.sin(2.0 * math.pi * 1550.0 * t_local)
                    sample_val = envelope * 0.65 * (c1 + c2)

                signal[i] += sample_val

        # Normaliza para ~ -12 dBFS a -15 dBFS de energia média
        peak = max(max(abs(s) for s in signal), 1e-6)
        target_peak = 0.78 if profile != "saudavel" else 0.55
        signal = [max(-0.99, min(0.99, (s / peak) * target_peak)) for s in signal]

    pcm_ints = [int(round(max(-1.0, min(1.0, s)) * 32767.0)) for s in signal]
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(struct.pack(f"<{len(pcm_ints)}h", *pcm_ints))
    return buf.getvalue()
