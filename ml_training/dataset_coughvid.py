"""
Ingestão e Balanceamento de Dados Clínicos — COUGHVID (EPFL) & Coswara (IISc).

Implementa a Seção 5.1 do PRD/Spec:
- Filtragem por qualidade de anotação médica (cough_detected >= 0.8).
- Mapeamento padronizado para as 3 classes clínicas:
  0 -> 'saudavel'
  1 -> 'tosse_seca'
  2 -> 'tosse_produtiva'
- Cálculo de pesos de balanceamento de classes (Inverse Class Frequency) para lidar
  com o desbalanceamento natural de dados clínicos.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


LABEL_TO_INDEX: Dict[str, int] = {
    "saudavel": 0,
    "tosse_seca": 1,
    "tosse_produtiva": 2,
}
INDEX_TO_LABEL: Dict[int, str] = {v: k for k, v in LABEL_TO_INDEX.items()}


@dataclass
class ClinicalCoughRecord:
    record_id: str
    audio_path: str
    label_name: str
    label_index: int
    cough_detected_prob: float
    dataset_source: str  # "coughvid" ou "coswara"


def map_clinical_annotation_to_label(
    status: str, cough_type: Optional[str] = None
) -> Optional[str]:
    """
    Converte os metadados clínicos do COUGHVID/Coswara para a taxonomia do EdgeBuds.
    """
    status_norm = (status or "").strip().lower()
    ctype_norm = (cough_type or "").strip().lower()

    if ctype_norm in ("wet", "productive", "tosse_produtiva"):
        return "tosse_produtiva"
    if ctype_norm in ("dry", "tosse_seca"):
        return "tosse_seca"
    if status_norm in ("healthy", "saudavel") and ctype_norm in ("", "unknown", "none"):
        return "saudavel"
    if status_norm in ("symptomatic", "covid-19", "positive_mild", "positive_moderate"):
        return "tosse_seca"
    return None


def compute_class_weights(label_indices: List[int], num_classes: int = 3) -> List[float]:
    """
    Calcula pesos inversamente proporcionais à frequência de cada classe:
    w_c = N / (num_classes * count_c)
    """
    total = len(label_indices)
    if total == 0:
        return [1.0] * num_classes

    counts = [0] * num_classes
    for idx in label_indices:
        if 0 <= idx < num_classes:
            counts[idx] += 1

    weights = [
        total / float(num_classes * max(1, c)) for c in counts
    ]
    return weights


def load_coughvid_metadata_csv(
    metadata_csv_path: str | Path,
    audio_root_dir: str | Path,
    min_cough_prob: float = 0.80,
) -> List[ClinicalCoughRecord]:
    """
    Carrega o arquivo `metadata_compiled.csv` oficial do COUGHVID (EPFL).
    """
    csv_path = Path(metadata_csv_path)
    audio_root = Path(audio_root_dir)
    records: List[ClinicalCoughRecord] = []

    if not csv_path.is_file():
        return records

    with open(csv_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                cough_prob = float(row.get("cough_detected", 0.0))
            except ValueError:
                continue
            if cough_prob < min_cough_prob:
                continue

            status = row.get("status", "")
            cough_type = (
                row.get("cough_type_4")
                or row.get("cough_type_1")
                or row.get("cough_type", "")
            )
            mapped = map_clinical_annotation_to_label(status, cough_type)
            if mapped is None:
                continue

            uuid_str = row.get("uuid", "").strip()
            wav_candidate = audio_root / f"{uuid_str}.wav"
            records.append(
                ClinicalCoughRecord(
                    record_id=uuid_str,
                    audio_path=str(wav_candidate),
                    label_name=mapped,
                    label_index=LABEL_TO_INDEX[mapped],
                    cough_detected_prob=cough_prob,
                    dataset_source="coughvid",
                )
            )
    return records
