# ARCHITECTURE SPEC & IMPLEMENTATION PLAN: ACOUSTIC RESPIRATORY BIOMARKER (HEARABLE MVP)

## 1. PROJECT OVERVIEW & SCOPE
* **Objective:** Construir um sistema de biomarcador acústico para triagem ativa de tosse e distúrbios respiratórios (saudável vs. tosse seca/irritativa vs. tosse produtiva/úmida).
* **Hardware Input:** Samsung Galaxy Buds (Buds Core / Buds 2) atuando como microfone sem fio via protocolo Bluetooth SCO.
* **Paradigma de Uso:** Check-in Ativo (o usuário executa um teste guiado de 5 a 10 segundos, tossindo voluntariamente).
* **Arquitetura:** Híbrida (Edge on-device para triagem/validação acústica + Cloud Backend para inferência pesada e explicabilidade).
* **Objetivo Final:** Estudo avançado e projeto de portfólio em Engenharia de Software, IA Médica e Processamento Digital de Sinais (DSP).

---

## 2. SYSTEM ARCHITECTURE

```text
[ Usuário c/ Galaxy Buds ]
           │
           │ Áudio via Bluetooth SCO (16 kHz, Mono)
           ▼
[ Mobile App (Edge / Client) ]
     ├─ 1. Audio Recording Manager (Força entrada via TYPE_BLUETOOTH_SCO)
     ├─ 2. Local Sanity Check (TFLite/DSP): Valida se houve tosse real ou ruído/silêncio
     └─ 3. Upload Gateway (Envia payload criptografado via HTTPS POST)
           │
           ▼
[ Cloud Backend (FastAPI + PyTorch/ONNX) ]
     ├─ 1. DSP Pipeline:
     │     ├─ Bandpass Filter (100 Hz - 7500 Hz)
     │     ├─ Voice/Cough Activity Detection (VAD por energia RMS)
     │     └─ Mel-Spectrogram Extraction (128 bandas Mel, n_fft=2048, hop_length=512)
     ├─ 2. Fine-Tuned Model (YAMNet / PANNs Backbone):
     │     └─ Classificador: [Saudável / Tosse Seca / Tosse Úmida]
     ├─ 3. Explainability Module (Grad-CAM):
     │     └─ Gera mapa de calor sobre o espectrograma destacando frequências patológicas
     └─ 4. Response Dispatcher:
           └─ Retorna JSON: Scores de probabilidade, VFC estimada e URL do laudo visual
           │
           ▼
[ Mobile App: Dashboard de Resultados ]
     └─ Exibe probabilidade clínica, características acústicas e espectrograma anotado.
```

---

## 3. TECH STACK & DEPENDENCIES

### A) Data Science & DSP (Python)
* `librosa` / `torchaudio`: Extração de Mel-Espectrogramas e MFCCs.
* `soundfile` / `scipy`: Normalização, filtros passa-faixa e manipulação de arquivos WAV.
* `torch` / `torchvision`: Treinamento e fine-tuning do modelo de classificação.
* `pytorch-grad-cam`: Geração de mapas de atenção visual do espectrograma.

### B) Backend Cloud
* `FastAPI`: Criação de APIs assíncronas de alta performance para inferência de áudio.
* `uvicorn`: ASGI server.
* `pydantic`: Validação estrita de esquemas de dados.
* `Docker`: Containerização do ambiente para deploy agnóstico.

### C) Mobile / Edge
* `Flutter` (Dart) ou `Kotlin` (Android Nativo):
  * Gerenciamento de hardware: `AudioManager` com `startBluetoothSco()` e `setBluetoothScoOn(true)`.
* `tflite_flutter` ou `onnxruntime-mobile`: Execução do filtro rápido local on-device.

---

## 4. PIPELINE DE PROCESSAMENTO DIGITAL DE SINAIS (DSP)

1. **Amostragem & Canais:**
   * Todo áudio deve ser forçado a **16.000 Hz (16 kHz)** e **1 canal (Mono)**. O Bluetooth SCO opera nativamente próximo a essa faixa com codec mSBC.
2. **Normalização de Amplitude:**
   * Aplicar normalização de pico para garantir escala dinâmica entre [-1.0, 1.0].
3. **Corte e Segmentação de Energia (VAD de Tosse):**
   * Janela deslizante de 50ms calculando *Root Mean Square (RMS)*.
   * Amostras com energia abaixo de limiar dinâmico são removidas (descarte de silêncio ambiente).
4. **Conversão de Domínio (Time -> Frequency):**
   * Transformada Rápida de Fourier em Tempo Reduzido (STFT).
   * Parâmetros padrão: `n_fft = 2048`, `hop_length = 512`, `n_mels = 128`.
   * Conversão de amplitude para decibéis: `librosa.power_to_db(mel_spectrogram)`.
   * Dimensão resultante formatada como imagem de entrada para CNN: `(1, 128, T)`.

---

## 5. ESTRATÉGIA DE DADOS & IA (TRANSFER LEARNING)

1. **Bases de Dados Clínicas:**
   * **COUGHVID (EPFL):** Base primária (~27.000 áudios com rotulação de diagnóstico médico, tosse seca vs produtiva).
   * **Coswara (Indian Institute of Science):** Base secundária para validação cruzada.
2. **Modelo Base:**
   * Utilizar **YAMNet** ou **MobileNetV3** pré-treinado em AudioSet.
3. **Treinamento e Fine-Tuning:**
   * Congelar blocos convolucionais primários para retenção de filtros acústicos genéricos.
   * Substituir o Head de Classificação:
     * `GlobalAveragePooling2D` -> `Dense(256, ReLU)` -> `Dropout(0.4)` -> `Dense(3, Softmax)`.
   * Data Augmentation específico para áudio médico:
     * *SpecAugment* (Time masking e Frequency masking).
     * Injeção de ruído branco de baixa intensidade (SNR entre 15dB e 30dB).
     * Pequenos desvios de tom (*Pitch shifting* de até ±1.5 semitons).
4. **Métricas de Sucesso:**
   * **ROC-AUC > 0.85** em validação clínica estratificada.
   * Foco em **Recall / Sensibilidade** para patologias (minimizar falsos negativos).

---

## 6. CONTRATO DE INTERFACE DA API (FASTAPI)

### `POST /api/v1/analyze-cough`
* **Input (multipart/form-data):**
  * `audio_file`: arquivo `.wav` (PCM 16-bit, 16kHz mono).
  * `device_type`: string (ex: `"samsung_buds_2"`).
  * `patient_metadata`: JSON string opcional (idade, presença de febre, tabagismo).
* **Output (JSON):**
  ```json
  {
    "status": "success",
    "analysis_id": "uuid4",
    "timestamp": "ISO-8601",
    "metrics": {
      "is_cough_detected": true,
      "cough_events_count": 3,
      "classification": {
        "label": "tosse_produtiva",
        "confidence": 0.892,
        "probabilities": {
          "saudavel": 0.041,
          "tosse_seca": 0.067,
          "tosse_produtiva": 0.892
        }
      },
      "acoustic_properties": {
        "mean_spectral_centroid_hz": 2341.2,
        "cough_duration_seconds": 0.68,
        "estimated_energy_db": -14.2
      }
    },
    "explainability": {
      "spectrogram_heatmap_url": "https://api.domain.com/static/gradcam/uuid4.png"
    }
  }
  ```

---

## 7. CRONOGRAMA DE SPRINTS & ENTREGÁVEIS (ROADMAP)

### Sprint 1: Setup de Dados, DSP e Baseline
- [x] Estruturação de ingestão do dataset COUGHVID / Coswara.
- [x] Pipeline de pré-processamento (`dsp_pipeline.py`) com filtro passa-faixa (100Hz-7500Hz), VAD RMS (50ms) e Mel-Spectrogram (`n_mels=128`, `n_fft=2048`, `hop_length=512`).
- [x] Arquitetura de classificador Baseline/Fine-Tuned para validação do pipeline.

### Sprint 2: Modelo Otimizado & Explainability
- [x] Técnicas de SpecAugment, injeção de ruído branco (SNR 15-30dB), Pitch Shifting (±1.5 semitons) e balanceamento de classes.
- [x] Integração de Grad-CAM para mapear em quais frequências a rede focou.
- [x] Scripts de exportação do modelo para formato ONNX e TensorFlow Lite (versão reduzida on-device).

### Sprint 3: Backend em Nuvem (Containerizado)
- [x] Serviço FastAPI com endpoints assíncronos (`POST /api/v1/analyze-cough`).
- [x] Validação e conversão de áudio recebido em tempo de execução.
- [x] `Dockerfile` e `docker-compose.yml` com suporte a bibliotecas de processamento de áudio (`libsndfile1`, `ffmpeg`).

### Sprint 4: Mobile Client & Integração Samsung Galaxy Buds
- [x] UI de teste guiado (Contagem regressiva de 3 tosses / 5-10s).
- [x] Código nativo Android (Kotlin) de controle de áudio (`BluetoothScoAudioManager.kt` com `startBluetoothSco()` e `TYPE_BLUETOOTH_SCO`).
- [x] Integração do pipeline HTTP do app com a API do FastAPI e Dashboard de Resultados com Grad-CAM.

### Sprint 5: Documentação de Portfólio de Alto Nível
- [x] README técnico no GitHub com diagramas de arquitetura (C4 Model e Fluxo DSP).
- [x] Seção de Ética Médica e Limitações (LGPD, conformidade SaMD e aviso de não-substituição médica).
- [x] Simulador Web Interativo integrado ao backend para demonstração imediata com captura de áudio ou amostras sintéticas.
