import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';

class SanityCheckResult {
  final bool isValidCoughCandidate;
  final int detectedBursts;
  final double maxRmsEnergy;
  final String message;

  const SanityCheckResult({
    required this.isValidCoughCandidate,
    required this.detectedBursts,
    required this.maxRmsEnergy,
    required this.message,
  });
}

/// Local Sanity Check (Edge On-Device — Seção 2 do PRD/Spec):
/// Valida localmente em 16 kHz Mono (janelas deslizantes de 50 ms RMS + filtro rápido)
/// se o áudio captado pelos Samsung Galaxy Buds contém eventos reais de tosse
/// ou apenas silêncio/ruído ambiente antes de realizar o upload HTTPS para a nuvem.
class LocalSanityCheckService {
  static const int sampleRate = 16000;
  static const int windowMs = 50;
  static const double minRmsThreshold = 0.025;

  Future<SanityCheckResult> validateRecordedWav(File wavFile) async {
    final bytes = await wavFile.readAsBytes();
    if (bytes.length <= 44) {
      return const SanityCheckResult(
        isValidCoughCandidate: false,
        detectedBursts: 0,
        maxRmsEnergy: 0.0,
        message: 'Arquivo WAV vazio ou inválido.',
      );
    }

    final pcmBytes = bytes.sublist(44);
    final byteData = ByteData.sublistView(Uint8List.fromList(pcmBytes));
    final totalSamples = byteData.lengthInBytes ~/ 2;
    final samples = List<double>.generate(
      totalSamples,
      (i) => byteData.getInt16(i * 2, Endian.little) / 32768.0,
    );

    final winSamples = (sampleRate * windowMs) ~/ 1000; // 800 amostras (50ms)
    final hopSamples = winSamples ~/ 2;

    double maxRms = 0.0;
    final rmsList = <double>[];

    for (int start = 0; start + winSamples <= samples.length; start += hopSamples) {
      double sumSq = 0.0;
      for (int i = start; i < start + winSamples; i++) {
        sumSq += samples[i] * samples[i];
      }
      final rms = math.sqrt(sumSq / winSamples);
      rmsList.add(rms);
      if (rms > maxRms) maxRms = rms;
    }

    if (maxRms < minRmsThreshold) {
      return SanityCheckResult(
        isValidCoughCandidate: false,
        detectedBursts: 0,
        maxRmsEnergy: maxRms,
        message: 'Apenas silêncio ou ruído ambiente detectado. Repita o teste tossindo próximo aos Galaxy Buds.',
      );
    }

    final dynamicThreshold = math.max(minRmsThreshold, maxRms * 0.22);
    int bursts = 0;
    bool inBurst = false;
    int quietFrames = 0;

    for (final rms in rmsList) {
      if (rms >= dynamicThreshold) {
        if (!inBurst) {
          bursts++;
          inBurst = true;
        }
        quietFrames = 0;
      } else if (inBurst) {
        quietFrames++;
        if (quietFrames >= 5) {
          inBurst = false;
        }
      }
    }

    return SanityCheckResult(
      isValidCoughCandidate: bursts >= 1,
      detectedBursts: bursts,
      maxRmsEnergy: maxRms,
      message: 'Amostra validada localmente ($bursts eventos acústicos detectados).',
    );
  }
}
