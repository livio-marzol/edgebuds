import 'dart:async';
import 'dart:io';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'services/api_client.dart';
import 'services/local_sanity_check.dart';

void main() {
  runApp(const EdgeBudsHearableApp());
}

class EdgeBudsHearableApp extends StatelessWidget {
  const EdgeBudsHearableApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'EdgeBuds — Acoustic Biomarker',
      debugShowCheckedModeBanner: false,
      theme: ThemeData.dark(useMaterial3: true).copyWith(
        colorScheme: ColorScheme.fromSeed(
          seedColor: const Color(0xFF38BDF8),
          brightness: Brightness.dark,
        ),
      ),
      home: const ActiveCheckInScreen(),
    );
  }
}

class ActiveCheckInScreen extends StatefulWidget {
  const ActiveCheckInScreen({super.key});

  @override
  State<ActiveCheckInScreen> createState() => _ActiveCheckInScreenState();
}

class _ActiveCheckInScreenState extends State<ActiveCheckInScreen> {
  static const _scoChannel = MethodChannel('com.edgebuds.hearable/bluetooth_sco');

  final _sanityChecker = LocalSanityCheckService();
  final _apiClient = const EdgeBudsApiClient(baseUrl: 'http://10.0.2.2:8000');

  bool _scoConnected = false;
  String _deviceName = 'Samsung Galaxy Buds2 (SCO)';
  bool _isRecording = false;
  int _coughStepCountdown = 0;
  String _statusMessage = 'Pronto para iniciar Check-in Ativo (3 tosses guiadas).';
  Map<String, dynamic>? _analysisResult;

  Future<void> _connectBluetoothSco() async {
    try {
      final res = await _scoChannel.invokeMapMethod<String, dynamic>('startBluetoothSco');
      setState(() {
        _scoConnected = res?['sco_active'] == true;
        _deviceName = (res?['device_product_name'] as String?) ?? 'Samsung Galaxy Buds SCO';
        _statusMessage = 'Canal Bluetooth SCO ativo (16 kHz Mono mSBC) em $_deviceName';
      });
    } on PlatformException catch (e) {
      setState(() {
        _statusMessage = 'Aviso Bluetooth SCO: ${e.message}';
      });
    }
  }

  Future<void> _startGuidedCoughCheckIn() async {
    await _connectBluetoothSco();
    setState(() {
      _isRecording = true;
      _coughStepCountdown = 3;
      _analysisResult = null;
      _statusMessage = 'Captando via Galaxy Buds (16 kHz Mono)... Tussa 3 vezes!';
    });

    Timer.periodic(const Duration(seconds: 2), (timer) {
      if (!mounted || !_isRecording) {
        timer.cancel();
        return;
      }
      setState(() {
        if (_coughStepCountdown > 1) {
          _coughStepCountdown--;
        } else {
          timer.cancel();
        }
      });
    });

    try {
      final wavPath = await _scoChannel.invokeMethod<String>(
        'recordCoughCheckIn',
        {'durationSeconds': 6},
      );
      if (wavPath == null) throw Exception('Caminho de áudio nulo.');

      final wavFile = File(wavPath);

      // 1. Local Sanity Check (Edge On-Device)
      final sanity = await _sanityChecker.validateRecordedWav(wavFile);
      if (!sanity.isValidCoughCandidate) {
        setState(() {
          _isRecording = false;
          _statusMessage = sanity.message;
        });
        return;
      }

      setState(() {
        _statusMessage = '${sanity.message} Enviando para inferência e Grad-CAM...';
      });

      // 2. Cloud Backend Inference + Grad-CAM
      final response = await _apiClient.analyzeCough(
        wavFile: wavFile,
        deviceType: 'samsung_buds_2',
        patientMetadata: const {'age': 30, 'has_fever': false, 'smoker': false},
      );

      setState(() {
        _isRecording = false;
        _analysisResult = response;
        _statusMessage = 'Laudo acústico concluído com sucesso.';
      });
    } catch (e) {
      setState(() {
        _isRecording = false;
        _statusMessage = 'Erro durante o Check-in: $e';
      });
    } finally {
      await _scoChannel.invokeMethod('stopBluetoothSco');
    }
  }

  @override
  Widget build(BuildContext context) {
    final metrics = _analysisResult?['metrics'] as Map<String, dynamic>?;
    final classification = metrics?['classification'] as Map<String, dynamic>?;
    final acoustic = metrics?['acoustic_properties'] as Map<String, dynamic>?;
    final explainability = _analysisResult?['explainability'] as Map<String, dynamic>?;

    return Scaffold(
      appBar: AppBar(
        title: const Text('EdgeBuds • Biomarcador Acústico'),
        actions: [
          Chip(
            label: Text(_scoConnected ? 'SCO 16kHz Ativo' : 'Bluetooth SCO'),
            avatar: Icon(
              Icons.headphones,
              color: _scoConnected ? Colors.greenAccent : Colors.lightBlueAccent,
              size: 18,
            ),
          ),
          const SizedBox(width: 12),
        ],
      ),
      body: ListView(
        padding: const EdgeInsets.all(16),
        children: [
          Card(
            child: Padding(
              padding: const EdgeInsets.all(16),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    '1. Check-in Ativo Guiado ($_deviceName)',
                    style: Theme.of(context).textTheme.titleMedium,
                  ),
                  const SizedBox(height: 8),
                  Text(_statusMessage),
                  const SizedBox(height: 16),
                  if (_isRecording)
                    Center(
                      child: Column(
                        children: [
                          Text(
                            'Tosse restante: $_coughStepCountdown',
                            style: Theme.of(context).textTheme.headlineMedium?.copyWith(
                                  color: Colors.amberAccent,
                                  fontWeight: FontWeight.bold,
                                ),
                          ),
                          const SizedBox(height: 8),
                          const LinearProgressIndicator(),
                        ],
                      ),
                    )
                  else
                    FilledButton.icon(
                      onPressed: _startGuidedCoughCheckIn,
                      icon: const Icon(Icons.mic),
                      label: const Text('Iniciar Teste Guiado (3 Tosses • 6s)'),
                    ),
                ],
              ),
            ),
          ),
          if (classification != null && acoustic != null) ...[
            const SizedBox(height: 16),
            Card(
              child: Padding(
                padding: const EdgeInsets.all(16),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      '2. Dashboard Clínico de Resultados',
                      style: Theme.of(context).textTheme.titleMedium,
                    ),
                    const Divider(),
                    Text(
                      'Classificação: ${classification['label']} '
                      '(${((classification['confidence'] as num) * 100).toStringAsFixed(1)}%)',
                      style: Theme.of(context).textTheme.titleLarge?.copyWith(
                            color: Colors.lightBlueAccent,
                            fontWeight: FontWeight.bold,
                          ),
                    ),
                    const SizedBox(height: 8),
                    Text('Eventos de tosse detectados: ${metrics?['cough_events_count']}'),
                    Text('Centroide Espectral Médio: ${acoustic['mean_spectral_centroid_hz']} Hz'),
                    Text('Duração Acústica Ativa: ${acoustic['cough_duration_seconds']} s'),
                    Text('Energia Estimada: ${acoustic['estimated_energy_db']} dB'),
                    if (explainability?['spectrogram_heatmap_url'] != null) ...[
                      const SizedBox(height: 12),
                      const Text(
                        'Mapa de Explicabilidade (Grad-CAM Mel-Spectrogram):',
                        style: TextStyle(fontWeight: FontWeight.w600),
                      ),
                      const SizedBox(height: 8),
                      ClipRRect(
                        borderRadius: BorderRadius.circular(8),
                        child: Image.network(
                          explainability!['spectrogram_heatmap_url'] as String,
                          errorBuilder: (_, __, ___) => const Text(
                            'Imagem Grad-CAM disponível na URL do servidor.',
                          ),
                        ),
                      ),
                    ],
                  ],
                ),
              ),
            ),
          ],
        ],
      ),
    );
  }
}
