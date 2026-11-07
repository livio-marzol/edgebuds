import 'dart:convert';
import 'dart:io';
import 'package:http/http.dart' as http;

/// Upload Gateway (Seção 2 e Seção 6 do PRD/Spec):
/// Envia o arquivo WAV (PCM 16-bit, 16 kHz Mono) e metadados do paciente
/// para o endpoint POST /api/v1/analyze-cough no Cloud Backend FastAPI.
class EdgeBudsApiClient {
  final String baseUrl;

  const EdgeBudsApiClient({required this.baseUrl});

  Future<Map<String, dynamic>> analyzeCough({
    required File wavFile,
    String deviceType = 'samsung_buds_2',
    Map<String, dynamic>? patientMetadata,
  }) async {
    final uri = Uri.parse('$baseUrl/api/v1/analyze-cough');
    final request = http.MultipartRequest('POST', uri)
      ..fields['device_type'] = deviceType
      ..files.add(
        await http.MultipartFile.fromPath(
          'audio_file',
          wavFile.path,
          filename: 'galaxy_buds_checkin.wav',
        ),
      );

    if (patientMetadata != null) {
      request.fields['patient_metadata'] = jsonEncode(patientMetadata);
    }

    final streamedResponse = await request.send();
    final response = await http.Response.fromStream(streamedResponse);

    if (response.statusCode != 200) {
      throw HttpException(
        'Erro na API (${response.statusCode}): ${response.body}',
        uri: uri,
      );
    }

    return jsonDecode(response.body) as Map<String, dynamic>;
  }
}
