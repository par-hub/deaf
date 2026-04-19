import 'dart:io';
import 'package:http/http.dart' as http;
import 'dart:convert';
import '../models/isl_data.dart';

class ApiService {
  final String baseUrl;

  const ApiService({this.baseUrl = 'http://172.16.71.27:8000'});

  // ── full pipeline ──────────────────────────────────────────────────────────
  Future<PipelineResult> runPipeline(File audioFile) async {
    final uri = Uri.parse('$baseUrl/pipeline');
    final request = http.MultipartRequest('POST', uri)
      ..files.add(await http.MultipartFile.fromPath(
        'audio',
        audioFile.path,
        filename: 'recording.m4a',
      ));

    final streamed = await request.send().timeout(const Duration(seconds: 60));
    final body = await streamed.stream.bytesToString();

    if (streamed.statusCode != 200) {
      throw ApiException(
        streamed.statusCode,
        _parseError(body) ?? 'Pipeline failed',
      );
    }

    final json = jsonDecode(body) as Map<String, dynamic>;
    return PipelineResult.fromJson(json);
  }

  // ── individual endpoints (for manual testing / step-by-step mode) ──────────
  Future<String> speechToText(File audioFile) async {
    final uri = Uri.parse('$baseUrl/speech');
    final request = http.MultipartRequest('POST', uri)
      ..files.add(await http.MultipartFile.fromPath(
        'audio',
        audioFile.path,
        filename: 'recording.m4a',
      ));

    final streamed = await request.send().timeout(const Duration(seconds: 45));
    final body = await streamed.stream.bytesToString();

    if (streamed.statusCode != 200) {
      throw ApiException(streamed.statusCode, _parseError(body) ?? 'STT failed');
    }

    final json = jsonDecode(body) as Map<String, dynamic>;
    return json['text'] as String? ?? '';
  }

  Future<String> textToGloss(String text) async {
    final resp = await http.post(
      Uri.parse('$baseUrl/gloss'),
      body: {'text': text},
    ).timeout(const Duration(seconds: 30));

    if (resp.statusCode != 200) {
      throw ApiException(resp.statusCode, _parseError(resp.body) ?? 'Gloss failed');
    }

    return (jsonDecode(resp.body) as Map<String, dynamic>)['gloss'] as String? ?? '';
  }

  Future<List<List<List<double>>>> glossToPose(String gloss) async {
    final resp = await http.post(
      Uri.parse('$baseUrl/pose'),
      body: {'gloss': gloss},
    ).timeout(const Duration(seconds: 30));

    if (resp.statusCode != 200) {
      throw ApiException(resp.statusCode, _parseError(resp.body) ?? 'Pose failed');
    }

    final raw = (jsonDecode(resp.body) as Map<String, dynamic>)['pose'] as List;
    return raw
        .map((f) => (f as List)
            .map((j) => (j as List).map((v) => (v as num).toDouble()).toList())
            .toList())
        .toList();
  }

  String? _parseError(String body) {
    try {
      return (jsonDecode(body) as Map<String, dynamic>)['detail'] as String?;
    } catch (_) {
      return null;
    }
  }
}

class ApiException implements Exception {
  final int statusCode;
  final String message;
  const ApiException(this.statusCode, this.message);

  @override
  String toString() => 'ApiException($statusCode): $message';
}
