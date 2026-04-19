import 'dart:async';
import 'dart:io';

import 'package:flutter/material.dart';
import 'package:flutter_animate/flutter_animate.dart';
import 'package:path/path.dart' as p;
import 'package:path_provider/path_provider.dart';
import 'package:permission_handler/permission_handler.dart';
import 'package:record/record.dart';

import '../models/isl_data.dart';
import '../services/api_service.dart';
import '../widgets/avatar_painter.dart';

// ── pipeline state ────────────────────────────────────────────────────────────
enum _Stage { idle, recording, processing, done, error }

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> with TickerProviderStateMixin {
  final _recorder = AudioRecorder();
  final _api = const ApiService();

  _Stage _stage = _Stage.idle;
  String _statusMsg = 'Tap the mic to begin';
  String _recognizedText = '';
  String _gloss = '';
  List<PoseFrame> _poseFrames = [];

  String? _tmpAudioPath;
  late final AnimationController _pulseCtrl;
  late final AnimationController _waveCtrl;

  @override
  void initState() {
    super.initState();
    _pulseCtrl = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 900),
    )..repeat(reverse: true);
    _waveCtrl = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 1500),
    )..repeat();
  }

  @override
  void dispose() {
    _pulseCtrl.dispose();
    _waveCtrl.dispose();
    _recorder.dispose();
    super.dispose();
  }

  // ── recording ──────────────────────────────────────────────────────────────
  Future<void> _startRecording() async {
    final micOk = await Permission.microphone.request();
    if (!micOk.isGranted) {
      _setError('Microphone permission denied.');
      return;
    }

    final dir = await getTemporaryDirectory();
    _tmpAudioPath = p.join(dir.path, 'isl_recording.m4a');

    await _recorder.start(
      const RecordConfig(encoder: AudioEncoder.aacLc, bitRate: 128000),
      path: _tmpAudioPath!,
    );

    setState(() {
      _stage = _Stage.recording;
      _statusMsg = 'Listening… tap again to stop';
      _recognizedText = '';
      _gloss = '';
      _poseFrames = [];
    });
  }

  Future<void> _stopAndProcess() async {
    await _recorder.stop();
    setState(() {
      _stage = _Stage.processing;
      _statusMsg = 'Processing…';
    });

    if (_tmpAudioPath == null || !File(_tmpAudioPath!).existsSync()) {
      _setError('Audio file not found.');
      return;
    }

    try {
      final result = await _api.runPipeline(File(_tmpAudioPath!));
      final frames = parsePoseFrames(result.pose);
      setState(() {
        _stage = _Stage.done;
        _recognizedText = result.text;
        _gloss = result.gloss;
        _poseFrames = frames;
        _statusMsg = 'Showing ISL translation';
      });
    } on ApiException catch (e) {
      _setError('Server error (${e.statusCode}): ${e.message}');
    } catch (e) {
      _setError('Failed: $e');
    }
  }

  Future<void> _onMicTap() async {
    if (_stage == _Stage.recording) {
      await _stopAndProcess();
    } else if (_stage == _Stage.idle || _stage == _Stage.done || _stage == _Stage.error) {
      await _startRecording();
    }
  }

  void _reset() => setState(() {
        _stage = _Stage.idle;
        _statusMsg = 'Tap the mic to begin';
        _recognizedText = '';
        _gloss = '';
        _poseFrames = [];
      });

  void _setError(String msg) => setState(() {
        _stage = _Stage.error;
        _statusMsg = msg;
      });

  // ── build ──────────────────────────────────────────────────────────────────
  @override
  Widget build(BuildContext context) {
    final cs = Theme.of(context).colorScheme;
    final isDark = Theme.of(context).brightness == Brightness.dark;

    return Scaffold(
      backgroundColor: cs.surface,
      appBar: _buildAppBar(cs),
      body: Column(
        children: [
          // ── avatar panel ──────────────────────────────────────────────────
          Expanded(
            flex: 5,
            child: _buildAvatarPanel(cs, isDark),
          ),

          // ── text + gloss strip ────────────────────────────────────────────
          _buildInfoStrip(cs),

          // ── status + mic ──────────────────────────────────────────────────
          _buildControlArea(cs),

          const SizedBox(height: 12),
        ],
      ),
    );
  }

  // ── app bar ────────────────────────────────────────────────────────────────
  AppBar _buildAppBar(ColorScheme cs) => AppBar(
        backgroundColor: cs.surface,
        elevation: 0,
        title: Row(
          children: [
            Container(
              width: 34,
              height: 34,
              decoration: BoxDecoration(
                color: cs.primary,
                borderRadius: BorderRadius.circular(9),
              ),
              child: Icon(Icons.sign_language, color: cs.onPrimary, size: 20),
            ),
            const SizedBox(width: 10),
            Text(
              'SpeakEasy ISL',
              style: TextStyle(
                fontWeight: FontWeight.bold,
                color: cs.onSurface,
                fontSize: 20,
              ),
            ),
          ],
        ),
        actions: [
          if (_stage != _Stage.idle)
            IconButton(
              icon: const Icon(Icons.refresh_rounded),
              tooltip: 'Reset',
              onPressed: _reset,
            ),
        ],
      );

  // ── avatar ─────────────────────────────────────────────────────────────────
  Widget _buildAvatarPanel(ColorScheme cs, bool isDark) {
    return Container(
      margin: const EdgeInsets.fromLTRB(16, 8, 16, 0),
      decoration: BoxDecoration(
        color: isDark ? cs.surfaceContainerHighest : cs.surfaceContainerLowest,
        borderRadius: BorderRadius.circular(24),
        border: Border.all(
          color: _stage == _Stage.done
              ? cs.primary.withAlpha(120)
              : cs.outlineVariant.withAlpha(80),
          width: 1.5,
        ),
      ),
      child: ClipRRect(
        borderRadius: BorderRadius.circular(24),
        child: _buildAvatarContent(cs, isDark),
      ),
    );
  }

  Widget _buildAvatarContent(ColorScheme cs, bool isDark) {
    // ── idle / error placeholder ─────────────────────────────────────────
    if (_stage == _Stage.idle || _stage == _Stage.error) {
      return _placeholder(cs);
    }

    // ── recording: show animated waveform ───────────────────────────────
    if (_stage == _Stage.recording) {
      return _recordingOverlay(cs);
    }

    // ── processing spinner ───────────────────────────────────────────────
    if (_stage == _Stage.processing) {
      return _processingOverlay(cs);
    }

    // ── done: render avatar ──────────────────────────────────────────────
    if (_poseFrames.isNotEmpty) {
      return AvatarView(
        frames: _poseFrames,
        fps: 15,
        isDark: isDark,
      ).animate().fadeIn(duration: 500.ms);
    }

    return _placeholder(cs);
  }

  Widget _placeholder(ColorScheme cs) => Center(
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            Icon(
              Icons.accessibility_new_rounded,
              size: 72,
              color: cs.outlineVariant,
            ),
            const SizedBox(height: 12),
            Text(
              _stage == _Stage.error ? _statusMsg : 'Avatar will appear here',
              textAlign: TextAlign.center,
              style: TextStyle(
                color: _stage == _Stage.error ? cs.error : cs.outline,
                fontSize: 14,
              ),
            ),
          ],
        ),
      );

  Widget _recordingOverlay(ColorScheme cs) => Center(
        child: AnimatedBuilder(
          animation: _waveCtrl,
          builder: (context, child) {
            return Row(
              mainAxisAlignment: MainAxisAlignment.center,
              crossAxisAlignment: CrossAxisAlignment.end,
              children: List.generate(7, (i) {
                final phase = (_waveCtrl.value * 2 * 3.14159 + i * 0.8) % (2 * 3.14159);
                final h = 16.0 + 28.0 * ((1 + _sinApprox(phase)) / 2);
                return Padding(
                  padding: const EdgeInsets.symmetric(horizontal: 3),
                  child: AnimatedContainer(
                    duration: const Duration(milliseconds: 80),
                    width: 6,
                    height: h,
                    decoration: BoxDecoration(
                      color: cs.primary.withAlpha(180 + (i * 10)),
                      borderRadius: BorderRadius.circular(4),
                    ),
                  ),
                );
              }),
            );
          },
        ),
      );

  Widget _processingOverlay(ColorScheme cs) => Center(
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            SizedBox(
              width: 48,
              height: 48,
              child: CircularProgressIndicator(
                strokeWidth: 3,
                color: cs.primary,
              ),
            ),
            const SizedBox(height: 16),
            Text(
              'Running ISL pipeline…',
              style: TextStyle(color: cs.outline, fontSize: 14),
            ),
          ],
        ),
      );

  // ── text / gloss strip ─────────────────────────────────────────────────────
  Widget _buildInfoStrip(ColorScheme cs) {
    if (_stage != _Stage.done ||
        (_recognizedText.isEmpty && _gloss.isEmpty)) {
      return const SizedBox.shrink();
    }

    return Container(
      margin: const EdgeInsets.fromLTRB(16, 10, 16, 0),
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
      decoration: BoxDecoration(
        color: cs.secondaryContainer.withAlpha(160),
        borderRadius: BorderRadius.circular(16),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          if (_recognizedText.isNotEmpty) ...[
            _label('Recognized Speech', cs),
            const SizedBox(height: 4),
            Text(
              _recognizedText,
              style: TextStyle(
                fontSize: 15,
                color: cs.onSurface,
                height: 1.4,
              ),
            ),
          ],
          if (_gloss.isNotEmpty) ...[
            const SizedBox(height: 8),
            _label('ISL Gloss', cs),
            const SizedBox(height: 4),
            Wrap(
              spacing: 6,
              runSpacing: 4,
              children: _gloss.trim().split(' ').map((word) {
                return Container(
                  padding: const EdgeInsets.symmetric(
                    horizontal: 10, vertical: 4),
                  decoration: BoxDecoration(
                    color: cs.primary.withAlpha(30),
                    borderRadius: BorderRadius.circular(20),
                    border: Border.all(
                      color: cs.primary.withAlpha(100),
                      width: 1,
                    ),
                  ),
                  child: Text(
                    word,
                    style: TextStyle(
                      fontSize: 13,
                      fontWeight: FontWeight.w600,
                      color: cs.primary,
                      letterSpacing: 0.4,
                    ),
                  ),
                );
              }).toList(),
            ),
          ],
        ],
      ),
    ).animate().fadeIn(duration: 400.ms).slideY(begin: 0.08, end: 0);
  }

  Widget _label(String text, ColorScheme cs) => Text(
        text,
        style: TextStyle(
          fontSize: 11,
          fontWeight: FontWeight.w600,
          color: cs.outline,
          letterSpacing: 0.5,
        ),
      );

  // ── mic + status ───────────────────────────────────────────────────────────
  Widget _buildControlArea(ColorScheme cs) {
    return Padding(
      padding: const EdgeInsets.fromLTRB(16, 16, 16, 0),
      child: Column(
        children: [
          // status text
          AnimatedSwitcher(
            duration: const Duration(milliseconds: 250),
            child: Text(
              _stage == _Stage.error ? _statusMsg : _statusMsg,
              key: ValueKey(_statusMsg),
              textAlign: TextAlign.center,
              style: TextStyle(
                fontSize: 13,
                color: _stage == _Stage.error ? cs.error : cs.outline,
              ),
            ),
          ),
          const SizedBox(height: 14),

          // mic button
          Stack(
            alignment: Alignment.center,
            children: [
              // pulse ring while recording
              if (_stage == _Stage.recording)
                AnimatedBuilder(
                  animation: _pulseCtrl,
                  builder: (context, child) => Container(
                    width: 90 + _pulseCtrl.value * 26,
                    height: 90 + _pulseCtrl.value * 26,
                    decoration: BoxDecoration(
                      shape: BoxShape.circle,
                      color: Colors.red.withAlpha(
                        (50 * (1 - _pulseCtrl.value)).toInt(),
                      ),
                    ),
                  ),
                ),

              // button
              GestureDetector(
                onTap: _stage == _Stage.processing ? null : _onMicTap,
                child: AnimatedContainer(
                  duration: const Duration(milliseconds: 200),
                  width: 72,
                  height: 72,
                  decoration: BoxDecoration(
                    shape: BoxShape.circle,
                    color: _micColor(cs),
                    boxShadow: [
                      BoxShadow(
                        color: _micColor(cs).withAlpha(90),
                        blurRadius: 18,
                        spreadRadius: 2,
                      ),
                    ],
                  ),
                  child: Icon(
                    _micIcon(),
                    color: Colors.white,
                    size: 32,
                  ),
                ),
              ),
            ],
          ),
        ],
      ),
    );
  }

  Color _micColor(ColorScheme cs) {
    switch (_stage) {
      case _Stage.recording:
        return Colors.red;
      case _Stage.processing:
        return cs.secondary;
      case _Stage.error:
        return cs.error;
      default:
        return cs.primary;
    }
  }

  IconData _micIcon() {
    switch (_stage) {
      case _Stage.recording:
        return Icons.stop_rounded;
      case _Stage.processing:
        return Icons.hourglass_top_rounded;
      default:
        return Icons.mic_rounded;
    }
  }
}

// Simple sine approximation to avoid dart:math import just for the waveform.
double _sinApprox(double x) {
  // Bhaskara I approximation
  x = x % (2 * 3.14159265358979);
  final sign = x < 3.14159265358979 ? 1.0 : -1.0;
  final xn = x < 3.14159265358979 ? x : x - 3.14159265358979;
  return sign * (16 * xn * (3.14159265358979 - xn)) /
      (5 * 3.14159265358979 * 3.14159265358979 -
          4 * xn * (3.14159265358979 - xn));
}
