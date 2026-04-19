import 'dart:math' as math;
import 'package:flutter/material.dart';
import '../models/isl_data.dart';

// ── Palette ───────────────────────────────────────────────────────────────────
const _kSkin      = Color(0xFFDEB897);
const _kSkinLight = Color(0xFFF2CFBA);
const _kSkinDark  = Color(0xFFBE8C6A);
const _kHair      = Color(0xFF180E06);
const _kShirt     = Color(0xFF1565C0);
const _kShirtDark = Color(0xFF0A3D8F);
const _kPants     = Color(0xFF263238);
const _kShoe      = Color(0xFF1A1A1A);
const _kShoeSole  = Color(0xFF424242);

// ── MediaPipe 33-pt indices ───────────────────────────────────────────────────
const _jNose = 0;
const _jLEar = 7;
const _jLS   = 11; const _jRS   = 12;
const _jLE   = 13; const _jRE   = 14;
const _jLW   = 15; const _jRW   = 16;
const _jLPin = 17; const _jRPin = 18;
const _jLIdx = 19; const _jRIdx = 20;
const _jLTh  = 21; const _jRTh  = 22;
const _jLH   = 23; const _jRH   = 24;
const _jLK   = 25; const _jRK   = 26;
const _jLA   = 27; const _jRA   = 28;
const _jLF   = 31; const _jRF   = 32;

// ── Helpers ───────────────────────────────────────────────────────────────────
Offset _mid(Offset a, Offset b) => Offset((a.dx + b.dx) / 2, (a.dy + b.dy) / 2);
Offset _lerp(Offset a, Offset b, double t) =>
    Offset(a.dx + (b.dx - a.dx) * t, a.dy + (b.dy - a.dy) * t);
double _dist(Offset a, Offset b) {
  final dx = a.dx - b.dx, dy = a.dy - b.dy;
  return math.sqrt(dx * dx + dy * dy);
}
Color _shade(Color c, double v) => Color.lerp(c, Colors.black, v.clamp(0, 1))!;
Color _tint(Color c, double v)  => Color.lerp(c, Colors.white, v.clamp(0, 1))!;

Offset _norm(Offset o) {
  final len = math.sqrt(o.dx * o.dx + o.dy * o.dy);
  return len < 0.001 ? const Offset(0, -1) : Offset(o.dx / len, o.dy / len);
}

// ── Perspective projection ────────────────────────────────────────────────────
Offset _project(Joint j, Size s) {
  const cx = 0.5, cy = 0.44;
  final sc = 1.0 + j.z * 0.18;
  return Offset((cx + (j.x - cx) * sc) * s.width,
                (cy + (j.y - cy) * sc) * s.height);
}

// ── Cylindrical limb ──────────────────────────────────────────────────────────
void _drawCylinder(Canvas canvas, Offset from, Offset to,
    double r1, double r2, Color fill,
    {bool caps = true, double shadowA = 55}) {
  final dx = to.dx - from.dx, dy = to.dy - from.dy;
  final len = math.sqrt(dx * dx + dy * dy);
  if (len < 0.5) return;
  final nx = -dy / len, ny = dx / len;

  final a = Offset(from.dx + nx * r1, from.dy + ny * r1);
  final b = Offset(from.dx - nx * r1, from.dy - ny * r1);
  final c = Offset(to.dx - nx * r2, to.dy - ny * r2);
  final d = Offset(to.dx + nx * r2, to.dy + ny * r2);

  final path = Path()
    ..moveTo(a.dx, a.dy) ..lineTo(b.dx, b.dy)
    ..lineTo(c.dx, c.dy) ..lineTo(d.dx, d.dy) ..close();

  final minX = math.min(a.dx, math.min(b.dx, math.min(c.dx, d.dx)));
  final minY = math.min(a.dy, math.min(b.dy, math.min(c.dy, d.dy)));
  final maxX = math.max(a.dx, math.max(b.dx, math.max(c.dx, d.dx)));
  final maxY = math.max(a.dy, math.max(b.dy, math.max(c.dy, d.dy)));
  final bounds = Rect.fromLTRB(minX, minY, maxX, maxY);

  // drop shadow
  if (shadowA > 0) {
    canvas.drawPath(path.shift(const Offset(2.5, 3.5)),
        Paint()..color = Colors.black.withAlpha(shadowA.toInt()));
  }
  // gradient fill
  canvas.drawPath(path, Paint()
    ..shader = LinearGradient(
      begin: const Alignment(-1, -0.4),
      end:   const Alignment(1,  0.4),
      colors: [_tint(fill, 0.30), fill, _shade(fill, 0.28)],
      stops:  const [0.0, 0.42, 1.0],
    ).createShader(bounds));
  // outline
  canvas.drawPath(path, Paint()
    ..color = _shade(fill, 0.45).withAlpha(160)
    ..style = PaintingStyle.stroke ..strokeWidth = 0.6);

  if (caps) {
    for (final (ctr, r) in [(from, r1), (to, r2)]) {
      canvas.drawOval(
        Rect.fromCenter(center: ctr, width: r * 2.0, height: r * 0.55),
        Paint()..color = _tint(fill, 0.15));
    }
  }
}

// ── Sphere ────────────────────────────────────────────────────────────────────
void _drawSphere(Canvas canvas, Offset c, double r, Color fill) {
  canvas.drawCircle(c + Offset(r * .12, r * .16), r,
      Paint()..color = Colors.black.withAlpha(50));
  canvas.drawCircle(c, r, Paint()
    ..shader = RadialGradient(
      center: const Alignment(-0.35, -0.45), radius: 0.85,
      colors: [_tint(fill, 0.38), fill, _shade(fill, 0.32)],
      stops: const [0.0, 0.5, 1.0],
    ).createShader(Rect.fromCircle(center: c, radius: r)));
  canvas.drawCircle(c, r, Paint()
    ..color = _shade(fill, 0.50).withAlpha(100)
    ..style = PaintingStyle.stroke ..strokeWidth = 0.7);
}

// ── Head ──────────────────────────────────────────────────────────────────────
void _drawHead(Canvas canvas, Offset center, double r) {
  // Neck stub below head
  canvas.drawOval(
    Rect.fromCenter(center: center + Offset(0, r * 0.82),
        width: r * 0.58, height: r * 0.48),
    Paint()..color = _kSkin);

  // Head sphere
  _drawSphere(canvas, center, r, _kSkin);

  // ── Hair — TOP half only ──────────────────────────────────────────────
  // sweepAngle = -π  → counter-clockwise from left (π) → top → right (0)
  // This correctly fills ONLY the upper hemisphere.
  final hairPath = Path()
    ..moveTo(center.dx - r, center.dy)
    ..arcTo(Rect.fromCircle(center: center, radius: r),
        math.pi, -math.pi, false)      // top arc: left → top → right
    ..lineTo(center.dx - r, center.dy) // straight back along the equator
    ..close();
  canvas.drawPath(hairPath, Paint()..color = _kHair);

  // Extra hair volume on top
  canvas.drawOval(
    Rect.fromCenter(
        center: center + Offset(0, -r * 0.82),
        width: r * 1.5, height: r * 0.55),
    Paint()..color = _kHair);

  // Ears
  for (final side in [-1.0, 1.0]) {
    canvas.drawOval(
      Rect.fromCenter(
          center: Offset(center.dx + side * r * 0.97, center.dy + r * 0.10),
          width: r * 0.22, height: r * 0.34),
      Paint()..color = _kSkinDark);
  }

  // ── Face features – placed in LOWER half of head ──────────────────────
  final eyeY = center.dy + r * 0.12;  // below center = clearly visible
  const eyeOff = 0.28;

  for (final side in [-1.0, 1.0]) {
    final ex = center.dx + side * r * eyeOff;
    // white
    canvas.drawOval(
        Rect.fromCenter(center: Offset(ex, eyeY), width: r * 0.24, height: r * 0.16),
        Paint()..color = Colors.white);
    // iris
    canvas.drawCircle(Offset(ex, eyeY), r * 0.09,
        Paint()..color = const Color(0xFF3E2000));
    // pupil
    canvas.drawCircle(Offset(ex, eyeY), r * 0.05,
        Paint()..color = Colors.black);
    // highlight
    canvas.drawCircle(Offset(ex - r * 0.025, eyeY - r * 0.025), r * 0.022,
        Paint()..color = Colors.white);
    // brow
    final brow = Path()
      ..moveTo(ex - r * 0.12, eyeY - r * 0.17)
      ..quadraticBezierTo(ex, eyeY - r * 0.22, ex + r * 0.12, eyeY - r * 0.16);
    canvas.drawPath(brow, Paint()
      ..color = _kHair ..style = PaintingStyle.stroke
      ..strokeWidth = r * 0.055 ..strokeCap = StrokeCap.round);
  }

  // Nose
  canvas.drawOval(
    Rect.fromCenter(
        center: Offset(center.dx, center.dy + r * 0.30),
        width: r * 0.18, height: r * 0.11),
    Paint()..color = _kSkinDark);

  // Smile
  final smile = Path()
    ..moveTo(center.dx - r * 0.15, center.dy + r * 0.44)
    ..quadraticBezierTo(
        center.dx, center.dy + r * 0.53,
        center.dx + r * 0.15, center.dy + r * 0.44);
  canvas.drawPath(smile, Paint()
    ..color = _kSkinDark ..style = PaintingStyle.stroke
    ..strokeWidth = r * 0.058 ..strokeCap = StrokeCap.round);
}

// ── Hand ──────────────────────────────────────────────────────────────────────
// Forces a minimum palm size so fingers are always visible regardless of
// how close the pose keypoints are in the normalised coordinate space.
void _drawHand(Canvas canvas, Offset wrist, Offset pinkyMCP,
    Offset indexMCP, Offset thumbRef, double canvasW) {

  // ── enforce minimum palm dimensions ───────────────────────────────────
  final minPalm = canvasW * 0.095;
  var rawW = _dist(indexMCP, pinkyMCP);
  if (rawW < minPalm) {
    // Re-derive index/pinky around the wrist using palm direction
    final dir = _norm(Offset(
      (indexMCP.dx + pinkyMCP.dx) / 2 - wrist.dx,
      (indexMCP.dy + pinkyMCP.dy) / 2 - wrist.dy,
    ));
    final perp = Offset(-dir.dy, dir.dx);
    final palmCtr = wrist + Offset(dir.dx * minPalm * 1.1, dir.dy * minPalm * 1.1);
    indexMCP = palmCtr + Offset(perp.dx * minPalm * 0.5,  perp.dy * minPalm * 0.5);
    pinkyMCP = palmCtr + Offset(-perp.dx * minPalm * 0.5, -perp.dy * minPalm * 0.5);
    rawW = minPalm;
  }

  // Palm geometry
  final palmDir  = _norm(Offset(
    (indexMCP.dx + pinkyMCP.dx) / 2 - wrist.dx,
    (indexMCP.dy + pinkyMCP.dy) / 2 - wrist.dy,
  ));
  final palmPerp = Offset(-palmDir.dy, palmDir.dx);
  final palmWidth = _dist(indexMCP, pinkyMCP);

  // Derive middle & ring MCP
  final middleMCP = _mid(indexMCP, pinkyMCP);
  final ringMCP   = _lerp(middleMCP, pinkyMCP, 0.50);
  final thumbMCP  = wrist + Offset(
    palmPerp.dx * palmWidth * 0.55 + palmDir.dx * palmWidth * 0.25,
    palmPerp.dy * palmWidth * 0.55 + palmDir.dy * palmWidth * 0.25,
  );

  final mcps = [thumbMCP, indexMCP, middleMCP, ringMCP, pinkyMCP];

  // Finger length ratios & fanning angles (relative to palmDir)
  const lenRatios  = [0.68, 0.90, 1.00, 0.88, 0.72];
  const fanAngles  = [0.55, 0.18, 0.00, -0.18, -0.36]; // radians
  final baseLen    = palmWidth * 1.10;
  final fingerW    = palmWidth * 0.115;

  // ── Palm shape ────────────────────────────────────────────────────────
  final palmPath = Path()
    ..moveTo(wrist.dx + palmPerp.dx * palmWidth * 0.38,
             wrist.dy + palmPerp.dy * palmWidth * 0.38)
    ..lineTo(indexMCP.dx, indexMCP.dy)
    ..lineTo(pinkyMCP.dx, pinkyMCP.dy)
    ..lineTo(wrist.dx - palmPerp.dx * palmWidth * 0.38,
             wrist.dy - palmPerp.dy * palmWidth * 0.38)
    ..close();

  canvas.drawPath(palmPath.shift(const Offset(1.5, 2.5)),
      Paint()..color = Colors.black.withAlpha(40));

  final palmBounds = Rect.fromPoints(wrist, middleMCP + Offset(palmWidth, palmWidth));
  canvas.drawPath(palmPath, Paint()
    ..shader = LinearGradient(
      begin: const Alignment(-0.7, -0.7),
      end:   const Alignment(0.7, 0.7),
      colors: [_kSkinLight, _kSkin, _kSkinDark],
    ).createShader(palmBounds));
  canvas.drawPath(palmPath, Paint()
    ..color = _kSkinDark.withAlpha(100)
    ..style = PaintingStyle.stroke ..strokeWidth = 0.6);

  // ── Fingers ───────────────────────────────────────────────────────────
  for (int fi = 0; fi < 5; fi++) {
    final mcp  = mcps[fi];
    final fLen = baseLen * lenRatios[fi];
    final fW   = fi == 0 ? fingerW * 1.0 : fingerW * 0.88;
    final fan  = fanAngles[fi];

    // Rotate palmDir by fan angle
    final cos = math.cos(fan), sin = math.sin(fan);
    final fDir = Offset(
      palmDir.dx * cos - palmDir.dy * sin,
      palmDir.dx * sin + palmDir.dy * cos,
    );

    // 2 segments: proximal (60%) + distal (40%)
    final mid = mcp + Offset(fDir.dx * fLen * 0.58, fDir.dy * fLen * 0.58);
    final tip = mcp + Offset(fDir.dx * fLen,        fDir.dy * fLen);

    _drawCylinder(canvas, mcp, mid, fW, fW * 0.88, _kSkin, caps: false, shadowA: 0);
    _drawCylinder(canvas, mid, tip, fW * 0.82, fW * 0.65, _kSkin, caps: false, shadowA: 0);

    // Knuckle & middle joint
    for (final pt in [mcp, mid]) {
      _drawSphere(canvas, pt, fW * 0.78, _kSkinDark);
      canvas.drawCircle(pt, fW * 0.55, Paint()..color = _tint(_kSkin, 0.08));
    }
    // Fingertip
    _drawSphere(canvas, tip, fW * 0.62, _kSkin);
  }

  // Wrist joint
  _drawSphere(canvas, wrist, fingerW * 1.1, _kSkinDark);
}

// ── Holistic hand renderer (uses actual 21 MediaPipe hand landmarks) ─────────
// MediaPipe Hand connections
const _kHandBones = [
  // thumb
  (0,1),(1,2),(2,3),(3,4),
  // index
  (0,5),(5,6),(6,7),(7,8),
  // middle
  (0,9),(9,10),(10,11),(11,12),
  // ring
  (0,13),(13,14),(14,15),(15,16),
  // pinky
  (0,17),(17,18),(18,19),(19,20),
  // knuckle bar
  (5,9),(9,13),(13,17),
];

void _drawHandHolistic(Canvas canvas, List<Joint> lm, Size s) {
  if (lm.length < 21) return;
  final pts = lm.map((j) => _project(j, s)).toList();

  // Compute scale from wrist→middle-MCP distance (fallback minimum)
  final wristToMid = math.max(_dist(pts[0], pts[9]), s.width * 0.035);
  final boneW  = wristToMid * 0.28;
  final jointR = wristToMid * 0.23;

  // Palm fill
  final palmPath = Path()
    ..moveTo(pts[0].dx, pts[0].dy)
    ..lineTo(pts[5].dx, pts[5].dy)
    ..lineTo(pts[9].dx, pts[9].dy)
    ..lineTo(pts[13].dx, pts[13].dy)
    ..lineTo(pts[17].dx, pts[17].dy)
    ..close();

  canvas.drawPath(palmPath.shift(const Offset(1.5, 2.0)),
      Paint()..color = Colors.black.withAlpha(40));

  final pBounds = Rect.fromPoints(pts[0], pts[12]);
  canvas.drawPath(palmPath, Paint()
    ..shader = LinearGradient(
      begin: const Alignment(-0.8,-0.8), end: const Alignment(0.8, 0.8),
      colors: [_kSkinLight, _kSkin, _kSkinDark],
    ).createShader(pBounds));
  canvas.drawPath(palmPath, Paint()
    ..color = _kSkinDark.withAlpha(90)
    ..style = PaintingStyle.stroke ..strokeWidth = 0.5);

  // Bones
  for (final (a, b) in _kHandBones) {
    _drawCylinder(canvas, pts[a], pts[b], boneW, boneW * 0.88,
        _kSkin, caps: false, shadowA: 0);
  }

  // Joints — fingertips larger and lighter
  const tips = {4, 8, 12, 16, 20};
  const knuckles = {5, 9, 13, 17};
  for (int i = 0; i < 21; i++) {
    final r = tips.contains(i)    ? jointR * 1.10 :
              knuckles.contains(i) ? jointR * 0.95 :
                                     jointR * 0.78;
    final col = tips.contains(i) ? _kSkin : _kSkinDark;
    _drawSphere(canvas, pts[i], r, col);
    if (tips.contains(i)) {
      // nail hint
      canvas.drawCircle(pts[i], r * 0.55,
          Paint()..color = Colors.white.withAlpha(70));
    }
  }
}

// ── Main Painter ──────────────────────────────────────────────────────────────
class AvatarPainter3D extends CustomPainter {
  final List<Joint> frame;
  final bool isDark;

  const AvatarPainter3D({required this.frame, this.isDark = false});

  Offset _j(int i, Size s) {
    if (i >= frame.length) return Offset(s.width / 2, s.height / 2);
    return _project(frame[i], s);
  }

  double _z(int i) => i < frame.length ? frame[i].z : 0.0;

  @override
  void paint(Canvas canvas, Size size) {
    if (frame.length < 29) return;
    final s = size;
    final sc = s.width;

    // Key joint positions
    final lS  = _j(_jLS, s); final rS  = _j(_jRS, s);
    final lE  = _j(_jLE, s); final rE  = _j(_jRE, s);
    final lW  = _j(_jLW, s); final rW  = _j(_jRW, s);
    final lH  = _j(_jLH, s); final rH  = _j(_jRH, s);
    final lK  = _j(_jLK, s); final rK  = _j(_jRK, s);
    final lA  = _j(_jLA, s); final rA  = _j(_jRA, s);
    final lFt = _j(_jLF, s); final rFt = _j(_jRF, s);
    final nose = _j(_jNose, s);

    final shoulderMid = _mid(lS, rS);
    final neckTop     = _lerp(nose, shoulderMid, 0.32);

    // Radii
    final rHead   = math.max(_dist(nose, _j(_jLEar, s)) * 0.76, sc * 0.09);
    final rArm    = sc * 0.036;
    final rFore   = sc * 0.030;
    final rThigh  = sc * 0.045;
    final rShin   = sc * 0.036;
    final rShoe   = sc * 0.028;
    final rNeck   = sc * 0.036;

    // Depth helpers
    final lArmZ = (_z(_jLS) + _z(_jLE) + _z(_jLW)) / 3;
    final rArmZ = (_z(_jRS) + _z(_jRE) + _z(_jRW)) / 3;
    final lLegZ = (_z(_jLH) + _z(_jLK) + _z(_jLA)) / 3;
    final rLegZ = (_z(_jRH) + _z(_jRK) + _z(_jRA)) / 3;
    final leftArmBack = lArmZ < rArmZ;
    final leftLegBack = lLegZ < rLegZ;

    // ── Draw arms (closures so we can call in depth order) ────────────────
    void drawArm(bool left) {
      final s1 = left ? lS : rS;
      final el = left ? lE : rE;
      final wr = left ? lW : rW;
      _drawCylinder(canvas, s1, el, rArm, rArm * 0.92, _kShirt);
      _drawCylinder(canvas, el, wr, rFore, rFore * 0.90, _kSkin);
      _drawSphere(canvas, el, rFore * 0.90, _kSkin);
    }

    // Use actual 21-pt hand landmarks when the pose has 75 joints (holistic).
    // Fall back to estimation when only 33-pt body data is available.
    final isHolistic = frame.length >= 75;

    void drawHand(bool left) {
      if (isHolistic) {
        final start = left ? 33 : 54;
        _drawHandHolistic(canvas, frame.sublist(start, start + 21), s);
      } else {
        _drawHand(
          canvas,
          left ? lW : rW,
          left ? _j(_jLPin, s) : _j(_jRPin, s),
          left ? _j(_jLIdx, s) : _j(_jRIdx, s),
          left ? _j(_jLTh,  s) : _j(_jRTh,  s),
          sc,
        );
      }
    }

    void drawLeg(bool left) {
      final hip   = left ? lH  : rH;
      final knee  = left ? lK  : rK;
      final ankle = left ? lA  : rA;
      final foot  = left ? lFt : rFt;
      _drawCylinder(canvas, hip,   knee,  rThigh, rThigh * 0.88, _kPants);
      _drawCylinder(canvas, knee,  ankle, rShin,  rShin  * 0.82, _kPants);
      _drawSphere(canvas, knee, rShin * 0.85, _shade(_kPants, 0.20));
      // Shoe
      _drawCylinder(canvas, ankle, foot, rShoe, rShoe * 0.85, _kShoe,
          caps: false, shadowA: 0);
      canvas.drawOval(
        Rect.fromCenter(center: foot, width: rShoe * 3.6, height: rShoe * 1.15),
        Paint()..color = _kShoeSole);
      canvas.drawOval(
        Rect.fromCenter(center: foot, width: rShoe * 3.3, height: rShoe * 0.88),
        Paint()..color = _kShoe);
    }

    // ── DRAW ORDER: back → front ───────────────────────────────────────────
    // Back leg
    drawLeg(!leftLegBack);   // draw the further leg first
    // Back arm
    if (leftArmBack) { drawArm(true); } else { drawArm(false); }

    // ── Torso ────────────────────────────────────────────────────────────
    final torsoPath = Path()
      ..moveTo(lS.dx, lS.dy) ..lineTo(rS.dx, rS.dy)
      ..lineTo(rH.dx, rH.dy) ..lineTo(lH.dx, lH.dy) ..close();

    canvas.drawPath(torsoPath.shift(const Offset(2, 3)),
        Paint()..color = Colors.black.withAlpha(50));

    final tBounds = Rect.fromLTRB(
      math.min(lS.dx, lH.dx) - 4, math.min(lS.dy, rS.dy),
      math.max(rS.dx, rH.dx) + 4, math.max(lH.dy, rH.dy));
    canvas.drawPath(torsoPath, Paint()
      ..shader = LinearGradient(
        begin: const Alignment(-1, 0), end: const Alignment(1, 0),
        colors: [_tint(_kShirt, 0.22), _kShirt, _kShirtDark],
        stops: const [0.0, 0.45, 1.0],
      ).createShader(tBounds));
    canvas.drawPath(torsoPath, Paint()
      ..color = _kShirtDark.withAlpha(150)
      ..style = PaintingStyle.stroke ..strokeWidth = 0.7);

    // Collar
    canvas.drawOval(
      Rect.fromCenter(center: shoulderMid + Offset(0, -rArm * 0.2),
          width: _dist(lS, rS) * 0.55, height: rArm * 0.65),
      Paint()..color = _kShirtDark);

    // Shoulder caps
    for (final pos in [lS, rS]) {
      _drawSphere(canvas, pos, rArm * 0.98, _kShirt);
    }

    // Front leg
    drawLeg(leftLegBack);

    // ── Neck ───────────────────────────────────────────────────────────────
    _drawCylinder(canvas, neckTop, shoulderMid, rNeck, rNeck * 0.88, _kSkin,
        shadowA: 20);

    // ── Front arm + both hands ─────────────────────────────────────────────
    if (leftArmBack) {
      drawArm(false);
      drawHand(false);
      drawHand(true);
    } else {
      drawArm(true);
      drawHand(true);
      drawHand(false);
    }

    // ── Head (always on top) ───────────────────────────────────────────────
    _drawHead(canvas, nose + Offset(0, -rHead * 0.08), rHead);
  }

  @override
  bool shouldRepaint(AvatarPainter3D old) =>
      old.frame != frame || old.isDark != isDark;
}

// ── Animated widget (same public API) ────────────────────────────────────────
class AvatarView extends StatefulWidget {
  final List<PoseFrame> frames;
  final double fps;
  final bool isDark;

  const AvatarView({
    super.key, required this.frames, this.fps = 15.0, this.isDark = false,
  });

  @override
  State<AvatarView> createState() => _AvatarViewState();
}

class _AvatarViewState extends State<AvatarView>
    with SingleTickerProviderStateMixin {
  late AnimationController _ctrl;

  @override
  void initState() {
    super.initState();
    _ctrl = AnimationController(vsync: this, duration: _dur())
      ..addListener(() => setState(() {}))
      ..repeat();
  }

  @override
  void didUpdateWidget(AvatarView old) {
    super.didUpdateWidget(old);
    if (old.frames != widget.frames) {
      _ctrl.duration = _dur();
      _ctrl..reset()..repeat();
    }
  }

  @override
  void dispose() { _ctrl.dispose(); super.dispose(); }

  Duration _dur() {
    final n = widget.frames.isEmpty ? 1 : widget.frames.length;
    return Duration(milliseconds: (n / widget.fps * 1000).round());
  }

  List<Joint> _interpFrame() {
    final frames = widget.frames;
    if (frames.isEmpty) return [];
    if (frames.length == 1) return frames[0];
    final raw = _ctrl.value * frames.length;
    final ai = raw.floor() % frames.length;
    final bi = (ai + 1) % frames.length;
    final t  = raw - raw.floor();
    final ts = t * t * (3 - 2 * t); // smoothstep
    final fa = frames[ai], fb = frames[bi];
    final n  = math.min(fa.length, fb.length);
    return List.generate(n, (i) => Joint(
      fa[i].x + (fb[i].x - fa[i].x) * ts,
      fa[i].y + (fb[i].y - fa[i].y) * ts,
      fa[i].z + (fb[i].z - fa[i].z) * ts,
    ));
  }

  @override
  Widget build(BuildContext context) => RepaintBoundary(
    child: CustomPaint(
      painter: AvatarPainter3D(frame: _interpFrame(), isDark: widget.isDark),
      child: const SizedBox.expand(),
    ),
  );
}
