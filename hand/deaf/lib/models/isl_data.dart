/// Data models for the ISL pipeline response.
library;

class PipelineResult {
  final String text;
  final String gloss;
  final List<List<List<double>>> pose; // [frame][joint][x,y,z]

  const PipelineResult({
    required this.text,
    required this.gloss,
    required this.pose,
  });

  factory PipelineResult.fromJson(Map<String, dynamic> json) {
    final rawPose = json['pose'] as List<dynamic>;
    final pose = rawPose
        .map((frame) => (frame as List<dynamic>)
            .map((joint) => (joint as List<dynamic>)
                .map((v) => (v as num).toDouble())
                .toList())
            .toList())
        .toList();
    return PipelineResult(
      text: json['text'] as String? ?? '',
      gloss: json['gloss'] as String? ?? '',
      pose: pose,
    );
  }

  bool get hasPose => pose.isNotEmpty;
  int get frameCount => pose.length;
  int get jointCount => pose.isEmpty ? 0 : pose[0].length;
}

/// A single joint position in 3-space, normalised to [0, 1].
class Joint {
  final double x, y, z;
  const Joint(this.x, this.y, this.z);
}

/// One frame = list of joint positions.
typedef PoseFrame = List<Joint>;

/// Converts the raw nested-list pose into typed PoseFrames.
List<PoseFrame> parsePoseFrames(List<List<List<double>>> raw) {
  return raw
      .map((frame) => frame.map((j) => Joint(j[0], j[1], j[2])).toList())
      .toList();
}
