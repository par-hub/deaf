# SpeakEasy ISL — Setup Guide

## Architecture

```
Flutter App (Android/iOS)
  ↓ audio file (m4a)
FastAPI Backend
  ├─ Whisper        → recognized text
  ├─ GlossModel     → ISL gloss tokens
  └─ PoseModel      → keypoint sequence (T × 33 × 3)
  ↓ JSON
Flutter App
  └─ AvatarView     → animated skeleton
```

---

## 1 · Backend Setup

### Prerequisites
- Python 3.10+
- pip / conda

### Install dependencies

```bash
cd backend
pip install -r requirements.txt
```

> Whisper will download the `base` model (~150 MB) on first run.
> Change the `WHISPER_MODEL` env var to `tiny`, `small`, `medium`, or `large`.

### Prepare model weights

Your two PyTorch checkpoints are stored as extracted directories under
`assets/best_gloss_model.pt/`.  The script below re-zips them into
`backend/model_weights/`.

```bash
cd backend
python setup_models.py
```

Expected output:
```
✓  best_gloss_model  →  model_weights/best_gloss_model.pt  (XXX KB)
✓  best_model (1).pth  →  model_weights/best_model.pth  (XXX KB)
```

### Wire up your model architectures (REQUIRED)

The backend wraps each checkpoint with a thin loader.  If your `.pt`/`.pth`
files were saved with `torch.save(model, path)` (full model), they load
automatically.

If they were saved with `torch.save(model.state_dict(), path)`, open the
corresponding file and follow the comments:

| File | What to edit |
|------|-------------|
| `backend/models/gloss_model.py` | Set `USE_STATE_DICT = True` and fill in `GlossModelArchitecture`, then update the `predict()` method to call your model correctly. |
| `backend/models/pose_model.py` | Set `USE_STATE_DICT = True` and fill in `PoseModelArchitecture`, then update the `predict()` method. |

Both files include a **fallback** that works without the real models so you
can test the Flutter UI immediately.

### Start the server

```bash
cd backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Test it:
```bash
curl http://localhost:8000/health
```

---

## 2 · Flutter Setup

### Prerequisites
- Flutter 3.22+
- Android SDK / Android device or emulator

### Install packages

```bash
flutter pub get
```

### Configure the backend URL

Open `lib/services/api_service.dart` and change `baseUrl`:

| Target | URL |
|--------|-----|
| Android emulator → local PC | `http://10.0.2.2:8000` *(default)* |
| Physical Android device (same Wi-Fi) | `http://<your-PC-LAN-IP>:8000` |
| iOS simulator → local Mac | `http://127.0.0.1:8000` |

### Run the app

```bash
flutter run
```

---

## 3 · Data Flow

```
[Tap mic]
    ↓
AudioRecorder.start()  →  records to /tmp/isl_recording.m4a
    ↓
[Tap mic again]
    ↓
POST /pipeline  multipart audio file
    ↓
Backend:
  Whisper.transcribe(audio)     →  "hello how are you"
  GlossModel.predict(text)      →  "HELLO HOW YOU"
  PoseModel.predict(gloss)      →  float32 array (T, 33, 3)
    ↓
JSON response  { text, gloss, pose }
    ↓
Flutter:
  parsePoseFrames(pose)         →  List<PoseFrame>
  AvatarView(frames: ...)       →  animated skeleton @ 15 fps
```

---

## 4 · Pose Coordinate Format

The backend returns pose as a nested JSON array:

```json
{
  "pose": [
    [                        // frame 0
      [0.50, 0.14, 0.00],    // joint 0  — nose
      [0.52, 0.12, 0.01],    // joint 1  — left eye inner
      ...                    // 33 joints total (MediaPipe layout)
    ],
    ...                      // T frames
  ]
}
```

- All coordinates are normalised to **[0, 1]** (x=0 left, x=1 right, y=0 top, y=1 bottom).
- `z` is depth; positive = closer to camera.

If your `PoseModel` outputs a different layout, adjust `NUM_JOINTS` and the
coordinate normalisation in `backend/models/pose_model.py`.

---

## 5 · Upgrading to a Full 3D Avatar

The current renderer uses Flutter's `CustomPainter` with a 33-joint skeleton.
To use a rigged Mixamo/GLB avatar instead:

1. Add `flutter_unity_widget` or `three_dart` to `pubspec.yaml`.
2. Import a `.glb` Mixamo character into Unity (for `flutter_unity_widget`)
   or load it with `three_dart`'s `GLTFLoader`.
3. Map the 33-joint pose array to the avatar's bone transforms in the
   `AvatarView.build()` method (replace `AvatarPainter` with the 3D widget).

The pose data format and `parsePoseFrames()` helper remain unchanged.

---

## 6 · Troubleshooting

| Symptom | Fix |
|---------|-----|
| `Connection refused` on device | Check `baseUrl` in `api_service.dart`; ensure PC firewall allows port 8000 |
| `Microphone permission denied` | Grant mic permission in Android Settings → Apps → deaf |
| Model loads but inference crashes | Implement your model's `forward()` call in the `predict()` method |
| Avatar doesn't animate | Verify `/pipeline` returns non-empty `pose` array; check server logs |
| Whisper is slow | Set `WHISPER_MODEL=tiny` env var before starting the server |
