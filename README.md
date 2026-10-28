# SpeakEasy ISL — Speech-to-Avatar Prototype

The runnable Flutter/FastAPI project is currently nested under [`hand/deaf/`](hand/deaf/). The repository root contains that project directory rather than the application files directly.

## What it explores

The project documents a prototype pipeline for converting audio to text, text to Indian Sign Language (ISL) gloss, and gloss to a sequence of pose keypoints that can be rendered as an animated skeleton/avatar.

- Flutter mobile app
- FastAPI backend with `/speech`, `/gloss`, `/pose`, and `/pipeline` endpoints
- Whisper-based speech transcription
- Gloss and pose model wrappers
- Avatar rendering assets and configuration

## Setup

Follow the detailed [setup guide](hand/deaf/SETUP.md), which covers backend dependencies, model weights, Flutter dependencies, API URL configuration, and troubleshooting.

The project code is under `hand/deaf/`. Example:

```bash
cd hand/deaf
flutter pub get
flutter run
```

For the backend, follow the setup guide and run its FastAPI server from the nested `backend/` folder.

## Important limitations

Model checkpoint formats and model architecture wiring may require local configuration. Confirm that the real model weights are present and loaded before expecting end-to-end inference. Treat generated text/gloss/pose as experimental output and test with fluent ISL users and accessibility specialists.
