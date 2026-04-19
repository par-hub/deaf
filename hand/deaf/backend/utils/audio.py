"""Audio helpers: save upload to a temp file, run faster-whisper transcription."""

import os
import tempfile
from faster_whisper import WhisperModel
from fastapi import UploadFile

_MODEL_SIZE = os.environ.get("WHISPER_MODEL", "base")
_whisper_model: WhisperModel | None = None


def _get_whisper() -> WhisperModel:
    global _whisper_model
    if _whisper_model is None:
        print(f"[Whisper] Loading '{_MODEL_SIZE}' model …")
        _whisper_model = WhisperModel(_MODEL_SIZE, device="cpu", compute_type="int8")
        print("[Whisper] Ready.")
    return _whisper_model


async def transcribe_upload(file: UploadFile) -> str:
    """Save the uploaded audio file to a temp path and return transcript."""
    suffix = _extension(file.filename or "audio.m4a")
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        model = _get_whisper()
        segments, _ = model.transcribe(tmp_path, language="en", beam_size=5)
        return " ".join(seg.text.strip() for seg in segments).strip()
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def _extension(filename: str) -> str:
    ext = os.path.splitext(filename)[-1].lower()
    return ext if ext else ".m4a"
