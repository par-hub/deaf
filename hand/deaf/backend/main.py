"""
ISL (Indian Sign Language) pipeline — FastAPI backend.

Endpoints:
  POST /speech   → { text }
  POST /gloss    → { gloss }
  POST /pose     → { pose }
  POST /pipeline → { text, gloss, pose }

Run locally:
  uvicorn main:app --reload --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Any, List

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from models.gloss_model import GlossModel
from models.pose_model import PoseModel
from utils.audio import transcribe_upload


# ── app lifecycle ─────────────────────────────────────────────────────────────
_gloss_model: GlossModel | None = None
_pose_model: PoseModel | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _gloss_model, _pose_model
    print("Loading models …")
    _gloss_model = GlossModel()
    _pose_model = PoseModel()
    print("All models ready. Server is up.")
    yield


app = FastAPI(title="ISL Pipeline API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── response schemas ──────────────────────────────────────────────────────────
class SpeechResponse(BaseModel):
    text: str


class GlossResponse(BaseModel):
    gloss: str


class PoseResponse(BaseModel):
    pose: List[List[List[float]]]   # (T, J, 3)


class PipelineResponse(BaseModel):
    text: str
    gloss: str
    pose: List[List[List[float]]]


# ── helpers ───────────────────────────────────────────────────────────────────
def _pose_to_list(pose: np.ndarray) -> List[List[List[float]]]:
    """Convert (T, J, 3) numpy array → nested Python list for JSON."""
    return pose.tolist()


# ── endpoints ─────────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {
        "status": "ok",
        "gloss_model_loaded": _gloss_model is not None and _gloss_model._loaded,
        "pose_model_loaded": _pose_model is not None and _pose_model._loaded,
    }


@app.post("/speech", response_model=SpeechResponse)
async def speech_endpoint(audio: UploadFile = File(...)):
    """Accept an audio file and return the Whisper transcript."""
    if not audio.content_type or not (
        audio.content_type.startswith("audio/")
        or audio.content_type == "application/octet-stream"
    ):
        raise HTTPException(
            status_code=400,
            detail=f"Expected audio file, got: {audio.content_type}",
        )

    text = await transcribe_upload(audio)
    return SpeechResponse(text=text)


@app.post("/gloss", response_model=GlossResponse)
async def gloss_endpoint(text: str = Form(...)):
    """Convert English text to ISL gloss."""
    if not text.strip():
        raise HTTPException(status_code=400, detail="Empty text.")
    gloss = _gloss_model.predict(text)
    return GlossResponse(gloss=gloss)


@app.post("/pose", response_model=PoseResponse)
async def pose_endpoint(gloss: str = Form(...)):
    """Convert ISL gloss to a pose keypoint sequence."""
    if not gloss.strip():
        raise HTTPException(status_code=400, detail="Empty gloss.")
    pose = _pose_model.predict(gloss)
    return PoseResponse(pose=_pose_to_list(pose))


@app.post("/pipeline", response_model=PipelineResponse)
async def pipeline_endpoint(audio: UploadFile = File(...)):
    """
    Full pipeline: audio file → text → gloss → pose.
    Returns all intermediate results for display in the Flutter app.
    """
    # Step 1 — speech to text
    text = await transcribe_upload(audio)
    if not text:
        raise HTTPException(status_code=422, detail="Could not transcribe audio.")

    # Step 2 — text to gloss
    gloss = _gloss_model.predict(text)

    # Step 3 — gloss to pose
    pose = _pose_model.predict(gloss)

    return PipelineResponse(
        text=text,
        gloss=gloss,
        pose=_pose_to_list(pose),
    )
