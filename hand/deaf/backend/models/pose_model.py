"""
ISL Gloss → Pose keypoints (MediaPipe Holistic – 75 joints).

Output shape: (T, 75, 3)
  joints  0-32  : body (MediaPipe Pose 33-pt)
  joints 33-53  : left  hand (MediaPipe Hand 21-pt)
  joints 54-74  : right hand (MediaPipe Hand 21-pt)

All coordinates are normalised to [0, 1].
"""

import os
import math
import torch
import torch.nn as nn
import numpy as np
from typing import List

# ── config ────────────────────────────────────────────────────────────────────
USE_STATE_DICT = False
DEVICE = torch.device("cpu")

_HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(_HERE, "..", "model_weights", "best_model.pth")

NUM_JOINTS   = 75
HAND_SCALE   = 0.088   # hand size in normalised image space
FPS          = 15
FRAMES_WORD  = 20      # frames per gloss word


# ── (optional) architecture ───────────────────────────────────────────────────
class PoseModelArchitecture(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        raise NotImplementedError


# ── Neutral 33-pt body skeleton ───────────────────────────────────────────────
_BODY = np.array([
    [0.500,0.140,0.000],[0.520,0.120,0.010],[0.535,0.115,0.010],
    [0.550,0.118,0.010],[0.480,0.120,0.010],[0.465,0.115,0.010],
    [0.450,0.118,0.010],[0.565,0.130,-0.02],[0.435,0.130,-0.02],
    [0.520,0.160,0.015],[0.480,0.160,0.015],
    [0.580,0.280,0.000],[0.420,0.280,0.000],   # shoulders 11,12
    [0.630,0.420,0.000],[0.370,0.420,0.000],   # elbows    13,14
    [0.660,0.555,0.000],[0.340,0.555,0.000],   # wrists    15,16
    [0.675,0.585,0.000],[0.325,0.585,0.000],   # pinky knuckle
    [0.670,0.592,0.005],[0.330,0.592,0.005],   # index knuckle
    [0.650,0.578,0.010],[0.350,0.578,0.010],   # thumb knuckle
    [0.550,0.590,0.000],[0.450,0.590,0.000],   # hips      23,24
    [0.560,0.750,0.000],[0.440,0.750,0.000],   # knees
    [0.555,0.910,0.000],[0.445,0.910,0.000],   # ankles
    [0.548,0.930,-0.02],[0.452,0.930,-0.02],   # heels
    [0.570,0.945,0.010],[0.430,0.945,0.010],   # feet
], dtype=np.float32)


# ── Hand shape presets ────────────────────────────────────────────────────────
# Offsets for the RIGHT hand (thumb toward +x).
# Mirror by negating x for the LEFT hand.
# Units: multiply by HAND_SCALE to get normalised image offsets.

def _shape(data):
    return np.array(data, dtype=np.float32)

# All five fingers fully extended and spread
_SH_OPEN = _shape([
    [ 0.000,  0.000, 0.00],  # 0  WRIST
    [ 0.310, -0.210, 0.03],  # 1  THUMB_CMC
    [ 0.540, -0.420, 0.08],  # 2  THUMB_MCP
    [ 0.720, -0.600, 0.13],  # 3  THUMB_IP
    [ 0.860, -0.750, 0.17],  # 4  THUMB_TIP
    [ 0.230, -0.540, 0.00],  # 5  INDEX_MCP
    [ 0.260, -0.810, 0.06],  # 6  INDEX_PIP
    [ 0.260, -1.020, 0.10],  # 7  INDEX_DIP
    [ 0.260, -1.190, 0.13],  # 8  INDEX_TIP
    [ 0.000, -0.570, 0.00],  # 9  MIDDLE_MCP
    [ 0.000, -0.860, 0.06],  # 10 MIDDLE_PIP
    [ 0.000, -1.090, 0.10],  # 11 MIDDLE_DIP
    [ 0.000, -1.260, 0.13],  # 12 MIDDLE_TIP
    [-0.220, -0.540, 0.00],  # 13 RING_MCP
    [-0.240, -0.810, 0.06],  # 14 RING_PIP
    [-0.240, -1.020, 0.10],  # 15 RING_DIP
    [-0.240, -1.190, 0.13],  # 16 RING_TIP
    [-0.400, -0.460, 0.00],  # 17 PINKY_MCP
    [-0.435, -0.690, 0.04],  # 18 PINKY_PIP
    [-0.435, -0.870, 0.07],  # 19 PINKY_DIP
    [-0.435, -1.010, 0.10],  # 20 PINKY_TIP
])

# Index finger only – pointing gesture
_SH_POINT = _shape([
    [ 0.000,  0.000, 0.00],
    [ 0.290, -0.200, 0.03],  # THUMB_CMC
    [ 0.460, -0.340, 0.08],  # THUMB_MCP (partially curled)
    [ 0.490, -0.240, 0.20],  # THUMB_IP
    [ 0.460, -0.170, 0.28],  # THUMB_TIP
    [ 0.230, -0.540, 0.00],  # INDEX_MCP
    [ 0.260, -0.810, 0.06],  # INDEX_PIP  (EXTENDED)
    [ 0.260, -1.020, 0.10],  # INDEX_DIP
    [ 0.260, -1.190, 0.13],  # INDEX_TIP
    [ 0.000, -0.570, 0.00],  # MIDDLE_MCP
    [-0.020, -0.490, 0.18],  # MIDDLE_PIP (curled)
    [-0.020, -0.410, 0.30],  # MIDDLE_DIP
    [-0.015, -0.370, 0.35],  # MIDDLE_TIP
    [-0.220, -0.540, 0.00],  # RING_MCP
    [-0.235, -0.465, 0.18],  # RING_PIP   (curled)
    [-0.235, -0.395, 0.30],  # RING_DIP
    [-0.230, -0.355, 0.35],  # RING_TIP
    [-0.400, -0.460, 0.00],  # PINKY_MCP
    [-0.415, -0.395, 0.16],  # PINKY_PIP  (curled)
    [-0.415, -0.340, 0.25],  # PINKY_DIP
    [-0.415, -0.305, 0.30],  # PINKY_TIP
])

# Closed fist
_SH_FIST = _shape([
    [ 0.000,  0.000, 0.00],
    [ 0.200, -0.140, 0.08],
    [ 0.300, -0.140, 0.18],  # THUMB over fingers
    [ 0.250, -0.215, 0.28],
    [ 0.175, -0.270, 0.30],
    [ 0.150, -0.490, 0.00],  # INDEX_MCP
    [ 0.075, -0.415, 0.24],  # curled
    [ 0.025, -0.355, 0.37],
    [ 0.000, -0.320, 0.42],
    [ 0.000, -0.510, 0.00],  # MIDDLE_MCP
    [-0.020, -0.435, 0.24],
    [-0.035, -0.370, 0.37],
    [-0.035, -0.330, 0.42],
    [-0.145, -0.490, 0.00],  # RING_MCP
    [-0.175, -0.415, 0.24],
    [-0.190, -0.355, 0.37],
    [-0.190, -0.315, 0.42],
    [-0.275, -0.435, 0.00],  # PINKY_MCP
    [-0.295, -0.370, 0.20],
    [-0.295, -0.315, 0.30],
    [-0.295, -0.275, 0.35],
])

# V-sign (index + middle extended)
_SH_V = _shape([
    [ 0.000,  0.000, 0.00],
    [ 0.290, -0.200, 0.03],
    [ 0.460, -0.340, 0.08],
    [ 0.490, -0.240, 0.20],
    [ 0.460, -0.170, 0.28],
    [ 0.280, -0.530, 0.00],  # INDEX_MCP (spread right)
    [ 0.320, -0.800, 0.06],  # EXTENDED
    [ 0.320, -1.010, 0.10],
    [ 0.320, -1.180, 0.13],
    [-0.050, -0.550, 0.00],  # MIDDLE_MCP (spread left)
    [-0.085, -0.820, 0.06],  # EXTENDED
    [-0.085, -1.040, 0.10],
    [-0.085, -1.210, 0.13],
    [-0.220, -0.540, 0.00],  # RING_MCP
    [-0.235, -0.465, 0.18],  # curled
    [-0.235, -0.395, 0.30],
    [-0.230, -0.355, 0.35],
    [-0.400, -0.460, 0.00],  # PINKY_MCP
    [-0.415, -0.395, 0.16],  # curled
    [-0.415, -0.340, 0.25],
    [-0.415, -0.305, 0.30],
])

# Thumbs-up
_SH_THUMB = _shape([
    [ 0.000,  0.000, 0.00],
    [ 0.310, -0.220, 0.03],  # THUMB_CMC
    [ 0.540, -0.440, 0.08],  # THUMB_MCP  (EXTENDED)
    [ 0.710, -0.630, 0.13],
    [ 0.840, -0.780, 0.17],
    [ 0.120, -0.480, 0.00],  # INDEX_MCP
    [ 0.050, -0.405, 0.24],  # curled
    [ 0.000, -0.345, 0.37],
    [-0.010, -0.305, 0.42],
    [ 0.000, -0.510, 0.00],  # MIDDLE_MCP
    [-0.020, -0.435, 0.24],
    [-0.035, -0.370, 0.37],
    [-0.035, -0.330, 0.42],
    [-0.120, -0.485, 0.00],  # RING_MCP
    [-0.145, -0.410, 0.24],
    [-0.155, -0.350, 0.37],
    [-0.155, -0.310, 0.42],
    [-0.250, -0.430, 0.00],  # PINKY_MCP
    [-0.265, -0.365, 0.20],
    [-0.265, -0.310, 0.30],
    [-0.265, -0.270, 0.35],
])

# Flat B-hand (fingers together, extended)
_SH_FLAT = _shape([
    [ 0.000,  0.000, 0.00],
    [ 0.200, -0.180, 0.03],
    [ 0.340, -0.330, 0.07],
    [ 0.430, -0.490, 0.10],
    [ 0.490, -0.640, 0.13],
    [ 0.130, -0.550, 0.00],
    [ 0.130, -0.820, 0.05],
    [ 0.130, -1.030, 0.09],
    [ 0.130, -1.190, 0.12],
    [ 0.000, -0.570, 0.00],
    [ 0.000, -0.850, 0.05],
    [ 0.000, -1.080, 0.09],
    [ 0.000, -1.240, 0.12],
    [-0.130, -0.550, 0.00],
    [-0.130, -0.820, 0.05],
    [-0.130, -1.030, 0.09],
    [-0.130, -1.190, 0.12],
    [-0.240, -0.470, 0.00],
    [-0.250, -0.700, 0.04],
    [-0.250, -0.880, 0.07],
    [-0.250, -1.020, 0.10],
])

_SHAPES = [_SH_OPEN, _SH_POINT, _SH_FIST, _SH_V, _SH_THUMB, _SH_FLAT]

# ISL-word → shape index mapping (heuristic)
_WORD_SHAPES = {
    "I": 1, "ME": 1, "YOU": 1, "HE": 1, "SHE": 1, "WE": 1, "THEY": 1,
    "YES": 4, "GOOD": 4, "OK": 4, "GREAT": 4,
    "NO": 2, "NOT": 2, "STOP": 2, "NEVER": 2,
    "WIN": 3, "TWO": 3, "PEACE": 3, "V": 3,
    "HELLO": 0, "HI": 0, "BYE": 0, "COME": 0, "GO": 0,
    "WILL": 5, "CAN": 5, "WANT": 5, "NEED": 5,
}

# Sign location presets: (left_wrist_xy, right_wrist_xy, left_elbow_xy, right_elbow_xy)
# Values are override positions for body joints during signing
_LOCS = {
    "neutral": dict(lw=[0.660,0.555], rw=[0.340,0.555],
                    le=[0.630,0.420], re=[0.370,0.420]),
    "chest":   dict(lw=[0.570,0.370], rw=[0.430,0.370],
                    le=[0.580,0.440], re=[0.420,0.440]),
    "chin":    dict(lw=[0.540,0.270], rw=[0.460,0.270],
                    le=[0.575,0.390], re=[0.425,0.390]),
    "fore":    dict(lw=[0.540,0.200], rw=[0.460,0.200],
                    le=[0.572,0.330], re=[0.428,0.330]),
    "side":    dict(lw=[0.680,0.380], rw=[0.320,0.380],
                    le=[0.645,0.430], re=[0.355,0.430]),
}

_WORD_LOCS = {
    "HELLO": "fore", "HI": "fore", "BYE": "fore",
    "I": "chest", "ME": "chest", "YOU": "chest",
    "YES": "chest", "GOOD": "chest", "NO": "chin",
    "WIN": "side", "TWO": "side", "GO": "side",
}


def _apply_shape(wrist_xy, shape_offsets, mirror=False):
    """Return 21×3 absolute positions for one hand."""
    offs = shape_offsets.copy()
    if mirror:
        offs[:, 0] = -offs[:, 0]   # flip x for left hand
    result = np.zeros((21, 3), dtype=np.float32)
    result[:, 0] = wrist_xy[0] + offs[:, 0] * HAND_SCALE
    result[:, 1] = wrist_xy[1] + offs[:, 1] * HAND_SCALE
    result[:, 2] = offs[:, 2] * 0.5
    return result


def _smoothstep(t):
    t = np.clip(t, 0, 1)
    return t * t * (3 - 2 * t)


def _generate_fallback_sequence(gloss: str) -> np.ndarray:
    words = [w for w in gloss.strip().upper().split() if w] or ["NEUTRAL"]

    # Neutral full-body frame (75 joints)
    def _neutral_frame():
        frame = np.zeros((75, 3), dtype=np.float32)
        frame[:33] = _BODY.copy()
        loc = _LOCS["neutral"]
        # left hand
        lw = loc["lw"]
        frame[33:54] = _apply_shape(lw, _SH_OPEN, mirror=True)
        frame[33, 0], frame[33, 1] = lw[0], lw[1]
        # right hand
        rw = loc["rw"]
        frame[54:75] = _apply_shape(rw, _SH_OPEN, mirror=False)
        frame[54, 0], frame[54, 1] = rw[0], rw[1]
        return frame

    all_frames: List[np.ndarray] = []

    for word in words:
        shape_idx = _WORD_SHAPES.get(word, abs(hash(word)) % len(_SHAPES))
        shape     = _SHAPES[shape_idx]
        loc_key   = _WORD_LOCS.get(word, "chest")
        loc       = _LOCS[loc_key]

        for f in range(FRAMES_WORD):
            t = f / max(FRAMES_WORD - 1, 1)
            # ease in, hold, ease out
            if t < 0.30:
                alpha = _smoothstep(t / 0.30)
            elif t < 0.70:
                alpha = 1.0
            else:
                alpha = _smoothstep(1.0 - (t - 0.70) / 0.30)

            frame = np.zeros((75, 3), dtype=np.float32)

            # Body: interpolate wrist/elbow positions
            base = _BODY.copy()
            n_loc = _LOCS["neutral"]

            for ji, key, base_key in [
                (15, "lw", "lw"), (16, "rw", "rw"),
                (13, "le", "le"), (14, "re", "re"),
            ]:
                bxy = np.array(n_loc[base_key], dtype=np.float32)
                txy = np.array(loc[key], dtype=np.float32)
                base[ji, :2] = bxy + (txy - bxy) * alpha

            frame[:33] = base

            # Hands
            lw = base[15, :2]
            rw = base[16, :2]

            neutral_lh = _apply_shape(_LOCS["neutral"]["lw"], _SH_OPEN, mirror=True)
            neutral_rh = _apply_shape(_LOCS["neutral"]["rw"], _SH_OPEN, mirror=False)

            target_lh = _apply_shape(loc["lw"], shape, mirror=True)
            target_rh = _apply_shape(loc["rw"], shape, mirror=False)

            lh = neutral_lh + (target_lh - neutral_lh) * alpha
            rh = neutral_rh + (target_rh - neutral_rh) * alpha

            # Anchor wrist to body wrist
            lh += np.array([lw[0] - lh[0, 0], lw[1] - lh[0, 1], 0], dtype=np.float32)
            rh += np.array([rw[0] - rh[0, 0], rw[1] - rh[0, 1], 0], dtype=np.float32)

            frame[33:54] = lh
            frame[54:75] = rh

            all_frames.append(frame)

    return np.array(all_frames, dtype=np.float32)  # (T, 75, 3)


# ── loader ────────────────────────────────────────────────────────────────────
class PoseModel:
    def __init__(self):
        self._model = None
        self._loaded = False
        self._load()

    def _load(self):
        if not os.path.exists(MODEL_PATH):
            print(f"[PoseModel] Not found at {MODEL_PATH}. Using procedural fallback.")
            return
        try:
            if USE_STATE_DICT:
                arch = PoseModelArchitecture()
                arch.load_state_dict(
                    torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False))
                arch.eval()
                self._model = arch
            else:
                self._model = torch.load(MODEL_PATH, map_location=DEVICE,
                                         weights_only=False)
                if hasattr(self._model, "eval"):
                    self._model.eval()
            self._loaded = True
            print("[PoseModel] Loaded successfully.")
        except Exception as e:
            print(f"[PoseModel] Load failed ({e}). Using fallback.")

    @torch.no_grad()
    def predict(self, gloss: str) -> np.ndarray:
        """Return (T, 75, 3) array, normalised [0,1]."""
        if not self._loaded or self._model is None:
            return _generate_fallback_sequence(gloss)
        try:
            result = self._model(gloss)
            if isinstance(result, torch.Tensor):
                result = result.cpu().numpy()
            result = np.array(result, dtype=np.float32)
            if result.ndim == 2:
                t = result.shape[0]
                result = result.reshape(t, -1, 3)
            return result
        except Exception as e:
            print(f"[PoseModel] Inference error ({e}). Falling back.")
            return _generate_fallback_sequence(gloss)
