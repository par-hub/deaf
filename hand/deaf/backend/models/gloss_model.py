"""
Text → ISL Gloss model wrapper.

The model receives an English sentence and returns a sequence of ISL gloss
tokens (space-separated). The saved checkpoint is expected to be a full
torch.save(model, path) export so that torch.load returns the nn.Module
directly. If your checkpoint only saves state_dict, define the model class
below under GlossModelArchitecture and set USE_STATE_DICT = True.
"""

import os
import torch
import torch.nn as nn
from typing import List

# ── config ───────────────────────────────────────────────────────────────────
USE_STATE_DICT = False          # flip to True if you saved only state_dict
DEVICE = torch.device("cpu")   # change to "cuda" if GPU is available

# Path is resolved relative to this file so the backend can run from any cwd
_HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(_HERE, "..", "model_weights", "best_gloss_model.pt")


# ── (optional) architecture ───────────────────────────────────────────────────
# Fill this in if your checkpoint is a state_dict (USE_STATE_DICT = True).
class GlossModelArchitecture(nn.Module):
    """Placeholder – replace with your actual model class."""
    def __init__(self):
        super().__init__()

    def forward(self, x):
        raise NotImplementedError("Provide the real architecture.")


# ── vocab helper ──────────────────────────────────────────────────────────────
# Many seq2seq models need a token→id map. If your model uses its own,
# replace this with an actual load from a saved vocab file.
_FALLBACK_VOCAB: List[str] = []


def _build_simple_gloss(text: str) -> str:
    """
    Minimal rule-based fallback used when the model cannot be loaded.
    Produces approximate ISL gloss by uppercasing content words and
    dropping function words (determiners, auxiliary verbs, etc.).
    """
    stopwords = {"is", "are", "am", "was", "were", "the", "a", "an",
                 "to", "of", "in", "on", "at", "for", "and", "or",
                 "but", "with", "from", "that", "this", "it", "be"}
    tokens = text.lower().split()
    gloss_tokens = [t.upper() for t in tokens if t not in stopwords and t.isalpha()]
    return " ".join(gloss_tokens) if gloss_tokens else text.upper()


# ── loader ────────────────────────────────────────────────────────────────────
class GlossModel:
    def __init__(self):
        self._model = None
        self._loaded = False
        self._load()

    def _load(self):
        if not os.path.exists(MODEL_PATH):
            print(f"[GlossModel] WARNING: model not found at {MODEL_PATH}. "
                  "Using rule-based fallback.")
            return

        try:
            if USE_STATE_DICT:
                arch = GlossModelArchitecture()
                state = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
                arch.load_state_dict(state)
                arch.eval()
                self._model = arch
            else:
                self._model = torch.load(
                    MODEL_PATH, map_location=DEVICE, weights_only=False
                )
                if hasattr(self._model, "eval"):
                    self._model.eval()

            self._loaded = True
            print("[GlossModel] Loaded successfully.")
        except Exception as exc:
            print(f"[GlossModel] Load failed ({exc}). Using rule-based fallback.")

    @torch.no_grad()
    def predict(self, text: str) -> str:
        """Return an ISL gloss string for the given English text."""
        if not self._loaded or self._model is None:
            return _build_simple_gloss(text)

        try:
            # ── adapt this section to your model's actual interface ──────────
            # Example for a HuggingFace-style generate() model:
            #   inputs = self._tokenizer(text, return_tensors="pt").to(DEVICE)
            #   out = self._model.generate(**inputs)
            #   return self._tokenizer.decode(out[0], skip_special_tokens=True)
            #
            # Example for a custom LSTM seq2seq model:
            #   tokens = [self._vocab.get(w, 0) for w in text.lower().split()]
            #   tensor = torch.tensor([tokens]).to(DEVICE)
            #   output = self._model(tensor)
            #   return " ".join([self._inv_vocab[i] for i in output.argmax(-1)[0]])

            result = self._model(text)  # adjust to your model's interface
            if isinstance(result, torch.Tensor):
                result = result.tolist()
            return str(result)
        except Exception as exc:
            print(f"[GlossModel] Inference error: {exc}. Falling back.")
            return _build_simple_gloss(text)
