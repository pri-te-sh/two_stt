# ──────────────────────────────────────────────────────────────────────────────
# File: app/asr/models.py
# Model loading and warmup
# ──────────────────────────────────────────────────────────────────────────────
from __future__ import annotations
import os
from dataclasses import dataclass
from faster_whisper import WhisperModel


@dataclass
class ModelHandles:
    interim: WhisperModel
    final: WhisperModel


def load_models() -> ModelHandles:
    interim_name = os.getenv("INTERIM_MODEL", "small")
    final_name = os.getenv("FINAL_MODEL", "distil-large-v3")
    interim_comp = os.getenv("INTERIM_COMPUTE", "int8_float16")
    final_comp = os.getenv("FINAL_COMPUTE", "float16")

    interim = WhisperModel(interim_name, device="cuda", compute_type=interim_comp)
    final = WhisperModel(final_name, device="cuda", compute_type=final_comp)

    # Simple warmup: 1 second of silence
    import numpy as np
    silent = np.zeros(16000, dtype=np.float32)
    list(interim.transcribe(silent, beam_size=1, language="en", without_timestamps=True))
    list(final.transcribe(silent, beam_size=1, language="en", without_timestamps=True))

    return ModelHandles(interim=interim, final=final)

