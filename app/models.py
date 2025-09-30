from __future__ import annotations
import threading
from faster_whisper import WhisperModel
from .config import SETTINGS


class _ModelBundle:
    __slots__ = ("interim", "final", "interim_lock", "final_lock")

    def __init__(self) -> None:
        # Load both models once at startup
        self.interim = WhisperModel(
            SETTINGS.INTERIM_MODEL,
            device=SETTINGS.DEVICE,
            compute_type=SETTINGS.INTERIM_COMPUTE_TYPE,
        )
        self.final = WhisperModel(
            SETTINGS.FINAL_MODEL,
            device=SETTINGS.DEVICE,
            compute_type=SETTINGS.FINAL_COMPUTE_TYPE,
        )
        # Locks to avoid over-scheduling the same model concurrently in WS mode
        self.interim_lock = threading.Lock()
        self.final_lock = threading.Lock()


_MODELS: _ModelBundle | None = None
_MODELS_LOCK = threading.Lock()


def get_models() -> _ModelBundle:
    global _MODELS
    if _MODELS is None:
        with _MODELS_LOCK:
            if _MODELS is None:
                _MODELS = _ModelBundle()
    return _MODELS


def model_info() -> dict:
    s = SETTINGS
    return {
        "interim_model": s.INTERIM_MODEL,
        "final_model": s.FINAL_MODEL,
        "device": s.DEVICE,
        "interim_compute_type": s.INTERIM_COMPUTE_TYPE,
        "final_compute_type": s.FINAL_COMPUTE_TYPE,
    }
