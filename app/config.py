from __future__ import annotations
import os
from dataclasses import dataclass


def _getenv(name: str, default: str) -> str:
    v = os.getenv(name)
    return v if v is not None and v != "" else default


@dataclass(frozen=True)
class Settings:
    # Model IDs accepted by faster-whisper (e.g., "tiny", "base", "small", "medium", "large-v3", or a local path)
    INTERIM_MODEL: str = _getenv("INTERIM_MODEL", "small")
    FINAL_MODEL: str = _getenv("FINAL_MODEL", "large-v3")

    # Device: "cuda" if you have GPU; otherwise "cpu"
    DEVICE: str = _getenv("DEVICE", "cuda")

    # Compute type: for GPU typically "float16"; for low VRAM interims you can try "int8_float16"
    INTERIM_COMPUTE_TYPE: str = _getenv("INTERIM_COMPUTE_TYPE", "int8_float16")
    FINAL_COMPUTE_TYPE: str = _getenv("FINAL_COMPUTE_TYPE", "float16")

    # WebSocket interim emit cooldown (ms)
    WS_INTERIM_COOLDOWN_MS: int = int(_getenv("WS_INTERIM_COOLDOWN_MS", "350"))

    # Max seconds of audio to attempt in WS interims (keeps interims cheap). 0 = no limit
    WS_INTERIM_MAX_SECONDS: float = float(_getenv("WS_INTERIM_MAX_SECONDS", "12"))

    # Language and task defaults
    LANGUAGE: str | None = None if _getenv("LANGUAGE", "auto") == "auto" else _getenv("LANGUAGE", None)  # "en" or None
    TASK: str = _getenv("TASK", "transcribe")  # or "translate"


SETTINGS = Settings()