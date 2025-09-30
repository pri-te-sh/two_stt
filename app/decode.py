from __future__ import annotations
from io import BytesIO
from typing import Optional, Tuple
import numpy as np
import soundfile as sf
import librosa

from faster_whisper import WhisperModel


SAMPLE_RATE = 16000


def load_audio_from_bytes(data: bytes, target_sr: int = SAMPLE_RATE) -> np.ndarray:
    """
    Load arbitrary audio bytes (wav/m4a/mp3/ogg/flac, etc.) and return mono float32 @ target_sr.
    """
    with BytesIO(data) as bio:
        audio, sr = sf.read(bio, dtype="float32", always_2d=False)
    if audio.ndim == 2:
        audio = np.mean(audio, axis=1)
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    return audio.astype(np.float32, copy=False)


def _common_opts(language: Optional[str], task: str) -> dict:
    return {
        "language": language,
        "task": task,
        "vad_filter": False,  # MVP: explicitly off
        "beam_size": 1,
        "temperature": 0.0,
        "without_timestamps": True,
        "condition_on_previous_text": False,
    }


def transcribe_interim(model: WhisperModel, audio: np.ndarray, language: Optional[str], task: str) -> Tuple[str, int]:
    """Fast, low-cost pass for interims."""
    opts = _common_opts(language, task)
    segments, info = model.transcribe(audio, **opts)
    text = "".join(seg.text for seg in segments).strip()
    return text, int(info.duration * 1000) if hasattr(info, "duration") else 0


def transcribe_final(model: WhisperModel, audio: np.ndarray, language: Optional[str], task: str) -> Tuple[str, int]:
    """Higher-quality pass for finals."""
    opts = _common_opts(language, task)
    opts.update({
        "beam_size": 5,
        "temperature": 0.0,
        "best_of": 5,
        "patience": 1.0,
        "compression_ratio_threshold": 2.4,
        "log_prob_threshold": -1.0,
        "no_speech_threshold": 0.6,
        "condition_on_previous_text": True,
        "initial_prompt": None,
    })
    segments, info = model.transcribe(audio, **opts)
    text = "".join(seg.text for seg in segments).strip()
    return text, int(info.duration * 1000) if hasattr(info, "duration") else 0
