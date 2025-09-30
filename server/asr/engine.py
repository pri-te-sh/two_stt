from __future__ import annotations
import threading
import numpy as np
from faster_whisper import WhisperModel
from server.config import settings

_model_lock = threading.Lock()
_model = None


def get_model() -> WhisperModel:
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                _model = WhisperModel(settings.MODEL_NAME,
                                      device=settings.DEVICE,
                                      compute_type=settings.COMPUTE_TYPE)
    return _model


def decode(audio_f32_mono: np.ndarray, lang: str, kind: str) -> str:
    """Decode using Faster-Whisper; different knobs for interim vs final."""
    model = get_model()
    if audio_f32_mono.ndim != 1:
        audio_f32_mono = audio_f32_mono.reshape(-1)

    if kind == "interim":
        beam = 1
        temperature = 0.0
        cond_prev = False
        ts = False
    else:  # final
        beam = 3
        temperature = [0.0, 0.2, 0.4]
        cond_prev = True
        ts = True

    segments, _ = model.transcribe(
        audio_f32_mono,
        language=None if lang == "auto" else lang,
        beam_size=beam,
        temperature=temperature,
        vad_filter=False,
        condition_on_previous_text=cond_prev,
        word_timestamps=ts,
    )
    text = "".join(seg.text for seg in segments).strip()
    return text
```python
from __future__ import annotations
import threading
import numpy as np
from faster_whisper import WhisperModel
from server.config import settings

_model_lock = threading.Lock()
_model = None


def get_model() -> WhisperModel:
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                _model = WhisperModel(settings.MODEL_NAME,
                                      device=settings.DEVICE,
                                      compute_type=settings.COMPUTE_TYPE)
    return _model


def decode(audio_f32_mono: np.ndarray, lang: str, *, kind: str) -> str:
    """Decode using Faster-Whisper; different knobs for interim vs final."""
    model = get_model()
    if audio_f32_mono.ndim != 1:
        audio_f32_mono = audio_f32_mono.reshape(-1)

    if kind == "interim":
        beam = 1
        temperature = 0.0
        cond_prev = False
        ts = False
    else:  # final
        beam = 3
        temperature = [0.0, 0.2, 0.4]
        cond_prev = True
        ts = True

    segments, _ = model.transcribe(
        audio_f32_mono,
        language=None if lang == "auto" else lang,
        beam_size=beam,
        temperature=temperature,
        vad_filter=False,
        condition_on_previous_text=cond_prev,
        word_timestamps=ts,
    )
    text = "".join(seg.text for seg in segments).strip()
    return text