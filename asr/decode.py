# ──────────────────────────────────────────────────────────────────────────────
# File: app/asr/decode.py
# Decode helpers for interim and final passes
# ──────────────────────────────────────────────────────────────────────────────
from __future__ import annotations
from typing import Dict, Any
import numpy as np
from faster_whisper import WhisperModel
from asr.config import ASRConfig, DecodeParams


def _params_dict(p: DecodeParams, language: str) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = dict(
        beam_size=p.beam_size,
        without_timestamps=p.without_timestamps,
        vad_filter=p.vad_filter,
        condition_on_previous_text=p.condition_on_previous_text,
        compression_ratio_threshold=p.compression_ratio_threshold,
        logprob_threshold=p.logprob_threshold,
        language=language,
    )
    if p.temperature is not None:
        kwargs["temperature"] = p.temperature
    if p.temperature_fallback:
        kwargs["temperature_fallback"] = p.temperature_fallback
    return kwargs


def decode_interim(model: WhisperModel, audio_f32: np.ndarray, language: str) -> str:
    params = _params_dict(ASRConfig.INTERIM, language)
    segments, _ = model.transcribe(audio_f32, **params)
    txt_parts = [seg.text for seg in segments]
    return " ".join(part.strip() for part in txt_parts).strip()


def decode_final(model: WhisperModel, audio_f32: np.ndarray, language: str) -> Dict[str, Any]:
    params = _params_dict(ASRConfig.FINAL, language)
    segments, info = model.transcribe(audio_f32, **params)
    out_segments = []
    for s in segments:
        out_segments.append({
            "text": s.text,
            "start": s.start,
            "end": s.end,
            "avg_logprob": s.avg_logprob,
            "no_speech_prob": s.no_speech_prob,
            "words": [
                {"word": w.word, "start": w.start, "end": w.end, "prob": w.prob}
                for w in (s.words or [])
            ],
        })
    return {
        "text": " ".join(seg["text"].strip() for seg in out_segments).strip(),
        "segments": out_segments,
        "language": info.language,
        "language_probability": info.language_probability,
    }

