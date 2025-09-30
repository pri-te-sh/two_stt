# ──────────────────────────────────────────────────────────────────────────────
# File: app/asr/config.py
# Stage-specific decode configuration
# ──────────────────────────────────────────────────────────────────────────────
from __future__ import annotations
from dataclasses import dataclass, asdict


@dataclass
class DecodeParams:
    beam_size: int
    temperature: float | None = 0.0
    temperature_fallback: tuple[float, ...] = ()
    without_timestamps: bool = True
    condition_on_previous_text: bool = False
    vad_filter: bool = False
    compression_ratio_threshold: float = 2.6
    logprob_threshold: float = -1.2


class ASRConfig:
    INTERIM = DecodeParams(
        beam_size=1,
        temperature=0.0,
        without_timestamps=True,
        condition_on_previous_text=False,
        vad_filter=False,
        compression_ratio_threshold=2.6,
        logprob_threshold=-1.2,
    )
    FINAL = DecodeParams(
        beam_size=5,
        temperature=None,
        temperature_fallback=(0.0, 0.2, 0.4),
        without_timestamps=False,
        condition_on_previous_text=True,
        vad_filter=True,
        compression_ratio_threshold=2.4,
        logprob_threshold=-0.9,
    )

    @staticmethod
    def snapshot():
        return {
            "interim": asdict(ASRConfig.INTERIM),
            "final": asdict(ASRConfig.FINAL),
        }

