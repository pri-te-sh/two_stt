from __future__ import annotations
from pydantic import BaseModel
from typing import Optional, Literal

class StartMsg(BaseModel):
    op: Literal["start"] = "start"
    sample_rate: int = 16000
    lang: str = "auto"

class AudioMsg(BaseModel):
    op: Literal["audio"] = "audio"
    payload: str  # base64 PCM16 little-endian mono

class StopMsg(BaseModel):
    op: Literal["stop"] = "stop"

class InterimOut(BaseModel):
    type: Literal["interim"] = "interim"
    text: str
    stable_chars: int

class FinalOut(BaseModel):
    type: Literal["final"] = "final"
    text: str

class StatusOut(BaseModel):
    type: Literal["status"] = "status"
    message: str