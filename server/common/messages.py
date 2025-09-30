from __future__ import annotations
from pydantic import BaseModel, Field
from typing import Optional

class StartMsg(BaseModel):
    op: str = Field("start", const=True)
    sample_rate: int = 16000
    lang: str = "auto"

class AudioMsg(BaseModel):
    op: str = Field("audio", const=True)
    payload: str  # base64 PCM16 little-endian mono

class StopMsg(BaseModel):
    op: str = Field("stop", const=True)

class InterimOut(BaseModel):
    type: str = Field("interim", const=True)
    text: str
    stable_chars: int

class FinalOut(BaseModel):
    type: str = Field("final", const=True)
    text: str

class StatusOut(BaseModel):
    type: str = Field("status", const=True)
    message: str