"""Message schemas for WebSocket protocol."""
from typing import Literal, Optional, List, Dict, Any
from pydantic import BaseModel, Field
from enum import Enum


class EventType(str, Enum):
    """Client event types."""
    START = "start"
    STOP = "stop"
    SET = "set"


class MessageType(str, Enum):
    """Server message types."""
    INTERIM = "interim"
    FINAL = "final"
    STATUS = "status"
    ERROR = "error"


class BackpressureLevel(str, Enum):
    """Backpressure levels."""
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


# Client -> Server messages
class StartEvent(BaseModel):
    """Client start event."""
    event: Literal[EventType.START]
    language: str = Field(default="auto")


class StopEvent(BaseModel):
    """Client stop event."""
    event: Literal[EventType.STOP]


class SetEvent(BaseModel):
    """Client configuration update."""
    event: Literal[EventType.SET]
    interim_rate: Optional[int] = None
    tail_seconds: Optional[float] = None


# Server -> Client messages
class InterimMessage(BaseModel):
    """Interim transcription result."""
    type: Literal[MessageType.INTERIM] = MessageType.INTERIM
    conn: str
    text: str
    stable_chars: int
    t0: float
    t1: float


class Segment(BaseModel):
    """Transcription segment with timestamps."""
    start: float
    end: float
    text: str


class FinalMessage(BaseModel):
    """Final transcription result."""
    type: Literal[MessageType.FINAL] = MessageType.FINAL
    conn: str
    text: str
    segments: List[Segment] = Field(default_factory=list)
    language: Optional[str] = None
    t0: float
    t1: float


class StatusMessage(BaseModel):
    """Status update message."""
    type: Literal[MessageType.STATUS] = MessageType.STATUS
    backpressure: BackpressureLevel
    cooldown_ms: int
    tail_s: float
    interim_paused: bool = False


class ErrorMessage(BaseModel):
    """Error message."""
    type: Literal[MessageType.ERROR] = MessageType.ERROR
    code: str
    detail: str


# Internal job representation
class JobType(str, Enum):
    """Job types for scheduler."""
    INTERIM = "interim"
    FINAL = "final"


class DecodeJob(BaseModel):
    """Decode job for scheduler."""
    job_id: str
    job_type: JobType
    conn_id: str
    audio_data: bytes
    language: str
    created_at: float
    t0: float  # Audio start timestamp
    t1: float  # Audio end timestamp
    
    class Config:
        arbitrary_types_allowed = True
