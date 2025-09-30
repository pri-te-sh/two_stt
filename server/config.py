from pydantic import BaseModel
import os

class Settings(BaseModel):
    MODEL_NAME: str = os.getenv("MODEL_NAME", "base")
    DEVICE: str = os.getenv("DEVICE", "auto")
    COMPUTE_TYPE: str = os.getenv("COMPUTE_TYPE", "float16")
    SAMPLE_RATE: int = int(os.getenv("SAMPLE_RATE", "16000"))

    # interim / final heuristics
    INTERIM_MIN_MS: int = int(os.getenv("INTERIM_MIN_MS", "350"))
    VAD_FRAME_MS: int = int(os.getenv("VAD_FRAME_MS", "20"))
    VAD_AGGRESSIVENESS: int = int(os.getenv("VAD_AGGRESSIVENESS", "2"))
    FINAL_SILENCE_MS_MIN: int = int(os.getenv("FINAL_SILENCE_MS_MIN", "400"))
    FINAL_SILENCE_MS_MAX: int = int(os.getenv("FINAL_SILENCE_MS_MAX", "700"))
    MAX_RING_SECONDS: float = float(os.getenv("MAX_RING_SECONDS", "25.0"))

    # workers
    NUM_WORKERS: int = int(os.getenv("NUM_WORKERS", "1"))  # MVP: single worker

settings = Settings()