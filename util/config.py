"""Configuration module for STT server."""
import os
from typing import Optional
from pydantic import BaseModel, Field, validator
from dotenv import load_dotenv

load_dotenv()


class Config(BaseModel):
    """Server configuration."""
    
    # Environment
    env: str = Field(default="dev")
    
    # Server
    bind_host: str = Field(default="0.0.0.0")
    bind_port: int = Field(default=8080)
    
    # Models
    interim_model: str = Field(default="small")
    final_model: str = Field(default="distil-large-v3")
    interim_compute: str = Field(default="int8_float16")
    final_compute: str = Field(default="float16")
    
    # Language
    asr_language: str = Field(default="auto")
    
    # Timing
    interim_cooldown_ms: int = Field(default=220)
    tail_seconds: float = Field(default=7.0)
    scheduler_tick_ms: int = Field(default=12)
    f_final_burst: int = Field(default=2)
    f_interim_burst: int = Field(default=3)
    
    # Watermarks
    final_hi: int = Field(default=6)
    final_crit: int = Field(default=12)
    interim_hi: int = Field(default=20)
    interim_crit: int = Field(default=40)
    
    # VAD
    vad_mode: int = Field(default=2)
    vad_end_silence_ms: int = Field(default=500)
    
    # Audio
    sample_rate: int = Field(default=16000)
    ring_buffer_seconds: int = Field(default=30)
    
    # Auth
    jwt_public_key_path: Optional[str] = Field(default=None)
    require_auth: bool = Field(default=False)
    
    # Logging
    log_level: str = Field(default="INFO")
    log_format: str = Field(default="json")
    
    # Observability
    enable_metrics: bool = Field(default=True)
    metrics_port: int = Field(default=9090)
    
    @validator("interim_cooldown_ms")
    def validate_cooldown(cls, v):
        if v < 50 or v > 1000:
            raise ValueError("interim_cooldown_ms must be between 50 and 1000")
        return v
    
    @validator("tail_seconds")
    def validate_tail(cls, v):
        if v < 1.0 or v > 30.0:
            raise ValueError("tail_seconds must be between 1.0 and 30.0")
        return v
    
    @classmethod
    def from_env(cls) -> "Config":
        """Load configuration from environment variables."""
        return cls(
            env=os.getenv("ENV", "dev"),
            bind_host=os.getenv("BIND_HOST", "127.0.0.1"),
            bind_port=int(os.getenv("BIND_PORT", "8081")),
            interim_model=os.getenv("INTERIM_MODEL", "small"),
            final_model=os.getenv("FINAL_MODEL", "distil-large-v3"),
            interim_compute=os.getenv("INTERIM_COMPUTE", "int8_float16"),
            final_compute=os.getenv("FINAL_COMPUTE", "float16"),
            asr_language=os.getenv("ASR_LANGUAGE", "auto"),
            interim_cooldown_ms=int(os.getenv("INTERIM_COOLDOWN_MS", "220")),
            tail_seconds=float(os.getenv("TAIL_SECONDS", "7.0")),
            scheduler_tick_ms=int(os.getenv("SCHEDULER_TICK_MS", "12")),
            f_final_burst=int(os.getenv("F_FINAL_BURST", "2")),
            f_interim_burst=int(os.getenv("F_INTERIM_BURST", "3")),
            final_hi=int(os.getenv("FINAL_HI", "6")),
            final_crit=int(os.getenv("FINAL_CRIT", "12")),
            interim_hi=int(os.getenv("INTERIM_HI", "20")),
            interim_crit=int(os.getenv("INTERIM_CRIT", "40")),
            vad_mode=int(os.getenv("VAD_MODE", "2")),
            vad_end_silence_ms=int(os.getenv("VAD_END_SILENCE_MS", "500")),
            jwt_public_key_path=os.getenv("JWT_PUBLIC_KEY_PATH"),
            require_auth=os.getenv("REQUIRE_AUTH", "false").lower() == "true",
            log_level=os.getenv("LOG_LEVEL", "INFO"),
            log_format=os.getenv("LOG_FORMAT", "json"),
            enable_metrics=os.getenv("ENABLE_METRICS", "true").lower() == "true",
            metrics_port=int(os.getenv("METRICS_PORT", "9090")),
        )


# Global config instance
config = Config.from_env()
