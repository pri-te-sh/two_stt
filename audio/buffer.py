"""Ring buffer for audio storage."""
import numpy as np
from typing import Optional
from collections import deque
from util.logging import get_logger

logger = get_logger(__name__)


class AudioRingBuffer:
    """
    Ring buffer for storing audio samples with fixed maximum duration.
    Supports efficient tail window extraction for interim transcriptions.
    """
    
    def __init__(self, sample_rate: int = 16000, max_duration_seconds: int = 30):
        """
        Initialize audio ring buffer.
        
        Args:
            sample_rate: Audio sample rate
            max_duration_seconds: Maximum buffer duration in seconds
        """
        self.sample_rate = sample_rate
        self.max_samples = sample_rate * max_duration_seconds
        self.buffer = np.zeros(self.max_samples, dtype=np.int16)
        self.write_pos = 0
        self.total_samples_written = 0
        self.utterance_start_pos = 0  # Track start of current utterance
        
        logger.debug(f"Initialized ring buffer: max_duration={max_duration_seconds}s, max_samples={self.max_samples}")
    
    def append(self, audio: np.ndarray):
        """
        Append audio samples to ring buffer.
        
        Args:
            audio: Audio samples as numpy array (will be converted to int16)
        """
        # Convert to int16 if needed
        if audio.dtype != np.int16:
            audio = (audio * 32767).astype(np.int16)
        
        num_samples = len(audio)
        
        # Write samples to buffer (wrapping around if needed)
        space_to_end = self.max_samples - self.write_pos
        
        if num_samples <= space_to_end:
            # Fits without wrapping
            self.buffer[self.write_pos:self.write_pos + num_samples] = audio
        else:
            # Needs to wrap around
            self.buffer[self.write_pos:] = audio[:space_to_end]
            remaining = num_samples - space_to_end
            self.buffer[:remaining] = audio[space_to_end:]
        
        # Update positions
        self.write_pos = (self.write_pos + num_samples) % self.max_samples
        self.total_samples_written += num_samples
    
    def get_tail(self, duration_seconds: float) -> np.ndarray:
        """
        Get the most recent N seconds of audio.
        
        Args:
            duration_seconds: Duration of tail to retrieve
            
        Returns:
            Audio samples as numpy array
        """
        num_samples = int(self.sample_rate * duration_seconds)
        num_samples = min(num_samples, self.total_samples_written, self.max_samples)
        
        if num_samples == 0:
            return np.array([], dtype=np.int16)
        
        # Calculate start position
        start_pos = (self.write_pos - num_samples) % self.max_samples
        
        if start_pos < self.write_pos:
            # Contiguous region
            return self.buffer[start_pos:self.write_pos].copy()
        else:
            # Wrapped around
            return np.concatenate([
                self.buffer[start_pos:],
                self.buffer[:self.write_pos]
            ])
    
    def get_utterance(self) -> np.ndarray:
        """
        Get audio from start of current utterance to present.
        
        Returns:
            Audio samples as numpy array
        """
        if self.utterance_start_pos <= self.write_pos:
            # Contiguous region
            return self.buffer[self.utterance_start_pos:self.write_pos].copy()
        else:
            # Wrapped around
            return np.concatenate([
                self.buffer[self.utterance_start_pos:],
                self.buffer[:self.write_pos]
            ])
    
    def mark_utterance_start(self):
        """Mark current position as start of new utterance."""
        self.utterance_start_pos = self.write_pos
        logger.debug(f"Marked utterance start at position {self.write_pos}")
    
    def get_duration_seconds(self) -> float:
        """Get total duration of audio in buffer."""
        num_samples = min(self.total_samples_written, self.max_samples)
        return num_samples / self.sample_rate
    
    def get_utterance_duration_seconds(self) -> float:
        """Get duration of current utterance."""
        if self.utterance_start_pos <= self.write_pos:
            num_samples = self.write_pos - self.utterance_start_pos
        else:
            num_samples = (self.max_samples - self.utterance_start_pos) + self.write_pos
        return num_samples / self.sample_rate
    
    def clear(self):
        """Clear buffer and reset positions."""
        self.buffer.fill(0)
        self.write_pos = 0
        self.total_samples_written = 0
        self.utterance_start_pos = 0
        logger.debug("Cleared ring buffer")


class ConnectionAudioState:
    """
    Manages audio state for a single WebSocket connection.
    Includes ring buffer, VAD state, and utterance tracking.
    """
    
    def __init__(
        self,
        conn_id: str,
        sample_rate: int = 16000,
        max_duration_seconds: int = 30
    ):
        """
        Initialize connection audio state.
        
        Args:
            conn_id: Connection identifier
            sample_rate: Audio sample rate
            max_duration_seconds: Maximum ring buffer duration
        """
        self.conn_id = conn_id
        self.sample_rate = sample_rate
        self.ring_buffer = AudioRingBuffer(sample_rate, max_duration_seconds)
        
        # Utterance tracking
        self.in_utterance = False
        self.utterance_id = 0
        
        # Timing
        self.last_audio_time = 0.0
        self.utterance_start_time = 0.0
        
        # Interim tracking
        self.last_interim_text = ""
        self.last_interim_time = 0.0
        self.interim_queued_or_inflight = False
        
        # Language
        self.detected_language: Optional[str] = None
        self.language_confidence: float = 0.0
        
        logger.info(f"Initialized audio state for connection {conn_id}")
    
    def start_utterance(self, timestamp: float):
        """Mark start of new utterance."""
        self.in_utterance = True
        self.utterance_id += 1
        self.utterance_start_time = timestamp
        self.ring_buffer.mark_utterance_start()
        self.last_interim_text = ""
        logger.debug(f"Started utterance {self.utterance_id} for {self.conn_id}")
    
    def end_utterance(self):
        """Mark end of utterance."""
        self.in_utterance = False
        self.interim_queued_or_inflight = False
        logger.debug(f"Ended utterance {self.utterance_id} for {self.conn_id}")
