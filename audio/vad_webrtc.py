"""WebRTC VAD wrapper for voice activity detection."""
import numpy as np
import webrtcvad
from typing import Optional
from util.logging import get_logger
from util.config import config

logger = get_logger(__name__)


class VADProcessor:
    """Voice Activity Detection processor using WebRTC VAD."""
    
    def __init__(self, sample_rate: int = 16000, mode: int = 2):
        """
        Initialize VAD processor.
        
        Args:
            sample_rate: Audio sample rate (8000, 16000, 32000, or 48000)
            mode: Aggressiveness mode (0-3). Higher = more aggressive filtering
        """
        self.sample_rate = sample_rate
        self.mode = mode
        self.vad = webrtcvad.Vad(mode)
        
        # Frame duration for VAD (10, 20, or 30 ms)
        self.frame_duration_ms = 30
        self.frame_length = int(sample_rate * self.frame_duration_ms / 1000)
        
        logger.info(f"Initialized VAD: mode={mode}, sample_rate={sample_rate}, frame_ms={self.frame_duration_ms}")
    
    def is_speech(self, audio: np.ndarray) -> bool:
        """
        Check if audio frame contains speech.
        
        Args:
            audio: Audio samples as int16 numpy array
            
        Returns:
            True if speech detected, False otherwise
        """
        # Ensure audio is int16
        if audio.dtype != np.int16:
            audio = (audio * 32767).astype(np.int16)
        
        # VAD requires exact frame length
        if len(audio) != self.frame_length:
            # Pad or truncate to frame length
            if len(audio) < self.frame_length:
                audio = np.pad(audio, (0, self.frame_length - len(audio)), mode='constant')
            else:
                audio = audio[:self.frame_length]
        
        # Convert to bytes
        audio_bytes = audio.tobytes()
        
        try:
            return self.vad.is_speech(audio_bytes, self.sample_rate)
        except Exception as e:
            logger.warning(f"VAD processing failed: {e}")
            return False
    
    def detect_silence(self, audio: np.ndarray, chunk_size_ms: int = 30) -> bool:
        """
        Detect if audio segment is silence by checking multiple frames.
        
        Args:
            audio: Audio samples as numpy array
            chunk_size_ms: Size of chunks to process
            
        Returns:
            True if segment is silence (no speech detected)
        """
        if len(audio) == 0:
            return True
        
        # Ensure int16
        if audio.dtype != np.int16:
            audio = (audio * 32767).astype(np.int16)
        
        # Process in frames
        chunk_samples = int(self.sample_rate * chunk_size_ms / 1000)
        speech_frames = 0
        total_frames = 0
        
        for i in range(0, len(audio), chunk_samples):
            frame = audio[i:i + chunk_samples]
            if len(frame) < chunk_samples:
                # Pad last frame
                frame = np.pad(frame, (0, chunk_samples - len(frame)), mode='constant')
            
            if self.is_speech(frame):
                speech_frames += 1
            total_frames += 1
        
        # Consider silence if < 20% frames have speech
        if total_frames == 0:
            return True
        
        speech_ratio = speech_frames / total_frames
        return speech_ratio < 0.2


class SilenceTracker:
    """Track continuous silence duration for utterance segmentation."""
    
    def __init__(self, sample_rate: int = 16000, end_silence_ms: int = 500):
        """
        Initialize silence tracker.
        
        Args:
            sample_rate: Audio sample rate
            end_silence_ms: Silence duration to trigger utterance end
        """
        self.sample_rate = sample_rate
        self.end_silence_samples = int(sample_rate * end_silence_ms / 1000)
        self.silence_samples = 0
        self.speech_started = False
        
        logger.info(f"Initialized silence tracker: end_silence_ms={end_silence_ms}")
    
    def update(self, is_speech: bool, num_samples: int):
        """
        Update silence tracker with new audio segment.
        
        Args:
            is_speech: Whether segment contains speech
            num_samples: Number of audio samples in segment
        """
        if is_speech:
            self.silence_samples = 0
            self.speech_started = True
        elif self.speech_started:
            self.silence_samples += num_samples
    
    def should_finalize(self) -> bool:
        """Check if utterance should be finalized based on silence duration."""
        return self.speech_started and self.silence_samples >= self.end_silence_samples
    
    def reset(self):
        """Reset silence tracker for new utterance."""
        self.silence_samples = 0
        self.speech_started = False
