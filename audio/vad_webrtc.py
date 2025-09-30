# ──────────────────────────────────────────────────────────────────────────────
# File: app/audio/vad_webrtc.py
# Streaming VAD wrapper using webrtcvad (CPU)
# ──────────────────────────────────────────────────────────────────────────────
from __future__ import annotations
import webrtcvad


class StreamingVAD:
    """Frame-by-frame VAD with simple start/stop detection.

    - Accepts PCM16 mono 16kHz bytes via process_bytes
    - Uses 20 ms frames
    - Signals just_started/just_ended flags for consumers
    """

    def __init__(self, sample_rate: int, aggressiveness: int = 2,
                 start_trigger_ms: int = 60, end_trigger_ms: int = 500):
        assert sample_rate == 16000, "WebRTC VAD requires 16kHz mono PCM16"
        self.v = webrtcvad.Vad(aggressiveness)
        self.sample_rate = sample_rate
        self.frame_bytes = int(0.02 * sample_rate) * 2  # 20ms * 2 bytes
        self.speaking = False
        self.speech_ms = 0
        self.silence_ms = 0
        self.just_started = False
        self.just_ended = False
        self._start_trigger_ms = start_trigger_ms
        self._end_trigger_ms = end_trigger_ms
        self._remainder = b""

    def process_bytes(self, raw: bytes):
        self.just_started = False
        self.just_ended = False
        data = self._remainder + raw
        pos = 0
        while pos + self.frame_bytes <= len(data):
            frame = data[pos:pos + self.frame_bytes]
            pos += self.frame_bytes
            is_speech = self.v.is_speech(frame, self.sample_rate)
            if is_speech:
                self.speech_ms += 20
                self.silence_ms = 0
                if not self.speaking and self.speech_ms >= self._start_trigger_ms:
                    self.speaking = True
                    self.just_started = True
            else:
                self.silence_ms += 20
                # decay speech counter a bit
                self.speech_ms = max(0, self.speech_ms - 10)
                if self.speaking and self.silence_ms >= self._end_trigger_ms:
                    self.speaking = False
                    self.just_ended = True
        self._remainder = data[pos:]
