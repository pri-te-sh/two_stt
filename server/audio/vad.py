from collections import deque
import webrtcvad
import numpy as np

class RingBuffer:
    def __init__(self, max_seconds: float, sample_rate: int):
        self.max_len = int(max_seconds * sample_rate)
        self.buf = deque(maxlen=self.max_len)
        self.sample_rate = sample_rate

    def extend_pcm16(self, pcm16_bytes: bytes):
        # little-endian int16 -> float32 in [-1,1]
        arr = np.frombuffer(pcm16_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        self.extend(arr)

    def extend(self, samples: np.ndarray):
        self.buf.extend(samples.tolist())

    def to_numpy(self) -> np.ndarray:
        if not self.buf:
            return np.zeros(0, dtype=np.float32)
        return np.array(self.buf, dtype=np.float32)

class VadGate:
    def __init__(self, sample_rate: int, frame_ms: int, aggressiveness: int,
                 final_silence_ms_min: int, final_silence_ms_max: int):
        self.vad = webrtcvad.Vad(aggressiveness)
        self.sr = sample_rate
        self.frame_size = int(sample_rate * frame_ms / 1000)
        self.final_min = final_silence_ms_min
        self.final_max = final_silence_ms_max
        self.silence_run = 0
        self.in_speech = False

    def _frame_speech(self, pcm16_bytes: bytes) -> bool:
        # WebRTC-VAD expects 10/20/30ms PCM16 frames
        return self.vad.is_speech(pcm16_bytes, self.sr)

    def update_and_check_finalize(self, pcm16_bytes: bytes) -> bool:
        # Split incoming bytes into frame-sized chunks
        finalize = False
        for i in range(0, len(pcm16_bytes), self.frame_size * 2):
            frame = pcm16_bytes[i:i + self.frame_size * 2]
            if len(frame) < self.frame_size * 2:
                break
            is_speech = self._frame_speech(frame)
            if is_speech:
                self.in_speech = True
                self.silence_run = 0
            else:
                self.silence_run += int(1000 * (len(frame) / 2) / self.sr)
                if self.in_speech and self.final_min <= self.silence_run <= self.final_max:
                    finalize = True
                    self.in_speech = False
        return finalize