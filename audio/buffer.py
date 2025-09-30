# ──────────────────────────────────────────────────────────────────────────────
# File: app/audio/buffer.py
# PCM16 ring buffer with slice helpers
# ──────────────────────────────────────────────────────────────────────────────
from __future__ import annotations
from collections import deque
import numpy as np


class PCM16RingBuffer:
    def __init__(self, sample_rate: int, max_seconds: int = 30):
        self.sample_rate = sample_rate
        self.max_samples = sample_rate * max_seconds
        self.chunks: deque[np.ndarray] = deque()
        self.chunk_samples: int = sample_rate  # 1s per internal chunk for trimming
        self.current_sample_index = 0  # absolute sample count appended so far

    def append_bytes(self, raw: bytes) -> int:
        # raw is PCM16 mono little-endian
        arr = np.frombuffer(raw, dtype=np.int16)
        self.current_sample_index += arr.size
        # Break into ~1s chunks for trimming
        pos = 0
        while pos < arr.size:
            take = min(self.chunk_samples, arr.size - pos)
            self.chunks.append(arr[pos:pos+take].copy())
            pos += take
        self._trim()
        return arr.size

    def _trim(self):
        # Keep only the last max_samples
        total = sum(len(c) for c in self.chunks)
        while total > self.max_samples and self.chunks:
            drop = self.chunks.popleft()
            total -= len(drop)

    def _collect_last(self, samples: int) -> np.ndarray | None:
        if samples <= 0:
            return None
        out = []
        need = samples
        for c in reversed(self.chunks):
            if need <= 0:
                break
            if len(c) <= need:
                out.append(c)
                need -= len(c)
            else:
                out.append(c[-need:])
                need = 0
        if not out:
            return None
        arr = np.concatenate(list(reversed(out)))
        return arr

    def tail_seconds(self, seconds: float) -> np.ndarray | None:
        samples = int(seconds * self.sample_rate)
        pcm = self._collect_last(samples)
        if pcm is None:
            return None
        return pcm.astype(np.float32) / 32768.0

    def get_since(self, start_abs_sample: int) -> np.ndarray | None:
        # Returns audio from absolute start sample to current end (float32)
        # We approximate by taking last N where N = current - start.
        cur = self.current_sample_index
        if start_abs_sample >= cur:
            return None
        need = cur - start_abs_sample
        pcm = self._collect_last(need)
        if pcm is None:
            return None
        return pcm.astype(np.float32) / 32768.0

