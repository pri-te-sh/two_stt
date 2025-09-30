# ──────────────────────────────────────────────────────────────────────────────
# File: app/scheduler/priority.py
# Priority scheduler, coalescing interim queue, decode workers
# ──────────────────────────────────────────────────────────────────────────────
from __future__ import annotations
import asyncio
import time
from dataclasses import dataclass
from typing import Dict, Optional
import numpy as np

from asr.models import ModelHandles
from asr.decode import decode_interim, decode_final
from runtime.state import GlobalRuntime


@dataclass
class Job:
    kind: str  # 'final' or 'interim'
    conn_id: str
    audio: np.ndarray
    language: str
    ts: float


class CoalescingInterimQueue:
    def __init__(self):
        self._by_conn: Dict[str, Job] = {}

    def put(self, job: Job):
        # Replace existing job for the same connection
        self._by_conn[job.conn_id] = job

    def pop_oldest(self) -> Optional[Job]:
        if not self._by_conn:
            return None
        # pick oldest by ts
        cid, job = min(self._by_conn.items(), key=lambda kv: kv[1].ts)
        del self._by_conn[cid]
        return job

    def __len__(self):
        return len(self._by_conn)


class PriorityScheduler:
    def __init__(self, models: ModelHandles, runtime: GlobalRuntime):
        self.models = models
        self.runtime = runtime
        self.q_final: asyncio.Queue[Job] = asyncio.Queue()
        self.q_interim = CoalescingInterimQueue()
        self._task = None
        self.running = False
        self._tick_ms = 12
        self._final_burst = 2
        self._interim_burst = 3
        # dynamic backpressure knobs
        self._global_interim_cooldown_ms = 0
        self._stop_evt = asyncio.Event()
        # per-model locks to avoid concurrent decode on same handle
        self._lock_final = asyncio.Lock()
        self._lock_interim = asyncio.Lock()

    async def start(self):
        self.running = True
        self._stop_evt.clear()
        self._task = asyncio.create_task(self._run())

    async def stop(self):
        self.running = False
        self._stop_evt.set()
        if self._task:
            await self._task

    def enqueue_final(self, conn_id: str, audio: np.ndarray, language: str):
        self.q_final.put_nowait(Job("final", conn_id, audio, language, time.time()))

    def enqueue_interim(self, conn_id: str, audio: np.ndarray, language: str):
        self.q_interim.put(Job("interim", conn_id, audio, language, time.time()))

    def dynamic_tail_seconds(self) -> float:
        # Shrink tail as interim backlog grows
        q = len(self.q_interim)
        if q >= 40:
            return 2.0
        if q >= 20:
            return 3.5
        return 7.0

    def global_interim_cooldown_ms(self) -> int:
        # Increase cooldown as interim backlog grows
        q = len(self.q_interim)
        if q >= 40:
            return 1000
        if q >= 20:
            return 350
        return 0

    async def _run(self):
        try:
            while not self._stop_evt.is_set():
                # Finals first
                f_served = 0
                while f_served < self._final_burst and not self.q_final.empty():
                    job = await self.q_final.get()
                    await self._serve_final(job)
                    f_served += 1

                # Interims (skip if finals backlog is high)
                i_served = 0
                if self.q_final.qsize() == 0:
                    while i_served < self._interim_burst and len(self.q_interim) > 0:
                        job = self.q_interim.pop_oldest()
                        if job:
                            await self._serve_interim(job)
                            i_served += 1

                await asyncio.sleep(self._tick_ms / 1000.0)
        except Exception as e:
            # In production, log this
            self.running = False

    async def _serve_interim(self, job: Job):
        state = self.runtime.connection(job.conn_id)
        if state is None:
            return
        async with self._lock_interim:
            text = decode_interim(self.models.interim, job.audio, job.language)
        # Stabilization: only send if changed meaningfully
        if _should_emit_interim(prev=state.last_interim_text, new=text, now=time.time(), last_sent_ms=state.last_emit_ts_ms):
            state.last_interim_text = text
            await state.outgoing.put({
                "type": "interim",
                "conn": job.conn_id,
                "text": text,
                "t0": None,
                "t1": None,
            })

    async def _serve_final(self, job: Job):
        state = self.runtime.connection(job.conn_id)
        if state is None:
            return
        async with self._lock_final:
            result = decode_final(self.models.final, job.audio, job.language)
        # Reset interim state for next utterance
        state.last_interim_text = ""
        state.phase = "idle"
        await state.outgoing.put({
            "type": "final",
            "conn": job.conn_id,
            **result,
        })


def _should_emit_interim(prev: str, new: str, now: float, last_sent_ms: int) -> bool:
    if not new:
        return False
    if not prev:
        return True
    if new == prev:
        return False
    # simple growth or edit distance proxy
    if abs(len(new) - len(prev)) >= 6:
        return True
    # time-based throttle backup (>= 0.35s)
    return (now * 1000 - last_sent_ms) >= 350

