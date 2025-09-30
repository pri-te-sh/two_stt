import asyncio
import logging
from dataclasses import dataclass, field
from typing import Callable
import numpy as np
from server.asr.engine import decode
from server.config import settings

log = logging.getLogger(__name__)

PRIORITY_FINAL = 0
PRIORITY_INTERIM = 1

@dataclass(order=True)
class Job:
    priority: int
    seq: int
    audio: np.ndarray = field(compare=False)
    lang: str = field(compare=False)
    kind: str = field(compare=False)  # "interim" | "final"
    on_done: Callable[[str, str], None] = field(compare=False)

class Scheduler:
    def __init__(self):
        self.q: asyncio.PriorityQueue[Job] = asyncio.PriorityQueue()
        self._seq = 0
        self._workers = []

    async def start(self):
        for _ in range(settings.NUM_WORKERS):
            self._workers.append(asyncio.create_task(self._worker()))

    async def stop(self):
        for w in self._workers:
            w.cancel()
        await asyncio.gather(*self._workers, return_exceptions=True)

    async def submit(self, *, audio: np.ndarray, lang: str, kind: str, cb: Callable[[str, str], None]):
        self._seq += 1
        prio = PRIORITY_FINAL if kind == "final" else PRIORITY_INTERIM
        job = Job(prio, self._seq, audio, lang, kind, cb)
        await self.q.put(job)
        log.debug("enqueued job #%d kind=%s len=%.2fs", self._seq, kind, len(audio)/settings.SAMPLE_RATE)

    async def _worker(self):
        loop = asyncio.get_running_loop()
        while True:
            job = await self.q.get()
            try:
                text = await loop.run_in_executor(None, decode, job.audio, job.lang, job.kind)
                try:
                    job.on_done(text, job.kind)
                except Exception as e:
                    log.exception("on_done error: %s", e)
            except Exception as e:
                log.exception("decode error for job #%d kind=%s: %s", job.seq, job.kind, e)
            finally:
                self.q.task_done()
```python
import asyncio
from dataclasses import dataclass, field
from typing import Callable
import numpy as np
from server.asr.engine import decode
from server.config import settings

PRIORITY_FINAL = 0
PRIORITY_INTERIM = 1

@dataclass(order=True)
class Job:
    priority: int
    seq: int
    audio: np.ndarray = field(compare=False)
    lang: str = field(compare=False)
    kind: str = field(compare=False)  # "interim" | "final"
    on_done: Callable[[str, str], None] = field(compare=False)

class Scheduler:
    def __init__(self):
        self.q: asyncio.PriorityQueue[Job] = asyncio.PriorityQueue()
        self._seq = 0
        self._workers = []

    async def start(self):
        for _ in range(settings.NUM_WORKERS):
            self._workers.append(asyncio.create_task(self._worker()))

    async def stop(self):
        for w in self._workers:
            w.cancel()
        await asyncio.gather(*self._workers, return_exceptions=True)

    async def submit(self, *, audio: np.ndarray, lang: str, kind: str, cb: Callable[[str, str], None]):
        self._seq += 1
        prio = PRIORITY_FINAL if kind == "final" else PRIORITY_INTERIM
        job = Job(prio, self._seq, audio, lang, kind, cb)
        await self.q.put(job)

    async def _worker(self):
        loop = asyncio.get_running_loop()
        while True:
            job = await self.q.get()
            try:
                text = await loop.run_in_executor(None, decode, job.audio, job.lang, job.kind)
                job.on_done(text, job.kind)
            finally:
                self.q.task_done()