from __future__ import annotations
import base64
import asyncio
import time
from typing import Optional

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

from server.config import settings
from server.audio.vad import RingBuffer, VadGate
from server.asr.scheduler import Scheduler
from server.common.messages import (
    StartMsg, AudioMsg, StopMsg, InterimOut, FinalOut, StatusOut
)

app = FastAPI(title="MVP STT Server")
scheduler = Scheduler()

@app.on_event("startup")
async def _startup():
    await scheduler.start()

@app.on_event("shutdown")
async def _shutdown():
    await scheduler.stop()

@app.get("/ready")
async def ready():
    # Touch the model once to load lazily
    from server.asr.engine import get_model
    get_model()
    return JSONResponse({"status":"ok"})

@app.get("/status")
async def status():
    return JSONResponse({
        "status": "ok",
        "model": settings.MODEL_NAME,
        "device": settings.DEVICE,
        "compute_type": settings.COMPUTE_TYPE,
    })

class Session:
    def __init__(self, ws: WebSocket, sample_rate: int, lang: str):
        self.ws = ws
        self.lang = lang
        self.ring = RingBuffer(settings.MAX_RING_SECONDS, sample_rate)
        self.vad = VadGate(sample_rate, settings.VAD_FRAME_MS, settings.VAD_AGGRESSIVENESS,
                           settings.FINAL_SILENCE_MS_MIN, settings.FINAL_SILENCE_MS_MAX)
        self.last_interim_emit_ms = 0.0
        self.prev_interim_text = ""
        self.interim_inflight = False

    async def maybe_emit_interim(self, text: str):
        now = time.monotonic() * 1000
        if not text:
            return
        # Gate: first, delta length >=6, or >= INTERIM_MIN_MS since last emit
        cond = (not self.prev_interim_text) or \
               (abs(len(text) - len(self.prev_interim_text)) >= 6) or \
               (now - self.last_interim_emit_ms >= settings.INTERIM_MIN_MS)
        if not cond:
            return
        # Stabilization: common prefix length
        stable = 0
        for a, b in zip(text, self.prev_interim_text):
            if a == b:
                stable += 1
            else:
                break
        await self.ws.send_json(InterimOut(text=text, stable_chars=stable).model_dump())
        self.prev_interim_text = text
        self.last_interim_emit_ms = now

    async def emit_final(self, text: str):
        if text:
            await self.ws.send_json(FinalOut(text=text).model_dump())
        # reset interim state after a final
        self.prev_interim_text = ""
        self.last_interim_emit_ms = 0.0
        self.interim_inflight = False

@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    await ws.send_json(StatusOut(message="connection open").model_dump())

    session: Optional[Session] = None

    async def on_decode_done(text: str, kind: str):
        # Called from scheduler worker thread via event loop
        if kind == "interim":
            session.interim_inflight = False
            await session.maybe_emit_interim(text)
        else:
            await session.emit_final(text)

    try:
        while True:
            raw = await ws.receive_text()
            # Parse by op
            try:
                if raw.strip().startswith("{"):
                    import json
                    obj = json.loads(raw)
                    op = obj.get("op")
                else:
                    await ws.send_json(StatusOut(message="invalid message").model_dump())
                    continue
            except Exception:
                await ws.send_json(StatusOut(message="json parse error").model_dump())
                continue

            if op == "start":
                msg = StartMsg(**obj)
                if session is None:
                    session = Session(ws, msg.sample_rate, msg.lang)
                    await ws.send_json(StatusOut(message="stream started").model_dump())
                else:
                    await ws.send_json(StatusOut(message="already started").model_dump())

            elif op == "audio":
                if session is None:
                    await ws.send_json(StatusOut(message="send start first").model_dump())
                    continue
                msg = AudioMsg(**obj)
                pcm = base64.b64decode(msg.payload)
                finalize = session.vad.update_and_check_finalize(pcm)
                session.ring.extend_pcm16(pcm)

                # Opportunistic interim (coalesced: only if no in-flight)
                if not session.interim_inflight:
                    session.interim_inflight = True
                    await scheduler.submit(audio=session.ring.to_numpy(), lang=session.lang,
                                           kind="interim", cb=lambda t,k: asyncio.create_task(on_decode_done(t,k)))

                if finalize:
                    # Final job on the buffered audio
                    await scheduler.submit(audio=session.ring.to_numpy(), lang=session.lang,
                                           kind="final", cb=lambda t,k: asyncio.create_task(on_decode_done(t,k)))
                    # After scheduling final, clear ring for next utterance
                    session.ring = RingBuffer(settings.MAX_RING_SECONDS, settings.SAMPLE_RATE)

            elif op == "stop":
                if session is None:
                    await ws.send_json(StatusOut(message="not started").model_dump())
                    continue
                # Force a final on remaining audio
                await scheduler.submit(audio=session.ring.to_numpy(), lang=session.lang,
                                       kind="final", cb=lambda t,k: asyncio.create_task(on_decode_done(t,k)))
                await ws.send_json(StatusOut(message="stream stopped").model_dump())

            else:
                await ws.send_json(StatusOut(message=f"unknown op: {op}").model_dump())

    except WebSocketDisconnect:
        return