from __future__ import annotations
import base64
import asyncio
import time
import logging
import json
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

log = logging.getLogger("server")

app = FastAPI(title="MVP STT Server")
scheduler = Scheduler()

@app.on_event("startup")
async def _startup():
    import os
    logging.basicConfig(level=os.getenv("PY_LOG_LEVEL", "INFO"))
    await scheduler.start()

@app.on_event("shutdown")
async def _shutdown():
    await scheduler.stop()

@app.get("/ready")
async def ready():
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
        cond = (not self.prev_interim_text) or \
               (abs(len(text) - len(self.prev_interim_text)) >= 6) or \
               (now - self.last_interim_emit_ms >= settings.INTERIM_MIN_MS)
        if not cond:
            return
        # Stabilized prefix length
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
        self.prev_interim_text = ""
        self.last_interim_emit_ms = 0.0
        self.interim_inflight = False

@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    await ws.send_json(StatusOut(message="connection open").model_dump())

    session: Optional[Session] = None

    async def on_decode_done(text: str, kind: str):
        if session is None:
            return
        if kind == "interim":
            session.interim_inflight = False
            await session.maybe_emit_interim(text)
        else:
            await session.emit_final(text)
        log.debug("emit %s len=%d", kind, len(text))

    try:
        while True:
            raw = await ws.receive_text()
            try:
                obj = json.loads(raw)
            except Exception:
                await ws.send_json(StatusOut(message="json parse error").model_dump())
                continue

            op = obj.get("op")

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

                # Opportunistic interim (coalesced)
                if not session.interim_inflight:
                    session.interim_inflight = True
                    await scheduler.submit(audio=session.ring.to_numpy(), lang=session.lang,
                                           kind="interim", cb=on_decode_done)

                if finalize:
                    log.debug("finalize triggered")
                    await scheduler.submit(audio=session.ring.to_numpy(), lang=session.lang,
                                           kind="final", cb=on_decode_done)
                    # reset VAD & ring for next utterance
                    session.vad.reset()
                    session.ring = RingBuffer(settings.MAX_RING_SECONDS, session.ring.sample_rate)

            elif op == "stop":
                if session is None:
                    await ws.send_json(StatusOut(message="not started").model_dump())
                    continue
                await scheduler.submit(audio=session.ring.to_numpy(), lang=session.lang,
                                       kind="final", cb=on_decode_done)
                await ws.send_json(StatusOut(message="stream stopped").model_dump())

            else:
                await ws.send_json(StatusOut(message=f"unknown op: {op}").model_dump())

    except WebSocketDisconnect:
        return