# ──────────────────────────────────────────────────────────────────────────────
# File: app/server/app.py
# Core FastAPI app with WebSocket, model warmup, scheduler startup
# ──────────────────────────────────────────────────────────────────────────────
from __future__ import annotations
import asyncio
import json
import os
import time
import uuid
from typing import Dict, Optional

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

from server.routes import router as http_router
from asr.models import load_models, ModelHandles
from scheduler.priority import PriorityScheduler
from runtime.state import ConnectionState, GlobalRuntime
from audio.vad_webrtc import StreamingVAD
from audio.buffer import PCM16RingBuffer

app = FastAPI(title="Hybrid Realtime STT Service", version="0.1.0")
app.include_router(http_router)

MODELS: Optional[ModelHandles] = None
SCHED: Optional[PriorityScheduler] = None
RUNTIME: Optional[GlobalRuntime] = None

# Defaults (tweak via env or .env in prod)
INTERIM_COOLDOWN_MS = int(os.getenv("INTERIM_COOLDOWN_MS", "220"))
TAIL_SECONDS_NORMAL = float(os.getenv("TAIL_SECONDS", "7"))
TAIL_SECONDS_HIGH = 3.0
TAIL_SECONDS_CRIT = 2.0
SAMPLE_RATE = 16000
MAX_BUFFER_SECONDS = 30


@app.on_event("startup")
async def _startup():
    global MODELS, SCHED, RUNTIME
    # Load models once (shared in-process)
    MODELS = load_models()
    # Init global runtime (metrics, registries)
    RUNTIME = GlobalRuntime(sample_rate=SAMPLE_RATE)
    # Start scheduler
    SCHED = PriorityScheduler(models=MODELS, runtime=RUNTIME)
    await SCHED.start()


@app.on_event("shutdown")
async def _shutdown():
    if SCHED:
        await SCHED.stop()


@app.get("/health")
async def health():
    return JSONResponse({"status": "ok"})


@app.get("/ready")
async def ready():
    ok = (MODELS is not None) and (SCHED is not None and SCHED.running)
    return JSONResponse({"ready": ok})


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    conn_id = str(uuid.uuid4())

    # Per-connection state
    rb = PCM16RingBuffer(sample_rate=SAMPLE_RATE, max_seconds=MAX_BUFFER_SECONDS)
    vad = StreamingVAD(sample_rate=SAMPLE_RATE)
    state = ConnectionState(
        conn_id=conn_id,
        language=os.getenv("ASR_LANGUAGE", "auto"),
        interim_cooldown_ms=INTERIM_COOLDOWN_MS,
        last_emit_ts_ms=0,
        last_interim_text="",
        last_commit_sample=0,
        phase="idle",
        outgoing=asyncio.Queue(maxsize=100),
        created_at=time.time(),
    )
    assert SCHED and RUNTIME
    RUNTIME.register_connection(conn_id, state)

    # Start a sender task to serialize server → client messages
    sender_task = asyncio.create_task(_sender_loop(ws, state))

    try:
        while True:
            msg = await ws.receive()
            if msg.get("type") == "websocket.disconnect":
                raise WebSocketDisconnect

            if "text" in msg and msg["text"] is not None:
                # Control JSON
                try:
                    data = json.loads(msg["text"]) if msg["text"] else {}
                except Exception:
                    continue
                await _handle_control(ws, data, state)
                continue

            if "bytes" in msg and msg["bytes"]:
                raw = msg["bytes"]
                # Append to ring buffer and run VAD frames
                appended_samples = rb.append_bytes(raw)
                vad.process_bytes(raw)

                # State machine: detect transitions
                now_ms = int(time.time() * 1000)
                if vad.just_started and state.phase != "listening":
                    state.phase = "listening"

                # Opportunistic interims (respect cooldown & backpressure)
                tail_sec = SCHED.dynamic_tail_seconds()
                if state.phase == "listening" and _cooldown_ok(state, now_ms, SCHED):
                    tail_audio = rb.tail_seconds(tail_sec)
                    if tail_audio is not None and tail_audio.size > 0:
                        SCHED.enqueue_interim(conn_id, tail_audio, state.language)
                        state.last_emit_ts_ms = now_ms

                # Finalization when end-of-utterance detected
                if vad.just_ended:
                    # Slice utterance from last_commit_sample to current end
                    end_sample = rb.current_sample_index
                    chunk = rb.get_since(state.last_commit_sample)
                    if chunk is not None and chunk.size > 0:
                        SCHED.enqueue_final(conn_id, chunk, state.language)
                        state.phase = "processing"
                        state.last_commit_sample = end_sample
            # else: unknown frame, ignore

    except WebSocketDisconnect:
        pass
    finally:
        # Cleanup
        if sender_task:
            sender_task.cancel()
            try:
                await sender_task
            except Exception:
                pass
        RUNTIME.unregister_connection(conn_id)


async def _handle_control(ws: WebSocket, data: Dict, state: ConnectionState):
    t = data.get("event")
    if t == "start":
        lang = data.get("language")
        if lang:
            state.language = lang
        await state.outgoing.put({"type": "status", "ok": True, "language": state.language})
    elif t == "set":
        if "language" in data:
            state.language = data["language"]
        if "interimRate" in data:
            try:
                # rate per second → cooldown ms
                r = float(data["interimRate"]) or 1.0
                state.interim_cooldown_ms = max(50, int(1000.0 / r))
            except Exception:
                pass
        await state.outgoing.put({
            "type": "status",
            "ok": True,
            "language": state.language,
            "cooldown_ms": state.interim_cooldown_ms,
        })
    elif t == "stop":
        # Force finalize current utterance via scheduler by emitting a small
        # end-of-speech signal from VAD perspective is handled by client; here we
        # just notify UX.
        await state.outgoing.put({"type": "status", "stopping": True})
    else:
        await state.outgoing.put({"type": "status", "ok": True})


def _cooldown_ok(state: ConnectionState, now_ms: int, sched: PriorityScheduler) -> bool:
    # Scheduler can globally increase cooldown under pressure
    cd = max(state.interim_cooldown_ms, sched.global_interim_cooldown_ms())
    return (now_ms - state.last_emit_ts_ms) >= cd


async def _sender_loop(ws: WebSocket, state: ConnectionState):
    """Serialize server → client sends from multiple producers (decode workers)."""
    try:
        while True:
            msg = await state.outgoing.get()
            await ws.send_text(json.dumps(msg))
    except Exception:
        pass

