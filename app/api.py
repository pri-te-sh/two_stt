from __future__ import annotations
import asyncio
import json
import time
from typing import Optional

from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

from .config import SETTINGS
from .models import get_models, model_info
from .decode import load_audio_from_bytes, transcribe_interim, transcribe_final
from .timing import Stopwatch


def make_app() -> FastAPI:
    app = FastAPI(title="MVP STT (two-model Faster-Whisper)", version="0.1.0")

    @app.get("/status")
    async def status():
        return {"status": "ok", **model_info()}

    @app.post("/transcribe")
    async def transcribe(file: UploadFile = File(...), language: Optional[str] = None, task: str = SETTINGS.TASK):
        """HTTP: run both passes on a file and return timings.
        Use this to sanity-check GPU throughput and model differences.
        """
        data = await file.read()
        audio = load_audio_from_bytes(data)
        models = get_models()

        sw = Stopwatch()
        with models.interim_lock:
            interim_text, _ = transcribe_interim(models.interim, audio, language or SETTINGS.LANGUAGE, task)
        interim_ms = sw.ms()

        with models.final_lock:
            final_text, _ = transcribe_final(models.final, audio, language or SETTINGS.LANGUAGE, task)
        final_ms = sw.ms()

        return JSONResponse({
            "file": file.filename,
            "language": language or SETTINGS.LANGUAGE or "auto",
            "task": task,
            "interim": {"text": interim_text, "elapsed_ms": interim_ms},
            "final": {"text": final_text, "elapsed_ms": final_ms},
            "meta": model_info(),
        })

    @app.websocket("/ws")
    async def ws_endpoint(ws: WebSocket):
        """Minimal WS demo (no VAD):
        - Client streams binary audio chunks (any container)
        - Send text message "DONE" when finished
        - Server emits {type:"interim"|"final", text, elapsed_ms}
        """
        await ws.accept()
        models = get_models()
        bufs: list[bytes] = []
        last_emit = 0.0
        interim_task: asyncio.Task | None = None

        async def run_interim():
            
            audio = load_audio_from_bytes(b"".join(bufs))
            # Optional: trim interims to the last N seconds to keep them snappy
            if SETTINGS.WS_INTERIM_MAX_SECONDS > 0:
                max_samps = int(SETTINGS.WS_INTERIM_MAX_SECONDS * 16000)
                audio = audio[-max_samps:]
            t0 = time.perf_counter()
            with models.interim_lock:
                text, _ = transcribe_interim(models.interim, audio, SETTINGS.LANGUAGE, SETTINGS.TASK)
            elapsed = int((time.perf_counter() - t0) * 1000)
            await ws.send_text(json.dumps({"type": "interim", "text": text, "elapsed_ms": elapsed}))

        try:
            while True:
                msg = await ws.receive()
                if msg["type"] == "websocket.disconnect":
                    break
                if "text" in msg and msg["text"]:
                    if msg["text"].strip().upper() == "DONE":
                        # Final pass on full audio
                        audio = load_audio_from_bytes(b"".join(bufs))
                        t0 = time.perf_counter()
                        with models.final_lock:
                            text, _ = transcribe_final(models.final, audio, SETTINGS.LANGUAGE, SETTINGS.TASK)
                        elapsed = int((time.perf_counter() - t0) * 1000)
                        await ws.send_text(json.dumps({"type": "final", "text": text, "elapsed_ms": elapsed}))
                        await ws.close(code=1000)
                        return
                elif "bytes" in msg and msg["bytes"]:
                    bufs.append(msg["bytes"])
                    now = time.perf_counter() * 1000.0
                    if now - last_emit >= SETTINGS.WS_INTERIM_COOLDOWN_MS:
                        last_emit = now
                        if interim_task and not interim_task.done():
                            # Drop older interim task in favor of newest snapshot
                            interim_task.cancel()
                        interim_task = asyncio.create_task(run_interim())
        except WebSocketDisconnect:
            pass
        finally:
            if interim_task and not interim_task.done():
                with contextlib.suppress(Exception):
                    interim_task.cancel()

    return app
