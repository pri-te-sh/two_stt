README.md

# MVP FastAPI STT Server (priority scheduler)

This MVP shows:
- WebSocket audio streaming
- CPU WebRTC-VAD gating with adaptive finalization
- Priority scheduling (final > interim) with interim coalescing (max 1 in-flight per session)
- Faster-Whisper decoding tuned differently for interim vs final
- Stabilized interim payloads with `stable_chars`

## Quick start

### 1) System deps
- Python 3.10+
- (Optional) NVIDIA GPU with CUDA 11.8+ for best performance

### 2) Create venv & install
```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

3) Run

# Recommended defaults are set via env vars in run.sh
bash run.sh
# or
UVICORN_WORKERS=1 uvicorn server.app:app --host 0.0.0.0 --port 8000 --log-level info --ws websockets

Env vars (defaults in server/config.py):

MODEL_NAME (default: base) – try small, medium, distil-large-v3, large-v3 if VRAM allows

DEVICE (default: auto)

COMPUTE_TYPE (default: float16) – try int8_float16 for small GPUs or CPU fallback

INTERIM_MIN_MS (default: 350)

FINAL_SILENCE_MS_MIN (default: 400), FINAL_SILENCE_MS_MAX (default: 700)

4) Simple test (Python client)

python - <<'PY'
import asyncio, websockets, json, base64, soundfile as sf

async def main():
    uri = "ws://localhost:8000/ws"
    async with websockets.connect(uri, max_size=2**23) as ws:
        await ws.send(json.dumps({"op":"start","sample_rate":16000,"lang":"en"}))
        # send a short wav file (mono 16k PCM16)
        audio, sr = sf.read("sample.wav", dtype='int16')
        assert sr==16000
        chunk = 16000//2  # 0.5s
        for i in range(0, len(audio), chunk):
            buf = audio[i:i+chunk].tobytes()
            await ws.send(json.dumps({"op":"audio","payload": base64.b64encode(buf).decode()}))
            await asyncio.sleep(0.05)
        await ws.send(json.dumps({"op":"stop"}))
        try:
            while True:
                msg = await asyncio.wait_for(ws.recv(), timeout=3)
                print(msg)
        except asyncio.TimeoutError:
            pass

asyncio.run(main())
PY

5) Health endpoints

GET /ready – model loaded

GET /status – simple JSON status

Notes

This is an MVP: single model instance, single decode worker, but priority ensures finals are processed ahead of interims.

Upgrade path: add distinct interim/final workers and a shared VRAM model pool.