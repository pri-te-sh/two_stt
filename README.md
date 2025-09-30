# MVP STT (two-model Faster‑Whisper) — No VAD

**Goal:** Demonstrate the two-model approach (fast interim vs higher-quality final) on a simple FastAPI server. Includes HTTP file upload and a minimal WebSocket demo that finalizes on a client "DONE" message.

## 1) Setup
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -U pip
pip install -r requirements.txt
```
> GPU recommended. Set `DEVICE=cuda` (default). For CPU, `export DEVICE=cpu`.

### Optional env
```bash
export INTERIM_MODEL=small
export FINAL_MODEL=large-v3
export INTERIM_COMPUTE_TYPE=int8_float16
export FINAL_COMPUTE_TYPE=float16
export WS_INTERIM_COOLDOWN_MS=350
export WS_INTERIM_MAX_SECONDS=12
```

## 2) Run server
```bash
python server.py
```

## 3) Quick file test (HTTP)
```bash
python client/upload_test.py path/to/audio.wav
```
Output JSON includes both passes and elapsed ms.

## 4) Streaming demo (WS)
```bash
python client/ws_test.py path/to/audio.wav
```
Client streams bytes. Send `DONE` to trigger the final pass.

## 5) Notes
- **No VAD** by design. The client decides when to finalize ("DONE").
- Interims are throttled by `WS_INTERIM_COOLDOWN_MS` and optionally trimmed to the last `WS_INTERIM_MAX_SECONDS` to keep them snappy.
- For best speed, use a small interim model (e.g., `small`) and a larger final model (e.g., `large-v3`).
- Audio parsing is via `soundfile`; arbitrary containers should work as long as libsndfile can decode them. If you hit a codec issue, convert to 16 kHz mono WAV:
  ```bash
  ffmpeg -i input.any -ac 1 -ar 16000 out.wav
  ```

## 6) Troubleshooting
- **CUDA/cuDNN errors**: Ensure `nvidia-smi` works and you installed GPU wheels for `faster-whisper` (ctranslate2). Consider `FINAL_COMPUTE_TYPE=int8_float16` on smaller GPUs.
- **Repeating interims**: Increase `WS_INTERIM_COOLDOWN_MS` or decrease `WS_INTERIM_MAX_SECONDS`.
- **High latency**: Try `INTERIM_MODEL=base` and keep `FINAL_MODEL=large-v3` for quality.
