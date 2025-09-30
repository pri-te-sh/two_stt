# Real-time Speech-to-Text Server

Production-grade dual-model STT server with interim (fast) and final (quality) transcriptions.

## Architecture

- **Single process, shared models**: One Uvicorn worker keeps both models in VRAM
- **Dual-model system**: 
  - Interim model (small/base): Fast, streaming results
  - Final model (distil-large-v3/large-v3): High-quality final transcriptions
- **Priority scheduler**: Finals prioritized over interims with coalescing
- **Backpressure management**: Dynamic throttling based on queue depths
- **WebRTC VAD**: CPU-based voice activity detection
- **Per-connection state**: Ring buffers, utterance tracking, language detection

## Requirements

- Python 3.11+
- CUDA-capable GPU (tested on RTX 4090 with 24GB VRAM)
- NVIDIA driver with CUDA 12.x
- Ubuntu 24.04 or similar

## Installation

### 1. System Dependencies

```bash
# Install system packages
sudo apt-get update
sudo apt-get install -y build-essential python3.11 python3.11-venv python3-pip \
    ffmpeg libsndfile1 nginx

# Verify NVIDIA driver and CUDA
nvidia-smi
```

### 2. Setup User and Directories

```bash
# Create user and directories (as root)
sudo useradd -m -s /bin/bash stt
sudo mkdir -p /opt/stt/{app,logs,models,venv}
sudo chown -R stt:stt /opt/stt
```

### 3. Install Python Dependencies

```bash
# Create virtual environment
sudo -u stt python3.11 -m venv /opt/stt/venv
sudo -u stt /opt/stt/venv/bin/pip install --upgrade pip wheel

# Install dependencies
sudo -u stt /opt/stt/venv/bin/pip install -r requirements.txt
```

### 4. Configure Environment

```bash
# Copy code to /opt/stt/app
sudo cp -r /home/claude/stt-server/* /opt/stt/app/
sudo chown -R stt:stt /opt/stt/app

# Create .env file
sudo -u stt cp /opt/stt/app/.env.example /opt/stt/app/.env

# Edit configuration as needed
sudo -u stt nano /opt/stt/app/.env
```

## Configuration

Edit `.env` to customize:

```bash
# Models
INTERIM_MODEL=small              # or base
FINAL_MODEL=distil-large-v3      # or large-v3
INTERIM_COMPUTE=int8_float16
FINAL_COMPUTE=float16

# Timing
INTERIM_COOLDOWN_MS=220          # Interim rate limit
TAIL_SECONDS=7                   # Tail window size
SCHEDULER_TICK_MS=12             # Scheduler poll interval

# Backpressure
FINAL_HI=6                       # Final queue high watermark
FINAL_CRIT=12                    # Final queue critical watermark
INTERIM_HI=20                    # Interim queue high watermark
INTERIM_CRIT=40                  # Interim queue critical watermark
```

## Warmup Models

Before starting the server, warm up the models:

```bash
sudo -u stt /opt/stt/venv/bin/python /opt/stt/app/scripts/warmup.py
```

## Running the Server

### Development Mode

```bash
cd /opt/stt/app
/opt/stt/venv/bin/python server/app.py
```

### Production with Systemd

1. Create systemd service:

```bash
sudo nano /etc/systemd/system/stt.service
```

```ini
[Unit]
Description=Realtime STT server
After=network-online.target

[Service]
User=stt
Group=stt
WorkingDirectory=/opt/stt/app
EnvironmentFile=/opt/stt/app/.env
ExecStart=/opt/stt/venv/bin/uvicorn server.app:app --host ${BIND_HOST} --port ${BIND_PORT} --loop uvloop --http httptools
Restart=on-failure
RestartSec=5
LimitNOFILE=65535

[Install]
WantedBy=multi-user.target
```

2. Enable and start:

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now stt
sudo systemctl status stt
```

3. View logs:

```bash
sudo journalctl -u stt -f
```

## Nginx Configuration (Optional)

For TLS termination and WebSocket proxy:

```nginx
server {
    listen 443 ssl http2;
    server_name stt.yourdomain.com;

    ssl_certificate /etc/letsencrypt/live/stt.yourdomain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/stt.yourdomain.com/privkey.pem;

    location /ws {
        proxy_pass http://127.0.0.1:8081/ws;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_read_timeout 3600s;
        proxy_send_timeout 3600s;
        proxy_buffering off;
    }

    location / {
        proxy_pass http://127.0.0.1:8081;
        proxy_set_header Host $host;
    }
}

server {
    listen 80;
    server_name stt.yourdomain.com;
    return 301 https://$host$request_uri;
}
```

## API Endpoints

### WebSocket: `/ws`

Binary protocol for audio streaming + JSON control messages.

**Client → Server:**

Audio frames (binary):
- PCM16 mono @ 16 kHz
- 20-60 ms chunks (640-1920 bytes)

Control messages (JSON):
```json
{"event": "start", "language": "auto"}
{"event": "stop"}
{"event": "set", "interimRate": 4}
```

**Server → Client (JSON):**

Interim result:
```json
{
  "type": "interim",
  "conn": "conn_...",
  "text": "hello world",
  "stable_chars": 7,
  "t0": 1234567890.123,
  "t1": 1234567892.456
}
```

Final result:
```json
{
  "type": "final",
  "conn": "conn_...",
  "text": "hello world",
  "segments": [
    {"start": 0.0, "end": 1.2, "text": "hello"},
    {"start": 1.2, "end": 2.3, "text": "world"}
  ],
  "language": "en",
  "t0": 1234567890.123,
  "t1": 1234567892.456
}
```

Status update:
```json
{
  "type": "status",
  "backpressure": "normal",
  "cooldown_ms": 220,
  "tail_s": 7.0,
  "interim_paused": false
}
```

### HTTP Endpoints

- `GET /health` - Health check (process alive)
- `GET /ready` - Readiness check (models loaded)
- `GET /status` - Detailed status (queues, workers, config)
- `GET /metrics` - Prometheus metrics

## Metrics

Prometheus metrics available at `/metrics`:

- `stt_active_connections` - Active WebSocket connections
- `stt_queue_depth{queue_type}` - Queue depths (final, interim)
- `stt_jobs_processed_total{job_type, status}` - Jobs processed
- `stt_decode_duration_seconds{job_type}` - Decode latencies
- `stt_backpressure_level` - Current backpressure level
- `stt_gpu_memory_used_bytes{gpu_id}` - GPU memory usage
- And more...

## Monitoring

### Queue Depths

```bash
curl http://localhost:8081/status | jq '.queues'
```

### Worker Stats

```bash
curl http://localhost:8081/status | jq '.workers'
```

### Logs

```bash
# Systemd
sudo journalctl -u stt -f

# Or if running directly
tail -f /opt/stt/logs/stt.log
```

## Troubleshooting

### High Final Latency

1. Check queue depths: `curl localhost:8081/status | jq '.queues'`
2. Verify backpressure: `curl localhost:8081/status | jq '.backpressure'`
3. Reduce final beam_size to 3 in `asr/models.py`

### GPU OOM

1. Check GPU memory: `nvidia-smi`
2. Reduce tail_seconds in `.env`
3. Use smaller models (base for interim, distil-large-v3 for final)

### Repeating Interims

1. Verify `condition_on_previous_text=False` for interims in `asr/models.py`
2. Check stabilization rules in `server/routes.py`

### cuDNN/Torch Errors

- VAD is CPU-only (WebRTC VAD)
- Faster-Whisper uses ctranslate2 (doesn't need cuDNN)
- Verify CUDA version matches: `nvcc --version`

## Performance

**Capacity (RTX 4090):**
- 10-25 active concurrent speakers
- p95 interim latency: <250ms
- p95 finalization latency: <900ms

**Optimization:**
- Drop final beam_size to 3 for lower latency
- Reduce tail_seconds under high load
- Scale horizontally with sticky load balancer

## License

MIT
