#!/usr/bin/env bash
set -euo pipefail
export MODEL_NAME=${MODEL_NAME:-base}
export DEVICE=${DEVICE:-auto}
export COMPUTE_TYPE=${COMPUTE_TYPE:-float16}
export INTERIM_MIN_MS=${INTERIM_MIN_MS:-350}
export FINAL_SILENCE_MS_MIN=${FINAL_SILENCE_MS_MIN:-400}
export FINAL_SILENCE_MS_MAX=${FINAL_SILENCE_MS_MAX:-700}
export LOG_LEVEL=${LOG_LEVEL:-info}

exec uvicorn server.app:app \
  --host 0.0.0.0 \
  --port 8000 \
  --log-level ${LOG_LEVEL} \
  --ws websockets