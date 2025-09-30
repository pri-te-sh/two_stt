# ──────────────────────────────────────────────────────────────────────────────
# File: app/server/routes.py
# Basic HTTP routes (status, config snapshot)
# ──────────────────────────────────────────────────────────────────────────────
from __future__ import annotations
from fastapi import APIRouter

from asr.config import ASRConfig

router = APIRouter()


@router.get("/status")
def get_status():
    return {
        "service": "stt-core",
        "config": ASRConfig.snapshot(),
    }

