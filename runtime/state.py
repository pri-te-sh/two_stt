# ──────────────────────────────────────────────────────────────────────────────
# File: app/runtime/state.py
# Global runtime registries and per-connection state
# ──────────────────────────────────────────────────────────────────────────────
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict
import asyncio


@dataclass
class ConnectionState:
    conn_id: str
    language: str
    interim_cooldown_ms: int
    last_emit_ts_ms: int
    last_interim_text: str
    last_commit_sample: int
    phase: str  # idle | listening | processing
    outgoing: asyncio.Queue
    created_at: float


class GlobalRuntime:
    def __init__(self, sample_rate: int):
        self.sample_rate = sample_rate
        self._connections: Dict[str, ConnectionState] = {}

    def register_connection(self, conn_id: str, state: ConnectionState):
        self._connections[conn_id] = state

    def unregister_connection(self, conn_id: str):
        self._connections.pop(conn_id, None)

    def connection(self, conn_id: str) -> ConnectionState | None:
        return self._connections.get(conn_id)

