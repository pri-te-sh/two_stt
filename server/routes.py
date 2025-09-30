"""WebSocket routes for real-time audio streaming."""
import asyncio
import json
import time
import numpy as np
from fastapi import WebSocket, WebSocketDisconnect
from typing import Optional

from server.schemas import (
    StartEvent, StopEvent, SetEvent, StatusMessage,
    ErrorMessage, BackpressureLevel
)
from runtime.state import get_runtime
from util.logging import get_logger
from util.ids import generate_conn_id
from util.config import config

logger = get_logger(__name__)


class WebSocketHandler:
    """Handles WebSocket connection for a single client."""
    
    def __init__(self, websocket: WebSocket):
        """
        Initialize WebSocket handler.
        
        Args:
            websocket: FastAPI WebSocket connection
        """
        self.websocket = websocket
        self.conn_id = generate_conn_id()
        self.runtime = get_runtime()
        self.running = False
        self.language = config.asr_language
        
        logger.info(f"Initialized WebSocket handler: {self.conn_id}")
    
    async def handle(self):
        """Main handler for WebSocket connection."""
        try:
            # Accept connection
            await self.websocket.accept()
            logger.info(f"WebSocket connection accepted: {self.conn_id}")
            
            # Add connection to runtime
            self.runtime.add_connection(self.conn_id, self.websocket)
            self.running = True
            
            # Send initial status
            await self._send_status()
            
            # Main message loop
            while self.running:
                try:
                    # Receive message (binary or text)
                    message = await self.websocket.receive()
                    
                    if "bytes" in message:
                        # Binary audio data
                        await self._handle_audio(message["bytes"])
                    elif "text" in message:
                        # JSON control message
                        await self._handle_control(message["text"])
                    else:
                        logger.warning(f"Unknown message type from {self.conn_id}")
                
                except WebSocketDisconnect:
                    logger.info(f"WebSocket disconnected: {self.conn_id}")
                    break
                except Exception as e:
                    logger.error(f"Error handling message from {self.conn_id}: {e}", exc_info=True)
                    await self._send_error("MESSAGE_ERROR", str(e))
        
        finally:
            # Cleanup
            self.running = False
            self.runtime.remove_connection(self.conn_id)
            logger.info(f"WebSocket connection closed: {self.conn_id}")
    
    async def _handle_audio(self, audio_bytes: bytes):
        """
        Handle incoming audio data.
        
        Args:
            audio_bytes: Raw PCM16 audio bytes
        """
        conn_state = self.runtime.connections.get(self.conn_id)
        if not conn_state:
            return
        
        # Convert bytes to numpy array
        audio = np.frombuffer(audio_bytes, dtype=np.int16)
        
        # Update timestamp
        current_time = time.time()
        conn_state.last_audio_time = current_time
        
        # Append to ring buffer
        conn_state.ring_buffer.append(audio)
        
        # VAD processing
        silence_tracker = self.runtime.silence_trackers.get(self.conn_id)
        if not silence_tracker:
            return
        
        # Check for speech
        is_speech = self.runtime.vad_processor.is_speech(audio)
        silence_tracker.update(is_speech, len(audio))
        
        # Check if utterance should be finalized
        if silence_tracker.should_finalize():
            await self._finalize_utterance()
            silence_tracker.reset()
            conn_state.start_utterance(current_time)
        
        # Start utterance if speech detected and not already in utterance
        if is_speech and not conn_state.in_utterance:
            conn_state.start_utterance(current_time)
        
        # Enqueue interim if in utterance
        if conn_state.in_utterance:
            await self._enqueue_interim()
    
    async def _enqueue_interim(self):
        """Enqueue interim transcription job."""
        conn_state = self.runtime.connections.get(self.conn_id)
        if not conn_state:
            return
        
        # Get current tail window size from backpressure manager
        tail_seconds = self.runtime.backpressure.current_tail_seconds
        
        # Get tail audio
        audio = conn_state.ring_buffer.get_tail(tail_seconds)
        if len(audio) == 0:
            return
        
        # Convert to bytes
        audio_bytes = audio.tobytes()
        
        # Get timing
        current_time = time.time()
        t1 = current_time
        t0 = t1 - (len(audio) / config.sample_rate)
        
        # Enqueue job
        self.runtime.enqueue_interim(
            self.conn_id,
            audio_bytes,
            self.language,
            t0,
            t1
        )
    
    async def _finalize_utterance(self):
        """Finalize current utterance."""
        conn_state = self.runtime.connections.get(self.conn_id)
        if not conn_state:
            return
        
        # Get utterance audio
        audio = conn_state.ring_buffer.get_utterance()
        if len(audio) == 0:
            return
        
        # Convert to bytes
        audio_bytes = audio.tobytes()
        
        # Get timing
        current_time = time.time()
        t1 = current_time
        t0 = t1 - (len(audio) / config.sample_rate)
        
        # Enqueue final job
        self.runtime.enqueue_final(
            self.conn_id,
            audio_bytes,
            self.language,
            t0,
            t1
        )
        
        # End utterance
        conn_state.end_utterance()
        
        logger.debug(f"Finalized utterance for {self.conn_id}, duration={t1-t0:.2f}s")
    
    async def _handle_control(self, text: str):
        """
        Handle JSON control message.
        
        Args:
            text: JSON string
        """
        try:
            data = json.loads(text)
            event = data.get("event")
            
            if event == "start":
                await self._handle_start(data)
            elif event == "stop":
                await self._handle_stop()
            elif event == "set":
                await self._handle_set(data)
            else:
                logger.warning(f"Unknown event from {self.conn_id}: {event}")
        
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON from {self.conn_id}: {e}")
            await self._send_error("INVALID_JSON", str(e))
        except Exception as e:
            logger.error(f"Error handling control message from {self.conn_id}: {e}", exc_info=True)
            await self._send_error("CONTROL_ERROR", str(e))
    
    async def _handle_start(self, data: dict):
        """Handle start event."""
        self.language = data.get("language", "auto")
        logger.info(f"Started session for {self.conn_id}, language={self.language}")
    
    async def _handle_stop(self):
        """Handle stop event - force finalize current utterance."""
        await self._finalize_utterance()
        logger.info(f"Stopped session for {self.conn_id}")
    
    async def _handle_set(self, data: dict):
        """Handle configuration update."""
        # Could update per-connection settings here
        logger.debug(f"Set event from {self.conn_id}: {data}")
    
    async def _send_status(self):
        """Send status update to client."""
        try:
            bp_state = self.runtime.backpressure.get_state()
            
            message = StatusMessage(
                backpressure=BackpressureLevel(bp_state["level"]),
                cooldown_ms=bp_state["cooldown_ms"],
                tail_s=bp_state["tail_s"],
                interim_paused=bp_state["interims_paused"]
            )
            
            await self.websocket.send_json(message.dict())
        except Exception as e:
            logger.error(f"Failed to send status: {e}")
    
    async def _send_error(self, code: str, detail: str):
        """Send error message to client."""
        try:
            message = ErrorMessage(code=code, detail=detail)
            await self.websocket.send_json(message.dict())
        except Exception as e:
            logger.error(f"Failed to send error: {e}")


async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for audio streaming.
    
    Args:
        websocket: FastAPI WebSocket connection
    """
    handler = WebSocketHandler(websocket)
    await handler.handle()
