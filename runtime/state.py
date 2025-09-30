"""Runtime state manager coordinating all system components."""
import asyncio
import time
from typing import Dict, Optional, Any
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict

from asr.models import ASRModels
from asr.workers import DecodeWorker
from scheduler.priority import PriorityScheduler
from runtime.backpressure import BackpressureManager, ConnectionThrottle
from audio.buffer import ConnectionAudioState
from audio.vad_webrtc import VADProcessor, SilenceTracker
from server.schemas import DecodeJob, JobType, InterimMessage, FinalMessage
from util.logging import get_logger
from util.config import config
from util.ids import generate_job_id
import metrics.prometheus as metrics

logger = get_logger(__name__)


class RuntimeState:
    """
    Global runtime state managing all system components.
    
    Components:
    - ASR models (interim + final)
    - Scheduler with priority queues
    - Decode workers (one per model)
    - Backpressure manager
    - Connection states
    - Thread pool for CPU work
    """
    
    def __init__(self):
        """Initialize runtime state."""
        logger.info("Initializing runtime state...")
        
        # Core components
        self.models: Optional[ASRModels] = None
        self.scheduler = PriorityScheduler()
        self.backpressure = BackpressureManager()
        self.executor = ThreadPoolExecutor(max_workers=8, thread_name_prefix="stt-cpu")
        
        # Workers
        self.worker_final: Optional[DecodeWorker] = None
        self.worker_interim: Optional[DecodeWorker] = None
        
        # VAD processor (shared across connections)
        self.vad_processor = VADProcessor(
            sample_rate=config.sample_rate,
            mode=config.vad_mode
        )
        
        # Per-connection state
        self.connections: Dict[str, ConnectionAudioState] = {}
        self.connection_throttles: Dict[str, ConnectionThrottle] = {}
        self.silence_trackers: Dict[str, SilenceTracker] = {}
        
        # WebSocket connections (for sending messages)
        self.websockets: Dict[str, Any] = {}  # conn_id -> websocket
        
        # Background tasks
        self.monitor_task: Optional[asyncio.Task] = None
        self.monitor_running = False
        
        logger.info("Runtime state initialized")
    
    async def startup(self):
        """Start all components."""
        logger.info("Starting runtime components...")
        
        # Load ASR models
        logger.info("Loading ASR models...")
        self.models = ASRModels()
        self.models.warmup()
        
        # Create decode workers
        self.worker_final = DecodeWorker(
            name="worker_final",
            job_type=JobType.FINAL,
            models=self.models,
            scheduler=self.scheduler,
            executor=self.executor,
            result_callback=self._handle_final_result
        )
        
        self.worker_interim = DecodeWorker(
            name="worker_interim",
            job_type=JobType.INTERIM,
            models=self.models,
            scheduler=self.scheduler,
            executor=self.executor,
            result_callback=self._handle_interim_result
        )
        
        # Start workers
        await self.worker_final.start()
        await self.worker_interim.start()
        
        # Start monitoring task
        self.monitor_running = True
        self.monitor_task = asyncio.create_task(self._monitor_loop())
        
        logger.info("Runtime components started successfully")
    
    async def shutdown(self):
        """Shutdown all components gracefully."""
        logger.info("Shutting down runtime components...")
        
        # Stop monitoring
        self.monitor_running = False
        if self.monitor_task:
            await self.monitor_task
        
        # Stop workers
        if self.worker_final:
            await self.worker_final.stop()
        if self.worker_interim:
            await self.worker_interim.stop()
        
        # Shutdown executor
        self.executor.shutdown(wait=True)
        
        logger.info("Runtime components shut down")
    
    def add_connection(self, conn_id: str, websocket: Any):
        """Add new connection."""
        self.connections[conn_id] = ConnectionAudioState(
            conn_id=conn_id,
            sample_rate=config.sample_rate,
            max_duration_seconds=config.ring_buffer_seconds
        )
        self.connection_throttles[conn_id] = ConnectionThrottle(
            conn_id=conn_id,
            base_cooldown_ms=config.interim_cooldown_ms
        )
        self.silence_trackers[conn_id] = SilenceTracker(
            sample_rate=config.sample_rate,
            end_silence_ms=config.vad_end_silence_ms
        )
        self.websockets[conn_id] = websocket
        
        metrics.active_connections.inc()
        metrics.total_connections.inc()
        
        logger.info(f"Added connection: {conn_id}")
    
    def remove_connection(self, conn_id: str):
        """Remove connection and clean up resources."""
        if conn_id in self.connections:
            del self.connections[conn_id]
        if conn_id in self.connection_throttles:
            del self.connection_throttles[conn_id]
        if conn_id in self.silence_trackers:
            del self.silence_trackers[conn_id]
        if conn_id in self.websockets:
            del self.websockets[conn_id]
        
        metrics.active_connections.dec()
        
        logger.info(f"Removed connection: {conn_id}")
    
    def enqueue_interim(self, conn_id: str, audio: bytes, language: str, t0: float, t1: float) -> bool:
        """
        Enqueue interim transcription job.
        
        Returns:
            True if enqueued, False if throttled/rejected
        """
        # Check backpressure
        conn_state = self.connections.get(conn_id)
        if not conn_state:
            return False
        
        # Check throttle
        throttle = self.connection_throttles.get(conn_id)
        if throttle and not throttle.should_allow_interim(self.backpressure.current_cooldown_ms):
            logger.debug(f"Interim throttled for {conn_id}")
            return False
        
        # Create job
        job = DecodeJob(
            job_id=generate_job_id(),
            job_type=JobType.INTERIM,
            conn_id=conn_id,
            audio_data=audio,
            language=language,
            created_at=time.time(),
            t0=t0,
            t1=t1
        )
        
        # Enqueue with coalescing
        if self.scheduler.enqueue_interim(job):
            if throttle:
                throttle.mark_interim_sent()
            metrics.jobs_enqueued.labels(job_type="interim").inc()
            return True
        
        return False
    
    def enqueue_final(self, conn_id: str, audio: bytes, language: str, t0: float, t1: float):
        """Enqueue final transcription job."""
        job = DecodeJob(
            job_id=generate_job_id(),
            job_type=JobType.FINAL,
            conn_id=conn_id,
            audio_data=audio,
            language=language,
            created_at=time.time(),
            t0=t0,
            t1=t1
        )
        
        self.scheduler.enqueue_final(job)
        metrics.jobs_enqueued.labels(job_type="final").inc()
    
    def _handle_interim_result(self, job: DecodeJob, result: Dict[str, Any]):
        """Handle completed interim transcription."""
        if "error" in result:
            logger.error(f"Interim job {job.job_id} failed: {result['error']}")
            metrics.record_decode_complete("interim", 0, 0, success=False)
            return
        
        # Record metrics
        metrics.record_decode_complete(
            "interim",
            result["decode_time"],
            result["job_wait_time"],
            success=True
        )
        
        # Send interim message to client
        conn_state = self.connections.get(job.conn_id)
        if conn_state:
            conn_state.last_interim_text = result["text"]
            conn_state.last_interim_time = time.time()
        
        # Send via WebSocket
        asyncio.create_task(self._send_interim_message(job, result))
    
    def _handle_final_result(self, job: DecodeJob, result: Dict[str, Any]):
        """Handle completed final transcription."""
        if "error" in result:
            logger.error(f"Final job {job.job_id} failed: {result['error']}")
            metrics.record_decode_complete("final", 0, 0, success=False)
            return
        
        # Record metrics
        metrics.record_decode_complete(
            "final",
            result["decode_time"],
            result["job_wait_time"],
            success=True
        )
        
        # Send final message to client
        asyncio.create_task(self._send_final_message(job, result))
    
    async def _send_interim_message(self, job: DecodeJob, result: Dict[str, Any]):
        """Send interim message to WebSocket client."""
        websocket = self.websockets.get(job.conn_id)
        if not websocket:
            return
        
        try:
            message = InterimMessage(
                conn=job.conn_id,
                text=result["text"],
                stable_chars=int(len(result["text"]) * 0.7),  # Rough estimate
                t0=job.t0,
                t1=job.t1
            )
            await websocket.send_json(message.dict())
        except Exception as e:
            logger.error(f"Failed to send interim message: {e}")
    
    async def _send_final_message(self, job: DecodeJob, result: Dict[str, Any]):
        """Send final message to WebSocket client."""
        websocket = self.websockets.get(job.conn_id)
        if not websocket:
            return
        
        try:
            message = FinalMessage(
                conn=job.conn_id,
                text=result["text"],
                segments=result["segments"],
                language=result.get("language"),
                t0=job.t0,
                t1=job.t1
            )
            await websocket.send_json(message.dict())
        except Exception as e:
            logger.error(f"Failed to send final message: {e}")
    
    async def _monitor_loop(self):
        """Background monitoring loop for backpressure and metrics."""
        logger.info("Started monitor loop")
        
        while self.monitor_running:
            try:
                # Get queue depths
                depths = self.scheduler.get_queue_depths()
                q_final_len = depths["q_final"]
                q_interim_len = depths["q_interim"]
                
                # Update backpressure
                bp_state = self.backpressure.update(q_final_len, q_interim_len)
                
                # Update burst limits in scheduler
                f_final_burst, f_interim_burst = self.backpressure.get_burst_limits()
                self.scheduler.set_burst_limits(f_final_burst, f_interim_burst)
                
                # Update metrics
                metrics.queue_depth.labels(queue_type="final").set(q_final_len)
                metrics.queue_depth.labels(queue_type="interim").set(q_interim_len)
                
                # Queue age metrics
                final_age = self.scheduler.get_oldest_job_age(JobType.FINAL)
                if final_age is not None:
                    metrics.queue_job_age_seconds.labels(queue_type="final").set(final_age)
                
                interim_age = self.scheduler.get_oldest_job_age(JobType.INTERIM)
                if interim_age is not None:
                    metrics.queue_job_age_seconds.labels(queue_type="interim").set(interim_age)
                
                # Backpressure metrics
                level_map = {"normal": 0, "high": 1, "critical": 2}
                metrics.update_backpressure_metrics(
                    level_map[bp_state["level"].value],
                    bp_state["cooldown_ms"],
                    bp_state["tail_seconds"],
                    bp_state["interims_paused"]
                )
                
                # GPU metrics
                metrics.update_gpu_metrics()
                
                # Sleep for a bit
                await asyncio.sleep(1.0)
                
            except Exception as e:
                logger.error(f"Monitor loop error: {e}", exc_info=True)
                await asyncio.sleep(1.0)
        
        logger.info("Monitor loop stopped")


# Global runtime instance
runtime_state: Optional[RuntimeState] = None


def get_runtime() -> RuntimeState:
    """Get global runtime state instance."""
    global runtime_state
    if runtime_state is None:
        runtime_state = RuntimeState()
    return runtime_state
