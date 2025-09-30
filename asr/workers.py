"""Decode workers for processing ASR jobs."""
import asyncio
import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Callable, Dict, Any
from server.schemas import DecodeJob, JobType
from asr.models import ASRModels
from scheduler.priority import PriorityScheduler
from util.logging import get_logger

logger = get_logger(__name__)


class DecodeWorker:
    """
    Background worker that processes decode jobs from scheduler.
    Runs one decode at a time per model to avoid GPU contention.
    """
    
    def __init__(
        self,
        name: str,
        job_type: JobType,
        models: ASRModels,
        scheduler: PriorityScheduler,
        executor: ThreadPoolExecutor,
        result_callback: Callable[[DecodeJob, Dict[str, Any]], None]
    ):
        """
        Initialize decode worker.
        
        Args:
            name: Worker name for logging
            job_type: Type of jobs to process (INTERIM or FINAL)
            models: ASR models instance
            scheduler: Job scheduler
            executor: Thread pool for CPU-bound work
            result_callback: Callback to invoke with results
        """
        self.name = name
        self.job_type = job_type
        self.models = models
        self.scheduler = scheduler
        self.executor = executor
        self.result_callback = result_callback
        
        # Select the appropriate model
        self.model = models.interim if job_type == JobType.INTERIM else models.final
        
        # Worker state
        self.running = False
        self.task: Optional[asyncio.Task] = None
        self.current_job: Optional[DecodeJob] = None
        
        # Stats
        self.stats = {
            "jobs_processed": 0,
            "total_decode_time": 0.0,
            "total_job_wait_time": 0.0,
        }
        
        logger.info(f"Initialized decode worker: {name}")
    
    async def start(self):
        """Start the worker."""
        if self.running:
            logger.warning(f"Worker {self.name} already running")
            return
        
        self.running = True
        self.task = asyncio.create_task(self._run())
        logger.info(f"Started worker: {self.name}")
    
    async def stop(self):
        """Stop the worker gracefully."""
        if not self.running:
            return
        
        logger.info(f"Stopping worker: {self.name}")
        self.running = False
        
        if self.task:
            await self.task
        
        logger.info(f"Stopped worker: {self.name}")
    
    async def _run(self):
        """Main worker loop."""
        logger.info(f"Worker {self.name} loop started")
        
        while self.running:
            try:
                # Try to get a job from scheduler
                # Note: scheduler.get_next_job() returns any job type based on priority
                # We'll let the scheduler handle priority, but we process our type
                job = self._get_job_for_type()
                
                if job is None:
                    # No jobs available, sleep briefly
                    await asyncio.sleep(0.01)  # 10ms
                    continue
                
                # Process the job
                self.current_job = job
                await self._process_job(job)
                self.current_job = None
                
            except Exception as e:
                logger.error(f"Worker {self.name} error: {e}", exc_info=True)
                await asyncio.sleep(0.1)
        
        logger.info(f"Worker {self.name} loop ended")
    
    def _get_job_for_type(self) -> Optional[DecodeJob]:
        """Get next job matching this worker's type from scheduler."""
        # For simplicity, we directly access the appropriate queue
        # In a more complex system, we'd have the scheduler give us type-specific jobs
        if self.job_type == JobType.FINAL:
            if len(self.scheduler.q_final) > 0:
                job = self.scheduler.q_final.popleft()
                self.scheduler.stats["total_dequeued_finals"] += 1
                self.scheduler.inflight_by_conn[job.conn_id] = job
                return job
        else:  # INTERIM
            if len(self.scheduler.q_interim) > 0:
                # Check burst limit
                if self.scheduler.f_interim_burst > 0:
                    job = self.scheduler.q_interim.popleft()
                    self.scheduler.stats["total_dequeued_interims"] += 1
                    self.scheduler.inflight_by_conn[job.conn_id] = job
                    
                    # Remove from queued tracking
                    if job.conn_id in self.scheduler.interim_queued_by_conn:
                        del self.scheduler.interim_queued_by_conn[job.conn_id]
                    
                    return job
        
        return None
    
    async def _process_job(self, job: DecodeJob):
        """
        Process a decode job.
        
        Args:
            job: Job to process
        """
        start_time = time.time()
        job_wait_time = start_time - job.created_at
        
        try:
            # Convert audio bytes to numpy array
            audio = np.frombuffer(job.audio_data, dtype=np.int16)
            
            # Run transcription in thread pool (CPU-intensive preprocessing + GPU)
            loop = asyncio.get_event_loop()
            text, language, confidence, segments = await loop.run_in_executor(
                self.executor,
                self.model.transcribe,
                audio,
                job.language
            )
            
            decode_time = time.time() - start_time
            
            # Update stats
            self.stats["jobs_processed"] += 1
            self.stats["total_decode_time"] += decode_time
            self.stats["total_job_wait_time"] += job_wait_time
            
            # Prepare result
            result = {
                "text": text,
                "language": language,
                "confidence": confidence,
                "segments": segments,
                "decode_time": decode_time,
                "job_wait_time": job_wait_time,
            }
            
            # Invoke callback with result
            self.result_callback(job, result)
            
            logger.debug(
                f"Worker {self.name} completed job {job.job_id}: "
                f"text_len={len(text)}, decode_time={decode_time:.3f}s, "
                f"wait_time={job_wait_time:.3f}s"
            )
            
        except Exception as e:
            logger.error(f"Worker {self.name} failed to process job {job.job_id}: {e}", exc_info=True)
            # Invoke callback with error
            self.result_callback(job, {"error": str(e)})
        
        finally:
            # Mark job complete in scheduler
            self.scheduler.mark_job_complete(job)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get worker statistics."""
        stats = self.stats.copy()
        if stats["jobs_processed"] > 0:
            stats["avg_decode_time"] = stats["total_decode_time"] / stats["jobs_processed"]
            stats["avg_wait_time"] = stats["total_job_wait_time"] / stats["jobs_processed"]
        else:
            stats["avg_decode_time"] = 0.0
            stats["avg_wait_time"] = 0.0
        
        stats["current_job_id"] = self.current_job.job_id if self.current_job else None
        return stats
