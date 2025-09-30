"""Priority scheduler for decode jobs with coalescing and fairness."""
import asyncio
import time
from typing import Dict, Optional, List
from collections import deque
from server.schemas import DecodeJob, JobType
from util.logging import get_logger
from util.config import config

logger = get_logger(__name__)


class PriorityScheduler:
    """
    Priority scheduler with deadline-aware fairness and interim coalescing.
    
    Features:
    - Finals always have priority over interims
    - Within each queue type, older jobs are prioritized (FIFO with age)
    - Interim coalescing: max 1 queued interim per connection
    - Burst limits can throttle interims when finals are backed up
    """
    
    def __init__(self):
        """Initialize scheduler."""
        self.q_final: deque[DecodeJob] = deque()
        self.q_interim: deque[DecodeJob] = deque()
        
        # Track queued interims per connection for coalescing
        self.interim_queued_by_conn: Dict[str, DecodeJob] = {}
        
        # Track in-flight jobs per connection
        self.inflight_by_conn: Dict[str, DecodeJob] = {}
        
        # Burst limits (can be dynamically adjusted)
        self.f_final_burst = config.f_final_burst
        self.f_interim_burst = config.f_interim_burst
        
        # Stats
        self.stats = {
            "total_enqueued_finals": 0,
            "total_enqueued_interims": 0,
            "total_dequeued_finals": 0,
            "total_dequeued_interims": 0,
            "total_coalesced": 0,
        }
        
        logger.info(f"Initialized scheduler: f_final_burst={self.f_final_burst}, f_interim_burst={self.f_interim_burst}")
    
    def enqueue_final(self, job: DecodeJob):
        """
        Enqueue a final transcription job.
        
        Args:
            job: DecodeJob for final transcription
        """
        self.q_final.append(job)
        self.stats["total_enqueued_finals"] += 1
        logger.debug(f"Enqueued final job {job.job_id} for {job.conn_id}, queue_len={len(self.q_final)}")
    
    def enqueue_interim(self, job: DecodeJob) -> bool:
        """
        Enqueue an interim transcription job with coalescing.
        
        If this connection already has an interim queued, replace it (coalesce).
        If this connection has an interim in-flight, reject the job.
        
        Args:
            job: DecodeJob for interim transcription
            
        Returns:
            True if enqueued, False if rejected (in-flight)
        """
        conn_id = job.conn_id
        
        # Check if connection has interim in-flight
        if conn_id in self.inflight_by_conn:
            inflight_job = self.inflight_by_conn[conn_id]
            if inflight_job.job_type == JobType.INTERIM:
                logger.debug(f"Rejected interim for {conn_id}: already in-flight")
                return False
        
        # Check if connection already has interim queued
        if conn_id in self.interim_queued_by_conn:
            # Coalesce: replace old interim with new one
            old_job = self.interim_queued_by_conn[conn_id]
            try:
                self.q_interim.remove(old_job)
                self.stats["total_coalesced"] += 1
                logger.debug(f"Coalesced interim for {conn_id}: replaced {old_job.job_id} with {job.job_id}")
            except ValueError:
                # Old job was already dequeued, proceed normally
                pass
        
        # Enqueue new interim
        self.q_interim.append(job)
        self.interim_queued_by_conn[conn_id] = job
        self.stats["total_enqueued_interims"] += 1
        logger.debug(f"Enqueued interim job {job.job_id} for {conn_id}, queue_len={len(self.q_interim)}")
        return True
    
    def get_next_job(self) -> Optional[DecodeJob]:
        """
        Get next job to process based on priority and burst limits.
        
        Priority order:
        1. Finals (up to f_final_burst)
        2. Interims (up to f_interim_burst)
        
        Returns:
            Next job to process, or None if no jobs available
        """
        current_time = time.time()
        
        # Try to get a final job first (if not exceeding burst)
        if len(self.q_final) > 0 and self.f_final_burst > 0:
            job = self.q_final.popleft()
            self.stats["total_dequeued_finals"] += 1
            self.inflight_by_conn[job.conn_id] = job
            
            job_age = current_time - job.created_at
            logger.debug(f"Dequeued final job {job.job_id}, age={job_age:.3f}s, remaining={len(self.q_final)}")
            return job
        
        # Try to get an interim job (if not exceeding burst and allowed)
        if len(self.q_interim) > 0 and self.f_interim_burst > 0:
            job = self.q_interim.popleft()
            self.stats["total_dequeued_interims"] += 1
            self.inflight_by_conn[job.conn_id] = job
            
            # Remove from queued tracking
            if job.conn_id in self.interim_queued_by_conn:
                del self.interim_queued_by_conn[job.conn_id]
            
            job_age = current_time - job.created_at
            logger.debug(f"Dequeued interim job {job.job_id}, age={job_age:.3f}s, remaining={len(self.q_interim)}")
            return job
        
        return None
    
    def mark_job_complete(self, job: DecodeJob):
        """
        Mark a job as complete and remove from in-flight tracking.
        
        Args:
            job: Completed job
        """
        if job.conn_id in self.inflight_by_conn:
            del self.inflight_by_conn[job.conn_id]
            logger.debug(f"Marked job {job.job_id} complete for {job.conn_id}")
    
    def set_burst_limits(self, f_final_burst: int, f_interim_burst: int):
        """
        Update burst limits dynamically for backpressure management.
        
        Args:
            f_final_burst: Max finals to process per scheduler cycle
            f_interim_burst: Max interims to process per scheduler cycle (0 = paused)
        """
        self.f_final_burst = f_final_burst
        self.f_interim_burst = f_interim_burst
        logger.info(f"Updated burst limits: final={f_final_burst}, interim={f_interim_burst}")
    
    def get_queue_depths(self) -> Dict[str, int]:
        """Get current queue depths."""
        return {
            "q_final": len(self.q_final),
            "q_interim": len(self.q_interim),
        }
    
    def get_stats(self) -> Dict[str, int]:
        """Get scheduler statistics."""
        return self.stats.copy()
    
    def get_oldest_job_age(self, job_type: JobType) -> Optional[float]:
        """
        Get age of oldest job in queue.
        
        Args:
            job_type: Type of job to check
            
        Returns:
            Age in seconds, or None if queue is empty
        """
        current_time = time.time()
        
        if job_type == JobType.FINAL and len(self.q_final) > 0:
            oldest_job = self.q_final[0]
            return current_time - oldest_job.created_at
        elif job_type == JobType.INTERIM and len(self.q_interim) > 0:
            oldest_job = self.q_interim[0]
            return current_time - oldest_job.created_at
        
        return None
