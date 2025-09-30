"""Backpressure management with watermarks and dynamic throttling."""
import time
from typing import Dict, Any
from server.schemas import BackpressureLevel
from util.logging import get_logger
from util.config import config

logger = get_logger(__name__)


class BackpressureManager:
    """
    Manages backpressure based on queue depths and system load.
    
    Adjusts:
    - Interim cooldown periods
    - Tail window size
    - Whether interims are paused
    """
    
    def __init__(self):
        """Initialize backpressure manager."""
        # Watermarks from config
        self.final_hi = config.final_hi
        self.final_crit = config.final_crit
        self.interim_hi = config.interim_hi
        self.interim_crit = config.interim_crit
        
        # Current state
        self.level = BackpressureLevel.NORMAL
        
        # Dynamic parameters
        self.base_cooldown_ms = config.interim_cooldown_ms
        self.base_tail_seconds = config.tail_seconds
        
        self.current_cooldown_ms = self.base_cooldown_ms
        self.current_tail_seconds = self.base_tail_seconds
        self.interims_paused = False
        
        # Timing for adaptive behavior
        self.last_update_time = time.time()
        
        logger.info(
            f"Initialized backpressure manager: "
            f"final_hi={self.final_hi}, final_crit={self.final_crit}, "
            f"interim_hi={self.interim_hi}, interim_crit={self.interim_crit}"
        )
    
    def update(self, q_final_len: int, q_interim_len: int) -> Dict[str, Any]:
        """
        Update backpressure state based on queue depths.
        
        Args:
            q_final_len: Length of final queue
            q_interim_len: Length of interim queue
            
        Returns:
            Dictionary with current backpressure state
        """
        current_time = time.time()
        old_level = self.level
        
        # Determine backpressure level
        if q_final_len >= self.final_crit or q_interim_len >= self.interim_crit:
            self.level = BackpressureLevel.CRITICAL
        elif q_final_len >= self.final_hi or q_interim_len >= self.interim_hi:
            self.level = BackpressureLevel.HIGH
        else:
            self.level = BackpressureLevel.NORMAL
        
        # Log level changes
        if self.level != old_level:
            logger.warning(
                f"Backpressure level changed: {old_level} -> {self.level}, "
                f"q_final={q_final_len}, q_interim={q_interim_len}"
            )
        
        # Adjust parameters based on level
        if self.level == BackpressureLevel.CRITICAL:
            # Critical: aggressive throttling
            self.current_cooldown_ms = self.base_cooldown_ms + 250
            self.current_tail_seconds = max(1.5, self.base_tail_seconds * 0.25)
            self.interims_paused = q_final_len >= self.final_crit
            
        elif self.level == BackpressureLevel.HIGH:
            # High: moderate throttling
            self.current_cooldown_ms = self.base_cooldown_ms + 150
            self.current_tail_seconds = max(3.0, self.base_tail_seconds * 0.5)
            self.interims_paused = q_final_len >= self.final_hi
            
        else:
            # Normal: no throttling
            self.current_cooldown_ms = self.base_cooldown_ms
            self.current_tail_seconds = self.base_tail_seconds
            self.interims_paused = False
        
        self.last_update_time = current_time
        
        return {
            "level": self.level,
            "cooldown_ms": self.current_cooldown_ms,
            "tail_seconds": self.current_tail_seconds,
            "interims_paused": self.interims_paused,
            "q_final_len": q_final_len,
            "q_interim_len": q_interim_len,
        }
    
    def get_burst_limits(self) -> tuple[int, int]:
        """
        Get current burst limits based on backpressure level.
        
        Returns:
            Tuple of (f_final_burst, f_interim_burst)
        """
        # Finals always get their burst limit
        f_final_burst = config.f_final_burst
        
        # Interims can be throttled or paused
        if self.interims_paused:
            f_interim_burst = 0
        elif self.level == BackpressureLevel.CRITICAL:
            f_interim_burst = max(1, config.f_interim_burst // 3)
        elif self.level == BackpressureLevel.HIGH:
            f_interim_burst = max(1, config.f_interim_burst // 2)
        else:
            f_interim_burst = config.f_interim_burst
        
        return f_final_burst, f_interim_burst
    
    def get_state(self) -> Dict[str, Any]:
        """Get current backpressure state."""
        return {
            "level": self.level.value,
            "cooldown_ms": self.current_cooldown_ms,
            "tail_s": self.current_tail_seconds,
            "interims_paused": self.interims_paused,
        }
    
    def should_allow_interim(self, conn_id: str, last_interim_time: float) -> bool:
        """
        Check if connection should be allowed to enqueue interim.
        
        Args:
            conn_id: Connection ID
            last_interim_time: Timestamp of last interim enqueue for this connection
            
        Returns:
            True if interim should be allowed
        """
        if self.interims_paused:
            return False
        
        current_time = time.time()
        elapsed_ms = (current_time - last_interim_time) * 1000
        
        return elapsed_ms >= self.current_cooldown_ms


class ConnectionThrottle:
    """
    Per-connection throttle to enforce interim rate limits.
    """
    
    def __init__(self, conn_id: str, base_cooldown_ms: int = 220):
        """
        Initialize connection throttle.
        
        Args:
            conn_id: Connection identifier
            base_cooldown_ms: Base cooldown between interims
        """
        self.conn_id = conn_id
        self.base_cooldown_ms = base_cooldown_ms
        
        # Add per-connection jitter to avoid synchronized bursts
        import random
        self.jitter_ms = random.uniform(-30, 30)
        self.effective_cooldown_ms = base_cooldown_ms + self.jitter_ms
        
        self.last_interim_time = 0.0
        
        logger.debug(f"Initialized throttle for {conn_id}: cooldown={self.effective_cooldown_ms:.1f}ms")
    
    def should_allow_interim(self, current_cooldown_ms: int) -> bool:
        """
        Check if interim should be allowed based on cooldown.
        
        Args:
            current_cooldown_ms: Current cooldown from backpressure manager
            
        Returns:
            True if interim should be allowed
        """
        current_time = time.time()
        elapsed_ms = (current_time - self.last_interim_time) * 1000
        
        # Use the higher of base + jitter or current cooldown
        effective_cooldown = max(self.effective_cooldown_ms, current_cooldown_ms)
        
        return elapsed_ms >= effective_cooldown
    
    def mark_interim_sent(self):
        """Mark that an interim was sent."""
        self.last_interim_time = time.time()
