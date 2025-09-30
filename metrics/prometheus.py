"""Prometheus metrics for observability."""
from prometheus_client import Counter, Gauge, Histogram, Info
from typing import Optional
from util.logging import get_logger

logger = get_logger(__name__)

# Connection metrics
active_connections = Gauge(
    "stt_active_connections",
    "Number of active WebSocket connections"
)

total_connections = Counter(
    "stt_total_connections",
    "Total number of WebSocket connections"
)

# Queue metrics
queue_depth = Gauge(
    "stt_queue_depth",
    "Current queue depth",
    ["queue_type"]  # final or interim
)

queue_job_age_seconds = Gauge(
    "stt_queue_oldest_job_age_seconds",
    "Age of oldest job in queue",
    ["queue_type"]
)

# Job metrics
jobs_enqueued = Counter(
    "stt_jobs_enqueued_total",
    "Total jobs enqueued",
    ["job_type"]
)

jobs_processed = Counter(
    "stt_jobs_processed_total",
    "Total jobs processed",
    ["job_type", "status"]  # status: success or error
)

jobs_coalesced = Counter(
    "stt_jobs_coalesced_total",
    "Total interim jobs coalesced"
)

# Latency metrics
decode_duration_seconds = Histogram(
    "stt_decode_duration_seconds",
    "Decode duration in seconds",
    ["job_type"],
    buckets=[0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0]
)

job_wait_duration_seconds = Histogram(
    "stt_job_wait_duration_seconds",
    "Job wait time in queue",
    ["job_type"],
    buckets=[0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0]
)

utterance_finalize_duration_seconds = Histogram(
    "stt_utterance_finalize_duration_seconds",
    "Time from VAD end to final transcription sent",
    buckets=[0.1, 0.3, 0.5, 0.7, 0.9, 1.2, 1.5, 2.0, 3.0]
)

# Backpressure metrics
backpressure_level = Gauge(
    "stt_backpressure_level",
    "Current backpressure level (0=normal, 1=high, 2=critical)"
)

interim_cooldown_ms = Gauge(
    "stt_interim_cooldown_ms",
    "Current interim cooldown in milliseconds"
)

tail_window_seconds = Gauge(
    "stt_tail_window_seconds",
    "Current tail window size in seconds"
)

interims_paused = Gauge(
    "stt_interims_paused",
    "Whether interims are currently paused (0=no, 1=yes)"
)

# GPU metrics (if available)
gpu_memory_used_bytes = Gauge(
    "stt_gpu_memory_used_bytes",
    "GPU memory used in bytes",
    ["gpu_id"]
)

gpu_utilization_percent = Gauge(
    "stt_gpu_utilization_percent",
    "GPU utilization percentage",
    ["gpu_id"]
)

gpu_temperature_celsius = Gauge(
    "stt_gpu_temperature_celsius",
    "GPU temperature in Celsius",
    ["gpu_id"]
)

# System info
system_info = Info(
    "stt_system",
    "System information"
)


def update_gpu_metrics():
    """Update GPU metrics using NVML."""
    try:
        import pynvml
        
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()
        
        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            
            # Memory
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            gpu_memory_used_bytes.labels(gpu_id=str(i)).set(mem_info.used)
            
            # Utilization
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            gpu_utilization_percent.labels(gpu_id=str(i)).set(util.gpu)
            
            # Temperature
            temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            gpu_temperature_celsius.labels(gpu_id=str(i)).set(temp)
        
    except Exception as e:
        logger.debug(f"Failed to update GPU metrics: {e}")


def record_decode_complete(job_type: str, decode_time: float, wait_time: float, success: bool = True):
    """Record decode completion metrics."""
    status = "success" if success else "error"
    jobs_processed.labels(job_type=job_type, status=status).inc()
    decode_duration_seconds.labels(job_type=job_type).observe(decode_time)
    job_wait_duration_seconds.labels(job_type=job_type).observe(wait_time)


def update_backpressure_metrics(level_value: int, cooldown: int, tail: float, paused: bool):
    """Update backpressure metrics."""
    backpressure_level.set(level_value)
    interim_cooldown_ms.set(cooldown)
    tail_window_seconds.set(tail)
    interims_paused.set(1 if paused else 0)


logger.info("Prometheus metrics initialized")
