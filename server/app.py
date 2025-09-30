"""Main FastAPI application."""
import uvicorn
from fastapi import FastAPI, WebSocket
from fastapi.responses import JSONResponse
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response
from contextlib import asynccontextmanager

from server.routes import websocket_endpoint
from runtime.state import get_runtime
from util.logging import setup_logging, get_logger
from util.config import config
import metrics.prometheus as metrics

# Setup logging first
setup_logging()
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup and shutdown."""
    # Startup
    logger.info("Starting STT server...")
    logger.info(f"Environment: {config.env}")
    logger.info(f"Bind: {config.bind_host}:{config.bind_port}")
    logger.info(f"Models: interim={config.interim_model}, final={config.final_model}")
    
    # Initialize runtime
    runtime = get_runtime()
    await runtime.startup()
    
    logger.info("STT server started successfully")
    
    # Set system info metrics
    metrics.system_info.info({
        "interim_model": config.interim_model,
        "final_model": config.final_model,
        "sample_rate": str(config.sample_rate),
    })
    
    yield
    
    # Shutdown
    logger.info("Shutting down STT server...")
    await runtime.shutdown()
    logger.info("STT server shut down")


# Create FastAPI app
app = FastAPI(
    title="Real-time STT Server",
    description="Dual-model speech-to-text server with interim and final transcriptions",
    version="1.0.0",
    lifespan=lifespan
)


@app.get("/health")
async def health():
    """Basic health check (process is alive)."""
    return {"status": "ok"}


@app.get("/ready")
async def ready():
    """Readiness check (models loaded and workers running)."""
    runtime = get_runtime()
    
    is_ready = (
        runtime.models is not None and
        runtime.worker_final is not None and
        runtime.worker_interim is not None and
        runtime.worker_final.running and
        runtime.worker_interim.running
    )
    
    if is_ready:
        return {"status": "ready"}
    else:
        return JSONResponse(
            status_code=503,
            content={"status": "not_ready"}
        )


@app.get("/status")
async def status():
    """Detailed status including queue depths and configuration."""
    runtime = get_runtime()
    
    queue_depths = runtime.scheduler.get_queue_depths()
    scheduler_stats = runtime.scheduler.get_stats()
    bp_state = runtime.backpressure.get_state()
    
    worker_stats = {}
    if runtime.worker_final:
        worker_stats["final"] = runtime.worker_final.get_stats()
    if runtime.worker_interim:
        worker_stats["interim"] = runtime.worker_interim.get_stats()
    
    return {
        "status": "running",
        "config": {
            "interim_model": config.interim_model,
            "final_model": config.final_model,
            "sample_rate": config.sample_rate,
            "base_cooldown_ms": config.interim_cooldown_ms,
            "base_tail_seconds": config.tail_seconds,
        },
        "queues": queue_depths,
        "scheduler_stats": scheduler_stats,
        "backpressure": bp_state,
        "workers": worker_stats,
        "connections": {
            "active": len(runtime.connections),
        }
    }


@app.get("/metrics")
async def metrics_endpoint():
    """Prometheus metrics endpoint."""
    if not config.enable_metrics:
        return Response(status_code=404)
    
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )


@app.websocket("/ws")
async def websocket_route(websocket: WebSocket):
    """WebSocket endpoint for audio streaming."""
    await websocket_endpoint(websocket)


def main():
    """Main entry point."""
    uvicorn.run(
        "server.app:app",
        host=config.bind_host,
        port=config.bind_port,
        loop="uvloop",
        log_config=None,  # We handle logging ourselves
        access_log=False
    )


if __name__ == "__main__":
    main()
