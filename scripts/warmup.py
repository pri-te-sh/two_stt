"""Warmup script to pre-load ASR models."""
import sys
import time
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from asr.models import ASRModels
from util.logging import setup_logging, get_logger
from util.config import config

setup_logging()
logger = get_logger(__name__)


def warmup():
    """Warmup ASR models by loading and running test inference."""
    logger.info("Starting model warmup...")
    start_time = time.time()
    
    try:
        # Load models
        models = ASRModels()
        
        # Run warmup
        models.warmup()
        
        elapsed = time.time() - start_time
        logger.info(f"Model warmup completed in {elapsed:.2f}s")
        
        return 0
    
    except Exception as e:
        logger.error(f"Model warmup failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(warmup())
