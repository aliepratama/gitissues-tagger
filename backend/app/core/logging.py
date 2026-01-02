import logging
import sys
from app.core.config import settings

def setup_logging():
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG if settings.DEBUG_MODE else logging.INFO)

    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    return logger

logger = setup_logging()
