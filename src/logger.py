import logging
from .config import settings

# Configure root logger based on settings
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger("rag_copilot")

def get_logger(name: str = __name__) -> logging.Logger:
    """Return a logger with the given name.

    All modules should call ``get_logger(__name__)`` to obtain a
    module‑specific logger that inherits the root configuration.
    """
    return logging.getLogger(name)
