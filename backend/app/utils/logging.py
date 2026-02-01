"""
Logging Configuration.

Sets up structured logging for the application.
"""

import logging
import sys
from typing import Any

from app.config import get_settings


def setup_logging() -> None:
    """
    Configure application logging.

    Sets up:
    - Console handler with colored output
    - Log level from settings
    - Structured format with timestamps
    """
    settings = get_settings()

    # Determine log level
    level = getattr(logging, settings.log_level.upper(), logging.INFO)

    # Configure root logger
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
        ],
    )

    # Adjust third-party loggers
    logging.getLogger("uvicorn").setLevel(logging.INFO)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)

    # Log startup info
    logger = logging.getLogger(__name__)
    logger.info(f"Logging configured: level={settings.log_level}")


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger for the given module.

    Args:
        name: Logger name (usually __name__).

    Returns:
        Configured logger instance.
    """
    return logging.getLogger(name)


class RequestLogger:
    """
    Request logging middleware helper.

    Logs request/response details for debugging.
    """

    def __init__(self, logger: logging.Logger):
        self.logger = logger

    def log_request(
        self,
        method: str,
        path: str,
        status_code: int,
        duration_ms: float,
        extra: dict[str, Any] | None = None,
    ) -> None:
        """
        Log a request/response.

        Args:
            method: HTTP method.
            path: Request path.
            status_code: Response status code.
            duration_ms: Request duration in milliseconds.
            extra: Additional context.
        """
        level = logging.INFO if status_code < 400 else logging.WARNING

        message = f"{method} {path} - {status_code} ({duration_ms:.1f}ms)"

        if extra:
            message += f" | {extra}"

        self.logger.log(level, message)
