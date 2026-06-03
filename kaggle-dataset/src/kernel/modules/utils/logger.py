"""
Structured logging for Atlético Intelligence.
"""

import logging
import json
from datetime import datetime
from typing import Optional


class JSONFormatter(logging.Formatter):
    """Format logs as JSON for easier parsing and analysis."""

    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "message": record.getMessage(),
        }

        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_data)


def get_logger(name: str, level: str = "INFO", json_output: bool = True) -> logging.Logger:
    """
    Get a configured logger instance.

    Args:
        name: Logger name (usually __name__)
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        json_output: Whether to format logs as JSON

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level))

    # Console handler
    handler = logging.StreamHandler()
    handler.setLevel(getattr(logging, level))

    if json_output:
        formatter = JSONFormatter()
    else:
        formatter = logging.Formatter(
            fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger


def log_event(logger: logging.Logger, event: str, **kwargs):
    """
    Log a structured event with metadata.

    Args:
        logger: Logger instance
        event: Event name/description
        **kwargs: Additional metadata to include in log
    """
    log_entry = {"event": event, **kwargs}
    logger.info(json.dumps(log_entry))
