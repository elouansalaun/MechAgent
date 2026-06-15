"""Logging setup shared by every MechAgent module."""

from __future__ import annotations

import logging
from pathlib import Path

LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def setup_logging(level: str = "INFO", log_file: Path | None = None) -> logging.Logger:
    """Configure the `mechagent` logger with a console handler and an optional file handler.

    Calling it twice replaces the handlers instead of stacking them.
    """
    logger = logging.getLogger("mechagent")
    logger.setLevel(level)
    logger.propagate = False

    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)

    formatter = logging.Formatter(LOG_FORMAT)

    console = logging.StreamHandler()
    console.setFormatter(formatter)
    logger.addHandler(console)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str) -> logging.Logger:
    """Return a child of the `mechagent` logger, e.g. get_logger(__name__)."""
    if not name.startswith("mechagent"):
        name = f"mechagent.{name}"
    return logging.getLogger(name)
