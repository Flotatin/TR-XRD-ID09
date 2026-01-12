"""Centralised logging configuration for the DRX application."""

from __future__ import annotations

import logging
from typing import Optional

_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


def setup_logging(debug: bool = False) -> None:
    """Configure the root logger for the application.

    Parameters
    ----------
    debug:
        When ``True`` the root logger is configured at :data:`logging.DEBUG`,
        otherwise :data:`logging.INFO`.
    """

    level = logging.DEBUG if debug else logging.INFO

    root_logger = logging.getLogger()
    handler: Optional[logging.Handler] = None

    for existing in root_logger.handlers:
        if isinstance(existing, logging.StreamHandler):
            handler = existing
            break

    if handler is None:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(_LOG_FORMAT))
        root_logger.addHandler(handler)
    else:
        handler.setFormatter(logging.Formatter(_LOG_FORMAT))

    for existing in root_logger.handlers:
        existing.setLevel(level)

    root_logger.setLevel(level)
    logging.captureWarnings(True)

