"""Utility modules for the BOPET research framework."""

import logging


def get_logger(name: str = "bopet") -> logging.Logger:
    """Return a configured logger.

    Falls back to the root configuration if logging.basicConfig has not
    been called by the entry point.
    """
    return logging.getLogger(name)


# Module-level default logger, importable as `from src.utils import logger`
logger = get_logger()


__all__ = ["get_logger", "logger"]