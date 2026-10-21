"""Jobs for the worker logging tests, run in spawned workers.

Kept apart from the test module so a worker imports only loguru, not the
fixtures and Qt the tests pull in.
"""

import faulthandler
import os

from loguru import logger


def log_a_page(name: str) -> int:
    """Log one line under a page context and one error with a traceback."""
    with logger.contextualize(page=name, stage="output"):
        logger.info("rendering")
        try:
            raise ValueError("content area exceeds the image")
        except ValueError:
            logger.exception("render failed")
    return os.getpid()


def crash_natively(name: str) -> None:
    """Log a line under a page context, then die with a segmentation fault."""
    with logger.contextualize(page=name, stage="output"):
        logger.warning("about to crash natively")
        faulthandler._sigsegv()
