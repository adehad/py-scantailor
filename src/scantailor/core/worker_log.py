"""Carry pool workers' log lines to the process that started them.

A spawned worker starts with a fresh loguru and only a stderr sink, which the
window build lacks, so its lines would never reach the session log. The parent
keeps a second logger whose one enqueued sink logs each record it receives
into the parent's ``logger``. Workers receive that logger through the pool
initializer and forward their own ``logger``'s records through it.

This module imports no Qt, so workers stay free of it.
"""

import atexit
import faulthandler
import multiprocessing
import os
import sys
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Final, TextIO

from loguru import logger

if TYPE_CHECKING:
    from loguru import Logger, Message

# The log format reads these keys, so a record logged without them would raise
# KeyError in the formatter.
EXTRA_DEFAULTS: Final = {"page": "-", "stage": "-"}


@dataclass(frozen=True)
class WorkerLogging:
    """What a pool worker needs to log into the session that started it.

    Attributes:
        logger (Logger): A logger whose one sink queues each record back to the
            session's process, which logs it there. It pickles into the worker.
        folder (Path | None): Where the worker writes its crash file; None
            when the session writes no log file.
    """

    logger: Logger
    folder: Path | None


_worker_logging: WorkerLogging | None = None


def worker_logging() -> WorkerLogging | None:
    """The logging a pool worker should use, to pass in its initializer.

    Returns:
        WorkerLogging | None: None when no session forwards worker lines.
    """
    return _worker_logging


def start_bridge(worker_logger: Logger, folder: Path | None) -> Callable[[], None]:
    """Log, in this process, every record pool workers send to ``worker_logger``.

    ``worker_logger`` must be a ``copy.deepcopy`` of ``logger`` taken while it
    had no sinks: deepcopy of a logger that holds a file sink fails with a
    RecursionError, and a logger that holds one does not pickle.

    Args:
        worker_logger (Logger): The independent logger to hand to workers.
        folder (Path | None): Where workers write their crash files.

    Returns:
        Callable[[], None]: Stops forwarding once the lines already sent are
            logged.
    """
    global _worker_logging
    sink = worker_logger.add(
        _log_worker_record,
        level="DEBUG",
        format="{message}",
        enqueue=True,
        context=multiprocessing.get_context("spawn"),
        backtrace=False,
        diagnose=False,
    )
    _worker_logging = WorkerLogging(worker_logger, folder)

    def stop() -> None:
        global _worker_logging
        _worker_logging = None
        # Removing an enqueued sink waits until its queue is drained.
        worker_logger.remove(sink)

    return stop


def _log_worker_record(message: Message) -> None:
    record = message.record
    logger.patch(lambda own: own.update(record)).log(
        record["level"].name, record["message"]
    )


def start_worker_logging(worker: WorkerLogging | None) -> None:
    """Send this pool worker's lines to the parent and a native crash to a file.

    Run it first in the pool initializer. The worker's own ``logger`` then
    reaches the parent's sinks with the worker's process, module, line, page
    and stage. Each line is in the pipe before the logging call returns, so the
    last line before a native crash is kept. Without a session the worker
    writes warnings and errors to stderr.

    Args:
        worker (WorkerLogging | None): From ``worker_logging`` in the parent.
    """
    logger.remove()
    logger.configure(extra=dict(EXTRA_DEFAULTS))
    if worker is None:
        if sys.stderr is not None:
            logger.add(sys.stderr, level="WARNING", backtrace=False, diagnose=False)
        return
    parent = worker.logger

    def forward(message: Message) -> None:
        record = message.record
        parent.patch(lambda sent: sent.update(record)).log(
            record["level"].name, record["message"]
        )

    logger.add(
        forward, level="DEBUG", format="{message}", backtrace=False, diagnose=False
    )
    if worker.folder is not None:
        _enable_crash_file(worker.folder)
    logger.debug("worker {} started", os.getpid())


def worker_crash_file(folder: Path) -> Path:
    """Name the crash file of this worker process.

    Args:
        folder (Path): The session's log folder.

    Returns:
        Path: ``crash-worker-YYYYMMDD-HHMMSS-<pid>.log`` in ``folder``.
    """
    return folder / f"crash-worker-{datetime.now():%Y%m%d-%H%M%S}-{os.getpid()}.log"


def _enable_crash_file(folder: Path) -> None:
    crash_file = worker_crash_file(folder)
    try:
        stream = crash_file.open("w", encoding="utf-8")
    except OSError as error:
        logger.warning("Could not write the crash file {}: {}", crash_file, error)
        return
    # Without file= faulthandler writes to sys.stderr, which a worker of the
    # window build does not have.
    faulthandler.enable(file=stream, all_threads=True)
    # A spawned worker ends through sys.exit, which runs atexit. A worker the
    # pool terminates leaves its empty file to prune_logs.
    atexit.register(_close_crash_file, stream, crash_file)


def _close_crash_file(stream: TextIO, crash_file: Path) -> None:
    faulthandler.disable()
    stream.close()
    try:
        if crash_file.stat().st_size == 0:
            crash_file.unlink()
    except OSError:
        pass
