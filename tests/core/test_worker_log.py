"""Pool workers log into the session of the process that started them."""

import copy
import multiprocessing
import os
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path
from typing import Any

import pytest
from loguru import logger

from scantailor.core import worker_log
from scantailor.core.worker_log import WorkerLogging, start_worker_logging
from tests.conftest import quiet_loguru
from tests.core import worker_log_jobs


@pytest.fixture
def bridge(tmp_path: Path) -> Iterator[WorkerLogging]:
    """Forward worker lines into this process, with crash files in tmp_path."""
    logger.remove()
    worker_logger = copy.deepcopy(logger)
    quiet_loguru()
    stop = worker_log.start_bridge(worker_logger, tmp_path)
    forwarded = worker_log.worker_logging()
    assert forwarded is not None
    yield forwarded
    stop()


@pytest.fixture
def parent_records() -> Iterator[list[dict[str, Any]]]:
    """Capture the records this process logs, with their origin."""
    records: list[dict[str, Any]] = []

    def keep(message: Any) -> None:
        record = message.record
        records.append(
            {
                "level": record["level"].name,
                "message": record["message"],
                "extra": dict(record["extra"]),
                "process": record["process"].id,
                "name": record["name"],
                "exception": record["exception"],
            }
        )

    handler_id = logger.add(keep, level="DEBUG")
    yield records
    logger.remove(handler_id)


def _one_worker(forwarded: WorkerLogging) -> ProcessPoolExecutor:
    return ProcessPoolExecutor(
        max_workers=1,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=start_worker_logging,
        initargs=(forwarded,),
    )


def _drain(forwarded: WorkerLogging) -> None:
    forwarded.logger.complete()


def test_a_spawned_worker_logs_into_the_parent_with_its_page_and_process(
    bridge: WorkerLogging, parent_records: list[dict[str, Any]]
) -> None:
    with _one_worker(bridge) as pool:
        worker_pid = pool.submit(worker_log_jobs.log_a_page, "scan07.tif").result()
    _drain(bridge)

    started = [
        r for r in parent_records if r["message"] == f"worker {worker_pid} started"
    ]
    assert started[0]["level"] == "DEBUG"
    rendering = next(r for r in parent_records if r["message"] == "rendering")
    assert rendering["extra"] == {"page": "scan07.tif", "stage": "output"}
    assert rendering["process"] == worker_pid != os.getpid()
    assert rendering["name"] == "tests.core.worker_log_jobs"
    failed = next(r for r in parent_records if r["message"] == "render failed")
    assert failed["level"] == "ERROR"
    assert isinstance(failed["exception"].value, ValueError)


def test_a_worker_that_exits_cleanly_leaves_no_crash_file(
    bridge: WorkerLogging, tmp_path: Path
) -> None:
    with _one_worker(bridge) as pool:
        pool.submit(worker_log_jobs.log_a_page, "scan01.tif").result()

    assert list(tmp_path.glob("crash-worker-*.log")) == []


def test_a_native_crash_keeps_the_last_line_and_writes_the_crash_file(
    bridge: WorkerLogging, parent_records: list[dict[str, Any]], tmp_path: Path
) -> None:
    with _one_worker(bridge) as pool:
        crashed = pool.submit(worker_log_jobs.crash_natively, "scan08.tif")
        with pytest.raises(BrokenProcessPool):
            crashed.result()
    _drain(bridge)

    last = parent_records[-1]
    assert last["message"] == "about to crash natively"
    assert last["extra"]["page"] == "scan08.tif"
    crash_files = list(tmp_path.glob("crash-worker-*.log"))
    assert [path.name.endswith(f"-{last['process']}.log") for path in crash_files] == [
        True
    ]
    assert "Fatal Python error" in crash_files[0].read_text(encoding="utf-8")


def test_the_crash_file_is_named_after_the_worker_process(tmp_path: Path) -> None:
    path = worker_log.worker_crash_file(tmp_path)

    assert path.parent == tmp_path
    assert path.name.startswith("crash-worker-")
    assert path.name.endswith(f"-{os.getpid()}.log")


def test_after_the_bridge_stops_workers_get_no_logger(tmp_path: Path) -> None:
    logger.remove()
    worker_logger = copy.deepcopy(logger)
    quiet_loguru()

    stop = worker_log.start_bridge(worker_logger, tmp_path)
    stop()

    assert worker_log.worker_logging() is None
