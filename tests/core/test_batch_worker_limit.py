"""The live worker limit caps how many jobs run at once and can change mid-run."""

import threading
import time
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from typing import cast

from scantailor.core.batch import run_pool

_POOL_SIZE = 4


class _ConcurrencyProbe:
    """Job function that records the most jobs it saw running together."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._running = 0
        self.peak = 0

    def __call__(self, job: int) -> int:
        with self._lock:
            self._running += 1
            self.peak = max(self.peak, self._running)
        time.sleep(0.02)
        with self._lock:
            self._running -= 1
        return job


def _run(
    probe: _ConcurrencyProbe,
    jobs: list[int],
    worker_limit: Callable[[], int | None] | None,
    after_each: Callable[[int], None] | None = None,
) -> list[int]:
    """Run ``jobs`` on a thread pool, which run_pool drives like a process pool."""
    finished: list[int] = []

    def record(job: int, result: int | None, error: BaseException | None) -> None:
        assert error is None
        finished.append(job)
        if after_each is not None:
            after_each(job)

    with ThreadPoolExecutor(max_workers=_POOL_SIZE) as threads:
        run_pool(
            probe,
            jobs,
            record,
            max_workers=_POOL_SIZE,
            executor=cast(ProcessPoolExecutor, threads),
            worker_limit=worker_limit,
        )
    return finished


def test_a_limit_of_one_runs_the_jobs_one_at_a_time_in_order():
    probe = _ConcurrencyProbe()

    finished = _run(probe, list(range(6)), lambda: 1)

    assert finished == list(range(6))
    assert probe.peak == 1


def test_no_limit_fills_the_pool():
    probe = _ConcurrencyProbe()

    _run(probe, list(range(8)), None)

    assert probe.peak == _POOL_SIZE


def test_a_limit_that_answers_none_fills_the_pool():
    probe = _ConcurrencyProbe()

    _run(probe, list(range(8)), lambda: None)

    assert probe.peak == _POOL_SIZE


def test_a_limit_above_the_pool_size_is_clamped_to_the_pool_size():
    probe = _ConcurrencyProbe()

    _run(probe, list(range(8)), lambda: 64)

    assert probe.peak == _POOL_SIZE


def test_a_limit_below_one_still_runs_every_job_one_at_a_time():
    probe = _ConcurrencyProbe()

    finished = _run(probe, list(range(3)), lambda: 0)

    assert finished == [0, 1, 2]
    assert probe.peak == 1


def test_raising_the_limit_mid_run_starts_more_jobs_at_once():
    probe = _ConcurrencyProbe()
    limit = {"value": 1}

    def raise_after_first(_job: int) -> None:
        limit["value"] = 3

    finished = _run(probe, list(range(10)), lambda: limit["value"], raise_after_first)

    assert sorted(finished) == list(range(10))
    assert probe.peak == 3
