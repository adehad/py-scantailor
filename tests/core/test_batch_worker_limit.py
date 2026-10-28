"""The live worker limit caps how many jobs run at once and can change mid-run."""

import threading
from collections.abc import Callable, Iterator
from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor
from typing import Any, cast

import pytest

from scantailor.core import batch
from scantailor.core.batch import run_pool

_POOL_SIZE = 4
_HANG_GUARD_S = 30


class _ConcurrencyProbe:
    """Job function whose ``peak`` is the most jobs the scheduler let run at once.

    A job holds until the scheduler has blocked waiting for a result and every
    job it has in flight is running, then all of them are counted together.
    So ``peak`` depends on the scheduler's decisions alone, not on how the
    threads happen to be timed.
    """

    def __init__(self) -> None:
        self._state = threading.Condition()
        self._running = 0
        self._in_flight = 0
        self._scheduler_waiting = False
        self.peak = 0

    def submitted(self) -> None:
        with self._state:
            self._in_flight += 1
            self._scheduler_waiting = False

    def scheduler_waits(self) -> None:
        with self._state:
            self._scheduler_waiting = True
            self._state.notify_all()

    def __call__(self, job: int) -> int:
        with self._state:
            self._running += 1
            self.peak = max(self.peak, self._running)
            self._state.notify_all()
            assert self._state.wait_for(
                lambda: self._scheduler_waiting and self._running == self._in_flight,
                timeout=_HANG_GUARD_S,
            )
            self._running -= 1
            self._in_flight -= 1
            self._state.notify_all()
        return job


class _ProbedThreads(ThreadPoolExecutor):
    def __init__(self, probe: _ConcurrencyProbe) -> None:
        super().__init__(max_workers=_POOL_SIZE)
        self._probe = probe

    def submit(self, fn: Callable[..., Any], /, *args: Any, **kwargs: Any) -> Future:
        self._probe.submitted()
        return super().submit(fn, *args, **kwargs)


@pytest.fixture
def probe(monkeypatch: pytest.MonkeyPatch) -> Iterator[_ConcurrencyProbe]:
    probe = _ConcurrencyProbe()
    real_wait = batch.wait

    def waiting(*args: Any, **kwargs: Any) -> Any:
        probe.scheduler_waits()
        return real_wait(*args, **kwargs)

    monkeypatch.setattr(batch, "wait", waiting)
    yield probe


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

    with _ProbedThreads(probe) as threads:
        run_pool(
            probe,
            jobs,
            record,
            max_workers=_POOL_SIZE,
            executor=cast(ProcessPoolExecutor, threads),
            worker_limit=worker_limit,
        )
    return finished


def test_a_limit_of_one_runs_the_jobs_one_at_a_time_in_order(probe: _ConcurrencyProbe):
    finished = _run(probe, list(range(6)), lambda: 1)

    assert finished == list(range(6))
    assert probe.peak == 1


def test_no_limit_fills_the_pool(probe: _ConcurrencyProbe):
    _run(probe, list(range(8)), None)

    assert probe.peak == _POOL_SIZE


def test_a_limit_that_answers_none_fills_the_pool(probe: _ConcurrencyProbe):
    _run(probe, list(range(8)), lambda: None)

    assert probe.peak == _POOL_SIZE


def test_a_limit_above_the_pool_size_is_clamped_to_the_pool_size(
    probe: _ConcurrencyProbe,
):
    _run(probe, list(range(8)), lambda: 64)

    assert probe.peak == _POOL_SIZE


def test_a_limit_below_one_still_runs_every_job_one_at_a_time(probe: _ConcurrencyProbe):
    finished = _run(probe, list(range(3)), lambda: 0)

    assert finished == [0, 1, 2]
    assert probe.peak == 1


def test_raising_the_limit_mid_run_starts_more_jobs_at_once(probe: _ConcurrencyProbe):
    limit = {"value": 1}

    def raise_after_first(_job: int) -> None:
        limit["value"] = 3

    finished = _run(probe, list(range(10)), lambda: limit["value"], raise_after_first)

    assert sorted(finished) == list(range(10))
    assert probe.peak == 3
