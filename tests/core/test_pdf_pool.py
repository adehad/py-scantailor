"""Ordered, cancellable pool runs for PDF page jobs."""

import os
import threading
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor
from typing import Any

import pytest

from scantailor.core.batch import spawn_pool
from scantailor.core.pdf_pool import PdfCancelled, map_pool_progress


def square(job: list[int]) -> int:
    return job[0] ** 2


def crash_the_worker(_job: list[int]) -> int:
    os._exit(1)


def fail_on_three(job: list[int]) -> int:
    if job[0] == 3:
        raise ValueError("three is unlucky")
    return job[0]


@pytest.fixture
def pool() -> Iterator[ProcessPoolExecutor]:
    with spawn_pool(2) as executor:
        yield executor


def test_the_same_job_object_twice_gets_one_result_per_position(
    pool: ProcessPoolExecutor,
) -> None:
    shared = [4]
    jobs = [shared, [5], shared]

    assert map_pool_progress(square, jobs, pool, 2) == [16, 25, 16]


def test_results_follow_job_order_and_each_is_reported_once(
    pool: ProcessPoolExecutor,
) -> None:
    jobs = [[n] for n in range(6)]
    done: list[int] = []

    results = map_pool_progress(
        square, jobs, pool, 2, on_done=lambda index, _job, _result: done.append(index)
    )

    assert results == [n * n for n in range(6)]
    assert sorted(done) == list(range(6))


def test_the_first_failed_job_in_order_is_raised(pool: ProcessPoolExecutor) -> None:
    jobs = [[n] for n in range(5)]

    with pytest.raises(ValueError, match="unlucky"):
        map_pool_progress(fail_on_three, jobs, pool, 2)


def test_a_broken_pool_is_reported_in_plain_words(pool: ProcessPoolExecutor) -> None:
    with pytest.raises(RuntimeError, match="worker process stopped unexpectedly"):
        map_pool_progress(crash_the_worker, [[1], [2]], pool, 2)


@pytest.mark.parametrize(
    ("name_job", "running"),
    [
        (None, "PDF page 1, PDF page 2"),
        (lambda index, job: f"job {index} of {job[0]}", "job 0 of 1, job 1 of 2"),
    ],
)
def test_a_broken_pool_names_the_running_jobs_in_the_log(
    pool: ProcessPoolExecutor,
    log_records: list[dict[str, Any]],
    name_job: Any,
    running: str,
) -> None:
    with pytest.raises(RuntimeError):
        map_pool_progress(crash_the_worker, [[1], [2]], pool, 2, name_job=name_job)

    died = [r for r in log_records if r["message"].startswith("worker process died")]
    assert [r["message"] for r in died] == [
        f"worker process died; running: {running}; 0 more not started"
    ]


def test_cancel_before_any_job_raises_pdf_cancelled(pool: ProcessPoolExecutor) -> None:
    cancel = threading.Event()
    cancel.set()

    with pytest.raises(PdfCancelled):
        map_pool_progress(square, [[1], [2]], pool, 2, cancel=cancel)
