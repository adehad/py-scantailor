"""Run PDF page jobs on a worker pool with progress, cancel and ordered results."""

from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from threading import Event

from scantailor.core.batch import run_pool


class PdfCancelled(Exception):
    """Raised by PDF work that noticed its cancel event and stopped."""


def _call_indexed[Job, Result](
    request: tuple[Callable[[Job], Result], int, Job],
) -> tuple[int, Result]:
    function, index, job = request
    return index, function(job)


def describe_pool_error(error: BaseException) -> BaseException:
    """Return ``error``, or a clear replacement when the pool itself broke."""
    if isinstance(error, BrokenProcessPool):
        message = "A worker process stopped unexpectedly; the job could not finish"
        replacement = RuntimeError(message)
        replacement.__cause__ = error
        return replacement
    return error


def map_pool_progress[Job, Result](
    function: Callable[[Job], Result],
    jobs: Sequence[Job],
    executor: ProcessPoolExecutor,
    max_workers: int | None = None,
    *,
    cancel: Event | None = None,
    on_done: Callable[[int, Job, Result], None] | None = None,
) -> list[Result]:
    """Run ``function`` over ``jobs`` on a pool and return results in job order.

    Results are keyed by job position, so the same job object may appear twice.

    Args:
        function (Callable[[Any], Any]): Picklable module-level callable.
        jobs (Sequence[Any]): Picklable job arguments.
        executor (concurrent.futures.ProcessPoolExecutor):
            A pool from ``spawn_pool``; left running.
        max_workers (int | None): The pool's size.
        cancel (threading.Event | None): Stops further jobs from being scheduled.
        on_done (Callable[[int, Any, Any], None] | None): Called from this
            thread with each finished job's position, the job and its result.

    Returns:
        list[Any]: One result per job, in job order.

    Raises:
        PdfCancelled: If ``cancel`` was set before every job ran.
        BaseException: The error of the first failed job in job order, after
            every scheduled job has finished. A broken pool is reported as a
            ``RuntimeError`` saying so.
    """
    requests = [(function, index, job) for index, job in enumerate(jobs)]
    results: dict[int, Result] = {}
    errors: dict[int, BaseException] = {}

    def on_outcome(
        request: tuple[Callable[[Job], Result], int, Job],
        outcome: tuple[int, Result] | None,
        error: BaseException | None,
    ) -> None:
        index = request[1]
        if error is not None or outcome is None:
            errors[index] = describe_pool_error(error or RuntimeError("no result"))
            return
        results[index] = outcome[1]
        if on_done is not None:
            on_done(index, request[2], outcome[1])

    unscheduled = run_pool(
        _call_indexed,
        requests,
        on_outcome,
        max_workers=max_workers,
        cancel=cancel,
        executor=executor,
    )
    if errors:
        raise errors[min(errors)]
    if unscheduled:
        raise PdfCancelled
    return [results[index] for index in range(len(jobs))]
