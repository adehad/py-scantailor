"""A batch logs its start, its end with counts and timings, and each failure."""

from collections.abc import Iterator
from concurrent.futures import Future
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path
from typing import Any

import pytest
from loguru import logger

from scantailor.core import batch
from scantailor.core.batch import BatchResult, RenderInfo, run_batch
from scantailor.core.models import Dpi, ImageId, PageId, SubPage
from scantailor.core.stage_sequence import StageSequence
from scantailor.core.transformation import ImageTransformation, Rect
from scantailor.core.worker_log import WorkerLogging


def _page(name: str) -> PageId:
    return PageId(image_id=ImageId(file_path=Path("books") / name))


@pytest.fixture
def records_with_exceptions() -> Iterator[list[dict[str, Any]]]:
    """Capture loguru records with the exception each one carries."""
    records: list[dict[str, Any]] = []

    def keep(message: Any) -> None:
        record = message.record
        records.append(
            {
                "level": record["level"].name,
                "message": record["message"],
                "extra": dict(record["extra"]),
                "exception": record["exception"],
            }
        )

    handler_id = logger.add(keep, level="DEBUG")
    yield records
    logger.remove(handler_id)


def _info(seconds: float) -> RenderInfo:
    geometry = ImageTransformation(Rect(0, 0, 100, 100), Dpi.uniform(300))
    return RenderInfo(geometry, Dpi.uniform(300), False, seconds=seconds)


def test_the_batch_summary_counts_outcomes_and_names_the_slowest_page() -> None:
    result = BatchResult(
        render_info={
            _page("a.tif"): _info(1.0),
            _page("b.tif"): _info(9.4),
            _page("c.tif"): _info(2.0),
        },
        failed={_page("d.tif"): "ValueError: bad"},
    )
    result.detected = [*result.render_info]
    result.written = dict.fromkeys(result.render_info, Path("out.tif"))

    summary = batch._batch_summary(result, 377.0)

    assert summary == (
        "3 detected, 3 written, 1 failed, 0 cancelled in 377.0 s "
        "(median page 2.0 s, slowest b.tif 9.4 s)"
    )


def test_the_batch_summary_counts_kept_pages_apart_from_written_ones() -> None:
    kept = _page("a.tif")
    result = BatchResult(
        detected=[kept, _page("b.tif")],
        written={kept: Path("a.tif"), _page("b.tif"): Path("b.tif")},
        reused=[kept],
        render_info={_page("b.tif"): _info(3.0)},
    )

    summary = batch._batch_summary(result, 10.0)

    assert summary.startswith("2 detected, 1 written, 1 kept, 0 failed")


def test_a_failed_page_is_logged_once_as_an_error_with_its_traceback(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    records_with_exceptions: list[dict[str, Any]],
) -> None:
    page = PageId(
        image_id=ImageId(file_path=tmp_path / "scan07.tif"), sub_page=SubPage.LEFT_PAGE
    )
    try:
        raise ValueError("content area exceeds the image")
    except ValueError as cause:
        error = RuntimeError("worker failed")
        error.__cause__ = cause

    def failing_pool(function: Any, jobs: Any, on_outcome: Any, **kwargs: Any) -> list:
        for job in jobs:
            on_outcome(job, None, error)
        return []

    monkeypatch.setattr(batch, "run_pool", failing_pool)

    result = run_batch(
        StageSequence(),
        [page],
        tmp_path / "out",
        executor=object(),  # type: ignore[arg-type]
    )

    assert page in result.failed
    errors = [r for r in records_with_exceptions if r["level"] == "ERROR"]
    assert len(errors) == 1
    assert errors[0]["message"].startswith("geometry pass failed")
    assert errors[0]["extra"] == {"page": "scan07.tif/left_page", "stage": "batch"}
    assert errors[0]["exception"].value is error
    end = [r for r in records_with_exceptions if r["message"].startswith("batch end")]
    assert end[0]["message"].startswith(
        "batch end: 0 detected, 0 written, 1 failed, 0 cancelled"
    )


class _PoolThatBreaks:
    """Takes jobs until ``workers`` run, then fails them as a dead worker would."""

    def __init__(self, workers: int) -> None:
        self.workers = workers
        self.running: list[Future[Any]] = []
        self.broken = False

    def submit(self, function: Any, job: Any) -> Future[Any]:
        if self.broken:
            raise BrokenProcessPool("A process in the process pool was terminated")
        future: Future[Any] = Future()
        self.running.append(future)
        if len(self.running) == self.workers:
            self.broken = True
            for running in self.running:
                running.set_exception(BrokenProcessPool("terminated abruptly"))
        return future


def _break_the_pool(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    pages = [
        _page("scan07.tif"),
        PageId(
            image_id=ImageId(file_path=Path("books") / "scan08.tif"),
            sub_page=SubPage.LEFT_PAGE,
        ),
        *(_page(f"scan{index:02}.tif") for index in range(9, 12)),
    ]
    failed: list[PageId] = []

    batch.run_pool(
        lambda page: page,
        pages,
        lambda page, result, error: failed.append(page),
        max_workers=2,
        executor=_PoolThatBreaks(2),  # type: ignore[arg-type]
        name_job=batch.page_name,
    )

    assert sorted(map(batch.page_name, failed)) == sorted(map(batch.page_name, pages))
    return [r for r in records if r["message"].startswith("worker process died")]


def test_a_broken_pool_names_the_running_pages_apart_from_those_never_started(
    records_with_exceptions: list[dict[str, Any]],
) -> None:
    died = _break_the_pool(records_with_exceptions)

    assert len(died) == 1
    assert died[0]["level"] == "ERROR"
    assert died[0]["message"] == (
        "worker process died; running: scan07.tif, scan08.tif/left_page; "
        "3 more not started"
    )
    assert died[0]["extra"]["stage"] == "batch"


def test_a_broken_pool_points_at_the_crash_files_when_workers_write_them(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    records_with_exceptions: list[dict[str, Any]],
) -> None:
    forwarded = WorkerLogging(logger, tmp_path)  # type: ignore[arg-type]
    monkeypatch.setattr(batch, "worker_logging", lambda: forwarded)

    died = _break_the_pool(records_with_exceptions)

    assert died[0]["message"].endswith("; see crash-worker-*.log")


def test_a_pool_without_job_names_logs_no_line_when_it_breaks(
    records_with_exceptions: list[dict[str, Any]],
) -> None:
    batch.run_pool(
        lambda job: job,
        [1, 2, 3],
        lambda job, result, error: None,
        max_workers=2,
        executor=_PoolThatBreaks(2),  # type: ignore[arg-type]
    )

    assert not any(
        r["message"].startswith("worker process died") for r in records_with_exceptions
    )


@pytest.mark.parametrize("phase", ["geometry pass", "render"])
def test_pages_failed_by_a_broken_pool_log_one_line_each_and_no_traceback(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    records_with_exceptions: list[dict[str, Any]],
    phase: str,
) -> None:
    pages = [_page(f"scan{index:02}.tif") for index in range(1, 6)]
    if phase == "render":
        monkeypatch.setattr(
            batch, "_detect_geometry", lambda stages, pages, *args: list(pages)
        )

    result = run_batch(
        StageSequence(),
        pages,
        tmp_path / "out",
        max_workers=2,
        executor=_PoolThatBreaks(2),  # type: ignore[arg-type]
    )

    assert set(result.failed) == set(pages)
    errors = [r for r in records_with_exceptions if r["level"] == "ERROR"]
    page_lines = [r for r in errors if r["message"].startswith(f"{phase} failed")]
    assert len(page_lines) == len(pages)
    assert all("BrokenProcessPool" in r["message"] for r in page_lines)
    assert [r["message"] for r in errors if r["exception"] is not None] == []
    assert sum(r["message"].startswith("worker process died;") for r in errors) == 1
