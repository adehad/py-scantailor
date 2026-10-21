"""A batch logs its start, its end with counts and timings, and each failure."""

from collections.abc import Iterator
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
    result.written = dict.fromkeys(result.render_info, Path("out.tif"))

    summary = batch._batch_summary(result, 377.0)

    assert summary == (
        "3 written, 1 failed, 0 cancelled in 377.0 s "
        "(median page 2.0 s, slowest b.tif 9.4 s)"
    )


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
        error = BrokenProcessPool("worker failed")
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
    assert end[0]["message"].startswith("batch end: 0 written, 1 failed, 0 cancelled")
