"""Lines logged while a page is processed carry its name and the stage."""

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from loguru import logger

from scantailor.core import batch, pipeline
from scantailor.core.image_io import write_image
from scantailor.core.models import Dpi, ImageId, PageId, SubPage
from scantailor.core.pipeline import PipelineOptions, process_page
from scantailor.core.stage_params import collect_stage_defaults
from scantailor.core.stage_sequence import FilterStage, StageSequence


def _page_pixels() -> np.ndarray:
    pixels = np.full((400, 300), 225, dtype=np.uint8)
    for top in range(80, 320, 14):
        pixels[top : top + 6, 50:250] = 30
    return pixels


def _records_with(records: list[dict[str, Any]], message: str) -> list[dict[str, Any]]:
    return [record for record in records if record["message"] == message]


def test_a_line_logged_inside_a_stage_carries_the_page_and_the_stage(
    monkeypatch: pytest.MonkeyPatch, log_records: list[dict[str, Any]]
) -> None:
    deskew = pipeline._process_deskew

    def logging_deskew(*args: Any, **kwargs: Any) -> dict:
        logger.debug("inside deskew")
        return deskew(*args, **kwargs)

    monkeypatch.setattr(pipeline, "_process_deskew", logging_deskew)
    page = PageId(
        image_id=ImageId(file_path=Path("books/scan07.tif")),
        sub_page=SubPage.LEFT_PAGE,
    )

    process_page(
        _page_pixels(),
        page,
        StageSequence(),
        PipelineOptions(end_stage=FilterStage.DESKEW),
    )

    [record] = _records_with(log_records, "inside deskew")
    assert record["extra"] == {"page": "scan07.tif/left_page", "stage": "deskew"}


def test_the_context_ends_with_the_pipeline_run(
    log_records: list[dict[str, Any]],
) -> None:
    page = PageId(image_id=ImageId(file_path=Path("scan.tif")))
    process_page(
        _page_pixels(),
        page,
        StageSequence(),
        PipelineOptions(end_stage=FilterStage.DESKEW),
    )

    logger.debug("after the run")

    [record] = _records_with(log_records, "after the run")
    assert record["extra"].get("page", "-") == "-"
    assert record["extra"].get("stage", "-") == "-"


def test_a_pool_job_logs_its_page_before_the_pipeline_starts(
    tmp_path: Path, log_records: list[dict[str, Any]]
) -> None:
    path = tmp_path / "scan03.tif"
    write_image(path, _page_pixels(), Dpi.uniform(300))
    page = PageId(image_id=ImageId(file_path=path))
    stages = StageSequence()
    job = batch._PageJob(
        page,
        batch._PhaseConfig(
            token="context-test",
            stage_params={},
            defaults=collect_stage_defaults(stages),
            default_dpi=300.0,
        ),
    )

    batch._detect_page(job)

    decode = next(r for r in log_records if r["message"].startswith("Loaded "))
    assert decode["extra"]["page"] == "scan03.tif"
