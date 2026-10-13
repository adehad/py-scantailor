"""A pipeline run asked to stop does so before its next stage."""

from pathlib import Path

import numpy as np
import pytest

from scantailor.core.models import ImageId, PageId, SubPage
from scantailor.core.pipeline import PipelineCancelled, PipelineOptions, process_page
from scantailor.core.stage_sequence import FilterStage, StageSequence

PAGE = PageId(image_id=ImageId(file_path=Path("/a.png")), sub_page=SubPage.SINGLE_PAGE)


def _image() -> np.ndarray:
    return np.full((300, 200), 240, dtype=np.uint8)


def test_a_run_that_is_cancelled_from_the_start_stops_at_once() -> None:
    options = PipelineOptions(should_cancel=lambda: True)

    with pytest.raises(PipelineCancelled):
        process_page(_image(), PAGE, StageSequence(), options)


def test_a_run_stops_before_the_stage_after_the_cancel() -> None:
    asked: list[int] = []

    def cancel_after_two_stages() -> bool:
        asked.append(1)
        return len(asked) > 2

    options = PipelineOptions(
        end_stage=FilterStage.PAGE_LAYOUT, should_cancel=cancel_after_two_stages
    )

    with pytest.raises(PipelineCancelled):
        process_page(_image(), PAGE, StageSequence(), options)

    assert len(asked) == 3


def test_a_run_that_is_not_cancelled_completes() -> None:
    options = PipelineOptions(end_stage=FilterStage.DESKEW, should_cancel=lambda: False)

    result = process_page(_image(), PAGE, StageSequence(), options)

    assert FilterStage.DESKEW in result.stages_completed
