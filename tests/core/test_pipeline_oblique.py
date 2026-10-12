"""The stored oblique angle shears the pages that later stages work on."""

from pathlib import Path

import numpy as np
import pytest

from scantailor.core.models import ImageId, PageId, SubPage
from scantailor.core.page_geometry import build_transformation
from scantailor.core.pipeline import PipelineOptions, process_page
from scantailor.core.stage_sequence import FilterStage, StageSequence

OBLIQUE_DEGREES = 4.0
DPI = 300.0


def _page() -> PageId:
    return PageId(
        image_id=ImageId(file_path=Path("scan.tiff")), sub_page=SubPage.SINGLE_PAGE
    )


def _image() -> np.ndarray:
    image = np.full((300, 200), 255, dtype=np.uint8)
    image[40:260, 40:160] = 0
    return image


def _wider_after(oblique: float) -> int:
    sequence = StageSequence()
    page = _page()
    sequence.deskew_filter.set_manual_angle(page, 0.0)
    sequence.deskew_filter.set_manual_oblique(page, oblique)
    result = process_page(
        _image(), page, sequence, PipelineOptions(end_stage=FilterStage.DESKEW, dpi=DPI)
    )
    return result.image.shape[1]


def test_pipeline_shears_the_image_by_the_stored_oblique_angle() -> None:
    assert _wider_after(OBLIQUE_DEGREES) != _wider_after(0.0)


def test_pipeline_reports_the_oblique_angle_in_its_metadata() -> None:
    sequence = StageSequence()
    page = _page()
    sequence.deskew_filter.set_manual_oblique(page, OBLIQUE_DEGREES)
    result = process_page(
        _image(), page, sequence, PipelineOptions(end_stage=FilterStage.DESKEW, dpi=DPI)
    )

    assert result.metadata["DESKEW"]["oblique_angle"] == pytest.approx(OBLIQUE_DEGREES)


def test_page_geometry_for_later_stages_includes_the_shear() -> None:
    sequence = StageSequence()
    page = _page()
    sequence.deskew_filter.set_manual_angle(page, 0.0)
    sheared = build_transformation(
        sequence, page, (200, 300), DPI, FilterStage.SELECT_CONTENT
    )
    sequence.deskew_filter.set_manual_oblique(page, OBLIQUE_DEGREES)
    unsheared_after = build_transformation(
        sequence, page, (200, 300), DPI, FilterStage.SELECT_CONTENT
    )

    assert sheared.post_oblique == pytest.approx(0.0)
    assert unsheared_after.post_oblique == pytest.approx(OBLIQUE_DEGREES)
