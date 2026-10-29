"""Every detection step sees one gray page, converted with qGray's weights."""

from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np
import pytest
from numpy.typing import NDArray

from scantailor.core import pipeline
from scantailor.core.black_on_white_estimator import is_black_on_white
from scantailor.core.content_mask import render_page_at_dpi
from scantailor.core.models import Dpi, ImageId, PageId, SubPage
from scantailor.core.pipeline import PipelineOptions, process_page
from scantailor.core.settings import BlackOnWhiteDetection
from scantailor.core.stage_sequence import FilterStage, StageSequence
from scantailor.core.transformation import ImageTransformation, render_transformed
from scantailor.filters.deskew.filter import Filter as DeskewFilter
from scantailor.filters.select_content.content_finder import find_content_box
from scantailor.filters.select_content.page_finder import find_page_box
from scantailor.imageproc import binary_threshold, filters
from scantailor.imageproc.analysis import find_skew
from scantailor.imageproc.filters import qgray
from tests.core.dark_pages import printed_page

# Cyan ink on green paper on a magenta bed, in BGR. qGray makes them 167, 127
# and 127, BT.601 179, 150 and 105: BT.601 halves the ink's contrast and shows
# a page edge where qGray has none. Each detection below gives a different
# result from the BT.601 gray of this page than from its qGray.
_INK = (255, 255, 0)
_PAPER = (0, 255, 0)
_BED = (255, 0, 255)


def _page_id() -> PageId:
    return PageId(
        image_id=ImageId(file_path=Path("colour.png")), sub_page=SubPage.SINGLE_PAGE
    )


def _colour_page() -> NDArray[np.uint8]:
    """Printed lines turned 2 degrees on coloured paper, on a coloured bed."""
    ink = printed_page(angle=2.0) < 128
    page = np.empty((*ink.shape, 3), dtype=np.uint8)
    page[...] = _PAPER
    page[ink] = _INK
    bed = np.empty((ink.shape[0] + 120, ink.shape[1] + 120, 3), dtype=np.uint8)
    bed[...] = _BED
    bed[60:-60, 60:-60] = page
    return bed


def _identity(image: NDArray[np.uint8]) -> ImageTransformation:
    height, width = image.shape[:2]
    return ImageTransformation.from_size(width, height, Dpi.uniform(300))


DETECTIONS: dict[str, Callable[[NDArray[np.uint8]], object]] = {
    "is_black_on_white": lambda image: is_black_on_white(image, _identity(image)),
    "render_page_at_dpi": lambda image: render_page_at_dpi(  # type: ignore[union-attr]
        image, _identity(image), 150
    ).gray.tobytes(),
    "detect_skew": lambda image: DeskewFilter().detect_skew(
        image, _page_id(), store_result=False, dpi=None
    ),
    "find_skew": lambda image: find_skew(image),
    "find_content_box": lambda image: find_content_box(image, 300.0).content_box,
    "find_page_box": lambda image: find_page_box(image, 300.0),
}


@pytest.mark.parametrize("name", list(DETECTIONS))
def test_a_colour_page_is_detected_as_its_qgray(name: str) -> None:
    page = _colour_page()
    detect = DETECTIONS[name]

    assert detect(page) == detect(qgray(page))


def test_a_page_is_converted_to_gray_once_for_every_detection_stage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    converted: list[tuple[int, ...]] = []
    to_gray = {cv2.COLOR_BGR2GRAY, cv2.COLOR_BGRA2GRAY}
    cvt_color = cv2.cvtColor

    def counting_qgray(image: NDArray[np.uint8]) -> NDArray[np.uint8]:
        converted.append(image.shape)
        return qgray(image)

    def counting_cvt_color(image: NDArray[np.uint8], code: int, *args: object):
        if code in to_gray:
            converted.append(image.shape)
        return cvt_color(image, code, *args)

    monkeypatch.setattr(filters, "qgray", counting_qgray)
    monkeypatch.setattr(binary_threshold, "qgray", counting_qgray)
    monkeypatch.setattr(cv2, "cvtColor", counting_cvt_color)
    page = _colour_page()

    process_page(
        page,
        _page_id(),
        StageSequence(),
        PipelineOptions(dpi=300.0, end_stage=FilterStage.PAGE_LAYOUT),
    )

    assert converted == [page.shape]


@pytest.mark.parametrize("black_on_white", [True, False])
def test_the_detection_view_is_the_gray_source_resampled(
    monkeypatch: pytest.MonkeyPatch, black_on_white: bool
) -> None:
    page = _colour_page()
    state = pipeline._PageState(page, 300.0)
    state.xform.set_post_rotation(2.0)
    monkeypatch.setattr(state, "is_black_on_white", lambda dbg=None: black_on_white)
    gray = qgray(page) if black_on_white else 255 - qgray(page)

    view = state.detection_view(BlackOnWhiteDetection())

    np.testing.assert_array_equal(view, render_transformed(gray, state.xform))
