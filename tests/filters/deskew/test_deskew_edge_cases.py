"""The Deskew stage measures the skew of sample-book pages turned by a known angle."""

from pathlib import Path
from types import ModuleType
from typing import Final

import cv2
import numpy as np
import pytest
from numpy.typing import NDArray

from scantailor.core.models import Dpi, ImageId, PageId, SubPage
from scantailor.filters.deskew.filter import Filter

PAGE: Final = PageId(
    image_id=ImageId(file_path=Path("/book.png")), sub_page=SubPage.SINGLE_PAGE
)
SAMPLE_DPI: Final = Dpi.uniform(150)
# The fine search halves its step from 0.5 degree until the step is under the
# 0.1 degree accuracy, so answers lie on a 0.125 degree grid and may sit one
# step from the true angle.
SKEW_TOLERANCE_DEG: Final = 0.15


def _measured_skew(rgb_page: NDArray[np.uint8]) -> float:
    """Skew the Deskew stage measures on a page, asserting it is confident.

    Args:
        rgb_page (NDArray[np.uint8]): (H, W, 3) RGB page at ``SAMPLE_DPI``.

    Returns:
        float: The measured skew in degrees.
    """
    gray = np.asarray(cv2.cvtColor(rgb_page, cv2.COLOR_RGB2GRAY), dtype=np.uint8)
    result = Filter().detect_skew(gray, PAGE, store_result=False, dpi=SAMPLE_DPI)
    assert result.is_confident()
    return result.angle


@pytest.mark.parametrize("degrees", [-4.0, -1.5, -0.5, 0.7, 2.0, 5.0])
def test_a_page_turned_by_a_known_angle_measures_that_skew(
    sample_book_tool: ModuleType, degrees: float
) -> None:
    """The generator turns counter-clockwise for a positive angle. The stage
    reports the skew as the negated clockwise-positive correction, which for such
    a page is the negated angle."""
    page = sample_book_tool.render_skewed_page(degrees)

    assert _measured_skew(page) == pytest.approx(-degrees, abs=SKEW_TOLERANCE_DEG)


def test_the_skewed_sample_page_measures_three_degrees(
    sample_book_tool: ModuleType,
) -> None:
    page = sample_book_tool.render_sample_page("05_skewed").image

    assert _measured_skew(page) == pytest.approx(-3.0, abs=SKEW_TOLERANCE_DEG)


def test_an_upright_page_measures_no_skew(sample_book_tool: ModuleType) -> None:
    page = sample_book_tool.render_sample_page("01_single_page").image

    assert _measured_skew(page) == pytest.approx(0.0, abs=SKEW_TOLERANCE_DEG)
