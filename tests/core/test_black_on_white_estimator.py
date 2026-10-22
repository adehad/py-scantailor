"""Telling dark content on light paper from light content on a dark page."""

import cv2
import numpy as np
import pytest

from scantailor.core.black_on_white_estimator import (
    is_black_on_white,
    is_black_on_white_in_area,
    is_black_on_white_refining,
)
from scantailor.core.models import Dpi
from scantailor.core.transformation import ImageTransformation
from tests.core.dark_pages import dark_page, dark_sheet_on_light_bed, printed_page


def _xform(image: np.ndarray) -> ImageTransformation:
    height, width = image.shape[:2]
    return ImageTransformation.from_size(width, height, Dpi.uniform(300))


def _whole(image: np.ndarray) -> np.ndarray:
    return np.ones(image.shape[:2], dtype=bool)


def test_printed_page_is_black_on_white():
    page = printed_page()

    assert is_black_on_white(page, _xform(page))


def test_light_text_on_a_black_page_is_not_black_on_white():
    page = dark_page()

    assert not is_black_on_white(page, _xform(page))


def test_colour_scan_of_a_dark_page_is_not_black_on_white():
    page = cv2.cvtColor(dark_page(), cv2.COLOR_GRAY2BGR)
    page[..., 0] = np.maximum(page[..., 0], 40)

    assert not is_black_on_white(page, _xform(page))


def test_printed_page_in_a_wide_black_surround_is_black_on_white():
    scan = np.zeros((2400, 1900), dtype=np.uint8)
    scan[450:1950, 400:1500] = printed_page()

    assert not is_black_on_white_in_area(scan, _whole(scan))
    assert is_black_on_white(scan, _xform(scan))


def test_dark_sheet_filling_most_of_a_light_bed_is_not_black_on_white():
    scan = dark_sheet_on_light_bed()

    assert not is_black_on_white(scan, _xform(scan))


def test_dark_sheet_on_a_bed_lighter_and_wider_than_it_counts_as_black_on_white():
    """The area check settles a mostly light scan; only a dark one is refined."""
    scan = dark_sheet_on_light_bed(bed_margin=600)

    assert not is_black_on_white_refining(scan, _xform(scan))
    assert is_black_on_white(scan, _xform(scan))


def test_only_the_pre_crop_area_is_measured():
    scan = np.hstack([dark_page(), printed_page()])
    left = np.array([(0, 0), (1100, 0), (1100, 1500), (0, 1500)], dtype=float)
    right = left + np.array([1100, 0])
    xform = _xform(scan)

    xform.set_pre_crop_area(left)
    assert not is_black_on_white(scan, xform)
    xform.set_pre_crop_area(right)
    assert is_black_on_white(scan, xform)


def test_rotation_and_crop_after_the_pre_crop_do_not_change_the_answer():
    scan = dark_sheet_on_light_bed()
    xform = _xform(scan)
    xform.set_post_rotation(12.0)
    xform.set_post_crop_area(
        np.array([(400, 400), (900, 400), (900, 900), (400, 900)], dtype=float)
    )

    assert is_black_on_white(scan, xform) is is_black_on_white(scan, _xform(scan))


def test_refining_reads_the_polarity_of_the_text_blocks():
    printed, dark = printed_page(), dark_page()

    assert is_black_on_white_refining(printed, _xform(printed))
    assert not is_black_on_white_refining(dark, _xform(dark))


def test_blank_page_counts_as_black_on_white():
    for level in (0, 255):
        page = np.full((300, 200), level, dtype=np.uint8)

        assert is_black_on_white(page, _xform(page))


def test_empty_area_counts_as_black_on_white():
    page = dark_page()

    assert is_black_on_white_in_area(page, np.zeros(page.shape, dtype=bool))


def test_area_of_another_shape_is_refused():
    page = dark_page()

    with pytest.raises(ValueError, match="differ"):
        is_black_on_white_in_area(page, np.ones((10, 10), dtype=bool))


def test_estimator_leaves_its_input_untouched():
    page = dark_page()
    before = page.copy()

    is_black_on_white(page, _xform(page))

    assert np.array_equal(page, before)
