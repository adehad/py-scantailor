"""Tests for the automatic oblique (shear) detection."""

import cv2
import numpy as np
import pytest

from scantailor.filters.deskew.oblique import (
    build_sausage_mask,
    find_oblique_degrees,
    rotate_mask,
)
from scantailor.imageproc.analysis import find_skew

from .synthetic_pages import sheared_page, text_page


def _binary(page: np.ndarray) -> np.ndarray:
    return np.where(page > 127, 255, 0).astype(np.uint8)


class TestBuildSausageMask:
    """The sausage keeps each row's ink length and mean position."""

    def test_row_becomes_one_run_of_the_same_length(self):
        mask = np.full((3, 20), 255, dtype=np.uint8)
        mask[1, 2:5] = 0
        mask[1, 12:15] = 0

        sausage = build_sausage_mask(mask)

        run = np.flatnonzero(sausage[1] == 0)
        assert run.size == 6
        assert np.array_equal(run, np.arange(run[0], run[0] + 6))
        assert run.mean() == pytest.approx(np.flatnonzero(mask[1] == 0).mean(), abs=1)

    def test_rows_without_ink_stay_white(self):
        mask = np.full((4, 10), 255, dtype=np.uint8)
        mask[2, 3:6] = 0

        sausage = build_sausage_mask(mask)

        assert np.all(sausage[[0, 1, 3]] == 255)

    def test_a_row_of_ink_touching_the_edge_stays_inside_the_image(self):
        mask = np.full((1, 10), 255, dtype=np.uint8)
        mask[0, 0:3] = 0

        sausage = build_sausage_mask(mask)

        assert sausage.shape == (1, 10)
        assert np.count_nonzero(sausage == 0) == 3

    def test_blank_image_stays_blank(self):
        mask = np.full((5, 5), 255, dtype=np.uint8)

        assert np.all(build_sausage_mask(mask) == 255)


class TestRotateMask:
    """Rotation levels the text lines before the oblique is measured."""

    def test_zero_degrees_returns_the_mask_unchanged(self):
        mask = _binary(text_page(100, 120))

        assert rotate_mask(mask, 0.0) is mask

    def test_canvas_grows_to_hold_the_rotated_page(self):
        mask = _binary(text_page(100, 120))

        rotated = rotate_mask(mask, 10.0)

        assert rotated.shape[0] > 100 and rotated.shape[1] > 120

    def test_result_is_binary_with_white_corners(self):
        mask = _binary(text_page(100, 120))

        rotated = rotate_mask(mask, 10.0)

        assert set(np.unique(rotated)) <= {0, 255}
        assert rotated[0, 0] == 255

    def test_rotating_by_the_correction_angle_levels_a_skewed_page(self):
        level = _binary(text_page())
        skew_degrees = 3.0
        height, width = level.shape
        matrix = cv2.getRotationMatrix2D((width / 2, height / 2), skew_degrees, 1.0)
        tilted = np.where(
            cv2.warpAffine(level, matrix, (width, height), borderValue=255) > 127,
            255,
            0,
        ).astype(np.uint8)
        before = abs(find_skew(tilted).angle)

        # The tilt above is counter-clockwise on screen, so clockwise levels it.
        after = abs(find_skew(rotate_mask(tilted, skew_degrees)).angle)

        assert after < before


class TestFindObliqueDegrees:
    """The estimate follows the shear of a synthetic page."""

    @pytest.mark.parametrize("shear", [-4.0, -2.0, 2.0, 4.0])
    def test_recovers_the_shear_within_tolerance(self, shear: float):
        found = find_oblique_degrees(_binary(sheared_page(shear)))

        assert found is not None
        # The correction applied later is the negative of what is found.
        assert -found == pytest.approx(shear, abs=1.0)
        assert np.sign(-found) == np.sign(shear)

    def test_an_upright_page_has_no_oblique(self):
        found = find_oblique_degrees(_binary(text_page()))

        assert found == pytest.approx(0.0, abs=0.2)

    def test_result_is_clamped_to_the_limit(self):
        found = find_oblique_degrees(_binary(sheared_page(5.0)), max_oblique_deg=1.0)

        assert found is not None
        assert abs(found) <= 1.0

    def test_empty_mask_has_no_estimate(self):
        assert find_oblique_degrees(np.zeros((0, 0), dtype=np.uint8)) is None

    def test_input_is_not_modified(self):
        mask = _binary(sheared_page(2.0))
        before = mask.copy()

        find_oblique_degrees(mask)

        assert np.array_equal(mask, before)
