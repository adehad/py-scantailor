"""Tests for the manual trim of Fix Orientation.

Expected values are worked out by hand from the upstream C++ in
``src/core/filters/fix_orientation/ImageTrim.cpp`` (toInnerRect) and
``ImageTransformation::origRectToPreCropSpace``.
"""

import numpy as np
import pytest

from scantailor.core.models import OrthogonalRotation
from scantailor.filters.fix_orientation.image_trim import (
    MIN_TRIM_INNER_SIDE,
    ImageTrim,
    merge_with_trim,
)


def _trim(left: int = 0, top: int = 0, right: int = 0, bottom: int = 0) -> ImageTrim:
    return ImageTrim(enabled=True, left=left, top=top, right=right, bottom=bottom)


class TestToInnerRect:
    def test_the_minimum_inner_side_is_32_pixels(self):
        assert MIN_TRIM_INNER_SIDE == 32

    def test_a_disabled_trim_keeps_the_whole_image_whatever_its_margins(self):
        trim = ImageTrim(enabled=False, left=50, top=50, right=50, bottom=50)

        assert trim.to_inner_rect(300, 400) == (0, 0, 300, 400)

    def test_an_empty_image_is_returned_as_it_is(self):
        assert _trim(left=5).to_inner_rect(0, 400) == (0, 0, 0, 400)
        assert _trim(left=5).to_inner_rect(300, 0) == (0, 0, 300, 0)

    def test_margins_are_cut_from_their_own_edges(self):
        # x = left, y = top, w = 300 - 10 - 30, h = 400 - 20 - 40.
        assert _trim(left=10, top=20, right=30, bottom=40).to_inner_rect(300, 400) == (
            10,
            20,
            260,
            340,
        )

    def test_negative_margins_count_as_zero(self):
        trim = _trim(left=-5, top=-6, right=-7, bottom=-8)

        assert trim.to_inner_rect(300, 400) == (0, 0, 300, 400)

    def test_margins_that_exactly_leave_the_minimum_side_are_kept(self):
        assert _trim(left=40, right=28).to_inner_rect(100, 100) == (40, 0, 32, 100)

    def test_excess_is_taken_back_from_both_margins_leading_edge_first(self):
        # Width 100, left 60, right 60: excess = 60 + 60 + 32 - 100 = 52.
        # fromLeading = min(60, (52 + 1) / 2 = 26) = 26, so left = 34 and
        # right = 60 - (52 - 26) = 34; width = max(32, 100 - 68) = 32.
        assert _trim(left=60, right=60).to_inner_rect(100, 100) == (34, 0, 32, 100)

    def test_a_vertical_excess_is_redistributed_the_same_way(self):
        assert _trim(top=60, bottom=60).to_inner_rect(100, 100) == (0, 34, 100, 32)

    def test_an_odd_excess_takes_the_extra_pixel_from_the_leading_edge(self):
        # Width 100, left 50, right 51: excess = 133 - 100 = 33,
        # fromLeading = min(50, 17) = 17, left = 33, right = 51 - 16 = 35.
        assert _trim(left=50, right=51).to_inner_rect(100, 100) == (33, 0, 32, 100)

    def test_a_single_huge_margin_slides_the_minimum_window_to_the_far_edge(self):
        # excess = 200 + 32 - 100 = 132, fromLeading = min(200, 66) = 66,
        # left = 134, right = max(0, 0 - 66) = 0; w = 32 and left + w > 100
        # moves left back to 100 - 32 = 68.
        assert _trim(left=200).to_inner_rect(100, 100) == (68, 0, 32, 100)

    def test_an_image_smaller_than_the_minimum_gives_a_window_at_the_origin(self):
        assert _trim(left=3, top=3).to_inner_rect(20, 20) == (0, 0, 32, 32)


class TestCropArea:
    """The trimmed region in the coordinates of the rotated image."""

    ORIGINAL = (300, 400)
    TRIM = _trim(left=10, top=20, right=30, bottom=40)

    @staticmethod
    def _bounds(area: np.ndarray) -> tuple[float, float, float, float]:
        return (
            float(area[:, 0].min()),
            float(area[:, 1].min()),
            float(area[:, 0].max()),
            float(area[:, 1].max()),
        )

    def test_a_disabled_trim_has_no_area(self):
        area = ImageTrim().crop_area(OrthogonalRotation(degrees=0), self.ORIGINAL)

        assert area is None

    def test_without_rotation_the_area_is_the_inner_rect(self):
        area = self.TRIM.crop_area(OrthogonalRotation(degrees=0), self.ORIGINAL)

        assert area is not None
        assert self._bounds(area) == (10, 20, 270, 360)

    @pytest.mark.parametrize(
        ("degrees", "expected"),
        [
            # (x, y) -> (H - y, x) with H = 400, so x in [400 - 360, 400 - 20].
            (90, (40, 10, 380, 270)),
            # (x, y) -> (W - x, H - y).
            (180, (30, 40, 290, 380)),
            # (x, y) -> (y, W - x) with W = 300.
            (270, (20, 30, 360, 290)),
        ],
    )
    def test_the_inner_rect_is_carried_through_the_rotation(
        self, degrees: int, expected: tuple[float, float, float, float]
    ):
        rotation = OrthogonalRotation(degrees=degrees)
        oriented = rotation.rotate_dimensions(*self.ORIGINAL)

        area = self.TRIM.crop_area(rotation, (round(oriented[0]), round(oriented[1])))

        assert area is not None
        assert self._bounds(area) == expected

    def test_the_trim_is_measured_on_the_unrotated_image(self):
        """A 90 degree turn swaps the image's sides but not what the margins mean."""
        rotation = OrthogonalRotation(degrees=90)
        area = _trim(left=100).crop_area(rotation, (400, 300))

        assert area is not None
        # The left edge of the stored image becomes the top edge after a
        # clockwise quarter turn, so 100 pixels go from the top.
        assert self._bounds(area) == (0, 100, 400, 300)


class TestMergeWithTrim:
    PAGE = np.array([[0, 0], [200, 0], [200, 300], [0, 300]], dtype=np.float64)

    def test_without_a_trim_the_page_outline_is_returned_unchanged(self):
        assert merge_with_trim(self.PAGE, None) is self.PAGE

    def test_the_intersection_of_outline_and_trim_is_returned(self):
        trim = np.array([[20, 10], [370, 10], [370, 290], [20, 290]], dtype=np.float64)

        merged = merge_with_trim(self.PAGE, trim)

        assert merged[:, 0].min() == pytest.approx(20)
        assert merged[:, 0].max() == pytest.approx(200)
        assert merged[:, 1].min() == pytest.approx(10)
        assert merged[:, 1].max() == pytest.approx(290)

    def test_the_clip_does_not_depend_on_the_winding_of_either_polygon(self):
        trim = np.array([[20, 10], [370, 10], [370, 290], [20, 290]], dtype=np.float64)

        merged = merge_with_trim(self.PAGE[::-1], trim[::-1])

        assert merged[:, 0].min() == pytest.approx(20)
        assert merged[:, 1].max() == pytest.approx(290)

    def test_a_slanted_outline_is_cut_along_the_trim_edge(self):
        outline = np.array([[0, 0], [100, 0], [100, 100]], dtype=np.float64)
        trim = np.array([[50, 0], [200, 0], [200, 200], [50, 200]], dtype=np.float64)

        merged = merge_with_trim(outline, trim)

        # Triangle (0,0)-(100,0)-(100,100) cut at x = 50: a trapezoid of area
        # 50 * (50 + 100) / 2 = 3750 at the right of the cut.
        x, y = merged[:, 0], merged[:, 1]
        area = 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
        assert area == pytest.approx(3750.0)

    def test_an_outline_that_misses_the_trim_is_kept_as_it_is(self):
        """Upstream falls back to the page outline when the intersection is empty."""
        trim = np.array([[250, 0], [300, 0], [300, 300], [250, 300]], dtype=np.float64)

        assert merge_with_trim(self.PAGE, trim) is self.PAGE
