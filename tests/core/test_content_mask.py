"""The content mask finds where a page has content."""

import numpy as np
import pytest

from scantailor.core.content_mask import MASK_DPI, ContentMask, compute_content_mask
from scantailor.core.models import Dpi
from scantailor.core.transformation import ImageTransformation

DPI = 300


def _page_with_bars() -> np.ndarray:
    page = np.full((1200, 900), 255, dtype=np.uint8)
    page[300:360, 200:600] = 30  # A line of text
    page[800:830, 100:300] = 20  # Another, lower and to the left
    return page


def _transformation(page: np.ndarray) -> ImageTransformation:
    height, width = page.shape[:2]
    return ImageTransformation.from_size(width, height, Dpi.uniform(DPI))


class TestComputing:
    def test_the_mask_is_made_at_one_hundred_and_fifty_dpi(self):
        page = _page_with_bars()

        mask = compute_content_mask(page, _transformation(page))

        assert mask.image.shape == (600, 450)
        assert mask.original_to_content[0, 0] == pytest.approx(MASK_DPI / DPI)
        assert mask.original_to_content[1, 1] == pytest.approx(MASK_DPI / DPI)

    def test_dark_marks_are_content_and_paper_is_not(self):
        page = _page_with_bars()

        mask = compute_content_mask(page, _transformation(page))

        assert mask.image[165, 200]
        assert not mask.image[50, 50]
        assert not mask.image[500, 300]

    def test_a_colour_page_gives_the_same_mask_as_its_gray_version(self):
        gray = _page_with_bars()
        colour = np.repeat(gray[:, :, None], 3, axis=2)

        from_gray = compute_content_mask(gray, _transformation(gray))
        from_colour = compute_content_mask(colour, _transformation(gray))

        assert np.array_equal(from_gray.image, from_colour.image)

    def test_content_outside_the_pre_crop_area_is_ignored(self):
        page = _page_with_bars()
        xform = _transformation(page)
        xform.set_pre_crop_area(
            np.array([[0, 0], [900, 0], [900, 600], [0, 600]], dtype=np.float64)
        )

        mask = compute_content_mask(page, xform)

        assert mask.find_content_in_area(0, 0, 450, 300) is not None
        assert mask.find_content_in_area(0, 380, 450, 150) is None

    def test_specks_are_not_content(self):
        page = _page_with_bars()
        page[1000:1002, 700:702] = 0

        mask = compute_content_mask(page, _transformation(page))

        assert mask.find_content_in_area(340, 490, 20, 20) is None

    def test_the_page_is_not_modified(self):
        page = _page_with_bars()
        before = page.copy()

        compute_content_mask(page, _transformation(page))

        assert np.array_equal(page, before)

    def test_an_empty_page_area_gives_a_null_mask(self):
        xform = ImageTransformation.from_size(0, 0, Dpi.uniform(DPI))

        mask = compute_content_mask(np.zeros((1, 1), dtype=np.uint8), xform)

        assert mask.is_null()


class TestFindingContent:
    @staticmethod
    def _mask() -> ContentMask:
        image = np.zeros((100, 200), dtype=bool)
        image[20:30, 50:120] = True
        return ContentMask(image, np.diag([0.5, 0.5, 1.0]))

    def test_the_box_of_the_content_in_the_area_is_returned_one_pixel_larger(self):
        found = self._mask().find_content_in_area(40, 10, 100, 40)

        assert found == (49, 19, 72, 12)

    def test_only_the_part_inside_the_area_counts(self):
        found = self._mask().find_content_in_area(60, 0, 20, 100)

        assert found == (59, 19, 22, 12)

    def test_an_area_without_content_gives_nothing(self):
        assert self._mask().find_content_in_area(0, 50, 30, 30) is None

    def test_an_area_partly_outside_the_mask_is_clipped(self):
        assert self._mask().find_content_in_area(-50, -50, 120, 100) == (
            49,
            19,
            22,
            12,
        )

    def test_an_area_wholly_outside_the_mask_gives_nothing(self):
        assert self._mask().find_content_in_area(500, 500, 10, 10) is None

    def test_a_null_mask_finds_nothing(self):
        assert ContentMask().find_content_in_area(0, 0, 10, 10) is None

    def test_the_inverse_transform_undoes_the_forward_one(self):
        mask = self._mask()

        assert np.allclose(
            mask.content_to_original @ mask.original_to_content, np.eye(3)
        )
