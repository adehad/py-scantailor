"""Tests for page box detection on the scanner background."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from scantailor.core.models import ImageId, PageId
from scantailor.filters.select_content.filter import Filter
from scantailor.filters.select_content.page_finder import find_page_box

FIXTURES = Path(__file__).parents[2] / "fixtures" / "select_content"
BACKGROUND = 133
PAGE = 252


def _page_on_background(
    shape: tuple[int, int], page: tuple[int, int, int, int]
) -> np.ndarray:
    """Gray scanner background around a white page with a line of text.

    ``page`` is ``(left, top, right, bottom)``, the right and bottom exclusive.
    """
    image = np.full(shape, BACKGROUND, dtype=np.uint8)
    left, top, right, bottom = page
    image[top:bottom, left:right] = PAGE
    image[top + 60 : top + 75, left + 60 : right - 60 : 2] = 20
    return image


def _edges(box) -> tuple[float, float, float, float]:
    return box.x, box.y, box.x + box.width, box.y + box.height


class TestReporterScan:
    """The scan attached to issue #5, reduced from 600 to 150 dpi.

    The Brother scanner trims the top edge of every scan and the scan was then
    turned 90 degrees anticlockwise, so the page touches the left border while
    the gray scanner background shows above, below and to the right of it.
    """

    @pytest.fixture
    def scan(self) -> np.ndarray:
        image = cv2.imread(
            str(FIXTURES / "issue5_trimmed_edge_page.png"), cv2.IMREAD_GRAYSCALE
        )
        assert image is not None
        return image

    def test_the_page_reaches_the_trimmed_edge_and_stops_at_the_background(self, scan):
        page_id = PageId(image_id=ImageId(file_path=Path("issue5.png")))
        box = Filter().detect_page(scan, page_id, dpi=150.0)

        # The reporter's corners at 600 dpi: (0, 71), (3241, 63), (0, 5022),
        # (3236, 5032); a quarter of them at 150 dpi.
        left, top, right, bottom = _edges(box)
        assert left == 0
        assert top == pytest.approx(16.5, abs=3)
        assert right == pytest.approx(809.5, abs=3)
        assert bottom == pytest.approx(1257, abs=3)

    def test_fine_tuning_trims_the_corners_to_the_size_scantailor_universal_finds(
        self, scan
    ):
        box = find_page_box(scan, dpi=150.0, fine_tune=True)

        # scantailor-universal crops this scan to 3220 x 4953 at 600 dpi.
        assert box.width == pytest.approx(3220 / 4, abs=3)
        assert box.height == pytest.approx(4953 / 4, abs=3)


class TestSyntheticPages:
    def test_background_on_all_four_sides_is_peeled_off(self):
        image = _page_on_background((600, 400), (30, 40, 370, 560))

        assert _edges(find_page_box(image, dpi=150.0)) == pytest.approx(
            (30, 40, 370, 560), abs=2
        )

    def test_a_trimmed_left_edge_leaves_the_page_at_the_image_border(self):
        image = _page_on_background((600, 400), (0, 40, 370, 560))

        left, top, right, bottom = _edges(find_page_box(image, dpi=150.0))

        assert left == 0
        assert (top, right, bottom) == pytest.approx((40, 370, 560), abs=2)

    def test_box_is_in_pixels_of_the_input_resolution(self):
        image = _page_on_background((1200, 800), (60, 80, 740, 1120))

        assert _edges(find_page_box(image, dpi=300.0)) == pytest.approx(
            (60, 80, 740, 1120), abs=3
        )

    def test_a_scan_without_background_is_all_page(self):
        image = np.full((300, 200), PAGE, dtype=np.uint8)
        image[100:110, 40:160] = 0

        assert _edges(find_page_box(image, dpi=150.0)) == (0, 0, 200, 300)

    def test_pixels_outside_the_source_count_as_background(self):
        image = _page_on_background((600, 400), (0, 40, 400, 560))
        outside = np.zeros(image.shape, dtype=bool)
        outside[:, 350:] = True
        image[outside] = 255

        right = _edges(find_page_box(image, dpi=150.0, outside=outside))[2]

        assert right == pytest.approx(350, abs=2)

    def test_an_expected_size_keeps_the_box_that_matches_it(self):
        # 340 x 520 pixels at 150 dpi.
        image = _page_on_background((600, 400), (30, 40, 370, 560))
        expected_mm = (340 * 25.4 / 150, 520 * 25.4 / 150)

        box = find_page_box(image, dpi=150.0, expected_size_mm=expected_mm)

        assert box is not None
        assert _edges(box) == pytest.approx((30, 40, 370, 560), abs=2)

    def test_no_box_when_nothing_is_near_the_expected_size(self):
        image = _page_on_background((600, 400), (30, 40, 370, 560))

        assert find_page_box(image, dpi=150.0, expected_size_mm=(5.0, 5.0)) is None

    def test_fine_tuning_stops_at_the_expected_size_less_the_tolerance(self):
        image = _page_on_background((600, 400), (30, 40, 370, 560))
        # A dark wedge in the top-left corner would pull that corner inwards.
        for offset in range(150):
            image[40 + offset, 30 : 30 + 150 - offset] = BACKGROUND
        expected_mm = (340 * 25.4 / 150, 520 * 25.4 / 150)

        box = find_page_box(
            image, dpi=150.0, expected_size_mm=expected_mm, fine_tune=True
        )

        assert box is not None
        assert box.width >= int(340 * 0.9) - 2
