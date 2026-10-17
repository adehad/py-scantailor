"""Finding the paper on a dark scanner background, with and without fine tuning."""

from pathlib import Path

import cv2
import numpy as np
import pytest
from numpy.typing import NDArray

from scantailor.core.models import ImageId, PageId
from scantailor.filters.select_content import page_finder
from scantailor.filters.select_content.content_box import PageBox
from scantailor.filters.select_content.content_box_finder import Rect
from scantailor.filters.select_content.filter import Filter
from scantailor.filters.select_content.page_finder import (
    detect_borders,
    find_page_box,
    fine_tune_corners,
)

DPI = 150.0
FIXTURES = Path(__file__).parents[2] / "fixtures" / "select_content"
BED = 30
PAPER = 235


def _upright_page(
    height: int = 400, width: int = 300, inset: tuple[int, int] = (40, 30)
) -> NDArray[np.uint8]:
    """Light paper with a dark bed showing on every side."""
    image = np.full((height, width), BED, dtype=np.uint8)
    top, left = inset
    image[top : height - top, left : width - left] = PAPER
    return image


def _crooked_page(angle: float, size: tuple[float, float] = (220.0, 300.0)) -> NDArray:
    """Paper turned by ``angle`` degrees on a dark bed."""
    image = np.full((400, 340), BED, dtype=np.uint8)
    corners = cv2.boxPoints(((170.0, 200.0), size, angle))
    cv2.fillPoly(image, [np.round(corners).astype(np.int32)], PAPER)
    return image


def _corners_on_paper(image: NDArray[np.uint8], box: PageBox) -> list[bool]:
    left, top = int(box.x), int(box.y)
    right = int(box.x + box.width) - 1
    bottom = int(box.y + box.height) - 1
    return [
        bool(image[y, x] == PAPER)
        for x, y in ((left, top), (right, top), (left, bottom), (right, bottom))
    ]


class TestDetectBorders:
    def test_each_side_stops_on_the_last_line_of_bed(self) -> None:
        background = _upright_page() == BED

        assert detect_borders(background) == Rect(29, 39, 270, 360)

    def test_a_scan_without_bed_is_the_whole_image(self) -> None:
        background = np.zeros((50, 40), dtype=bool)

        assert detect_borders(background) == Rect(0, 0, 39, 49)

    def test_a_thin_paper_coloured_streak_on_the_bed_is_crossed(self) -> None:
        image = _upright_page()
        image[:, 10:13] = PAPER  # dust or a reflection, narrower than the gap

        assert detect_borders(image == BED).left == 29


class TestFineTuneCorners:
    def test_corners_on_a_crooked_page_move_inward_until_they_reach_paper(
        self,
    ) -> None:
        image = _crooked_page(8.0)
        background = image == BED
        found = detect_borders(background)

        tuned = fine_tune_corners(background, found, None, 1.0)

        assert found.left < tuned.left and tuned.right < found.right
        assert found.top < tuned.top and tuned.bottom < found.bottom
        for x, y in (
            (tuned.left, tuned.top),
            (tuned.right, tuned.top),
            (tuned.left, tuned.bottom),
            (tuned.right, tuned.bottom),
        ):
            assert not background[y, x]

    def test_an_expected_size_keeps_the_box_within_the_tolerance(self) -> None:
        # Square paper turned 45 degrees: unbounded tuning halves the box.
        background = _crooked_page(45.0, size=(200.0, 200.0)) == BED
        found = detect_borders(background)
        expected = (found.width, found.height)

        free = fine_tune_corners(background, found, None, 1.0)
        bounded = fine_tune_corners(background, found, expected, 0.1)

        assert free.width < 0.7 * found.width
        assert 0.89 * found.width <= bounded.width < found.width
        assert 0.89 * found.height <= bounded.height < found.height

    def test_corners_already_on_paper_stay(self) -> None:
        background = np.zeros((30, 30), dtype=bool)
        rect = Rect(2, 3, 25, 26)

        assert fine_tune_corners(background, rect, None, 1.0) == rect


class TestFindPageBox:
    def test_finds_the_paper_on_a_dark_bed(self) -> None:
        box = find_page_box(_upright_page(), dpi=DPI)

        assert box == PageBox(x=29.0, y=39.0, width=242.0, height=322.0)

    def test_fine_tuning_trims_the_corners_of_a_crooked_page(self) -> None:
        image = _crooked_page(8.0)

        plain = find_page_box(image, dpi=DPI)
        tuned = find_page_box(image, dpi=DPI, fine_tune=True)

        assert plain is not None and tuned is not None
        assert tuned.width < plain.width and tuned.height < plain.height
        assert not any(_corners_on_paper(image, plain))
        assert all(_corners_on_paper(image, tuned))

    def test_the_box_is_returned_in_pixels_of_the_image(self) -> None:
        image = cv2.resize(
            _upright_page(), None, fx=2.0, fy=2.0, interpolation=cv2.INTER_NEAREST
        )

        box = find_page_box(image, dpi=2 * DPI)

        assert box is not None
        assert box.x == pytest.approx(58.0, abs=2.0)
        assert box.width == pytest.approx(484.0, abs=4.0)

    def test_a_color_scan_is_read_as_gray(self) -> None:
        gray = _upright_page()

        box = find_page_box(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), dpi=DPI)

        assert box == find_page_box(gray, dpi=DPI)

    def test_an_expected_size_picks_the_closest_binarisation(self) -> None:
        width_mm = 242 * 25.4 / DPI
        height_mm = 322 * 25.4 / DPI

        box = find_page_box(
            _upright_page(), dpi=DPI, expected_size_mm=(width_mm, height_mm)
        )

        assert box == PageBox(x=29.0, y=39.0, width=242.0, height=322.0)

    def test_an_expected_size_far_from_every_result_finds_nothing(self) -> None:
        box = find_page_box(_upright_page(), dpi=DPI, expected_size_mm=(10, 10))

        assert box is None

    def test_pixels_outside_the_source_count_as_bed(self) -> None:
        image = np.full((400, 300), PAPER, dtype=np.uint8)
        image[100:120, 100:200] = BED  # some content, so the bed has a level
        outside = np.zeros(image.shape, dtype=bool)
        outside[:, :60] = True

        box = find_page_box(image, dpi=DPI, outside=outside)

        assert box is not None
        assert box.x == 59.0

    def test_an_image_entirely_outside_the_source_has_no_page(self) -> None:
        image = _upright_page()

        assert find_page_box(image, outside=np.ones(image.shape, bool)) is None

    def test_the_input_is_not_modified(self) -> None:
        image = _upright_page()
        before = image.copy()
        outside = np.zeros(image.shape, dtype=bool)
        outside[:5] = True

        find_page_box(image, dpi=DPI, fine_tune=True, outside=outside)

        assert np.array_equal(image, before)


def _light_bed_page() -> NDArray[np.uint8]:
    """Paper on a light gray bed, as under a closed scanner lid, with some text."""
    image = np.full((600, 440), 206, dtype=np.uint8)
    image[40:560, 40:400] = 238
    for top in range(100, 500, 40):
        image[top : top + 12, 80:360] = 40
    return image


class TestBackgroundPaths:
    def test_a_light_bed_falls_back_to_the_straight_paper_edges(self) -> None:
        box = find_page_box(_light_bed_page(), dpi=DPI)

        assert box is not None
        assert box.x == pytest.approx(40.0, abs=3.0)
        assert box.y == pytest.approx(40.0, abs=3.0)
        assert box.x + box.width == pytest.approx(400.0, abs=3.0)
        assert box.y + box.height == pytest.approx(560.0, abs=3.0)

    def test_fine_tuning_leaves_the_box_from_a_light_bed_alone(self) -> None:
        image = _light_bed_page()

        assert find_page_box(image, dpi=DPI, fine_tune=True) == find_page_box(
            image, dpi=DPI
        )

    def test_a_dark_bed_never_reaches_the_fallback(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def unexpected(*_args: object) -> None:
            raise AssertionError("the straight line fallback ran")

        monkeypatch.setattr(page_finder, "_find_line_edges", unexpected)

        box = find_page_box(_upright_page(), dpi=DPI)

        assert box == PageBox(x=29.0, y=39.0, width=242.0, height=322.0)

    def test_a_scan_with_neither_bed_nor_edges_has_no_page(self) -> None:
        assert find_page_box(np.full((300, 200), 238, np.uint8), dpi=DPI) is None


class TestReporterScan:
    """The scan attached to issue #5, reduced from 600 to 150 dpi.

    The Brother scanner trims the top edge of every scan and the scan was then
    turned 90 degrees anticlockwise, so the page touches the left border while
    the gray scanner background shows above, below and to the right of it.
    """

    @pytest.fixture
    def scan(self) -> NDArray[np.uint8]:
        image = cv2.imread(
            str(FIXTURES / "issue5_trimmed_edge_page.png"), cv2.IMREAD_GRAYSCALE
        )
        assert image is not None
        return np.asarray(image, dtype=np.uint8)

    def test_the_page_reaches_the_trimmed_edge_and_stops_at_the_background(
        self, scan: NDArray[np.uint8]
    ) -> None:
        page_id = PageId(image_id=ImageId(file_path=Path("issue5.png")))

        box = Filter().detect_page(scan, page_id, dpi=DPI)

        # The reporter's corners at 600 dpi: (0, 71), (3241, 63), (0, 5022),
        # (3236, 5032); a quarter of them at 150 dpi.
        assert box.x == 0
        assert box.y == pytest.approx(16.5, abs=3)
        assert box.x + box.width == pytest.approx(809.5, abs=3)
        assert box.y + box.height == pytest.approx(1257, abs=3)

    def test_fine_tuning_trims_the_corners_to_the_size_scantailor_universal_finds(
        self, scan: NDArray[np.uint8]
    ) -> None:
        box = find_page_box(scan, dpi=DPI, fine_tune=True)

        # scantailor-universal crops this scan to 3220 x 4953 at 600 dpi.
        assert box is not None
        assert box.width == pytest.approx(3220 / 4, abs=3)
        assert box.height == pytest.approx(4953 / 4, abs=3)

    def test_the_trimmed_edge_page_never_reaches_the_fallback(
        self, scan: NDArray[np.uint8], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def unexpected(*_args: object) -> None:
            raise AssertionError("the straight line fallback ran")

        monkeypatch.setattr(page_finder, "_find_line_edges", unexpected)

        assert find_page_box(scan, dpi=DPI) is not None


class TestFilterDetectPage:
    PAGE = PageId(image_id=ImageId(file_path=Path("crooked.png")))

    def _detected(self, fine_tune: bool) -> PageBox:
        filter_ = Filter()
        params = filter_.get_params(self.PAGE).with_fine_tune(fine_tune)
        filter_.set_params(self.PAGE, params)
        return filter_.detect_page(_crooked_page(8.0), self.PAGE, dpi=DPI)

    def test_the_fine_tune_setting_of_the_page_changes_the_detected_box(
        self,
    ) -> None:
        plain = self._detected(fine_tune=False)
        tuned = self._detected(fine_tune=True)

        assert tuned.width < plain.width
        assert tuned.height < plain.height

    def test_the_expected_page_size_in_the_settings_is_used(self) -> None:
        filter_ = Filter()
        filter_.settings.set_page_detection_box(10.0, 10.0)
        image = _upright_page()

        box = filter_.detect_page(image, self.PAGE, dpi=DPI)

        assert box == PageBox.from_image_size(300, 400)
