"""Content-based versus top-edge skew detection in the Deskew filter."""

import math
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from scantailor.core.models import ImageId, PageId, SubPage
from scantailor.filters.deskew.filter import Filter
from scantailor.filters.deskew.settings import Settings
from scantailor.filters.deskew.top_edge import find_skew_from_top_edge
from tests.filters.deskew.synthetic_pages import sheared_page, text_page

PAGE = PageId(
    image_id=ImageId(file_path=Path("/book.tiff")), sub_page=SubPage.SINGLE_PAGE
)


def book_scan(
    edge_degrees: float, height: int = 600, width: int = 800, edge_row: int = 60
) -> NDArray[np.uint8]:
    """Return a black surround above a white page with a tilted top edge."""
    columns = np.arange(width)
    edge = edge_row + np.tan(math.radians(edge_degrees)) * (columns - width / 2)
    rows = np.arange(height)[:, None]
    return np.where(rows < edge[None, :], 0, 255).astype(np.uint8)


class TestTopEdgeDetection:
    @pytest.mark.parametrize("degrees", [-4.0, -1.5, 2.0, 5.0])
    def test_the_tilt_of_the_top_edge_is_found_with_good_confidence(self, degrees):
        result = find_skew_from_top_edge(book_scan(degrees))

        assert result.is_confident()
        assert result.angle == pytest.approx(degrees, abs=0.3)

    def test_a_tilt_beyond_the_maximum_is_clamped(self):
        result = find_skew_from_top_edge(book_scan(20.0), max_angle=7.0)

        assert result.angle == pytest.approx(7.0)

    def test_a_straight_edge_gives_zero_with_good_confidence(self):
        result = find_skew_from_top_edge(book_scan(0.0))

        assert result.angle == 0.0
        assert result.is_confident()

    def test_a_tilt_below_the_minimum_is_reported_as_zero(self):
        result = find_skew_from_top_edge(book_scan(0.3), min_angle=0.5)

        assert result.angle == 0.0
        assert result.is_confident()

    def test_a_page_without_a_dark_surround_has_no_confident_result(self):
        # Every column is light from its first row, so no surround ends anywhere.
        result = find_skew_from_top_edge(sheared_page(0.0))

        assert result.angle == 0.0
        assert not result.is_confident()

    def test_a_surround_that_does_not_reach_the_top_is_not_an_edge(self):
        # Ink that starts below a light strip could as well be a line of text.
        scan = book_scan(3.0)
        scan[:8, :] = 255

        assert not find_skew_from_top_edge(scan).is_confident()

    def test_one_light_row_at_the_very_top_is_tolerated(self):
        scan = book_scan(3.0)
        scan[0, :] = 255

        result = find_skew_from_top_edge(scan)

        assert result.is_confident()
        assert result.angle == pytest.approx(3.0, abs=0.3)

    def test_the_edge_row_is_the_last_row_of_the_first_light_window(self):
        from scantailor.filters.deskew.top_edge import _edge_row

        scan = book_scan(0.0, edge_row=60)

        # Rows 59, 60, 61 are the first three with at most one ink pixel.
        assert _edge_row(scan[:200, 400] == 0) == 61

    def test_a_column_that_is_light_from_its_first_row_has_no_edge(self):
        from scantailor.filters.deskew.top_edge import _edge_row

        column = np.zeros(200, dtype=bool)
        column[100:] = True

        assert _edge_row(column) is None

    def test_an_image_that_is_all_ink_in_the_search_band_has_no_edge(self):
        result = find_skew_from_top_edge(np.zeros((600, 800), dtype=np.uint8))

        assert result.angle == 0.0
        assert not result.is_confident()

    @pytest.mark.parametrize("shape", [(9, 100), (100, 9)])
    def test_a_tiny_image_has_no_confident_result(self, shape):
        result = find_skew_from_top_edge(np.zeros(shape, dtype=np.uint8))

        assert not result.is_confident()

    def test_the_search_stops_at_a_third_of_the_height_or_200_rows(self):
        # An edge 250 rows down is beyond the 200-row band of a tall image.
        tall = book_scan(0.0, height=900, edge_row=250)

        assert not find_skew_from_top_edge(tall).is_confident()


class TestFilterModes:
    def test_content_based_is_the_default(self):
        assert Filter().content_based is True

    def test_content_mode_ignores_the_page_edge(self):
        filter_ = Filter()
        page = text_page()
        page[:40, :] = 0  # dark surround, perfectly straight text below

        result = filter_.detect_skew(page, PAGE, store_result=False)

        assert abs(result.angle) < 0.5

    def test_top_edge_mode_follows_the_page_edge_not_the_content(self):
        filter_ = Filter()
        filter_.content_based = False

        result = filter_.detect_skew(book_scan(3.0), PAGE)

        assert result.angle == pytest.approx(3.0, abs=0.3)
        assert filter_.get_deskew_angle(PAGE) == pytest.approx(-3.0, abs=0.3)

    def test_the_two_modes_disagree_on_a_tilted_edge_above_straight_text(self):
        scan = book_scan(3.0)
        text = text_page(height=600, width=800)
        scan[120:, :] = np.minimum(scan[120:, :], text[120:, :])

        content = Filter()
        edge = Filter()
        edge.content_based = False

        assert abs(content.detect_skew(scan, PAGE, store_result=False).angle) < 1.0
        assert edge.detect_skew(scan, PAGE, store_result=False).angle == (
            pytest.approx(3.0, abs=0.3)
        )

    def test_a_page_without_a_dark_surround_falls_back_to_its_content(self):
        filter_ = Filter()
        filter_.content_based = False

        result = filter_.detect_skew(sheared_page(0.0), PAGE, store_result=False)
        content = Filter().detect_skew(sheared_page(0.0), PAGE, store_result=False)

        assert result == content

    def test_a_weak_top_edge_result_stores_no_rotation(self):
        filter_ = Filter()
        filter_.content_based = False

        filter_.detect_skew(np.zeros((600, 800), dtype=np.uint8), PAGE)

        assert filter_.get_deskew_angle(PAGE) == 0.0


class TestPersistence:
    def test_settings_start_content_based(self):
        assert Settings().get_content_based() is True

    def test_flag_survives_a_json_round_trip(self):
        settings = Settings()
        settings.set_content_based(False)

        restored = Settings.model_validate_json(settings.model_dump_json())

        assert restored.get_content_based() is False

    def test_a_settings_dump_without_the_flag_loads_as_content_based(self):
        restored = Settings.model_validate({"auto_oblique_default": True})

        assert restored.get_content_based() is True

    def test_the_filter_reads_and_writes_the_settings_flag(self):
        settings = Settings()
        filter_ = Filter(settings)

        filter_.content_based = False

        assert settings.get_content_based() is False


class TestSignAgreesWithContentDetection:
    @pytest.mark.parametrize("degrees", [-3.0, 3.0])
    def test_top_edge_and_content_corrections_turn_the_same_way(self, degrees):
        import cv2

        surround = np.full((800, 600), 0, dtype=np.uint8)
        page = text_page(height=700, width=500)
        surround[50:750, 50:550] = page
        matrix = cv2.getRotationMatrix2D((300, 400), degrees, 1.0)
        scan = cv2.warpAffine(surround, matrix, (600, 800), borderValue=0)

        content = Filter()
        edge = Filter()
        edge.content_based = False
        content.detect_skew(scan, PAGE)
        edge.detect_skew(scan, PAGE)

        content_angle = content.get_deskew_angle(PAGE)
        edge_angle = edge.get_deskew_angle(PAGE)
        assert content_angle * edge_angle > 0
