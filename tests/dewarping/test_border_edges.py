"""The top and bottom edge tracer ignores the scan's own border.

A straight edge hugging the first or the last row of the traced image is the
scanner's border, not the paper's edge, and upstream's habit of taking it for
one flattens the bottom of the model (issue 227). Paper edges further in, and
edges that bend, still count.
"""

import numpy as np
import pytest

from scantailor.dewarping.text_line_tracer import trace_text_lines
from scantailor.dewarping.top_bottom_edge_tracer import (
    is_image_border,
    trace_top_bottom_edges,
)
from tests.dewarping.book_pages import BentPage, bent_book_page, justified_text_page

CONTENT = (160, 200, 890, 1090)
BOUNDS = (((172.0, 0.0), (172.0, 1600.0)), ((1032.0, 0.0), (1032.0, 1600.0)))
PAPER = 235
BORDER = 40


def _page_on_light_ground() -> np.ndarray:
    """A flat page of text whose paper fills the image: no edge to trace."""
    flat, _ = justified_text_page()
    page = np.full((1600, 1200), PAPER, dtype=np.uint8)
    page[150:1450, 100:1100] = flat[:1300, :1000]
    return page


@pytest.fixture(scope="module")
def bent_page() -> BentPage:
    return bent_book_page(bow=45.0, squeeze=-25.0)


@pytest.fixture(scope="module")
def bent_bounds(bent_page: BentPage):
    result = trace_text_lines(bent_page.image, 200.0, CONTENT).vertical_bounds
    assert result is not None
    return (result.first, result.second)


class TestTracedEdges:
    def test_a_bent_page_on_a_dark_surround_keeps_both_real_edges(
        self, bent_page: BentPage, bent_bounds
    ) -> None:
        curves = sorted(
            trace_top_bottom_edges(bent_page.image, bent_bounds),
            key=lambda curve: curve[:, 1].mean(),
        )

        assert len(curves) == 2
        top, bottom = curves
        assert np.abs(top[:, 1] - np.interp(top[:, 0], *bent_page.top.T)).max() < 3.0
        assert (
            np.abs(bottom[:, 1] - np.interp(bottom[:, 0], *bent_page.bottom.T)).max()
            < 3.0
        )

    def test_a_dark_band_along_the_last_rows_gives_no_edge(self) -> None:
        page = _page_on_light_ground()
        page[-12:, :] = BORDER

        assert trace_top_bottom_edges(page, BOUNDS) == []

    def test_a_dark_band_along_the_first_rows_gives_no_edge(self) -> None:
        page = _page_on_light_ground()
        page[:10, :] = BORDER

        assert trace_top_bottom_edges(page, BOUNDS) == []

    def test_a_straight_paper_edge_away_from_the_border_still_counts(self) -> None:
        # A flat page whose paper ends well inside the scan, on a dark ground.
        page = np.full((1600, 1200), BORDER, dtype=np.uint8)
        page[100:1500, :] = _page_on_light_ground()[100:1500, :]

        curves = trace_top_bottom_edges(page, BOUNDS)

        assert len(curves) == 2
        heights = sorted(float(curve[:, 1].mean()) for curve in curves)
        assert heights[0] == pytest.approx(100.0, abs=4.0)
        assert heights[1] == pytest.approx(1500.0, abs=4.0)


class TestBorderRule:
    HEIGHT = 1000

    @staticmethod
    def _curve(ys: list[float]) -> np.ndarray:
        xs = np.linspace(700.0, 50.0, len(ys))
        return np.column_stack([xs, ys])

    def test_a_straight_curve_within_the_band_of_the_last_row_is_the_border(
        self,
    ) -> None:
        assert is_image_border(self._curve([990.0, 991.0, 990.5, 991.0]), self.HEIGHT)

    def test_a_straight_curve_within_the_band_of_the_first_row_is_the_border(
        self,
    ) -> None:
        assert is_image_border(self._curve([3.0, 4.0, 3.5, 4.0]), self.HEIGHT)

    def test_the_band_is_one_and_a_half_percent_of_the_height(self) -> None:
        assert is_image_border(self._curve([986.0, 986.0, 986.0]), self.HEIGHT)
        assert not is_image_border(self._curve([983.0, 983.0, 983.0]), self.HEIGHT)

    def test_the_band_is_at_least_three_pixels_on_a_tiny_image(self) -> None:
        assert is_image_border(self._curve([2.0, 2.5, 2.0]), 100)
        assert not is_image_border(self._curve([4.0, 4.0, 4.0]), 100)

    def test_a_curve_that_bends_more_than_two_pixels_is_a_paper_edge(self) -> None:
        bent = self._curve([992.0, 989.0, 986.0, 989.0, 992.0])

        assert not is_image_border(bent, self.HEIGHT)

    def test_a_tilted_straight_curve_is_still_the_border(self) -> None:
        tilted = self._curve([996.0, 995.0, 994.0, 993.0, 992.0])

        assert is_image_border(tilted, self.HEIGHT)

    def test_a_curve_leaving_the_band_at_one_end_is_not_the_border(self) -> None:
        assert not is_image_border(self._curve([995.0, 995.0, 960.0]), self.HEIGHT)

    def test_an_empty_curve_is_not_the_border(self) -> None:
        assert not is_image_border(np.empty((0, 2)), self.HEIGHT)
