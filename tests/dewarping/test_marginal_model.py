"""Marginal dewarping model: curves bend to the page edges near the spine."""

import numpy as np
import pytest

from scantailor.dewarping.marginal_model import build_marginal_model

HEIGHT, WIDTH = 300, 400
CONTENT = (50.0, 60.0, 300.0, 180.0)
PAPER = 220
BACKGROUND = 10


def _page_with_curled_edge(spine_on_right: bool) -> np.ndarray:
    """Dark background; the paper's top and bottom edges recede near the spine."""
    page = np.full((HEIGHT, WIDTH), BACKGROUND, dtype=np.uint8)
    for x in range(WIDTH):
        distance = (WIDTH - x) if spine_on_right else x
        recess = max(0, 120 - distance) // 4
        page[20 + recess : HEIGHT - 20 - recess, x] = PAPER
    return page


class TestMarginalModel:
    """Models from a synthetic page."""

    def test_curves_keep_the_content_ends_and_bend_toward_the_edges_at_the_spine(
        self,
    ) -> None:
        model = build_marginal_model(
            _page_with_curled_edge(True), CONTENT, spine_on_right=True
        )

        assert model is not None
        top = model.top_curve.polyline
        bottom = model.bottom_curve.polyline
        assert top[0, 0] == pytest.approx(CONTENT[0])
        assert top[-1, 0] == pytest.approx(CONTENT[0] + CONTENT[2])
        # Away from the spine the edge is the paper's top edge at row 20.
        assert top[0, 1] == pytest.approx(20.0, abs=1.0)
        assert bottom[0, 1] == pytest.approx(HEIGHT - 21.0, abs=1.0)
        # At the spine end the paper has receded by about 17 px.
        assert top[-1, 1] > top[0, 1] + 10.0
        assert bottom[-1, 1] < bottom[0, 1] - 10.0

    def test_left_page_binding_bends_the_other_end(self) -> None:
        model = build_marginal_model(
            _page_with_curled_edge(False), CONTENT, spine_on_right=False
        )

        assert model is not None
        top = model.top_curve.polyline
        assert top[0, 1] > top[-1, 1] + 10.0

    def test_flat_page_gives_straight_curves(self) -> None:
        page = np.full((HEIGHT, WIDTH), PAPER, dtype=np.uint8)

        model = build_marginal_model(page, CONTENT, spine_on_right=True)

        assert model is not None
        assert np.ptp(model.top_curve.polyline[:, 1]) == pytest.approx(0.0, abs=1e-6)
        assert np.ptp(model.bottom_curve.polyline[:, 1]) == pytest.approx(0.0, abs=1e-6)

    def test_empty_content_rect_gives_no_model(self) -> None:
        page = np.full((HEIGHT, WIDTH), PAPER, dtype=np.uint8)

        assert build_marginal_model(page, (10.0, 10.0, 0.0, 50.0), True) is None

    def test_curves_are_monotone_in_x_and_never_overshoot_their_control_points(
        self,
    ) -> None:
        model = build_marginal_model(
            _page_with_curled_edge(True), CONTENT, spine_on_right=True
        )

        assert model is not None
        top = model.top_curve.polyline
        assert np.all(np.diff(top[:, 0]) > 0)
        assert top[:, 1].min() >= 19.0
