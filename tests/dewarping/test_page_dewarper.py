"""Dewarpers made from manual edges and detected from a page."""

import numpy as np
import pytest

from scantailor.dewarping.distortion_model import create_distortion_model
from scantailor.dewarping.page_dewarper import (
    auto_dewarper,
    dewarper_from_model,
    dewarper_from_polylines,
    marginal_dewarper,
)
from tests.dewarping.book_pages import bent_book_page

TOP = [(0.0, 10.0), (50.0, 5.0), (100.0, 10.0)]
BOTTOM = [(0.0, 90.0), (50.0, 85.0), (100.0, 90.0)]
CONTENT = (160.0, 200.0, 890.0, 1090.0)
DPI = (200.0, 200.0)


class TestManual:
    """Dewarpers from given edges."""

    def test_valid_edges_give_a_dewarper(self) -> None:
        assert dewarper_from_polylines(TOP, BOTTOM, 2.0) is not None

    def test_an_edge_with_one_point_gives_none(self) -> None:
        assert dewarper_from_polylines(TOP[:1], BOTTOM, 2.0) is None
        assert dewarper_from_polylines(TOP, [], 2.0) is None

    def test_crossed_edges_give_none(self) -> None:
        assert dewarper_from_polylines(BOTTOM, TOP[::-1], 2.0) is None

    def test_missing_model_gives_none(self) -> None:
        assert dewarper_from_model(None, 2.0) is None

    def test_model_gives_the_same_dewarper_as_its_edges(self) -> None:
        model = create_distortion_model(np.array(TOP), np.array(BOTTOM))

        dewarper = dewarper_from_model(model, 2.0)

        assert dewarper is not None
        reference = dewarper_from_polylines(TOP, BOTTOM, 2.0)
        assert reference is not None
        assert dewarper.directrix_bounding_box() == pytest.approx(
            reference.directrix_bounding_box()
        )


class TestDetected:
    """Dewarpers found from page content."""

    def test_a_bent_page_gives_a_dewarper_around_the_text(self) -> None:
        page = bent_book_page()

        dewarper = auto_dewarper(page.image, DPI, CONTENT, 2.0)

        assert dewarper is not None
        left, top, right, bottom = dewarper.directrix_bounding_box()
        assert 90.0 < left < 200.0
        assert right > 1000.0
        assert top < bottom

    def test_the_resolution_may_be_given_as_one_number(self) -> None:
        page = bent_book_page()

        assert auto_dewarper(page.image, 200.0, CONTENT, 2.0) is not None

    def test_a_blank_page_gives_none(self) -> None:
        blank = np.full((1600, 1200), 235, dtype=np.uint8)

        assert auto_dewarper(blank, DPI, CONTENT, 2.0) is None

    def test_marginal_dewarper_is_built_for_a_page_with_paper_edges(self) -> None:
        page = np.full((300, 400), 220, dtype=np.uint8)

        dewarper = marginal_dewarper(page, (50.0, 60.0, 300.0, 180.0), True, 2.0)

        assert dewarper is not None
