"""Folding line detection reproduces upstream's VertLineFinder."""

import cv2
import numpy as np
import pytest
from numpy.typing import NDArray

from scantailor.filters.page_split.vert_line_finder import find_lines

from . import cpp_reference as cpp
from . import scan_pages as pages


def _assert_same_lines(page: NDArray[np.uint8], dpi: float, max_lines: int = 8) -> list:
    got = find_lines(page, dpi, max_lines)
    want = cpp.vert_line_finder(page, dpi, max_lines)

    assert len(got) == len(want)
    for mine, theirs in zip(got, want, strict=True):
        np.testing.assert_allclose(mine, theirs, atol=1e-6)
    return got


def _page_with_lines(
    xs: list[int], width: int = 700, height: int = 500
) -> NDArray[np.uint8]:
    page = pages.blank(width, height)
    for x in xs:
        pages.add_vertical_band(page, x, x + 6, 45)
    return page


@pytest.mark.slow
@pytest.mark.parametrize(
    "factory",
    [
        pages.two_page_spread,
        pages.spread_with_folding_line,
        pages.spread_with_black_gutter,
        pages.spread_with_dark_borders,
        pages.single_page_with_edge_lines,
        pages.single_page_with_right_edge_line,
        pages.single_page_with_dark_gutter_strip,
    ],
)
def test_lines_match_upstream_at_300_dpi(factory) -> None:
    _assert_same_lines(factory(), 300.0)


@pytest.mark.slow
def test_lines_match_upstream_at_150_dpi() -> None:
    _assert_same_lines(pages.spread_with_folding_line(750, 500), 150.0)


@pytest.mark.slow
def test_lines_match_upstream_at_a_resolution_that_is_no_multiple_of_100() -> None:
    _assert_same_lines(_page_with_lines([300]), 210.0)


@pytest.mark.slow
def test_a_skewed_page_gives_the_same_tilted_line() -> None:
    page = pages.skewed_spread(2.0)
    page[:, 740:748] = np.minimum(page[:, 740:748], 45)

    _assert_same_lines(page, 300.0)


def test_a_thin_dark_line_is_found_in_the_middle_of_the_page() -> None:
    lines = find_lines(pages.spread_with_folding_line(), 300.0)

    assert len(lines) >= 1
    xs = [0.5 * (a[0] + b[0]) for a, b in lines]
    assert any(abs(x - 773) < 8 for x in xs)


def test_a_line_runs_from_the_top_to_the_bottom_of_the_image() -> None:
    lines = find_lines(_page_with_lines([300]), 300.0)

    (x1, y1), (x2, y2) = lines[0]
    assert y1 == pytest.approx(0.0)
    assert y2 == pytest.approx(500.0, abs=3.0)
    assert x1 == pytest.approx(x2, abs=3.0)
    assert x1 == pytest.approx(303, abs=6.0)


def test_a_clean_page_has_no_lines() -> None:
    assert find_lines(pages.two_page_spread(), 300.0) == []


def test_a_blank_page_has_no_lines() -> None:
    assert find_lines(pages.blank(400, 300), 300.0) == []


def test_lines_closer_than_the_margin_to_the_edge_are_ignored() -> None:
    # 3.5 mm at 100 dpi is 14 pixels, so 42 pixels at 300 dpi.
    assert find_lines(_page_with_lines([10]), 300.0) == []


def test_never_returns_more_lines_than_asked() -> None:
    page = _page_with_lines([60, 160, 260, 360, 460, 560], width=700)

    assert len(find_lines(page, 300.0, max_lines=3)) == 3
    assert len(find_lines(page, 300.0, max_lines=8)) == 6


def test_lines_close_together_are_reported_once() -> None:
    page = _page_with_lines([300, 312])

    assert len(find_lines(page, 300.0)) == 1


def test_a_short_dark_mark_is_not_a_line() -> None:
    page = pages.blank(700, 500)
    pages.add_vertical_band(page, 300, 306, 45, y0=100, y1=170)

    assert find_lines(page, 300.0) == []


@pytest.mark.parametrize("dpi", [100.0, 150.0, 200.0, 300.0])
def test_letters_stacked_in_columns_are_not_a_folding_line_at_any_resolution(
    dpi: float,
) -> None:
    assert find_lines(pages.spread_of_one_repeated_sentence(dpi), dpi) == []


def test_a_thin_dark_line_is_found_in_a_100_dpi_scan() -> None:
    page = cv2.resize(
        pages.spread_with_folding_line(), (500, 333), interpolation=cv2.INTER_AREA
    )

    xs = [0.5 * (a[0] + b[0]) for a, b in find_lines(page, 100.0)]

    assert any(abs(x - 773 / 3) < 4 for x in xs)
