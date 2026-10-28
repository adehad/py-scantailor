"""Folding line detection: thin dark vertical lines, found where they were drawn."""

import cv2
import numpy as np
import pytest
from numpy.typing import NDArray

from scantailor.filters.page_split.vert_line_finder import find_lines

from . import scan_pages as pages


def _page_with_lines(
    xs: list[int], width: int = 700, height: int = 500
) -> NDArray[np.uint8]:
    page = pages.blank(width, height)
    for x in xs:
        pages.add_vertical_band(page, x, x + 6, 45)
    return page


MM_PER_INCH = 25.4
# The finder works at 100 dpi (0.25 mm a pixel) and votes for lines in a band 5
# of those pixels (1.3 mm) wide, in 0.25 degree steps. A found line may lean
# anywhere inside the band, so its ends may sit a band width and a pixel off.
LINE_TOLERANCE_MM = 1.5


def _tolerance(dpi: float) -> float:
    """``LINE_TOLERANCE_MM`` in pixels.

    Args:
        dpi (float): Resolution of the page.

    Returns:
        float: The tolerance in pixels at ``dpi``.
    """
    return LINE_TOLERANCE_MM / MM_PER_INCH * dpi


def _x_at(line: tuple[tuple[float, float], tuple[float, float]], y: float) -> float:
    """Column at which a line, extended if need be, crosses row ``y``.

    Args:
        line (tuple[tuple[float, float], tuple[float, float]]): Two points on a
            line that is not horizontal.
        y (float): The row.

    Returns:
        float: The column.
    """
    (x1, y1), (x2, y2) = line
    return x1 + (x2 - x1) * (y - y1) / (y2 - y1)


@pytest.mark.parametrize(
    ("dpi", "size", "stroke_x"),
    [(300.0, (1500, 1000), 768.0), (150.0, (750, 500), 393.0)],
    ids=["300dpi", "150dpi"],
)
def test_a_folding_line_is_found_within_1_5_mm_of_its_stroke(
    dpi: float, size: tuple[int, int], stroke_x: float
) -> None:
    """The stroke is 8 pixels wide, 14 to 22 pixels right of the centre."""
    lines = find_lines(pages.spread_with_folding_line(*size), dpi)

    assert len(lines) == 1
    for y in (0.0, float(size[1])):
        assert _x_at(lines[0], y) == pytest.approx(stroke_x, abs=_tolerance(dpi))


def test_a_line_is_found_at_a_resolution_that_is_no_multiple_of_100() -> None:
    lines = find_lines(_page_with_lines([300]), 210.0)

    assert len(lines) == 1
    for y in (0.0, 500.0):
        assert _x_at(lines[0], y) == pytest.approx(303.0, abs=_tolerance(210.0))


def test_a_tilted_line_is_found_with_its_tilt() -> None:
    page = pages.blank(700, 500)
    foot = 330 + 500 * np.tan(np.radians(3.0))
    cv2.line(page, (330, 0), (round(foot), 500), pages.INK, 6)

    lines = find_lines(page, 300.0)

    assert len(lines) == 1
    assert _x_at(lines[0], 0.0) == pytest.approx(330.0, abs=_tolerance(300.0))
    assert _x_at(lines[0], 500.0) == pytest.approx(foot, abs=_tolerance(300.0))


def test_lines_near_both_side_edges_of_a_page_are_both_found() -> None:
    lines = sorted(
        find_lines(pages.single_page_with_edge_lines(), 300.0),
        key=lambda line: line[0][0],
    )

    assert len(lines) == 2
    assert _x_at(lines[0], 700.0) == pytest.approx(81.0, abs=_tolerance(300.0))
    assert _x_at(lines[1], 700.0) == pytest.approx(919.0, abs=_tolerance(300.0))


def test_a_dark_strip_along_the_edge_is_one_line_at_its_inner_edge() -> None:
    """The strip covers columns 0 to 60; the edge of the image is no line."""
    lines = find_lines(pages.single_page_with_dark_gutter_strip(), 300.0)

    assert len(lines) == 1
    assert _x_at(lines[0], 700.0) == pytest.approx(60.0, abs=_tolerance(300.0))


def test_a_wide_black_gutter_is_one_line_inside_the_gutter() -> None:
    """The gutter covers columns 710 to 790."""
    lines = find_lines(pages.spread_with_black_gutter(), 300.0)

    assert len(lines) == 1
    assert 710.0 <= _x_at(lines[0], 500.0) <= 790.0


def test_black_borders_inside_the_edge_margin_are_no_lines() -> None:
    """The borders are 36 pixels wide, under the 3.5 mm margin of 41 pixels."""
    assert find_lines(pages.spread_with_dark_borders(), 300.0) == []


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
