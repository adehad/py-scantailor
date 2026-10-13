"""The estimator reproduces the cutters of upstream 1.2.1 on generated scans.

``cpp_reference`` follows the C++ step by step, and every case compares the
layout type and where each cutter crosses the top and bottom of the image.
"""

import numpy as np
import pytest
from numpy.typing import NDArray

from scantailor.filters.page_split.layout_type import LayoutType
from scantailor.filters.page_split.page_layout import PageLayout
from scantailor.filters.page_split.split_finder import detect_split

from . import cpp_reference as cpp
from . import scan_pages as pages

# Half a pixel of the input, which is a quarter of a pixel at the 150 dpi the
# whitespace analysis works in.
TOLERANCE = 0.5

UPSTREAM_TYPES = {
    LayoutType.AUTO_LAYOUT_TYPE: cpp.AUTO,
    LayoutType.PAGE_PLUS_OFFCUT: cpp.PLUS_OFFCUT,
    LayoutType.TWO_PAGES: cpp.TWO_PAGES,
}

SCENARIOS = {
    "clean_spread": pages.two_page_spread,
    "spread_with_folding_line": pages.spread_with_folding_line,
    "spread_with_black_gutter": pages.spread_with_black_gutter,
    "spread_with_dark_borders": pages.spread_with_dark_borders,
    "skewed_spread": lambda: pages.skewed_spread(1.5),
    "spread_with_empty_page": pages.spread_with_empty_page,
    "single_page": pages.single_page,
    "single_page_left_offcut": pages.single_page_with_left_offcut,
    "single_page_right_offcut": pages.single_page_with_right_offcut,
    "single_page_missing_inside_edge": pages.single_page_missing_inside_edge,
    "single_page_dark_strip": pages.single_page_with_dark_gutter_strip,
    "single_page_edge_lines": pages.single_page_with_edge_lines,
    "single_page_right_edge_line": pages.single_page_with_right_edge_line,
}


def _crossings(
    line: tuple[tuple[float, float], tuple[float, float]], height: int
) -> tuple[float, float]:
    """X of a line at the top and at the bottom of the image."""
    return cpp.x_at(line, 0.0), cpp.x_at(line, float(height))


def assert_matches_upstream(
    page: NDArray[np.uint8], requested: LayoutType, dpi: float
) -> PageLayout:
    height = page.shape[0]
    layout = detect_split(page, requested, dpi).layout
    kind, upstream1, upstream2 = cpp.estimate_page_layout(
        UPSTREAM_TYPES[requested], page, dpi
    )

    assert layout.layout_type.value == kind
    if upstream1 is None:
        assert layout.cutter1 is None
        return layout
    assert layout.cutter1 is not None
    np.testing.assert_allclose(
        _crossings(layout.cutter1, height),
        _crossings(upstream1, height),
        atol=TOLERANCE,
    )
    if kind == cpp.CUT:
        assert layout.cutter2 is not None and upstream2 is not None
        np.testing.assert_allclose(
            _crossings(layout.cutter2, height),
            _crossings(upstream2, height),
            atol=TOLERANCE,
        )
    return layout


@pytest.mark.slow
@pytest.mark.parametrize("requested", list(UPSTREAM_TYPES), ids=lambda t: t.value)
@pytest.mark.parametrize("scenario", list(SCENARIOS))
def test_cutters_match_upstream_at_300_dpi(
    scenario: str, requested: LayoutType
) -> None:
    assert_matches_upstream(SCENARIOS[scenario](), requested, 300.0)


@pytest.mark.slow
@pytest.mark.parametrize("requested", list(UPSTREAM_TYPES), ids=lambda t: t.value)
def test_cutters_match_upstream_at_600_dpi(requested: LayoutType) -> None:
    """Twice the working resolution: the 300 dpi step reduces exact 2x2 blocks."""
    page = np.repeat(np.repeat(pages.two_page_spread(1000, 680), 2, axis=0), 2, axis=1)

    assert_matches_upstream(page, requested, 600.0)


@pytest.mark.slow
@pytest.mark.parametrize("dpi", [150.0, 450.0, 72.0], ids=lambda d: f"{d:g}dpi")
def test_cutters_match_upstream_at_resolutions_that_need_fractional_scaling(
    dpi: float,
) -> None:
    height = int(680 * dpi / 300)
    width = int(1000 * dpi / 300)
    page = pages.two_page_spread(width, height)

    assert_matches_upstream(page, LayoutType.AUTO_LAYOUT_TYPE, dpi)
    assert_matches_upstream(page, LayoutType.TWO_PAGES, dpi)


@pytest.mark.slow
@pytest.mark.parametrize("angle", [-3.0, -0.8, 0.8, 2.6, 5.0])
def test_cutters_follow_the_skew_the_way_upstream_does(angle: float) -> None:
    layout = assert_matches_upstream(
        pages.skewed_spread(angle), LayoutType.TWO_PAGES, 300.0
    )

    assert layout.cutter1 is not None
    top, bottom = _crossings(layout.cutter1, 1000)
    # OpenCV rotates counter-clockwise for a positive angle, which moves the
    # bottom of the gutter to the right.
    assert np.sign(bottom - top) == np.sign(angle)
