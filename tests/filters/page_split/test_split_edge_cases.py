"""Page split on sample-book pages whose right answer the generator knows.

Each case states where the cutters belong from how ``tools/sample_book.py`` drew
the page: the spine column of a spread, the gap beside an offcut strip, the text
block of a single page. Tolerances are in millimetres so they mean the same at
every resolution.
"""

from types import ModuleType
from typing import Final

import cv2
import numpy as np
import pytest
from numpy.typing import NDArray

from scantailor.filters.page_split.layout_type import LayoutType
from scantailor.filters.page_split.page_layout import PageLayout, PageLayoutType
from scantailor.filters.page_split.split_finder import detect_split

from . import scan_pages as pages

MM_PER_INCH: Final = 25.4
SAMPLE_DPI: Final = 150.0
# Ink darker than this is text; paper, bed and the light gutter are all lighter.
INK_LEVEL: Final = 120
# A cut within 3 mm of the spine leaves each page its own gutter margin; the
# sample pages have about 15 mm of margin either side of the spine.
SPINE_TOLERANCE_MM: Final = 3.0
# The split of one scan at different resolutions may differ by a working pixel of
# the folding line finder (100 dpi, 0.25 mm) or of the whitespace analysis
# (150 dpi, 0.17 mm), plus resampling blur; 1 mm allows for both.
RESOLUTION_TOLERANCE_MM: Final = 1.0
# A cutter that lands on the paper edge is right; the folding line finder places
# an edge to within about 1 mm.
PAPER_EDGE_TOLERANCE_MM: Final = 1.5
# With content on one page only, the estimator cuts 15 working pixels (2.5 mm at
# 150 dpi) clear of it; 1 mm more allows for where the content box ends.
EMPTY_PAGE_CLEARANCE_MM: Final = 3.5
# The cutters take their tilt from the skew finder, whose 0.1 degree search
# lands up to one 0.125 degree step from the true angle.
CUTTER_TILT_TOLERANCE_DEG: Final = 0.15
SPREADS: Final = {"light": "02_spread_light_gutter", "black": "03_spread_black_gutter"}


def _gray(rgb: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Gray version of an RGB render.

    Args:
        rgb (NDArray[np.uint8]): (H, W, 3) RGB image.

    Returns:
        NDArray[np.uint8]: (H, W) gray image.
    """
    return np.asarray(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), dtype=np.uint8)


def _mm(pixels: float, dpi: float) -> float:
    """A distance in pixels converted to millimetres.

    Args:
        pixels (float): Distance in pixels.
        dpi (float): Resolution the distance was measured at.

    Returns:
        float: The distance in millimetres.
    """
    return pixels / dpi * MM_PER_INCH


def _mid_x(line: tuple[tuple[float, float], tuple[float, float]] | None) -> float:
    """Column halfway between a cutter's ends, asserting the cutter exists.

    Args:
        line (tuple[tuple[float, float], tuple[float, float]] | None): A cutter.

    Returns:
        float: The mean of its two x coordinates.
    """
    assert line is not None
    return 0.5 * (line[0][0] + line[1][0])


def _ink_columns(gray: NDArray[np.uint8]) -> NDArray[np.intp]:
    """Columns holding at least one pixel darker than ``INK_LEVEL``.

    Args:
        gray (NDArray[np.uint8]): (H, W) gray image.

    Returns:
        NDArray[np.intp]: The column indices, ascending.
    """
    return np.nonzero((gray < INK_LEVEL).any(axis=0))[0]


def _split_x(layout: PageLayout) -> float:
    """Middle of a layout's split line, asserting it has one.

    Args:
        layout (PageLayout): A two-page layout.

    Returns:
        float: The split column.
    """
    x = layout.get_split_line_x()
    assert x is not None
    return x


def _split(gray: NDArray[np.uint8], layout_type: LayoutType, dpi: float) -> PageLayout:
    """The layout page split detects on a gray page.

    Args:
        gray (NDArray[np.uint8]): (H, W) gray page.
        layout_type (LayoutType): Requested layout.
        dpi (float): Resolution of the page.

    Returns:
        PageLayout: The detected layout.
    """
    return detect_split(gray, layout_type, dpi).layout


@pytest.fixture(scope="module")
def sample_gray(sample_book_tool: ModuleType) -> dict[str, NDArray[np.uint8]]:
    """Gray renders of the sample pages these tests split.

    Returns:
        dict[str, NDArray[np.uint8]]: (H, W) gray pages keyed by sample name.
    """
    names = ("01_single_page", "04_offcut", *SPREADS.values())
    return {
        name: _gray(sample_book_tool.render_sample_page(name).image) for name in names
    }


class TestSpreads:
    @pytest.mark.parametrize("gutter", list(SPREADS))
    def test_a_spread_is_cut_in_the_blank_gutter_between_its_text_blocks(
        self,
        sample_book_tool: ModuleType,
        sample_gray: dict[str, NDArray[np.uint8]],
        gutter: str,
    ) -> None:
        name = SPREADS[gutter]
        (_, left_text_end), (right_text_start, _) = sample_book_tool.text_block_columns(
            name
        )

        layout = _split(sample_gray[name], LayoutType.AUTO_LAYOUT_TYPE, SAMPLE_DPI)

        assert layout.layout_type is PageLayoutType.TWO_PAGES
        assert layout.cutter1 is not None
        for end in layout.cutter1:
            assert left_text_end < end[0] < right_text_start

    @pytest.mark.parametrize("gutter", list(SPREADS))
    def test_a_spread_is_cut_within_3_mm_of_its_spine(
        self,
        sample_book_tool: ModuleType,
        sample_gray: dict[str, NDArray[np.uint8]],
        gutter: str,
    ) -> None:
        layout = _split(
            sample_gray[SPREADS[gutter]], LayoutType.AUTO_LAYOUT_TYPE, SAMPLE_DPI
        )

        offset = _split_x(layout) - sample_book_tool.SPREAD_SPINE_X
        assert abs(_mm(offset, SAMPLE_DPI)) <= SPINE_TOLERANCE_MM

    @pytest.mark.slow
    @pytest.mark.parametrize("gutter", list(SPREADS))
    def test_a_spread_splits_at_the_same_place_in_mm_at_any_resolution(
        self, sample_book_tool: ModuleType, gutter: str
    ) -> None:
        splits_mm = {}
        for dpi in (72.0, 150.0, 300.0, 600.0):
            rgb, _spine = sample_book_tool.render_spread_at(dpi, gutter == "black")
            layout = _split(_gray(rgb), LayoutType.AUTO_LAYOUT_TYPE, dpi)
            splits_mm[dpi] = _mm(_split_x(layout), dpi)

        spread = max(splits_mm.values()) - min(splits_mm.values())
        assert spread <= RESOLUTION_TOLERANCE_MM, splits_mm

    @pytest.mark.parametrize("dpi", [72.0, 150.0, 300.0, 450.0, 600.0])
    def test_a_spread_without_a_folding_line_splits_at_its_gutter_at_any_resolution(
        self, dpi: float
    ) -> None:
        """No dark line here, so the whitespace analysis decides at every dpi."""
        drawn = pages.two_page_spread()
        factor = dpi / 300.0
        size = (round(drawn.shape[1] * factor), round(drawn.shape[0] * factor))
        interpolation = cv2.INTER_AREA if factor < 1 else cv2.INTER_NEAREST
        page = np.asarray(cv2.resize(drawn, size, interpolation=interpolation))

        layout = _split(page.astype(np.uint8), LayoutType.AUTO_LAYOUT_TYPE, dpi)

        gutter_mm = _mm(drawn.shape[1] / 2, 300.0)
        split_mm = _mm(_split_x(layout), dpi)
        assert layout.layout_type is PageLayoutType.TWO_PAGES
        assert split_mm == pytest.approx(gutter_mm, abs=RESOLUTION_TOLERANCE_MM)

    @pytest.mark.parametrize("angle", [-3.0, 3.0])
    def test_a_skewed_spread_is_cut_along_its_tilted_gutter(self, angle: float) -> None:
        """The generator turns counter-clockwise for a positive angle, which moves
        the foot of the gutter to the right."""
        page = pages.skewed_spread(angle)
        height, width = page.shape

        layout = _split(page, LayoutType.TWO_PAGES, 300.0)

        assert layout.cutter1 is not None
        (x_top, _), (x_bottom, _) = layout.cutter1
        drift = x_bottom - x_top
        drift_tolerance = height * np.tan(np.radians(CUTTER_TILT_TOLERANCE_DEG))
        assert np.sign(drift) == np.sign(angle)
        assert drift == pytest.approx(
            height * np.tan(np.radians(angle)), abs=drift_tolerance
        )
        assert _mm(_split_x(layout) - width / 2, 300.0) == pytest.approx(
            0.0, abs=SPINE_TOLERANCE_MM
        )


class TestSinglePages:
    def test_a_single_page_without_a_fold_stays_one_page_with_all_its_text(
        self,
        sample_book_tool: ModuleType,
        sample_gray: dict[str, NDArray[np.uint8]],
    ) -> None:
        """Cutting the scanner bed off at the paper edges is allowed; cutting into
        the paper is not."""
        gray = sample_gray["01_single_page"]
        paper_left = sample_book_tool.BED_MARGIN
        paper_right = paper_left + sample_book_tool.PAPER_WIDTH
        tolerance = PAPER_EDGE_TOLERANCE_MM / MM_PER_INCH * SAMPLE_DPI

        layout = _split(gray, LayoutType.AUTO_LAYOUT_TYPE, SAMPLE_DPI)

        assert layout.layout_type in {
            PageLayoutType.SINGLE_PAGE_UNCUT,
            PageLayoutType.SINGLE_PAGE_CUT,
        }
        if layout.layout_type is PageLayoutType.SINGLE_PAGE_CUT:
            assert _mid_x(layout.cutter1) <= paper_left + tolerance
            assert _mid_x(layout.cutter2) >= paper_right - tolerance

    @pytest.mark.parametrize(
        "layout_type", [LayoutType.AUTO_LAYOUT_TYPE, LayoutType.PAGE_PLUS_OFFCUT]
    )
    def test_an_offcut_strip_is_cut_off_in_the_gap_beside_the_page(
        self,
        sample_book_tool: ModuleType,
        sample_gray: dict[str, NDArray[np.uint8]],
        layout_type: LayoutType,
    ) -> None:
        layout = _split(sample_gray["04_offcut"], layout_type, SAMPLE_DPI)

        assert layout.layout_type is PageLayoutType.SINGLE_PAGE_CUT
        assert layout.cutter2 is not None
        for x, _y in layout.cutter2:
            assert (
                sample_book_tool.OFFCUT_PAGE_RIGHT
                <= x
                <= sample_book_tool.OFFCUT_STRIP_LEFT
            )

    def test_an_offcut_strip_never_costs_the_page_its_text(
        self,
        sample_book_tool: ModuleType,
        sample_gray: dict[str, NDArray[np.uint8]],
    ) -> None:
        ((text_left, _),) = sample_book_tool.text_block_columns("04_offcut")
        page_ink = _ink_columns(
            sample_gray["04_offcut"][:, : sample_book_tool.OFFCUT_PAGE_RIGHT]
        )

        layout = _split(
            sample_gray["04_offcut"], LayoutType.AUTO_LAYOUT_TYPE, SAMPLE_DPI
        )

        assert layout.layout_type is not PageLayoutType.TWO_PAGES
        if layout.layout_type is PageLayoutType.SINGLE_PAGE_CUT:
            assert _mid_x(layout.cutter1) < text_left
            assert _mid_x(layout.cutter2) > page_ink.max()

    @pytest.mark.parametrize("side", ["left", "right"])
    def test_a_shadowed_edge_is_not_taken_for_a_spine_and_the_text_stays_whole(
        self, sample_book_tool: ModuleType, side: str
    ) -> None:
        """A dark band down one side is a shadow, not a fold between two pages.

        The page stays one page. A cutter may trim the shadow off, at its inner
        edge or further out, but never reaches the text.
        """
        gray = _gray(sample_book_tool.render_shadowed_edge_page(side))
        width = gray.shape[1]
        ((text_left, text_right),) = sample_book_tool.text_block_columns(
            "01_single_page"
        )
        shadow = sample_book_tool.SHADOW_WIDTH
        tolerance = PAPER_EDGE_TOLERANCE_MM / MM_PER_INCH * SAMPLE_DPI

        layout = _split(gray, LayoutType.AUTO_LAYOUT_TYPE, SAMPLE_DPI)

        assert layout.layout_type is not PageLayoutType.TWO_PAGES
        if layout.layout_type is PageLayoutType.SINGLE_PAGE_CUT:
            left, right = _mid_x(layout.cutter1), _mid_x(layout.cutter2)
            assert left < text_left and right > text_right
            if side == "left":
                assert left <= shadow + tolerance
            else:
                assert right >= width - shadow - tolerance

    def test_two_pages_forced_on_a_single_page_cut_just_clear_of_its_text(
        self, sample_gray: dict[str, NDArray[np.uint8]]
    ) -> None:
        """All the content lies on one page; the other is the empty margin beside it."""
        gray = sample_gray["01_single_page"]
        ink = _ink_columns(gray)

        layout = _split(gray, LayoutType.TWO_PAGES, SAMPLE_DPI)

        assert layout.layout_type is PageLayoutType.TWO_PAGES
        x = _split_x(layout)
        nearest_ink = ink.max() if x > ink.mean() else ink.min()
        assert x < ink.min() or x > ink.max()
        assert _mm(abs(x - nearest_ink), SAMPLE_DPI) <= EMPTY_PAGE_CLEARANCE_MM
