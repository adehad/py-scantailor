"""A page without content comes out blank, as upstream's Output leaves it."""

import numpy as np
import pytest

from scantailor.filters.output.color_mode import ColorMode
from scantailor.filters.output.color_options import ColorCommonOptions
from scantailor.filters.output.generator import generate_output
from scantailor.filters.output.params import Params
from scantailor.filters.output.raster import encloses_no_pixel
from scantailor.filters.output.splitting_options import SplittingOptions

# Page Layout's stand-in for a page without content: a square of 0.02 px.
_SPECK = [(49.99, 49.99), (50.01, 49.99), (50.01, 50.01), (49.99, 50.01)]


def _page_with_an_edge_band() -> np.ndarray:
    """A light page with only a dark band along its left edge, like a book's edge."""
    page = np.full((100, 120), 200, dtype=np.uint8)
    page[:, :8] = 40
    return page


class TestEnclosesNoPixel:
    def test_page_layouts_stand_in_for_no_content_covers_no_pixel(self) -> None:
        assert encloses_no_pixel(_SPECK)

    def test_a_polygon_without_height_covers_no_pixel(self) -> None:
        assert encloses_no_pixel([(0.0, 5.0), (40.0, 5.0), (40.0, 5.2), (0.0, 5.2)])

    def test_a_one_pixel_square_covers_a_pixel(self) -> None:
        assert not encloses_no_pixel([(3.0, 3.0), (4.0, 3.0), (4.0, 4.0), (3.0, 4.0)])

    @pytest.mark.parametrize("polygon", [[], [(1.0, 1.0)], [(0.0, 0.0), (9.0, 9.0)]])
    def test_fewer_than_three_vertices_is_no_polygon_and_not_too_small(
        self, polygon: list[tuple[float, float]]
    ) -> None:
        assert not encloses_no_pixel(polygon)


class TestBlankOutput:
    @pytest.mark.parametrize(
        "mode", [ColorMode.BLACK_AND_WHITE, ColorMode.COLOR_GRAYSCALE, ColorMode.MIXED]
    )
    def test_content_too_small_for_a_pixel_gives_a_white_page(
        self, mode: ColorMode
    ) -> None:
        page = _page_with_an_edge_band()

        result = generate_output(page, Params(color_mode=mode), content_polygon=_SPECK)

        assert result.is_binary
        assert result.image.shape == page.shape
        assert (result.image == 255).all()

    def test_without_filled_margins_the_whole_page_is_kept(self) -> None:
        params = Params(
            color_mode=ColorMode.BLACK_AND_WHITE,
            color_common=ColorCommonOptions(fill_margins=False),
        )

        result = generate_output(
            _page_with_an_edge_band(), params, content_polygon=_SPECK
        )

        assert (result.image[:, :8] == 0).any(axis=1).all()

    def test_split_output_of_a_blank_page_has_white_layers(self) -> None:
        params = Params(
            color_mode=ColorMode.MIXED,
            splitting=SplittingOptions(split_output=True, original_background=True),
        )

        result = generate_output(
            _page_with_an_edge_band(), params, content_polygon=_SPECK
        )

        assert result.split is not None
        assert (result.split.foreground == 255).all()
        assert (result.split.background == 255).all()
