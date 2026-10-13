"""Pixels with no source behind them take the margin colour in every mode."""

import numpy as np
import pytest

from scantailor.filters.output.color_options import ColorCommonOptions, FillingColor
from scantailor.filters.output.generator import generate_output
from scantailor.filters.output.params import ColorMode, Params

PAPER = 120
HEIGHT, WIDTH = 80, 120
SOURCE_LEFT = 30


def _view() -> np.ndarray:
    """A paper-coloured page whose left 30 columns are outside the source."""
    image = np.full((HEIGHT, WIDTH), 255, dtype=np.uint8)
    image[:, SOURCE_LEFT:] = PAPER
    return image


def _source_outline() -> list[tuple[float, float]]:
    return [
        (SOURCE_LEFT, 0),
        (WIDTH, 0),
        (WIDTH, HEIGHT),
        (SOURCE_LEFT, HEIGHT),
    ]


def _grayscale(**options: object) -> Params:
    return Params(
        color_mode=ColorMode.COLOR_GRAYSCALE,
        color_common=ColorCommonOptions(**options),
    )


@pytest.mark.parametrize(
    "options",
    [
        {"fill_margins": False, "fill_offcut": False},
        {"fill_margins": False, "fill_offcut": True, "fill_outside_page_box": True},
    ],
)
def test_added_margins_take_the_paper_colour_not_white(options: dict) -> None:
    result = generate_output(
        _view(), _grayscale(**options), source_outline=_source_outline()
    )

    assert set(np.unique(result.image[:, :SOURCE_LEFT])) == {PAPER}
    assert set(np.unique(result.image[:, SOURCE_LEFT:])) == {PAPER}


def test_without_an_outline_the_added_margin_stays_as_rendered() -> None:
    params = _grayscale(fill_margins=False, fill_offcut=False)

    result = generate_output(_view(), params)

    assert set(np.unique(result.image[:, :SOURCE_LEFT])) == {255}


def test_the_filling_colour_black_fills_outside_the_source() -> None:
    params = _grayscale(
        fill_margins=False, fill_offcut=False, filling_color=FillingColor.BLACK
    )

    result = generate_output(_view(), params, source_outline=_source_outline())

    assert set(np.unique(result.image[:, :SOURCE_LEFT])) == {0}


def test_an_outline_covering_the_image_changes_nothing() -> None:
    params = _grayscale(fill_margins=False, fill_offcut=False)
    whole = [(0, 0), (WIDTH, 0), (WIDTH, HEIGHT), (0, HEIGHT)]

    with_outline = generate_output(_view(), params, source_outline=whole)

    assert np.array_equal(with_outline.image, generate_output(_view(), params).image)


def test_binary_output_keeps_the_added_margin_white() -> None:
    params = Params(
        color_mode=ColorMode.BLACK_AND_WHITE,
        color_common=ColorCommonOptions(fill_margins=False, fill_offcut=False),
    )

    result = generate_output(_view(), params, source_outline=_source_outline())

    assert set(np.unique(result.image[:, :SOURCE_LEFT])) == {255}
