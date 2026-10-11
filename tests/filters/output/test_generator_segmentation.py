"""Colour segmentation, original background and split output through generate_output."""

import numpy as np
import pytest

from scantailor.core.models import Dpi
from scantailor.filters.output.binarization import (
    BinarizationOptions,
    ColorSegmenterOptions,
)
from scantailor.filters.output.color_mode import ColorMode
from scantailor.filters.output.color_options import ColorCommonOptions
from scantailor.filters.output.despeckle import DespeckleLevel
from scantailor.filters.output.generator import generate_output
from scantailor.filters.output.params import Params
from scantailor.filters.output.picture_shape import PictureShape, PictureShapeOptions
from scantailor.filters.output.split_output import ForegroundType
from scantailor.filters.output.splitting_options import SplittingOptions

_H, _W = 200, 300
_RED = (0, 0, 200)
_BLUE = (200, 0, 0)


def _page() -> np.ndarray:
    """White BGR page with a red and a blue bar of text-like ink."""
    page = np.full((_H, _W, 3), 255, dtype=np.uint8)
    page[40:56, 20:140] = _RED
    page[100:116, 20:140] = _BLUE
    return page


def _params(color_mode: ColorMode, *, segmented: bool, **kwargs) -> Params:
    binarization = BinarizationOptions(
        normalize_illumination=False,
        savitzky_golay_smoothing=False,
        morphological_smoothing=False,
        color_segmenter=ColorSegmenterOptions(enabled=segmented, noise_reduction=0),
    )
    return Params(
        color_mode=color_mode,
        output_dpi=Dpi.uniform(300),
        despeckle_level=DespeckleLevel.OFF,
        binarization=binarization,
        color_common=ColorCommonOptions(fill_margins=False, fill_offcut=False),
        picture_shape=PictureShapeOptions(shape=PictureShape.OFF),
        **kwargs,
    )


class TestBinaryColourSegmentation:
    def test_ink_takes_the_colours_of_the_page(self):
        result = generate_output(
            _page(), _params(ColorMode.BLACK_AND_WHITE, segmented=True)
        )

        assert result.is_binary is False
        assert tuple(result.image[48, 80]) == _RED
        assert tuple(result.image[108, 80]) == _BLUE

    def test_paper_stays_white(self):
        result = generate_output(
            _page(), _params(ColorMode.BLACK_AND_WHITE, segmented=True)
        )

        assert (result.image[150:, :] == 255).all()

    def test_without_segmentation_the_output_is_bilevel(self):
        result = generate_output(
            _page(), _params(ColorMode.BLACK_AND_WHITE, segmented=False)
        )

        assert result.is_binary is True
        assert set(np.unique(result.image)) <= {0, 255}

    def test_white_on_black_inverts_the_coloured_page(self):
        black_on_white = generate_output(
            _page(), _params(ColorMode.BLACK_AND_WHITE, segmented=True)
        )
        white_on_black = generate_output(
            _page(),
            _params(ColorMode.BLACK_AND_WHITE, segmented=True, black_on_white=False),
        )

        assert white_on_black.image.shape == black_on_white.image.shape
        assert not np.array_equal(white_on_black.image, black_on_white.image)

    def test_the_input_is_not_modified(self):
        page = _page()
        before = page.copy()

        generate_output(page, _params(ColorMode.BLACK_AND_WHITE, segmented=True))

        np.testing.assert_array_equal(page, before)


class TestMixedColourSegmentation:
    def test_text_is_coloured_in_the_text_region(self):
        result = generate_output(_page(), _params(ColorMode.MIXED, segmented=True))

        assert tuple(result.image[48, 80]) == _RED
        assert tuple(result.image[108, 80]) == _BLUE

    def test_plain_mixed_text_is_black(self):
        result = generate_output(_page(), _params(ColorMode.MIXED, segmented=False))

        assert tuple(result.image[48, 80]) == (0, 0, 0)

    def test_ink_mask_marks_the_drawn_text(self):
        result = generate_output(_page(), _params(ColorMode.MIXED, segmented=True))

        assert result.ink_mask is not None
        assert result.ink_mask[48, 80]
        assert not result.ink_mask[150, 10]


class TestOriginalBackground:
    def _params(self, **kwargs) -> Params:
        splitting = SplittingOptions(split_output=True, original_background=True)
        return _params(ColorMode.MIXED, segmented=False, splitting=splitting, **kwargs)

    def test_paper_keeps_its_original_colour(self):
        page = _page()
        page[150:, :] = (230, 235, 240)

        result = generate_output(page, self._params())

        assert tuple(result.image[180, 10]) == (230, 235, 240)

    def test_ink_is_pure_black(self):
        result = generate_output(_page(), self._params())

        assert tuple(result.image[48, 80]) == (0, 0, 0)

    def test_split_layers_are_produced(self):
        result = generate_output(_page(), self._params())

        assert result.split is not None
        assert result.split.original_background is not None


class TestSplitOutput:
    def _split(self, **options):
        splitting = SplittingOptions(split_output=True, **options)
        return generate_output(
            _page(),
            _params(ColorMode.MIXED, segmented=False, splitting=splitting),
        ).split

    def test_no_layers_unless_asked_for(self):
        result = generate_output(_page(), _params(ColorMode.MIXED, segmented=False))

        assert result.split is None

    def test_binary_foreground_holds_the_text(self):
        split = self._split()

        assert split is not None
        assert split.foreground_type == ForegroundType.BINARY
        assert split.foreground.ndim == 2
        assert split.foreground[48, 80] == 0
        assert split.foreground[150, 10] == 255

    def test_colour_foreground_is_not_binarised(self):
        split = self._split(color_foreground=True)

        assert split is not None
        assert split.foreground_type == ForegroundType.COLOR
        assert tuple(split.foreground[48, 80]) == _RED

    def test_binary_output_never_splits(self):
        splitting = SplittingOptions(split_output=True)
        result = generate_output(
            _page(),
            _params(ColorMode.BLACK_AND_WHITE, segmented=False, splitting=splitting),
        )

        assert result.split is None

    @pytest.mark.parametrize("black_on_white", [True, False])
    def test_layers_use_black_text_on_white_either_way(self, black_on_white):
        splitting = SplittingOptions(split_output=True)
        page = _page() if black_on_white else 255 - _page()
        result = generate_output(
            page,
            _params(
                ColorMode.MIXED,
                segmented=False,
                splitting=splitting,
                black_on_white=black_on_white,
            ),
        )

        assert result.split is not None
        assert result.split.foreground[48, 80] == 0
        assert result.split.foreground[150, 10] == 255
