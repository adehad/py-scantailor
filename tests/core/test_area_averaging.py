"""Averaging every source pixel an output pixel covers when a warp reduces."""

import cv2
import numpy as np
import pytest
from numpy.typing import NDArray

from scantailor.core.models import Dpi
from scantailor.core.transformation import (
    ImageTransformation,
    _box_taps,
    _then,
    _translation_matrix,
    _whiten_outside,
    render_transformed,
    warp_area_averaged,
)

_PAPER = 236
_INK = 30


def _transformation(
    width: int, height: int, source_dpi: int, output_dpi: int, skew: float = 0.0
) -> ImageTransformation:
    xform = ImageTransformation.from_size(width, height, Dpi.uniform(source_dpi))
    if skew:
        xform.set_post_rotation(skew)
    xform.post_scale_to_dpi(Dpi.uniform(output_dpi))
    return xform


def _centres_matrix(xform: ImageTransformation) -> NDArray[np.float64]:
    """Output pixel centres to source pixel centres, as render_transformed builds it."""
    rect = xform.resulting_rect
    return _then(
        _translation_matrix(rect.x + 0.5, rect.y + 0.5),
        xform.transform_back,
        _translation_matrix(-0.5, -0.5),
    )


def _plain_bilinear(
    image: NDArray[np.uint8], xform: ImageTransformation
) -> NDArray[np.uint8]:
    rect = xform.resulting_rect
    size = (max(1, round(rect.width)), max(1, round(rect.height)))
    return np.asarray(
        cv2.warpAffine(
            image,
            _centres_matrix(xform)[:2],
            size,
            flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(255,) * 4,
        ),
        dtype=np.uint8,
    )


def _line_ladder(width: int, height: int, gap: int) -> tuple[NDArray[np.uint8], int]:
    """Horizontal one pixel lines on paper, ``gap`` blank rows apart."""
    page = np.full((height, width), _PAPER, dtype=np.uint8)
    rows = range(20, height - 20, gap + 1)
    for row in rows:
        page[row, 20 : width - 20] = _INK
    return page, len(rows)


def _ink_components(image: NDArray[np.uint8]) -> int:
    _, ink = cv2.threshold(image, 0, 1, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    count, _ = cv2.connectedComponents(ink, connectivity=8)
    return count - 1


def _test_page(seed: int) -> NDArray[np.uint8]:
    """A small scan: line ladders, a halftone screen, text and sensor noise."""
    rng = np.random.default_rng(seed)
    page = np.full((300, 240), _PAPER, dtype=np.uint8)
    for index, row in enumerate(range(10, 90, 4)):
        page[row : row + 1 + index % 2, 10:230] = _INK
    for column in range(10, 120, 5):
        page[100:180, column] = _INK
    yy, xx = np.mgrid[0:110, 0:100].astype(np.float64)
    period = 600 / 150
    u, v = (xx + yy) / np.sqrt(2), (yy - xx) / np.sqrt(2)
    screen = (np.cos(2 * np.pi * u / period) + np.cos(2 * np.pi * v / period)) / 4
    page[100:210, 130:230][screen + 0.5 < 0.2 + 0.6 * xx / 99] = _INK
    cv2.putText(page, "Hamburg 0123", (10, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.6, _INK)
    page = cv2.GaussianBlur(page, (0, 0), 0.6)
    return np.clip(page + rng.normal(0, 2.0, page.shape), 0, 255).astype(np.uint8)


class TestNoReduction:
    """Warps that do not shrink the source stay plain bilinear."""

    @pytest.mark.parametrize(
        ("source_dpi", "output_dpi", "skew"),
        [(300, 300, 0.0), (300, 300, 7.0), (300, 300, -45.0), (300, 600, 3.0)],
    )
    def test_equal_or_higher_dpi_is_byte_identical_to_bilinear(
        self, source_dpi: int, output_dpi: int, skew: float
    ) -> None:
        """At the scan's DPI or above, with any deskew, no pixel changes."""
        image = _test_page(1)
        xform = _transformation(240, 300, source_dpi, output_dpi, skew)

        result = render_transformed(image, xform)

        expected = _plain_bilinear(image, xform)
        rect = xform.resulting_rect
        _whiten_outside(expected, xform.resulting_pre_crop_area - [rect.x, rect.y])
        np.testing.assert_array_equal(result, expected)

    def test_footprint_within_upstream_rounding_of_one_pixel_is_not_filtered(
        self,
    ) -> None:
        """Upstream rounds the box to 1/32 pixel, so 1.01 pixels is still one."""
        assert _box_taps(1.0) is None
        assert _box_taps(1.01) is None
        assert _box_taps(0.5) is None
        assert _box_taps(1.05) is not None


class TestBoxTaps:
    """The taps on the source that, with the bilinear warp, stand in for the box."""

    @pytest.mark.parametrize("width", [2.0, 3.0, 4.0, 5.0, 6.0])
    def test_whole_width_is_a_plain_box(self, width: float) -> None:
        """A box of whole pixels averages that many pixels with equal weights."""
        taps = _box_taps(width)

        assert taps is not None
        np.testing.assert_allclose(taps.weights, np.full(int(width), 1 / width))
        assert taps.centre == (0.5 if width % 2 == 0 else 0.0)

    @pytest.mark.parametrize("width", [1.25, 43 / 32, 1.5, 2.5, 3.0, 3.5, 5.25])
    def test_taps_add_the_variance_bilinear_lacks(self, width: float) -> None:
        """Taps plus the bilinear tent spread a pixel as upstream's box does."""
        taps = _box_taps(width)

        assert taps is not None
        offsets = np.arange(len(taps.weights)) - taps.anchor - taps.centre
        assert taps.weights.sum() == pytest.approx(1.0, abs=1e-6)
        assert np.sum(taps.weights * offsets) == pytest.approx(0.0, abs=1e-6)
        variance = np.sum(taps.weights * offsets**2)
        assert variance == pytest.approx((width**2 - 1) / 12, abs=1e-5)

    def test_widths_just_under_two_keep_the_two_pixel_box(self) -> None:
        """Centred on a pixel edge, two taps are the least spread available."""
        taps = _box_taps(1.75)

        assert taps is not None
        np.testing.assert_allclose(taps.weights, [0.5, 0.5])


class TestReduction:
    """Warps that shrink the source average every pixel an output pixel covers."""

    @pytest.mark.parametrize("output_dpi", [300, 200])
    def test_one_pixel_lines_stay_whole_when_reduced(self, output_dpi: int) -> None:
        """A slightly skewed ladder of 1 px lines keeps one component per line."""
        gap = 600 // output_dpi * 3
        page, lines = _line_ladder(600, 400, gap)
        xform = _transformation(600, 400, 600, output_dpi, skew=0.7)

        result = render_transformed(page, xform)

        assert _ink_components(result) == lines

    def test_colour_channels_are_each_averaged_alike(self) -> None:
        """A BGR page reduces channel by channel, as the grey channels would."""
        rng = np.random.default_rng(3)
        bgr = rng.integers(0, 256, size=(120, 90, 3), dtype=np.uint8)
        xform = _transformation(90, 120, 600, 250, skew=2.0)

        result = render_transformed(bgr, xform)

        for channel in range(3):
            np.testing.assert_array_equal(
                result[..., channel],
                render_transformed(np.ascontiguousarray(bgr[..., channel]), xform),
            )

    def test_single_channel_axis_survives_a_reduction(self) -> None:
        """An (H, W, 1) page comes back (H, W, 1) when the filter runs."""
        image = _test_page(4)[..., np.newaxis]
        xform = _transformation(240, 300, 600, 200)

        result = render_transformed(image, xform)

        assert result.ndim == 3
        assert result.shape[2] == 1

    def test_filtering_only_the_sampled_window_matches_filtering_everything(
        self,
    ) -> None:
        """Skipping source no output pixel reaches leaves the result unchanged."""
        image = np.random.default_rng(5).integers(0, 256, (200, 300), dtype=np.uint8)
        matrix = np.array([[3.0, 0.2, 150.5], [-0.1, 3.0, 40.0], [0.0, 0.0, 1.0]])
        taps = _box_taps(3.0)
        assert taps is not None

        result = warp_area_averaged(image, matrix, (40, 30))

        whole = cv2.sepFilter2D(
            image,
            -1,
            taps.weights,
            taps.weights,
            anchor=(taps.anchor, taps.anchor),
            borderType=cv2.BORDER_REPLICATE,
        )
        expected = cv2.warpAffine(
            whole,
            matrix[:2],
            (40, 30),
            flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=255,
        )
        np.testing.assert_array_equal(result, expected)

    def test_output_wholly_outside_the_source_is_white(self) -> None:
        """A reduction that samples no source pixel leaves the output blank."""
        image = np.zeros((50, 50), dtype=np.uint8)
        matrix = np.array([[2.0, 0.0, 500.0], [0.0, 2.0, 500.0], [0.0, 0.0, 1.0]])

        result = warp_area_averaged(image, matrix, (10, 10))

        assert np.all(result == 255)


def _halftone(coverage: float) -> NDArray[np.uint8]:
    """A 240x300 page of a 150 lpi screen at 600 DPI.

    Args:
        coverage (float): Fraction of the page the screen inks, from 0 to 1.

    Returns:
        NDArray[np.uint8]: The page, ink at ``_INK`` and paper at ``_PAPER``.
    """
    yy, xx = np.mgrid[0:300, 0:240].astype(np.float64)
    period = 600 / 150
    u, v = (xx + yy) / np.sqrt(2), (yy - xx) / np.sqrt(2)
    screen = (np.cos(2 * np.pi * u / period) + np.cos(2 * np.pi * v / period)) / 4
    return np.where(screen + 0.5 < coverage, _INK, _PAPER).astype(np.uint8)


# Pixels this far from the output edge may mix in the white outside the scan.
_INNER = (slice(8, -8), slice(8, -8))


class TestAreaAveragingProperties:
    """What averaging the covered source area guarantees, whatever the arithmetic."""

    @pytest.mark.parametrize("output_dpi", [300, 200])
    def test_a_whole_factor_reduction_without_skew_is_the_block_mean(
        self, output_dpi: int
    ) -> None:
        """At 2x and 3x each output pixel is the mean of its source block.

        The filter works in floating point and rounds once, so a block mean
        ending in exactly .5 may round either way: one level of tolerance.
        """
        page = _test_page(280)
        factor = 600 // output_dpi
        xform = _transformation(240, 300, 600, output_dpi)

        result = render_transformed(page, xform)

        height, width = 300 // factor, 240 // factor
        blocks = page[: height * factor, : width * factor].reshape(
            height, factor, width, factor
        )
        expected = blocks.mean(axis=(1, 3))
        assert result.shape == (height, width)
        assert np.abs(result - expected).max() <= 1.0

    @pytest.mark.parametrize("coverage", [0.3, 0.5])
    @pytest.mark.parametrize(
        ("output_dpi", "skew"), [(300, 0.7), (200, 0.0), (200, 0.7), (450, 0.7)]
    )
    def test_a_halftone_keeps_its_grey_and_ripples_less_than_bilinear(
        self, coverage: float, output_dpi: int, skew: float
    ) -> None:
        """Averaging conserves the ink, and a screen finer than the output
        pixel is smoothed rather than aliased into moire.

        Measured on these screens bilinear sampling leaves a spread of 65 to
        103 levels; area averaging cuts it by at least 8% at 450 DPI and by a
        third or more at 300 and 200 DPI, so 5% is a margin, not a fit.
        """
        page = _halftone(coverage)
        xform = _transformation(240, 300, 600, output_dpi, skew)

        result = render_transformed(page, xform)[_INNER].astype(np.float64)
        bilinear = _plain_bilinear(page, xform)[_INNER].astype(np.float64)

        assert result.mean() == pytest.approx(page.mean(), abs=1.0)
        assert result.std() < 0.95 * bilinear.std()

    @pytest.mark.parametrize("output_dpi", [300, 200])
    def test_a_dense_ladder_of_thin_lines_never_breaks_into_dashes(
        self, output_dpi: int
    ) -> None:
        """Twenty lines of 1 and 2 px on a 4 px pitch, slightly skewed.

        Reduced, neighbouring lines may merge, but averaging never leaves a
        gap along a line, so there are at most as many pieces as lines.
        Bilinear sampling skips source rows and breaks some lines apart.
        """
        page = _test_page(281)[:95]
        xform = _transformation(240, 95, 600, output_dpi, skew=0.7)

        result = render_transformed(page, xform)

        assert 1 <= _ink_components(result) <= 20

    def test_bilinear_sampling_does_break_the_dense_ladder(self) -> None:
        """Shows the ladder test above can fail: the same page sampled alone."""
        page = _test_page(281)[:95]
        xform = _transformation(240, 95, 600, 300, skew=0.7)

        assert _ink_components(_plain_bilinear(page, xform)) > 20

    @pytest.mark.parametrize("output_dpi", [450, 400])
    def test_a_fractional_reduction_is_closer_to_the_exact_area_mean_than_bilinear(
        self, output_dpi: int
    ) -> None:
        """Without skew OpenCV's area resize is the exact area mean to compare to.

        The taps match the box's spread, not its shape, so single pixels can
        differ from the exact mean by more than bilinear's do; the root mean
        square error over the page is still lower.
        """
        page = _test_page(282)
        xform = _transformation(240, 300, 600, output_dpi)

        result = render_transformed(page, xform)
        bilinear = _plain_bilinear(page, xform)

        height, width = result.shape
        exact = cv2.resize(page, (width, height), interpolation=cv2.INTER_AREA)
        inner = (slice(3, -3), slice(3, -3))

        def error(image: NDArray[np.uint8]) -> float:
            difference = image[inner].astype(np.float64) - exact[inner]
            return float(np.sqrt(np.mean(difference**2)))

        assert error(result) < error(bilinear)
