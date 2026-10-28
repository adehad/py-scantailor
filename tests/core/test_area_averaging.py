"""Averaging source pixels when a warp reduces, as upstream's transformGeneric does."""

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


def _upstream_transform_generic(
    image: NDArray[np.uint8], xform: ImageTransformation
) -> NDArray[np.uint8]:
    """NumPy port of upstream imageproc::transformGeneric for grey pixels.

    Every output pixel averages an axis-aligned box of source pixels, centred on
    its mapped centre and sized by upstream's calcSrcUnitSize, with the box
    edges in 1/32 pixel units and partly covered pixels weighted by coverage.
    Pixels outside the source are ignored rather than mixed in (upstream's weak
    outside mode). Box sums come from the integral image, which is bilinear
    within each cell, so fractional edges are exact. Too slow for full pages.
    """
    source_height, source_width = image.shape
    rect = xform.resulting_rect
    inverse = np.diag([32.0, 32.0, 1.0]) @ _then(
        _translation_matrix(rect.x, rect.y), xform.transform_back
    )
    edge_midpoints = np.array([[0.5, 1, 0.5, 0], [0, 0.5, 1, 0.5], [1, 1, 1, 1]])
    mapped = inverse @ edge_midpoints
    unit_width = max(1, round(max(0.9 * 32, float(np.ptp(mapped[0])))))
    unit_height = max(1, round(max(0.9 * 32, float(np.ptp(mapped[1])))))
    integral = cv2.integral(image, sdepth=cv2.CV_64F)

    def integral_at(x: NDArray[np.float64], y: NDArray[np.float64]):
        x0 = np.clip(np.floor(x).astype(np.int64), 0, source_width - 1)
        y0 = np.clip(np.floor(y).astype(np.int64), 0, source_height - 1)
        fx, fy = x - x0, y - y0
        return (
            integral[y0, x0] * (1 - fx) * (1 - fy)
            + integral[y0, x0 + 1] * fx * (1 - fy)
            + integral[y0 + 1, x0] * (1 - fx) * fy
            + integral[y0 + 1, x0 + 1] * fx * fy
        )

    out_width = max(1, round(rect.width))
    out_height = max(1, round(rect.height))
    grid_x, grid_y = np.meshgrid(
        np.arange(out_width) + 0.5, np.arange(out_height) + 0.5
    )
    centre_x = grid_x * inverse[0, 0] + grid_y * inverse[0, 1] + inverse[0, 2]
    centre_y = grid_x * inverse[1, 0] + grid_y * inverse[1, 1] + inverse[1, 2]
    left = np.trunc(centre_x).astype(np.int64) - (unit_width >> 1)
    top = np.trunc(centre_y).astype(np.int64) - (unit_height >> 1)
    right, bottom = left + unit_width, top + unit_height
    outside = (
        (right <= 0)
        | (bottom <= 0)
        | (left >= source_width * 32)
        | (top >= source_height * 32)
    )
    x0 = np.clip(left, 0, source_width * 32) / 32
    x1 = np.clip(right, 0, source_width * 32) / 32
    y0 = np.clip(top, 0, source_height * 32) / 32
    y1 = np.clip(bottom, 0, source_height * 32) / 32
    area = (x1 - x0) * (y1 - y0)
    total = (
        integral_at(x1, y1)
        - integral_at(x0, y1)
        - integral_at(x1, y0)
        + integral_at(x0, y0)
    )
    average = np.floor(total / np.where(area > 0, area, 1) + 0.5)
    average[outside | (area <= 0)] = 255
    return np.clip(average, 0, 255).astype(np.uint8)


def _psnr(a: NDArray[np.uint8], b: NDArray[np.uint8]) -> float:
    mse = float(np.mean((a.astype(np.float64) - b) ** 2))
    return float("inf") if mse == 0 else 10 * np.log10(255**2 / mse)


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


class TestAgainstUpstream:
    """The reduction against a NumPy port of upstream's transformGeneric."""

    @pytest.mark.parametrize("output_dpi", [300, 200])
    @pytest.mark.parametrize("skew", [0.0, 0.7])
    def test_whole_factor_reduction_matches_upstream_box_average(
        self, output_dpi: int, skew: float
    ) -> None:
        """At 2x and 3x the result is within rounding noise of upstream's."""
        page = _test_page(280)
        xform = _transformation(240, 300, 600, output_dpi, skew)

        result = render_transformed(page, xform)
        upstream = _upstream_transform_generic(page, xform)

        # Upstream ignores pixels beyond the scan where this port reads white,
        # so the outermost output pixels are left out.
        inner = (slice(3, -3), slice(3, -3))
        assert _psnr(result[inner], upstream[inner]) > 45
        assert np.abs(result[inner].astype(int) - upstream[inner]).max() <= 6

    @pytest.mark.parametrize("output_dpi", [300, 200])
    def test_line_ladder_keeps_upstreams_component_count(self, output_dpi: int) -> None:
        """Binarized, the reduced ladder has exactly as many pieces as upstream's."""
        page = _test_page(281)[:95]
        xform = _transformation(240, 95, 600, output_dpi, skew=0.7)

        result = render_transformed(page, xform)
        upstream = _upstream_transform_generic(page, xform)

        assert _ink_components(result) == _ink_components(upstream)

    @pytest.mark.parametrize("output_dpi", [450, 400])
    def test_fractional_reduction_is_closer_to_upstream_than_bilinear(
        self, output_dpi: int
    ) -> None:
        """Between whole factors the taps still beat bilinear sampling alone."""
        page = _test_page(282)
        xform = _transformation(240, 300, 600, output_dpi, skew=0.7)

        result = render_transformed(page, xform)
        upstream = _upstream_transform_generic(page, xform)
        bilinear = _plain_bilinear(page, xform)

        inner = (slice(3, -3), slice(3, -3))
        assert _psnr(result[inner], upstream[inner]) > _psnr(
            bilinear[inner], upstream[inner]
        )
