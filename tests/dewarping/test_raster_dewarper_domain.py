"""Precomputed dewarp maps and placing the model inside an output of a given size."""

import numpy as np
import pytest

from scantailor.dewarping.dewarper import CylindricalSurfaceDewarper
from scantailor.dewarping.raster_dewarper import (
    InterpolationMethod,
    compute_dewarp_maps,
    dewarp_image,
    output_model_domain,
    remap_with_maps,
)

WIDTH, HEIGHT = 100, 80


def _straight_dewarper() -> CylindricalSurfaceDewarper:
    """A model whose directrices are the page's own top and bottom edges."""
    top = np.array([[0.0, 0.0], [float(WIDTH), 0.0]])
    bottom = np.array([[0.0, float(HEIGHT)], [float(WIDTH), float(HEIGHT)]])
    return CylindricalSurfaceDewarper.from_directrices(top, bottom)


def _textured_image() -> np.ndarray:
    rng = np.random.default_rng(3)
    return rng.integers(0, 256, size=(HEIGHT, WIDTH), dtype=np.uint8)


class TestIdentityModel:
    """A model covering the whole page leaves its pixels where they are."""

    def test_domain_of_the_whole_image_is_the_unit_square(self) -> None:
        domain = output_model_domain(_straight_dewarper(), (WIDTH, HEIGHT))

        assert domain == pytest.approx((0.0, 0.0, 1.0, 1.0), abs=1e-9)

    def test_dewarping_reproduces_the_source_exactly(self) -> None:
        """Pixel centres map onto pixel centres, so bilinear sampling adds no blur."""
        image = _textured_image()
        dewarper = _straight_dewarper()
        domain = output_model_domain(dewarper, (WIDTH, HEIGHT))

        result = dewarp_image(image, dewarper, (WIDTH, HEIGHT), domain)

        np.testing.assert_array_equal(result, image)

    def test_a_smaller_model_is_stretched_over_its_own_box(self) -> None:
        """Margins stay margins: the model box keeps its place in the output."""
        top = np.array([[20.0, 10.0], [80.0, 10.0]])
        bottom = np.array([[20.0, 70.0], [80.0, 70.0]])
        dewarper = CylindricalSurfaceDewarper.from_directrices(top, bottom)

        left, upper, right, lower = output_model_domain(dewarper, (WIDTH, HEIGHT))

        # Model x = 0 sits at output x = 20, so the domain starts at -20 / 60.
        assert left == pytest.approx(-20.0 / 60.0)
        assert right == pytest.approx((WIDTH - 20.0) / 60.0)
        assert upper == pytest.approx(-10.0 / 60.0)
        assert lower == pytest.approx((HEIGHT - 10.0) / 60.0)

    def test_a_model_without_extent_is_rejected(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        dewarper = _straight_dewarper()
        monkeypatch.setattr(dewarper, "directrix_bounding_box", lambda: (0, 5, 0, 5))

        with pytest.raises(ValueError, match="no extent"):
            output_model_domain(dewarper, (WIDTH, HEIGHT))


class TestMaps:
    """Maps are computed once and reused for every image of a page."""

    def test_remapping_with_maps_equals_dewarp_image(self) -> None:
        image = _textured_image()
        dewarper = _straight_dewarper()
        domain = (0.05, 0.0, 0.95, 1.0)
        maps = compute_dewarp_maps(dewarper, (WIDTH, HEIGHT), domain)

        expected = dewarp_image(image, dewarper, (WIDTH, HEIGHT), domain)
        result = remap_with_maps(image, maps, 255)

        np.testing.assert_array_equal(result, expected)

    def test_maps_have_the_output_shape(self) -> None:
        map_x, map_y = compute_dewarp_maps(
            _straight_dewarper(), (30, 20), (0.0, 0.0, 1.0, 1.0)
        )

        assert map_x.shape == map_y.shape == (20, 30)

    def test_empty_domain_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="positive width and height"):
            compute_dewarp_maps(_straight_dewarper(), (10, 10), (0.5, 0.0, 0.5, 1.0))

    def test_outside_pixels_take_the_background_colour(self) -> None:
        image = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
        maps = compute_dewarp_maps(
            _straight_dewarper(), (WIDTH, HEIGHT), (-0.5, 0.0, 0.5, 1.0)
        )

        result = remap_with_maps(image, maps, 200)

        assert np.all(result[:, : WIDTH // 2] == 200)
        assert np.all(result[:, WIDTH // 2 + 2 :] == 0)

    def test_nearest_sampling_keeps_binary_masks_binary(self) -> None:
        mask = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
        mask[20:60, 30:70] = 255
        maps = compute_dewarp_maps(
            _straight_dewarper(), (WIDTH, HEIGHT), (0.0, 0.0, 1.0, 1.0)
        )

        result = remap_with_maps(mask, maps, 0, InterpolationMethod.NEAREST)

        assert set(np.unique(result)) <= {0, 255}
        np.testing.assert_array_equal(result, mask)

    def test_unsupported_shape_is_rejected(self) -> None:
        maps = compute_dewarp_maps(_straight_dewarper(), (10, 10), (0, 0, 1, 1))

        with pytest.raises(ValueError, match="Unsupported image format"):
            remap_with_maps(np.zeros((10, 10, 2), dtype=np.uint8), maps)
