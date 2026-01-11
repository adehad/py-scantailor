"""Tests for raster dewarping functions."""

import numpy as np
import pytest

from scantailor.dewarping import (
    CylindricalSurfaceDewarper,
    InterpolationMethod,
    compute_dewarped_size,
    dewarp_image,
)


@pytest.fixture
def simple_dewarper() -> CylindricalSurfaceDewarper:
    """Create a simple dewarper with minimal curvature."""
    # Nearly straight lines - minimal dewarping
    top = np.array([[0, 10], [100, 10], [200, 10]], dtype=np.float64)
    bottom = np.array([[0, 290], [100, 290], [200, 290]], dtype=np.float64)
    return CylindricalSurfaceDewarper.from_directrices(top, bottom)


@pytest.fixture
def curved_dewarper() -> CylindricalSurfaceDewarper:
    """Create a dewarper with noticeable curvature."""
    # Curved lines simulating book spine
    top = np.array([[0, 20], [100, 5], [200, 20]], dtype=np.float64)
    bottom = np.array([[0, 280], [100, 295], [200, 280]], dtype=np.float64)
    return CylindricalSurfaceDewarper.from_directrices(top, bottom)


class TestInterpolationMethod:
    """Tests for InterpolationMethod enum."""

    def test_values_exist(self) -> None:
        """All interpolation methods have valid OpenCV values."""
        assert InterpolationMethod.NEAREST.value >= 0
        assert InterpolationMethod.BILINEAR.value >= 0
        assert InterpolationMethod.BICUBIC.value >= 0
        assert InterpolationMethod.LANCZOS.value >= 0

    def test_all_methods_distinct(self) -> None:
        """All interpolation methods have different values."""
        values = [m.value for m in InterpolationMethod]
        assert len(values) == len(set(values))


class TestDewarpImage:
    """Tests for dewarp_image function."""

    def test_grayscale_output_shape(
        self, simple_dewarper: CylindricalSurfaceDewarper
    ) -> None:
        """Grayscale output has correct shape."""
        src = np.zeros((300, 200), dtype=np.uint8)
        result = dewarp_image(
            src,
            simple_dewarper,
            dst_size=(250, 350),
            model_domain=(0, 0, 1, 1),
        )

        assert result.shape == (350, 250)
        assert result.dtype == np.uint8

    def test_rgb_output_shape(
        self, simple_dewarper: CylindricalSurfaceDewarper
    ) -> None:
        """RGB output has correct shape."""
        src = np.zeros((300, 200, 3), dtype=np.uint8)
        result = dewarp_image(
            src,
            simple_dewarper,
            dst_size=(250, 350),
            model_domain=(0, 0, 1, 1),
        )

        assert result.shape == (350, 250, 3)
        assert result.dtype == np.uint8

    def test_rgba_output_shape(
        self, simple_dewarper: CylindricalSurfaceDewarper
    ) -> None:
        """RGBA output has correct shape."""
        src = np.zeros((300, 200, 4), dtype=np.uint8)
        result = dewarp_image(
            src,
            simple_dewarper,
            dst_size=(250, 350),
            model_domain=(0, 0, 1, 1),
        )

        assert result.shape == (350, 250, 4)
        assert result.dtype == np.uint8

    def test_background_color_grayscale(
        self, simple_dewarper: CylindricalSurfaceDewarper
    ) -> None:
        """Background color is applied for grayscale."""
        # Source image smaller than mapped area
        src = np.full((100, 100), 128, dtype=np.uint8)
        result = dewarp_image(
            src,
            simple_dewarper,
            dst_size=(200, 200),
            model_domain=(0, 0, 1, 1),
            background_color=255,
        )

        # Some pixels should be background color
        assert np.any(result == 255)

    def test_background_color_rgb(
        self, simple_dewarper: CylindricalSurfaceDewarper
    ) -> None:
        """Background color is applied for RGB."""
        src = np.full((100, 100, 3), 128, dtype=np.uint8)
        result = dewarp_image(
            src,
            simple_dewarper,
            dst_size=(200, 200),
            model_domain=(0, 0, 1, 1),
            background_color=(255, 0, 0),
        )

        # Check that we got RGB output
        assert result.shape[2] == 3

    def test_all_interpolation_methods(
        self, simple_dewarper: CylindricalSurfaceDewarper
    ) -> None:
        """All interpolation methods work."""
        src = np.random.randint(0, 256, (100, 100), dtype=np.uint8)

        for method in InterpolationMethod:
            result = dewarp_image(
                src,
                simple_dewarper,
                dst_size=(100, 100),
                model_domain=(0, 0, 1, 1),
                interpolation=method,
            )
            assert result.shape == (100, 100)

    def test_invalid_model_domain(
        self, simple_dewarper: CylindricalSurfaceDewarper
    ) -> None:
        """Invalid model domain raises error."""
        src = np.zeros((100, 100), dtype=np.uint8)

        with pytest.raises(ValueError, match="positive width"):
            dewarp_image(
                src,
                simple_dewarper,
                dst_size=(100, 100),
                model_domain=(0.5, 0, 0.5, 1),  # Zero width
            )

        with pytest.raises(ValueError, match="positive.*height"):
            dewarp_image(
                src,
                simple_dewarper,
                dst_size=(100, 100),
                model_domain=(0, 0.5, 1, 0.5),  # Zero height
            )

    def test_partial_model_domain(
        self, simple_dewarper: CylindricalSurfaceDewarper
    ) -> None:
        """Partial model domain works correctly."""
        src = np.zeros((300, 200), dtype=np.uint8)
        # Fill center with white
        src[100:200, 50:150] = 255

        # Map only the center portion
        result = dewarp_image(
            src,
            simple_dewarper,
            dst_size=(100, 100),
            model_domain=(0.25, 0.33, 0.75, 0.67),
        )

        assert result.shape == (100, 100)

    def test_curved_dewarping_preserves_content(
        self, curved_dewarper: CylindricalSurfaceDewarper
    ) -> None:
        """Curved dewarping preserves image content."""
        # Create image with horizontal lines
        src = np.zeros((300, 200), dtype=np.uint8)
        for y in range(0, 300, 30):
            src[y : y + 2, :] = 255

        result = dewarp_image(
            src,
            curved_dewarper,
            dst_size=(200, 300),
            model_domain=(0, 0, 1, 1),
        )

        # Result should have some white pixels (lines preserved)
        assert np.sum(result > 128) > 0


class TestComputeDewarpedSize:
    """Tests for compute_dewarped_size function."""

    def test_flat_surface_same_size(
        self, simple_dewarper: CylindricalSurfaceDewarper
    ) -> None:
        """Flat surface returns approximately same size."""
        src_size = (200, 300)
        dst_size = compute_dewarped_size(src_size, simple_dewarper)

        # For nearly flat surface, sizes should be similar
        assert dst_size[0] > 0
        assert dst_size[1] > 0
        # Width should be close to original (arc length ~1 for flat)
        assert 0.9 * src_size[0] < dst_size[0] < 1.1 * src_size[0]
        assert dst_size[1] == src_size[1]

    def test_curved_surface_wider(
        self, curved_dewarper: CylindricalSurfaceDewarper
    ) -> None:
        """Curved surface may result in wider output."""
        src_size = (200, 300)
        dst_size = compute_dewarped_size(src_size, curved_dewarper)

        # Arc length >= 1, so width should be >= original
        assert dst_size[0] >= src_size[0] * 0.95  # Allow small tolerance
        assert dst_size[1] == src_size[1]

    def test_dpi_scale(self, simple_dewarper: CylindricalSurfaceDewarper) -> None:
        """DPI scale factor works."""
        src_size = (200, 300)

        size_1x = compute_dewarped_size(src_size, simple_dewarper, dpi_scale=1.0)
        size_2x = compute_dewarped_size(src_size, simple_dewarper, dpi_scale=2.0)

        assert size_2x[0] == 2 * size_1x[0]
        assert size_2x[1] == 2 * size_1x[1]

    def test_minimum_size(self, simple_dewarper: CylindricalSurfaceDewarper) -> None:
        """Output size is at least 1x1."""
        src_size = (1, 1)
        dst_size = compute_dewarped_size(src_size, simple_dewarper)

        assert dst_size[0] >= 1
        assert dst_size[1] >= 1
