"""Tests for zone system."""

import numpy as np
import pytest

from scantailor.core.zones import (
    FillZoneProperties,
    PictureLayer,
    PictureZoneProperties,
    Zone,
    ZoneCategory,
    ZoneSpline,
)


class TestZoneCategory:
    """Tests for ZoneCategory enum."""

    def test_values(self) -> None:
        """All category values exist."""
        assert ZoneCategory.MANUAL.value == "manual"
        assert ZoneCategory.AUTO.value == "auto"


class TestPictureLayer:
    """Tests for PictureLayer enum."""

    def test_all_layers_exist(self) -> None:
        """All layer values exist."""
        assert PictureLayer.NOOP.value == "noop"
        assert PictureLayer.ERASER1.value == "eraser1"
        assert PictureLayer.PAINTER2.value == "painter2"
        assert PictureLayer.ERASER3.value == "eraser3"
        assert PictureLayer.FOREGROUND.value == "foreground"
        assert PictureLayer.BACKGROUND.value == "background"

    def test_all_unique(self) -> None:
        """All layer values are unique."""
        values = [layer.value for layer in PictureLayer]
        assert len(values) == len(set(values))


class TestZoneSpline:
    """Tests for ZoneSpline class."""

    def test_empty_spline(self) -> None:
        """Empty spline is invalid."""
        spline = ZoneSpline()
        assert not spline.is_valid()
        assert len(spline.points) == 0

    def test_from_list(self) -> None:
        """Create spline from list of points."""
        points = [[0, 0], [100, 0], [100, 100], [0, 100]]
        spline = ZoneSpline(points=points)
        assert spline.is_valid()
        assert len(spline.points) == 4
        assert spline.points.dtype == np.float64

    def test_from_numpy_array(self) -> None:
        """Create spline from numpy array."""
        points = np.array([[0, 0], [100, 0], [100, 100]], dtype=np.float32)
        spline = ZoneSpline(points=points)
        assert spline.is_valid()
        assert spline.points.dtype == np.float64  # Converted to float64

    def test_minimum_points(self) -> None:
        """Spline needs at least 3 points to be valid."""
        spline1 = ZoneSpline(points=[[0, 0], [100, 0]])
        assert not spline1.is_valid()

        spline2 = ZoneSpline(points=[[0, 0], [100, 0], [50, 100]])
        assert spline2.is_valid()

    def test_smooth_flag(self) -> None:
        """Smooth flag is stored correctly."""
        spline1 = ZoneSpline(points=[[0, 0], [100, 0], [50, 100]])
        assert not spline1.is_smooth

        spline2 = ZoneSpline(points=[[0, 0], [100, 0], [50, 100]], is_smooth=True)
        assert spline2.is_smooth

    def test_bounding_rect_empty(self) -> None:
        """Empty spline returns zero rect."""
        spline = ZoneSpline()
        rect = spline.bounding_rect()
        assert rect == (0.0, 0.0, 0.0, 0.0)

    def test_bounding_rect(self) -> None:
        """Bounding rect is computed correctly."""
        points = [[10, 20], [110, 20], [110, 120], [10, 120]]
        spline = ZoneSpline(points=points)
        x, y, w, h = spline.bounding_rect()
        assert x == 10.0
        assert y == 20.0
        assert w == 100.0
        assert h == 100.0

    def test_to_polygon_non_smooth(self) -> None:
        """Non-smooth spline returns copy of points."""
        points = [[0, 0], [100, 0], [100, 100], [0, 100]]
        spline = ZoneSpline(points=points)
        polygon = spline.to_polygon()

        np.testing.assert_array_equal(polygon, spline.points)
        assert polygon is not spline.points  # Should be a copy

    def test_to_polygon_smooth(self) -> None:
        """Smooth spline returns interpolated polygon."""
        points = [[0, 0], [100, 0], [100, 100], [0, 100]]
        spline = ZoneSpline(points=points, is_smooth=True)
        polygon = spline.to_polygon(num_samples=50)

        # Should have more points than input
        assert len(polygon) == 50
        # Should be different from input
        assert polygon.shape != spline.points.shape

    @pytest.mark.parametrize(
        ("points", "expected"),
        [
            (
                [[10, 10], [200, 30], [220, 180], [60, 210], [5, 120]],
                [
                    [10.0, 10.0],
                    [91.593690493, -17.02661551],
                    [190.554159173, 21.407507969],
                    [237.757266918, 109.206495574],
                    [207.278770587, 193.415169053],
                    [114.982358717, 223.960638869],
                    [34.507670973, 187.227912711],
                    [0.14277718, 97.901700745],
                    [10.0, 10.0],
                ],
            ),
            (
                [[0, 0], [100, 0], [100, 100], [0, 100]],
                [
                    [0.0, 0.0],
                    [50.0, -18.75],
                    [100.0, 0.0],
                    [118.75, 50.0],
                    [100.0, 100.0],
                    [50.0, 118.75],
                    [0.0, 100.0],
                    [-18.75, 50.0],
                    [0.0, 0.0],
                ],
            ),
        ],
    )
    def test_to_polygon_smooth_matches_the_closed_spline_scipy_fits(
        self, points: list[list[int]], expected: list[list[float]]
    ) -> None:
        """Samples equal scipy's splprep(s=0, per=True) curve, rounded to 1e-9."""
        polygon = ZoneSpline(points=points, is_smooth=True).to_polygon(num_samples=9)

        np.testing.assert_allclose(polygon, expected, rtol=0, atol=1e-8)

    def test_to_polygon_smooth_passes_through_every_vertex(self) -> None:
        """Equal sides put the vertices at every other of 9 samples."""
        points = np.array([[0, 0], [300, 0], [300, 300], [0, 300]], dtype=np.float64)
        polygon = ZoneSpline(points=points, is_smooth=True).to_polygon(num_samples=9)

        np.testing.assert_allclose(polygon[::2][:4], points, atol=1e-9)

    def test_to_polygon_smooth_repeated_point_falls_back_to_polygon(self) -> None:
        """A vertex given twice in a row has no curve through it."""
        points = [[0, 0], [0, 0], [100, 0], [100, 100], [0, 100]]
        spline = ZoneSpline(points=points, is_smooth=True)

        np.testing.assert_array_equal(spline.to_polygon(), spline.points)

    def test_to_polygon_smooth_insufficient_points(self) -> None:
        """Smooth spline with <4 points returns original."""
        points = [[0, 0], [100, 0], [50, 100]]
        spline = ZoneSpline(points=points, is_smooth=True)
        polygon = spline.to_polygon()

        np.testing.assert_array_equal(polygon, spline.points)


class TestPictureZoneProperties:
    """Tests for PictureZoneProperties."""

    def test_defaults(self) -> None:
        """Default values are correct."""
        props = PictureZoneProperties()
        assert props.layer == PictureLayer.NOOP
        assert props.category == ZoneCategory.MANUAL

    def test_custom_values(self) -> None:
        """Custom values are stored correctly."""
        props = PictureZoneProperties(
            layer=PictureLayer.FOREGROUND,
            category=ZoneCategory.AUTO,
        )
        assert props.layer == PictureLayer.FOREGROUND
        assert props.category == ZoneCategory.AUTO


class TestFillZoneProperties:
    """Tests for FillZoneProperties."""

    def test_defaults(self) -> None:
        """Default values are correct."""
        props = FillZoneProperties()
        assert props.color == (255, 255, 255)
        assert props.category == ZoneCategory.MANUAL

    def test_custom_color(self) -> None:
        """Custom color is stored correctly."""
        props = FillZoneProperties(color=(128, 64, 32))
        assert props.color == (128, 64, 32)


class TestZone:
    """Tests for Zone class."""

    @pytest.fixture
    def triangle_spline(self) -> ZoneSpline:
        """A valid triangular spline."""
        return ZoneSpline(points=[[0, 0], [100, 0], [50, 100]])

    @pytest.fixture
    def invalid_spline(self) -> ZoneSpline:
        """An invalid spline with too few points."""
        return ZoneSpline(points=[[0, 0], [100, 0]])

    def test_picture_zone(self, triangle_spline: ZoneSpline) -> None:
        """Create a picture zone."""
        props = PictureZoneProperties(layer=PictureLayer.BACKGROUND)
        zone = Zone(spline=triangle_spline, properties=props)

        assert zone.is_valid()
        assert zone.is_picture_zone()
        assert not zone.is_fill_zone()

    def test_fill_zone(self, triangle_spline: ZoneSpline) -> None:
        """Create a fill zone."""
        props = FillZoneProperties(color=(0, 0, 0))
        zone = Zone(spline=triangle_spline, properties=props)

        assert zone.is_valid()
        assert zone.is_fill_zone()
        assert not zone.is_picture_zone()

    def test_invalid_zone(self, invalid_spline: ZoneSpline) -> None:
        """Zone with invalid spline is invalid."""
        props = PictureZoneProperties()
        zone = Zone(spline=invalid_spline, properties=props)

        assert not zone.is_valid()
