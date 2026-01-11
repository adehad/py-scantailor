"""Tests for zone system."""

import numpy as np
import pytest

from scantailor.core.zones import (
    FillZoneProperties,
    PictureLayer,
    PictureZoneProperties,
    Zone,
    ZoneCategory,
    ZoneSet,
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
        pytest.importorskip("scipy")

        points = [[0, 0], [100, 0], [100, 100], [0, 100]]
        spline = ZoneSpline(points=points, is_smooth=True)
        polygon = spline.to_polygon(num_samples=50)

        # Should have more points than input
        assert len(polygon) == 50
        # Should be different from input
        assert polygon.shape != spline.points.shape

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


class TestZoneSet:
    """Tests for ZoneSet class."""

    @pytest.fixture
    def spline1(self) -> ZoneSpline:
        """First test spline."""
        return ZoneSpline(points=[[0, 0], [100, 0], [100, 100]])

    @pytest.fixture
    def spline2(self) -> ZoneSpline:
        """Second test spline."""
        return ZoneSpline(points=[[200, 0], [300, 0], [300, 100]])

    def test_empty_zoneset(self) -> None:
        """Empty ZoneSet has no zones."""
        zs = ZoneSet()
        assert zs.is_empty()
        assert len(zs) == 0
        assert zs.picture_zones == []
        assert zs.fill_zones == []

    def test_add_picture_zone(self, spline1: ZoneSpline) -> None:
        """Add a picture zone."""
        zs = ZoneSet()
        zone = zs.add_picture_zone(spline1, layer=PictureLayer.FOREGROUND)

        assert not zs.is_empty()
        assert len(zs) == 1
        assert len(zs.picture_zones) == 1
        assert len(zs.fill_zones) == 0
        assert zone.is_picture_zone()
        assert zone.properties.layer == PictureLayer.FOREGROUND

    def test_add_fill_zone(self, spline1: ZoneSpline) -> None:
        """Add a fill zone."""
        zs = ZoneSet()
        zone = zs.add_fill_zone(spline1, color=(128, 128, 128))

        assert not zs.is_empty()
        assert len(zs) == 1
        assert len(zs.picture_zones) == 0
        assert len(zs.fill_zones) == 1
        assert zone.is_fill_zone()
        assert zone.properties.color == (128, 128, 128)

    def test_add_multiple_zones(self, spline1: ZoneSpline, spline2: ZoneSpline) -> None:
        """Add multiple zones of different types."""
        zs = ZoneSet()
        zs.add_picture_zone(spline1)
        zs.add_fill_zone(spline2)

        assert len(zs) == 2
        assert len(zs.picture_zones) == 1
        assert len(zs.fill_zones) == 1

    def test_remove_picture_zone(self, spline1: ZoneSpline) -> None:
        """Remove a picture zone."""
        zs = ZoneSet()
        zone = zs.add_picture_zone(spline1)

        result = zs.remove_picture_zone(zone)
        assert result is True
        assert zs.is_empty()

    def test_remove_nonexistent_zone(
        self, spline1: ZoneSpline, spline2: ZoneSpline
    ) -> None:
        """Removing non-existent zone returns False."""
        zs = ZoneSet()
        zone1 = zs.add_picture_zone(spline1)
        zone2 = Zone(spline=spline2, properties=PictureZoneProperties())

        result = zs.remove_picture_zone(zone2)
        assert result is False
        assert len(zs) == 1

    def test_remove_fill_zone(self, spline1: ZoneSpline) -> None:
        """Remove a fill zone."""
        zs = ZoneSet()
        zone = zs.add_fill_zone(spline1)

        result = zs.remove_fill_zone(zone)
        assert result is True
        assert zs.is_empty()

    def test_clear_picture_zones(
        self, spline1: ZoneSpline, spline2: ZoneSpline
    ) -> None:
        """Clear only picture zones."""
        zs = ZoneSet()
        zs.add_picture_zone(spline1)
        zs.add_picture_zone(spline2)
        zs.add_fill_zone(spline1)

        zs.clear_picture_zones()
        assert len(zs.picture_zones) == 0
        assert len(zs.fill_zones) == 1

    def test_clear_fill_zones(self, spline1: ZoneSpline, spline2: ZoneSpline) -> None:
        """Clear only fill zones."""
        zs = ZoneSet()
        zs.add_picture_zone(spline1)
        zs.add_fill_zone(spline1)
        zs.add_fill_zone(spline2)

        zs.clear_fill_zones()
        assert len(zs.picture_zones) == 1
        assert len(zs.fill_zones) == 0

    def test_clear_all(self, spline1: ZoneSpline, spline2: ZoneSpline) -> None:
        """Clear all zones."""
        zs = ZoneSet()
        zs.add_picture_zone(spline1)
        zs.add_fill_zone(spline2)

        zs.clear_all()
        assert zs.is_empty()
        assert len(zs) == 0

    def test_zone_category_default(self, spline1: ZoneSpline) -> None:
        """Default category is MANUAL."""
        zs = ZoneSet()
        zone = zs.add_picture_zone(spline1)
        assert zone.properties.category == ZoneCategory.MANUAL

    def test_zone_category_auto(self, spline1: ZoneSpline) -> None:
        """Auto category can be specified."""
        zs = ZoneSet()
        zone = zs.add_picture_zone(spline1, category=ZoneCategory.AUTO)
        assert zone.properties.category == ZoneCategory.AUTO

    def test_picture_zones_returns_copy(self, spline1: ZoneSpline) -> None:
        """picture_zones returns a copy of the list."""
        zs = ZoneSet()
        zs.add_picture_zone(spline1)

        zones = zs.picture_zones
        zones.clear()

        assert len(zs.picture_zones) == 1  # Original unchanged

    def test_fill_zones_returns_copy(self, spline1: ZoneSpline) -> None:
        """fill_zones returns a copy of the list."""
        zs = ZoneSet()
        zs.add_fill_zone(spline1)

        zones = zs.fill_zones
        zones.clear()

        assert len(zs.fill_zones) == 1  # Original unchanged
