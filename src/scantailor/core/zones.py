"""Zone system for defining image regions with special properties.

Zones are user-defined regions (polygons or splines) that receive special
treatment during output processing. There are two main zone types:

- PictureZones: Regions to preserve as images (photos, diagrams)
- FillZones: Regions to fill with a solid color

Each zone has:
- A spline defining its boundary (can be polygon or smooth curve)
- Properties specific to its type (layer, category, fill color, etc.)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

import numpy as np
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from numpy.typing import NDArray


class ZoneCategory(str, Enum):
    """How the zone was created.

    MANUAL: User-defined zone.
    AUTO: Automatically detected zone.
    """

    MANUAL = "manual"
    AUTO = "auto"


class PictureLayer(str, Enum):
    """Layer assignment for picture zones.

    Controls how the zone interacts with binarization in mixed mode.

    NOOP: No special handling (default).
    ERASER1: First eraser layer - removes from picture mask.
    PAINTER2: Painter layer - adds to picture mask.
    ERASER3: Second eraser layer.
    FOREGROUND: Force zone to foreground (binarized).
    BACKGROUND: Force zone to background (preserved as-is).
    """

    NOOP = "noop"
    ERASER1 = "eraser1"
    PAINTER2 = "painter2"
    ERASER3 = "eraser3"
    FOREGROUND = "foreground"
    BACKGROUND = "background"


@dataclass
class ZoneSpline:
    """A spline defining a zone boundary.

    The spline is defined by control points. For simple polygon zones,
    these are the vertex positions. For smooth zones, the spline
    interpolates through these points.

    Attributes:
        points: Control points as (N, 2) array.
        is_smooth: If True, interpolate smoothly between points.
    """

    points: NDArray[np.float64] = field(default_factory=lambda: np.empty((0, 2)))
    is_smooth: bool = False

    def __post_init__(self) -> None:
        self.points = np.asarray(self.points, dtype=np.float64)
        if self.points.ndim == 1:
            self.points = self.points.reshape(-1, 2)

    def is_valid(self) -> bool:
        """Return True if the spline has at least 3 points."""
        return len(self.points) >= 3

    def bounding_rect(self) -> tuple[float, float, float, float]:
        """Return bounding rectangle as (x, y, width, height)."""
        if len(self.points) == 0:
            return (0.0, 0.0, 0.0, 0.0)
        min_pt = self.points.min(axis=0)
        max_pt = self.points.max(axis=0)
        return (min_pt[0], min_pt[1], max_pt[0] - min_pt[0], max_pt[1] - min_pt[1])

    def to_polygon(self, num_samples: int = 100) -> NDArray[np.float64]:
        """Convert to polygon for rasterization.

        Args:
            num_samples: Number of samples for smooth splines.

        Returns:
            Polygon vertices as (N, 2) array.
        """
        if not self.is_smooth or len(self.points) < 4:
            return self.points.copy()

        # For smooth splines, use cubic interpolation
        from scipy.interpolate import splev, splprep

        try:
            # Close the spline by appending start point
            pts = np.vstack([self.points, self.points[0]])
            tck, _ = splprep([pts[:, 0], pts[:, 1]], s=0, per=True)
            u = np.linspace(0, 1, num_samples)
            x, y = splev(u, tck)
            return np.column_stack([x, y])
        except Exception:
            # Fall back to polygon
            return self.points.copy()


class PictureZoneProperties(BaseModel):
    """Properties specific to picture zones.

    Attributes:
        layer: The layer assignment for mixed-mode output.
        category: Whether manually or auto-created.
    """

    layer: PictureLayer = PictureLayer.NOOP
    category: ZoneCategory = ZoneCategory.MANUAL


class FillZoneProperties(BaseModel):
    """Properties specific to fill zones.

    Attributes:
        color: Fill color as (R, G, B) tuple, 0-255.
        category: Whether manually or auto-created.
    """

    color: tuple[int, int, int] = (255, 255, 255)
    category: ZoneCategory = ZoneCategory.MANUAL


@dataclass
class Zone:
    """A zone with boundary and properties.

    Attributes:
        spline: The boundary spline.
        properties: Zone-specific properties.
    """

    spline: ZoneSpline
    properties: PictureZoneProperties | FillZoneProperties

    def is_valid(self) -> bool:
        """Return True if the zone has a valid boundary."""
        return self.spline.is_valid()

    def is_picture_zone(self) -> bool:
        """Return True if this is a picture zone."""
        return isinstance(self.properties, PictureZoneProperties)

    def is_fill_zone(self) -> bool:
        """Return True if this is a fill zone."""
        return isinstance(self.properties, FillZoneProperties)


class ZoneSet(BaseModel):
    """A collection of zones.

    Supports iteration and provides methods for adding/removing zones.
    """

    model_config = {"arbitrary_types_allowed": True}

    _picture_zones: list[Zone] = []
    _fill_zones: list[Zone] = []

    def __init__(self, **data) -> None:
        super().__init__(**data)
        object.__setattr__(self, "_picture_zones", [])
        object.__setattr__(self, "_fill_zones", [])

    @property
    def picture_zones(self) -> list[Zone]:
        """Get all picture zones."""
        return list(self._picture_zones)

    @property
    def fill_zones(self) -> list[Zone]:
        """Get all fill zones."""
        return list(self._fill_zones)

    def add_picture_zone(
        self,
        spline: ZoneSpline,
        layer: PictureLayer = PictureLayer.NOOP,
        category: ZoneCategory = ZoneCategory.MANUAL,
    ) -> Zone:
        """Add a picture zone.

        Args:
            spline: Zone boundary.
            layer: Layer assignment.
            category: Zone category.

        Returns:
            The created zone.
        """
        props = PictureZoneProperties(layer=layer, category=category)
        zone = Zone(spline=spline, properties=props)
        self._picture_zones.append(zone)
        return zone

    def add_fill_zone(
        self,
        spline: ZoneSpline,
        color: tuple[int, int, int] = (255, 255, 255),
        category: ZoneCategory = ZoneCategory.MANUAL,
    ) -> Zone:
        """Add a fill zone.

        Args:
            spline: Zone boundary.
            color: Fill color as (R, G, B).
            category: Zone category.

        Returns:
            The created zone.
        """
        props = FillZoneProperties(color=color, category=category)
        zone = Zone(spline=spline, properties=props)
        self._fill_zones.append(zone)
        return zone

    def remove_picture_zone(self, zone: Zone) -> bool:
        """Remove a picture zone.

        Args:
            zone: The zone to remove.

        Returns:
            True if zone was found and removed.
        """
        try:
            self._picture_zones.remove(zone)
            return True
        except ValueError:
            return False

    def remove_fill_zone(self, zone: Zone) -> bool:
        """Remove a fill zone.

        Args:
            zone: The zone to remove.

        Returns:
            True if zone was found and removed.
        """
        try:
            self._fill_zones.remove(zone)
            return True
        except ValueError:
            return False

    def clear_picture_zones(self) -> None:
        """Remove all picture zones."""
        self._picture_zones.clear()

    def clear_fill_zones(self) -> None:
        """Remove all fill zones."""
        self._fill_zones.clear()

    def clear_all(self) -> None:
        """Remove all zones."""
        self._picture_zones.clear()
        self._fill_zones.clear()

    def is_empty(self) -> bool:
        """Return True if there are no zones."""
        return len(self._picture_zones) == 0 and len(self._fill_zones) == 0

    def __len__(self) -> int:
        """Return total number of zones."""
        return len(self._picture_zones) + len(self._fill_zones)
