"""Zone system for defining image regions with special properties.

Zones are user-defined regions (polygons or splines) that receive special
treatment during output processing. There are two main zone types:

- PictureZones: Regions to preserve as images (photos, diagrams)
- FillZones: Regions to fill with a solid color

Each zone has:
- A spline defining its boundary (can be polygon or smooth curve)
- Properties specific to its type (layer, category, fill color, etc.)
"""

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel


class ZoneCategory(StrEnum):
    """How the zone was created.

    MANUAL: User-defined zone.
    AUTO: Automatically detected zone.
    """

    MANUAL = "manual"
    AUTO = "auto"


class PictureLayer(StrEnum):
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
        """Validate and normalize the points array."""
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
            tck_list, _ = splprep([pts[:, 0], pts[:, 1]], s=0, per=True)
            tck: Any = tck_list
            u = np.linspace(0, 1, num_samples)
            x, y = splev(u, tck)
            return np.column_stack([x, y])
        except ValueError, TypeError:
            # splprep rejects repeated or too few points; draw the polygon instead.
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
