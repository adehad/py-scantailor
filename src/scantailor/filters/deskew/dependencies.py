"""What a stored deskew angle was detected from."""

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict

# Outline coordinates are pixels; anything below this is float noise.
_OUTLINE_TOLERANCE_PX = 1e-3


class Dependencies(BaseModel):
    """Upstream geometry a detected skew angle depends on.

    Attributes:
        page_outline: Polygon of the page, in oriented source pixels, as left
            by Page Split; empty when unknown.
        rotation_degrees: Orthogonal rotation applied by Fix Orientation.
    """

    model_config = ConfigDict(frozen=True)

    page_outline: tuple[tuple[float, float], ...] = ()
    rotation_degrees: int = 0

    @classmethod
    def of(
        cls, page_outline: NDArray[np.float64], rotation_degrees: int
    ) -> Dependencies:
        """Build from a polygon given as an (N, 2) array of (x, y)."""
        return cls(
            page_outline=tuple((float(x), float(y)) for x, y in page_outline),
            rotation_degrees=rotation_degrees,
        )

    def is_null(self) -> bool:
        """Return True when nothing is known about the original detection."""
        return not self.page_outline

    def matches(self, other: Dependencies) -> bool:
        """Tell whether both describe the same upstream geometry."""
        if self.rotation_degrees != other.rotation_degrees:
            return False
        if len(self.page_outline) != len(other.page_outline):
            return False
        return bool(
            np.allclose(
                np.asarray(self.page_outline, dtype=np.float64),
                np.asarray(other.page_outline, dtype=np.float64),
                rtol=0.0,
                atol=_OUTLINE_TOLERANCE_PX,
            )
        )
