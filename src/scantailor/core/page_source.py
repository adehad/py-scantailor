"""Loading a page for processing with the project's resolution applied.

A resolution the user fixed lives in the project's ``ImageMetadata`` and takes
precedence over the one stored in the image file, for every stage. Callers on
worker threads or in worker processes receive the overrides as a plain mapping
snapshot rather than the live ``Project``.
"""

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from scantailor.core.image_io import load_image
from scantailor.core.models import Dpi, ImageId
from scantailor.core.project import Project

DpiOverrides = Mapping[ImageId, Dpi]

# Resolution assumed for images that carry none.
FALLBACK_DPI = 300.0


@dataclass(frozen=True)
class LoadedPage:
    """Pixels of an image together with the resolution to process them at.

    Attributes:
        pixels: ``(H, W)`` gray or ``(H, W, 3)`` BGR ``uint8``.
        dpi: Effective resolution; null when neither project nor file defines one.
    """

    pixels: NDArray[np.uint8]
    dpi: Dpi

    def horizontal_dpi(self, default: float = FALLBACK_DPI) -> float:
        """Horizontal resolution to size the page by, or ``default`` when null."""
        return float(self.dpi.horizontal) if not self.dpi.is_null() else default


def dpi_overrides_of(project: Project | None) -> dict[ImageId, Dpi]:
    """Snapshot the project's defined resolutions, keyed by image.

    Args:
        project (Project | None): Project to read; None yields no overrides.

    Returns:
        dict[ImageId, Dpi]: Images whose project metadata has a defined DPI.
    """
    if project is None:
        return {}
    return {
        info.id: info.metadata.dpi
        for info in project.images
        if info.metadata.is_dpi_ok()
    }


def load_page(
    image_id: ImageId, dpi_overrides: DpiOverrides | None = None
) -> LoadedPage:
    """Decode an image and resolve its effective DPI.

    Args:
        image_id (ImageId): File and page to decode.
        dpi_overrides (DpiOverrides | None): Project resolutions; an entry for
            ``image_id`` wins over the file's own DPI.

    Returns:
        LoadedPage: Pixels and effective DPI.

    Raises:
        FileNotFoundError: If the file does not exist.
        ImageLoadError: If the file cannot be decoded.
    """
    override = dpi_overrides.get(image_id) if dpi_overrides else None
    pixels, metadata = load_image(image_id, override)
    return LoadedPage(pixels, metadata.dpi)
