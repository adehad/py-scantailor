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

from scantailor.core.image_io import (
    decodes_shrunk,
    load_image,
    shrink_decoded,
    shrink_factor,
)
from scantailor.core.models import Dpi, ImageId
from scantailor.core.project import Project

DpiOverrides = Mapping[ImageId, Dpi]

# Resolution assumed for images that carry none.
FALLBACK_DPI = 300.0


@dataclass(frozen=True)
class LoadedPage:
    """Pixels of an image together with the resolution to process them at.

    Attributes:
        pixels: ``(H, W)`` gray or ``(H, W, 3)`` BGR ``uint8``, possibly shrunk
            (see ``load_page``).
        dpi: Effective resolution of the full-size image; null when neither
            project nor file defines one.
        full_size: Width and height of the full-size image; None when that
            is the size of ``pixels``.
    """

    pixels: NDArray[np.uint8]
    dpi: Dpi
    full_size: tuple[int, int] | None = None

    @property
    def size(self) -> tuple[int, int]:
        """Width and height of the full-size image."""
        if self.full_size is not None:
            return self.full_size
        height, width = self.pixels.shape[:2]
        return width, height

    def resolution(self, default: float = FALLBACK_DPI) -> Dpi:
        """Resolution to process the page at, ``default`` on both axes when null.

        Args:
            default (float): Dots per inch assumed when neither the project nor
                the file defines a resolution.

        Returns:
            Dpi: ``dpi``, whose axes may differ, or ``default`` on both.
        """
        return Dpi.uniform(round(default)) if self.dpi.is_null() else self.dpi


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
    image_id: ImageId,
    dpi_overrides: DpiOverrides | None = None,
    min_dpi: float | None = None,
    default_dpi: float = FALLBACK_DPI,
) -> LoadedPage:
    """Decode an image and resolve its effective DPI.

    Args:
        image_id (ImageId): File and page to decode.
        dpi_overrides (DpiOverrides | None): Project resolutions; an entry for
            ``image_id`` wins over the file's own DPI.
        min_dpi (float | None): Shrink the pixels as far as this resolution
            allows, as ``load_image`` does; None decodes in full.
        default_dpi (float): Resolution assumed for an image that has none,
            when shrinking it.

    Returns:
        LoadedPage: Pixels and effective DPI.

    Raises:
        FileNotFoundError: If the file does not exist.
        ImageLoadError: If the file cannot be decoded.
    """
    override = dpi_overrides.get(image_id) if dpi_overrides else None
    pixels, metadata = load_image(image_id, override, min_dpi, default_dpi)
    full_size = (metadata.width, metadata.height)
    height, width = pixels.shape[:2]
    return LoadedPage(
        pixels, metadata.dpi, None if full_size == (width, height) else full_size
    )


def shrink_page(
    page: LoadedPage,
    image_id: ImageId,
    dpi_overrides: DpiOverrides | None = None,
    min_dpi: float | None = None,
    default_dpi: float = FALLBACK_DPI,
) -> LoadedPage:
    """Return the pixels ``load_page`` gives with ``min_dpi``, from a full decode.

    Decoding again is needed only for a JPEG that the decoder shrinks itself;
    any other image is shrunk from ``page``, which gives the same pixels.

    Args:
        page (LoadedPage): ``image_id`` decoded in full by ``load_page``.
        image_id (ImageId): File and page.
        dpi_overrides (DpiOverrides | None): As given to ``load_page``.
        min_dpi (float | None): As for ``load_page``.
        default_dpi (float): As for ``load_page``.

    Returns:
        LoadedPage: ``page`` itself when it is not shrunk.

    Raises:
        FileNotFoundError: If the file does not exist.
        ImageLoadError: If the file cannot be decoded.
    """
    factor = shrink_factor(page.dpi, min_dpi, default_dpi)
    if factor == 1:
        return page
    if decodes_shrunk(image_id):
        return load_page(image_id, dpi_overrides, min_dpi, default_dpi)
    return LoadedPage(shrink_decoded(page.pixels, factor), page.dpi, page.size)
