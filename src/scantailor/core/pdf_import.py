"""Build a project's page images from a PDF.

Pages are rasterised with pypdfium2 at a fixed resolution and stored as PNG
files carrying that resolution, so the rest of the pipeline treats them as
ordinary scans. Pages are always rendered rather than extracted at native
resolution: a scan PDF stores its image in one of several codecs (DCT, JBIG2,
CCITT, JPX) with per-page masks, and re-rendering gives one uniform path.
"""

from pathlib import Path
from typing import Final

import cv2
import numpy as np
import pypdfium2 as pdfium
from loguru import logger
from numpy.typing import NDArray

from scantailor.core.image_io import write_image
from scantailor.core.models import Dpi

_POINTS_PER_INCH: Final = 72.0
# Largest per-pixel channel spread, in grey levels, still treated as neutral;
# anti-aliasing and JPEG noise on a grayscale page stay below this.
_COLOUR_SPREAD_THRESHOLD: Final = 12


def _has_colour(bgr: NDArray[np.uint8]) -> bool:
    """Return True when any pixel's channels differ by more than the noise floor."""
    channels = bgr.astype(np.int16)
    spread = channels.max(axis=2) - channels.min(axis=2)
    return bool((spread > _COLOUR_SPREAD_THRESHOLD).any())


def import_pdf(pdf_path: Path, out_dir: Path, dpi: int = 300) -> list[Path]:
    """Render every page of a PDF to a PNG image.

    Args:
        pdf_path (Path): Source PDF.
        out_dir (Path): Directory for the images; created when missing.
        dpi (int): Rendering resolution, also stored as the PNG's DPI.

    Returns:
        list[Path]: ``page_0001.png``, ``page_0002.png``, ... in page order.
        Pages with colour content are written as BGR, all others as grayscale.

    Raises:
        ValueError: If ``dpi`` is not positive, or the file is unreadable or
            has no pages.
    """
    if dpi <= 0:
        raise ValueError(f"dpi must be positive, got {dpi}")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    scale = dpi / _POINTS_PER_INCH
    resolution = Dpi.uniform(dpi)

    paths: list[Path] = []
    try:
        document = pdfium.PdfDocument(str(pdf_path))
    except pdfium.PdfiumError as error:
        raise ValueError(f"{pdf_path} is not a readable PDF with pages") from error
    try:
        for index in range(len(document)):
            page = document[index]
            try:
                bitmap = page.render(scale=scale, may_draw_forms=False)
                bgr = np.ascontiguousarray(bitmap.to_numpy()[..., :3])
            finally:
                page.close()
            pixels = bgr if _has_colour(bgr) else cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            path = out_dir / f"page_{index + 1:04d}.png"
            write_image(path, pixels, resolution)
            paths.append(path)
    finally:
        document.close()

    logger.info("Imported {} pages from {} at {} dpi", len(paths), pdf_path, dpi)
    return paths
