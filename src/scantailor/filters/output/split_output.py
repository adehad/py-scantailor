"""Split output: the text and picture layers of a mixed page as separate images.

With split output on, a mixed page is saved as a foreground image and a
background image in sub-folders of the output folder, instead of one combined
image. Everything is white wherever a layer has nothing to say, so the layers
overlay by "darker wins" and can be merged back into the combined page.

Two modes exist. In the plain mode the foreground is the text region
(binarised or segmented text on white) and the background is the picture
region. With the original background kept, the foreground holds only the ink,
the background the pictures, and a third image the original page with the ink
whitened and the pictures blacked out.
"""

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from scantailor.core.image_io import write_image
from scantailor.core.models import Dpi
from scantailor.imageproc.filters import bgr_to_gray

from .render_params import RenderParams

FOREGROUND_DIR = "foreground"
BACKGROUND_DIR = "background"
ORIGINAL_BACKGROUND_DIR = "original_background"

_WHITE = 255
_BLACK = 0
_COLOR_CHANNELS = 3
# Gray level below which a pixel counts as black when a layer is stored bilevel.
_BILEVEL_SPLIT = 128


class ForegroundType(Enum):
    """How the foreground layer is stored.

    BINARY: Black and white, 1 bit per pixel.
    INDEXED: A palette of at most 256 colours (segmented, posterised text).
    COLOR: Full colour or gray.
    """

    BINARY = "binary"
    INDEXED = "indexed"
    COLOR = "color"


def foreground_type_for(render: RenderParams) -> ForegroundType:
    """Pick the storage type of the foreground layer.

    Args:
        render (RenderParams): Flags of the page being rendered.

    Returns:
        ForegroundType: BINARY for plain binarisation, INDEXED when the text is
            colour-segmented, COLOR when the foreground is not binarised.
    """
    if not render.need_binarization:
        return ForegroundType.COLOR
    return (
        ForegroundType.INDEXED if render.color_segmentation else ForegroundType.BINARY
    )


@dataclass(frozen=True)
class SplitImages:
    """The layers of a split page.

    Attributes:
        foreground: Text layer, white outside the foreground. 2-D 0/255 for
            ``BINARY``, otherwise the dtype and channels of the source image.
        background: Picture layer, white outside the pictures.
        original_background: Original page with the ink whitened and the
            pictures black; None unless the original background is kept.
        foreground_type: Storage type of ``foreground``.
    """

    foreground: NDArray[np.uint8]
    background: NDArray[np.uint8]
    original_background: NDArray[np.uint8] | None
    foreground_type: ForegroundType


def split_mixed_output(
    image: NDArray[np.uint8],
    render: RenderParams,
    text_region: NDArray[np.bool_],
    ink: NDArray[np.bool_],
) -> SplitImages:
    """Split a composed mixed page according to the render flags.

    Args:
        image (NDArray[np.uint8]): The composed page, gray or (H, W, 3|4) BGR(A).
            With the original background kept, it is the original page with
            the ink drawn over it. Not modified.
        render (RenderParams): Decides the mode and the foreground type.
        text_region (NDArray[np.bool_]): True where the page is text, not picture.
        ink (NDArray[np.bool_]): True where text ink is drawn.

    Returns:
        SplitImages: The layers.
    """
    kind = foreground_type_for(render)
    if render.original_background:
        return split_with_original_background(
            image, ink & text_region, text_region, kind
        )
    return split_by_foreground_mask(image, text_region, kind)


def split_by_foreground_mask(
    image: NDArray[np.uint8],
    foreground_mask: NDArray[np.bool_],
    foreground_type: ForegroundType,
) -> SplitImages:
    """Split a page into the part inside a mask and the part outside it.

    Args:
        image (NDArray[np.uint8]): Gray or (H, W, 3|4) BGR(A) page. Not modified.
        foreground_mask (NDArray[np.bool_]): True where the foreground is.
        foreground_type (ForegroundType): Storage type of the foreground.

    Returns:
        SplitImages: Foreground (the page inside the mask, white outside),
            background (the page outside the mask, white inside) and no original
            background.
    """
    foreground = _bilevel_if_needed(
        _keep_where(image, foreground_mask), foreground_type
    )
    return SplitImages(
        foreground=foreground,
        background=_keep_where(image, ~foreground_mask),
        original_background=None,
        foreground_type=foreground_type,
    )


def split_with_original_background(
    image: NDArray[np.uint8],
    ink_mask: NDArray[np.bool_],
    text_region: NDArray[np.bool_],
    foreground_type: ForegroundType,
) -> SplitImages:
    """Split a page that keeps its original background.

    Args:
        image (NDArray[np.uint8]): Original page with the ink drawn over it,
            gray or (H, W, 3|4) BGR(A). Not modified.
        ink_mask (NDArray[np.bool_]): True on ink pixels; they form the foreground.
        text_region (NDArray[np.bool_]): True outside the pictures; the pictures
            form the background.
        foreground_type (ForegroundType): Storage type of the foreground.

    Returns:
        SplitImages: Foreground (ink on white), background (pictures on white)
            and the original background (ink whitened, pictures black).
    """
    picture = ~text_region
    foreground = _bilevel_if_needed(_keep_where(image, ink_mask), foreground_type)
    without_ink = _keep_where(image, ~ink_mask)
    return SplitImages(
        foreground=foreground,
        background=_keep_where(without_ink, picture),
        original_background=_fill_where(without_ink, picture, _BLACK),
        foreground_type=foreground_type,
    )


def merge_split_images(
    foreground: NDArray[np.uint8],
    background: NDArray[np.uint8],
    original_background: NDArray[np.uint8] | None = None,
) -> NDArray[np.uint8]:
    """Rebuild the combined page from stored layers.

    Args:
        foreground (NDArray[np.uint8]): Foreground layer, white where empty.
        background (NDArray[np.uint8]): Background layer, white where empty.
        original_background (NDArray[np.uint8] | None): Original background
            layer; when given, the other two are laid over it on the ink
            (white pixels of it) and the pictures (black pixels of it).

    Returns:
        NDArray[np.uint8]: New page with the layout of ``background`` (or of
            ``original_background`` when given).

    Raises:
        ValueError: If the layers differ in height and width.
    """
    layers = [foreground, background]
    if original_background is not None:
        layers.append(original_background)
    if any(layer.shape[:2] != foreground.shape[:2] for layer in layers):
        raise ValueError("layers must have the same height and width")

    if original_background is None:
        page = background.copy()
        _paint_non_white(page, foreground)
        return page
    page = original_background.copy()
    luma = _luma(original_background)
    _paint_where(page, foreground, luma == _WHITE)
    _paint_where(page, background, luma == _BLACK)
    return page


def write_split_images(
    out_dir: Path,
    file_name: str,
    split: SplitImages,
    dpi: Dpi | None = None,
) -> list[Path]:
    """Write the layers into the foreground, background and original background folders.

    Args:
        out_dir (Path): Output folder; the sub-folders are created inside it.
        file_name (str): File name of the page, with a TIFF or PNG suffix.
        split (SplitImages): Layers to write.
        dpi (Dpi | None): Resolution tags to store.

    Returns:
        list[Path]: Paths written: foreground, background and, when present,
            original background.
    """
    layers = [
        (FOREGROUND_DIR, _writable(split.foreground, split.foreground_type)),
        (BACKGROUND_DIR, _writable(split.background, ForegroundType.COLOR)),
    ]
    if split.original_background is not None:
        layers.append(
            (
                ORIGINAL_BACKGROUND_DIR,
                _writable(split.original_background, ForegroundType.COLOR),
            )
        )
    written = []
    for folder, pixels in layers:
        path = Path(out_dir) / folder / file_name
        path.parent.mkdir(parents=True, exist_ok=True)
        write_image(path, pixels, dpi)
        written.append(path)
    return written


def _writable(image: NDArray[np.uint8], kind: ForegroundType) -> NDArray:
    """Drop alpha and use the bilevel array form for binary layers."""
    if kind == ForegroundType.BINARY:
        return _luma(image) < _BILEVEL_SPLIT
    return image[..., :_COLOR_CHANNELS] if image.ndim == 3 else image


def _luma(image: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Gray level of an image."""
    return image if image.ndim == 2 else bgr_to_gray(image)


def _bilevel_if_needed(
    image: NDArray[np.uint8], foreground_type: ForegroundType
) -> NDArray[np.uint8]:
    """Reduce a layer to 2-D 0/255 when it is stored as binary."""
    if foreground_type != ForegroundType.BINARY:
        return image
    return np.where(_luma(image) < _BILEVEL_SPLIT, _BLACK, _WHITE).astype(np.uint8)


def _fill_where(
    image: NDArray[np.uint8], mask: NDArray[np.bool_], level: int
) -> NDArray[np.uint8]:
    """Copy of an image with a gray level in the colour channels where mask is set."""
    result = image.copy()
    if image.ndim == 2:
        result[mask] = level
    else:
        result[mask, :_COLOR_CHANNELS] = level
    return result


def _keep_where(image: NDArray[np.uint8], mask: NDArray[np.bool_]) -> NDArray[np.uint8]:
    """Copy of an image that is white wherever ``mask`` is not set; alpha is kept."""
    return _fill_where(image, ~mask, _WHITE)


def _paint_where(
    page: NDArray[np.uint8], layer: NDArray[np.uint8], mask: NDArray[np.bool_]
) -> None:
    """Copy layer pixels onto the page where ``mask`` is set, in place."""
    if page.ndim == 2:
        page[mask] = _luma(layer)[mask]
    elif layer.ndim == 2:
        page[mask, :_COLOR_CHANNELS] = layer[mask][:, None]
    else:
        page[mask, :_COLOR_CHANNELS] = layer[mask, :_COLOR_CHANNELS]


def _paint_non_white(page: NDArray[np.uint8], layer: NDArray[np.uint8]) -> None:
    """Copy the pixels of a layer that are not pure white onto the page, in place."""
    channels = layer if layer.ndim == 2 else layer[..., :_COLOR_CHANNELS]
    covered = channels != _WHITE
    _paint_where(page, layer, covered if layer.ndim == 2 else covered.any(axis=2))
