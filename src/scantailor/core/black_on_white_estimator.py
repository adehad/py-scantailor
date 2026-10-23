"""Tell a page of dark content on a light background from the reverse.

Detection in Deskew and Select Content, and Output's thresholds, expect dark
content on light paper. A page of light text on a dark background is inverted
before it is measured once this module says it is one.
"""

import cv2
import numpy as np
from numpy.typing import NDArray

from scantailor.core.content_mask import render_page_at_dpi
from scantailor.core.debug_images import DebugImages
from scantailor.core.transformation import ImageTransformation
from scantailor.imageproc.background_color import otsu_threshold
from scantailor.imageproc.binarize import binarize_otsu
from scantailor.imageproc.despeckle import despeckle
from scantailor.imageproc.filters import gray_histogram

REFINING_DPI = 150
_NORMAL_DESPECKLE_LEVEL = 2.0
# Strokes thinner than this, dark or light, mark where the content is.
_STROKE_BRICK = (13, 13)
# Joins the strokes of a text block into one area, then widens it a little.
_BLOCK_CLOSE_BRICK = (200, 200)
_BLOCK_DILATE_BRICK = (30, 30)


def is_black_on_white(
    pixels: NDArray[np.uint8],
    xform: ImageTransformation,
    dbg: DebugImages | None = None,
) -> bool:
    """Tell whether a page has dark content on a light background.

    A page whose pixels are mostly light inside its pre-crop area is black on
    white. A mostly dark one may still be, when a dark surround around the
    paper is what tips the balance, so it is looked at again where its content
    is (see ``is_black_on_white_refining``).

    Only the pre-crop area and the resolution of ``xform`` count, so every stage
    gets the answer Deskew gets, whatever rotation or crop it has added since.

    Args:
        pixels (NDArray[np.uint8]): The image ``xform`` starts from, gray or BGR.
            Not modified.
        xform (ImageTransformation): The page's transformation. Not modified.
        dbg (DebugImages | None): Receives the refining pass's images when the
            page is looked at again.

    Returns:
        bool: True for dark content on a light background, and for a page
            whose pre-crop area holds no pixel.
    """
    page_xform = _pre_crop_only(xform)
    gray = _gray(pixels)
    if is_black_on_white_in_area(gray, _source_pre_crop_mask(page_xform, gray.shape)):
        return True
    return is_black_on_white_refining(gray, page_xform, dbg)


def is_black_on_white_in_area(gray: NDArray[np.uint8], area: NDArray[np.bool_]) -> bool:
    """Tell whether at most half of an area is darker than its Otsu threshold.

    Args:
        gray (NDArray[np.uint8]): (H, W) gray image. Not modified.
        area (NDArray[np.bool_]): (H, W) mask, True for the pixels that count.

    Returns:
        bool: True when the dark pixels are at most half of the area, and for
            an empty area.

    Raises:
        ValueError: If ``gray`` and ``area`` differ in shape.
    """
    if gray.shape != area.shape:
        msg = f"image of shape {gray.shape} and area of shape {area.shape} differ"
        raise ValueError(msg)
    counted = gray[area]
    if counted.size == 0:
        return True
    threshold = otsu_threshold(gray_histogram(counted))
    dark = int(np.count_nonzero(counted < threshold))
    return 2 * dark <= counted.size


def is_black_on_white_refining(
    pixels: NDArray[np.uint8],
    xform: ImageTransformation,
    dbg: DebugImages | None = None,
) -> bool:
    """Tell the polarity from the areas where a page has strokes.

    At 150 dpi the page is binarised and cleaned of specks in both polarities.
    Strokes thinner than 13 pixels, dark ones on light and light ones on dark,
    mark the content; closing them into blocks gives the content area, which
    leaves out a dark surround or a margin of plain paper. The page is black on
    white when at most half of that area is dark.

    Args:
        pixels (NDArray[np.uint8]): The image ``xform`` starts from, gray or BGR.
            Not modified.
        xform (ImageTransformation): Maps the image to the page. Not modified.
        dbg (DebugImages | None): Receives the cleaned 150 dpi page and the
            content area; True, drawn black, is content.

    Returns:
        bool: True for dark content on a light background, and for a page with
            no area at 150 dpi.
    """
    page = render_page_at_dpi(pixels, xform, REFINING_DPI)
    if page is None:
        return True
    gray = page.gray.copy()
    gray[~page.inside] = 255
    binary = binarize_otsu(gray)
    binary = despeckle(binary, REFINING_DPI, _NORMAL_DESPECKLE_LEVEL)
    light_specks_removed = despeckle(
        255 - binary, REFINING_DPI, _NORMAL_DESPECKLE_LEVEL
    )
    dark = light_specks_removed != 0
    if dbg is not None:
        dbg.add(dark, "bw150")

    content = _content_area(dark)
    if dbg is not None:
        dbg.add(content, "content_mask")
    return 2 * int(np.count_nonzero(dark & content)) <= int(np.count_nonzero(content))


def _content_area(dark: NDArray[np.bool_]) -> NDArray[np.bool_]:
    """Return the blocks of thin strokes of a binary page, dark or light."""
    stroke_kernel = np.ones(_STROKE_BRICK[::-1], dtype=np.uint8)
    dark_strokes = dark & ~_morphology(dark, cv2.MORPH_OPEN, stroke_kernel)
    light_strokes = _morphology(dark, cv2.MORPH_CLOSE, stroke_kernel) & ~dark
    strokes = dark_strokes | light_strokes
    blocks = _morphology(
        strokes, cv2.MORPH_CLOSE, np.ones(_BLOCK_CLOSE_BRICK[::-1], dtype=np.uint8)
    )
    return _morphology(
        blocks, cv2.MORPH_DILATE, np.ones(_BLOCK_DILATE_BRICK[::-1], dtype=np.uint8)
    )


def _morphology(
    image: NDArray[np.bool_], operation: int, kernel: NDArray[np.uint8]
) -> NDArray[np.bool_]:
    """Apply a morphological operation as if the image were surrounded by white.

    The image is padded by the kernel's size, so a closing near the border
    behaves as though the page went on, blank, past it.
    """
    pad_y, pad_x = kernel.shape
    padded = cv2.copyMakeBorder(
        image.view(np.uint8),
        pad_y,
        pad_y,
        pad_x,
        pad_x,
        cv2.BORDER_CONSTANT,
        value=0,
    )
    result = cv2.morphologyEx(
        padded, operation, kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0
    )
    height, width = image.shape
    return result[pad_y : pad_y + height, pad_x : pad_x + width].astype(np.bool_)


def _gray(pixels: NDArray[np.uint8]) -> NDArray[np.uint8]:
    if pixels.ndim == 3 and pixels.shape[2] == 3:
        return np.asarray(cv2.cvtColor(pixels, cv2.COLOR_BGR2GRAY), dtype=np.uint8)
    return np.ascontiguousarray(pixels)


def _pre_crop_only(xform: ImageTransformation) -> ImageTransformation:
    """Return a transformation of the same image that only pre-crops it."""
    page = ImageTransformation(orig_rect=xform.orig_rect, orig_dpi=xform.orig_dpi)
    page.set_pre_rotation(xform.pre_rotation)
    area = xform.pre_crop_area
    if len(area) >= 3:
        page.set_pre_crop_area(area)
    return page


def _source_pre_crop_mask(
    xform: ImageTransformation, shape: tuple[int, ...]
) -> NDArray[np.bool_]:
    """Mark the pixels of the source image inside the pre-crop area."""
    mask = np.zeros(shape[:2], dtype=np.uint8)
    area = xform.map_polygon_back(xform.resulting_pre_crop_area)
    if len(area) < 3:
        mask[:] = 1
    else:
        cv2.fillPoly(mask, [np.round(area).astype(np.int32)], 1)
    return mask.astype(np.bool_)
