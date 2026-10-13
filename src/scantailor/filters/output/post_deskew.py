"""Skew left in a page after dewarping, and the rotation that removes it."""

from collections.abc import Sequence

import numpy as np
from loguru import logger
from numpy.typing import NDArray

from scantailor.core.estimate_background import normalize_illumination_gray
from scantailor.dewarping.dewarper import CylindricalSurfaceDewarper
from scantailor.dewarping.raster_dewarper import (
    compute_dewarp_maps,
    output_model_domain,
    remap_with_maps,
)
from scantailor.imageproc.analysis import find_skew
from scantailor.imageproc.binarize import binarize_otsu
from scantailor.imageproc.filters import bgr_to_gray

from .dewarping_options import DewarpingMode
from .generator import DewarperSource
from .params import Params
from .render_params import RenderParams

# Only a page dewarped from the edges the user drew or from the binding strip
# is measured again: the lines of an automatically dewarped page come out
# straight by construction.
POST_DESKEW_MODES = frozenset({DewarpingMode.MANUAL, DewarpingMode.MARGINAL})
_WHITE = 255
_DEWARPING_ERRORS = (ValueError, ArithmeticError, np.linalg.LinAlgError, IndexError)


def find_post_deskew_angle(dewarped_gray: NDArray[np.uint8]) -> float:
    """Measure the rotation that levels the text lines of a dewarped page.

    Args:
        dewarped_gray (NDArray[np.uint8]): 2-D gray dewarped page, dark text on
            a light background.

    Returns:
        float: The rotation in degrees, positive turning the page clockwise;
            0 when the skew is zero or not found with good confidence.
    """
    skew = find_skew(binarize_otsu(dewarped_gray))
    if skew.angle == 0.0 or not skew.is_confident():
        return 0.0
    return -skew.angle


def measure_post_deskew(
    image: NDArray[np.uint8],
    params: Params,
    source: DewarperSource | None,
    page_outline: Sequence[tuple[float, float]] | None = None,
) -> tuple[float, DewarperSource | None]:
    """Dewarp the page once to find the rotation to apply after dewarping.

    The page is prepared as the output generator prepares it, so a dewarper
    built from it is the one the generator would build.

    Args:
        image (NDArray[np.uint8]): The page the generator is given, 2-D gray or
            BGR(A).
        params (Params): Output parameters; the angle is only measured when the
            dewarping mode and the post-deskew option ask for it.
        source (DewarperSource | None): The page's dewarper or the function
            making it from the normalised gray page.
        page_outline (Sequence[tuple[float, float]] | None): Page outline in the
            pixels of ``image``, limiting where illumination is estimated.

    Returns:
        tuple[float, DewarperSource | None]: The angle in degrees, positive
        turning the page clockwise (0 when it is not to be applied or cannot be
        measured), and the source to give the generator in place of ``source``:
        a made dewarper, so that building it is not repeated.
    """
    options = params.dewarping
    if options.mode not in POST_DESKEW_MODES or source is None:
        return 0.0, source
    if not options.post_deskew:
        return 0.0, source

    gray = _generator_gray(image, params, page_outline)
    dewarper = (
        source if isinstance(source, CylindricalSurfaceDewarper) else source(gray)
    )
    if dewarper is None:
        return 0.0, _no_dewarper
    height, width = gray.shape
    try:
        maps = compute_dewarp_maps(
            dewarper, (width, height), output_model_domain(dewarper, (width, height))
        )
    except _DEWARPING_ERRORS as error:
        logger.warning(
            "Post-dewarp skew not measured, the model is unusable: {}", error
        )
        return 0.0, dewarper
    return find_post_deskew_angle(remap_with_maps(gray, maps, _WHITE)), dewarper


def _no_dewarper(_gray: NDArray[np.uint8]) -> None:
    """Stand-in source for a page whose dewarper could not be made."""
    return None


def _generator_gray(
    image: NDArray[np.uint8],
    params: Params,
    page_outline: Sequence[tuple[float, float]] | None,
) -> NDArray[np.uint8]:
    """Return the gray page the output generator builds its dewarper from."""
    oriented = image
    if not params.black_on_white:
        oriented = image.copy()
        if image.ndim == 2:
            oriented[:] = 255 - image
        else:
            oriented[..., :3] = 255 - image[..., :3]
    gray = bgr_to_gray(oriented)
    render = RenderParams.from_params(params)
    if render.normalize_illumination or render.normalize_illumination_color:
        return normalize_illumination_gray(gray, page_outline)
    return gray
