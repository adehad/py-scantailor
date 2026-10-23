"""Page background estimation for illumination normalisation.

Models the paper background as a smooth polynomial surface fitted to a small
version of the page after text and shadows have been suppressed morphologically.
"""

from collections.abc import Sequence

import cv2
import numpy as np
from numpy.typing import NDArray

from scantailor.core.debug_images import DebugImages
from scantailor.imageproc.polynomial_line import fit_polynomial_lines
from scantailor.imageproc.polynomial_surface import PolynomialSurface, _axis_basis
from scantailor.imageproc.reconstruction import reconstruction

# Longest side of the working image. Fitting at this size is cheap; the surface
# is then evaluated exactly at every pixel of the page.
_WORKING_SIZE = 300
# Height of the vertical opening that removes letter remnants. Chosen for the
# working size above.
_LETTER_REMNANT_OPENING = 20
# Share of pixels that may differ from the shadow model before the page is
# assumed to hold a picture rather than a shadow.
_PICTURE_PIXEL_FRACTION = 0.01
_PICTURE_DIFFERENCE = 10
# A pixel this much darker than its fitted line is background; lighter pixels
# on the line are rejected as outliers.
_LINE_OUTLIER_MARGIN = 30
# Rows and columns with fewer accepted pixels than 1/4 are dropped entirely.
_MIN_LINE_COVERAGE_DIVISOR = 4

SURFACE_HOR_DEGREE = 8
SURFACE_VERT_DEGREE = 5

# Rows of the page whose background is evaluated and applied together; small
# enough that the float64 intermediate stays in the CPU cache.
_BAND_ROWS = 32


def _fill_dark_holes(image: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Fill dark regions that do not touch the image border (8-connectivity)."""
    seed = np.full_like(image, 255)
    seed[0, :] = image[0, :]
    seed[-1, :] = image[-1, :]
    seed[:, 0] = image[:, 0]
    seed[:, -1] = image[:, -1]
    return reconstruction(seed, image, "erosion", 8)


def _fill_vertically(image: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Fill dark runs that are bounded by lighter pixels above and below.

    Each pixel becomes the larger of itself and the smaller of the running
    maximum above it and the largest value below it.
    """
    above = np.maximum.accumulate(image, axis=0)
    below = np.zeros_like(image)
    below[:-1] = np.maximum.accumulate(image[:0:-1], axis=0)[::-1]
    return np.maximum(image, np.minimum(above, below))


def _open_vertical(image: NDArray[np.uint8], height: int) -> NDArray[np.uint8]:
    """Grayscale opening with a 1 x height brick, treating outside pixels as black."""
    pad = height
    padded = np.pad(image, ((pad, pad), (0, 0)), constant_values=0)
    kernel = np.ones((height, 1), np.uint8)
    opened = cv2.morphologyEx(padded, cv2.MORPH_OPEN, kernel)
    return np.asarray(opened[pad:-pad], dtype=np.uint8)


def _suppress_text_and_shadow(
    small: NDArray[np.uint8], dbg: DebugImages | None = None
) -> NDArray[np.uint8]:
    """Remove text from a downscaled page, choosing between two strategies.

    Hole filling followed by a vertical opening copes with shadows that touch a
    side edge. Vertical filling copes with pictures lying partly in a spine
    shadow. The first is used unless the difference between the two is not
    explained by a smooth shadow.

    Args:
        small (NDArray[np.uint8]): Downscaled grayscale page.
        dbg (DebugImages | None): Receives both candidates, their difference,
            its smooth fit, the residual and the chosen candidate.

    Returns:
        NDArray[np.uint8]: Page with dark text suppressed.
    """
    method1 = _open_vertical(_fill_dark_holes(small), _LETTER_REMNANT_OPENING)
    method2 = _fill_vertically(small)

    # uint8 wrap-around is intended: it keeps the residual large wherever the
    # methods disagree in sign, which pushes the decision towards method2.
    diff = method2 - method1
    approximated = PolynomialSurface.fit(diff, 3, 3).render(diff.shape)
    residual = cv2.absdiff(diff, approximated)
    if dbg is not None:
        dbg.add(method1, "preproc_method1")
        dbg.add(method2, "preproc_method2")
        dbg.add(diff, "raw_diff")
        dbg.add(approximated, "approx_diff")
        dbg.add(residual, "raw_vs_approx_diff")

    unexplained = np.count_nonzero(residual > _PICTURE_DIFFERENCE)
    if unexplained < _PICTURE_PIXEL_FRACTION * residual.size:
        if dbg is not None:
            dbg.add(method1, "use_method1")
        return method1
    if dbg is not None:
        dbg.add(method2, "use_method2")
    return method2


def _mask_outliers(
    background: NDArray[np.uint8],
    mask: NDArray[np.bool_],
) -> NDArray[np.bool_]:
    """Reject pixels that sit well below a smooth fit of their column or row."""
    column_fit = fit_polynomial_lines(background, 2, axis=0).astype(np.int16)
    row_fit = fit_polynomial_lines(background, 4, axis=1).astype(np.int16)
    darker = background.astype(np.int16) + _LINE_OUTLIER_MARGIN
    return mask & ~(darker < column_fit) & ~(darker < row_fit)


def _drop_sparse_lines(mask: NDArray[np.bool_]) -> NDArray[np.bool_]:
    """Clear rows then columns that are mostly rejected."""
    height, width = mask.shape
    result = mask.copy()
    result[result.sum(axis=1) < width // _MIN_LINE_COVERAGE_DIVISOR, :] = False
    result[:, result.sum(axis=0) < height // _MIN_LINE_COVERAGE_DIVISOR] = False
    return result


def estimate_background(
    gray: NDArray[np.uint8],
    area_to_consider: Sequence[tuple[float, float]] | None = None,
    dbg: DebugImages | None = None,
) -> PolynomialSurface:
    """Estimate the paper background of a page as a polynomial surface.

    The page is scaled to 300 px on its longer side, text and
    shadows are suppressed morphologically, outlying pixels are rejected, and a
    degree (8, 5) surface is fitted by least squares to what remains.

    Args:
        gray (NDArray[np.uint8]): 2-D grayscale page, dark ink on light paper.
            Not modified.
        area_to_consider (Sequence[tuple[float, float]] | None): Polygon of (x, y)
            vertices in ``gray`` pixel coordinates. Pixels outside it do not take
            part in the fit. None, or a polygon that covers no pixel of the
            image, uses the whole image.
        dbg (DebugImages | None): Receives the downscaled page, the text
            suppression steps and the mask of pixels the fit uses after each
            refinement; True, drawn black, takes part.

    Returns:
        PolynomialSurface: Background model. ``render(gray.shape)`` gives a
        uint8 image of the same size as the input.

    Raises:
        ValueError: If ``gray`` is not a non-empty 2-D array.
    """
    if gray.ndim != 2 or gray.size == 0:
        raise ValueError("gray must be a non-empty 2-D array")

    height, width = gray.shape
    # Small pages are enlarged too: the vertical opening height is tuned for a
    # longer side of exactly _WORKING_SIZE.
    factor = _WORKING_SIZE / max(height, width)
    small_size = (max(1, round(width * factor)), max(1, round(height * factor)))
    small = np.asarray(
        cv2.resize(
            gray,
            small_size,
            interpolation=cv2.INTER_AREA if factor < 1 else cv2.INTER_LINEAR,
        ),
        dtype=np.uint8,
    )
    if dbg is not None:
        dbg.add(small, "downscaled")
    background = _suppress_text_and_shadow(small, dbg)

    mask = np.ones(background.shape, dtype=bool)
    if area_to_consider is not None and len(area_to_consider) > 0:
        scale = np.array([small_size[0] / width, small_size[1] / height])
        polygon = np.round(
            np.asarray(area_to_consider, dtype=np.float64) * scale
        ).astype(np.int32)
        inside = np.zeros(background.shape, dtype=np.uint8)
        cv2.fillPoly(inside, [polygon], 1)
        if inside.any():
            mask = inside.astype(bool)
    if dbg is not None:
        dbg.add(mask, "areaToConsider")

    mask = _mask_outliers(background, mask)
    if dbg is not None:
        dbg.add(mask, "mask")
    mask = cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    if dbg is not None:
        dbg.add(mask, "eroded")
    mask = _drop_sparse_lines(mask)
    if dbg is not None:
        dbg.add(mask, "lines_extended")
    return PolynomialSurface.fit(
        background, SURFACE_HOR_DEGREE, SURFACE_VERT_DEGREE, mask
    )


# cv2.divide rounds exact halves to even; the normalised level rounds them up.
# Scaling by a little over 255 lifts every exact half just above it. Any other
# quotient gray * 255 / background is at least 1 / (2 * 255) away from a half,
# far more than the at most 255 * 1e-6 this adds, so no other result changes.
_SCALE_ROUNDING_HALVES_UP = 255 * (1 + 1e-6)


def _divide_by_background(
    gray: NDArray[np.uint8],
    background: NDArray[np.uint8],
    out: NDArray[np.uint8],
) -> None:
    """Write the normalised level of every pixel into ``out``.

    The level is ``(gray * 255 + background // 2) // background``, and 255
    where ``gray`` is not below ``background``. All three arrays have one shape
    and rows OpenCV can address.
    """
    cv2.divide(gray, background, out, scale=_SCALE_ROUNDING_HALVES_UP)
    # cv2.divide gives 0 where the background is 0, and every level is at or above that.
    cv2.max(out, cv2.compare(gray, background, cv2.CMP_GE), dst=out)


def raise_above_background(
    gray: NDArray[np.uint8],
    background: NDArray[np.uint8],
) -> NDArray[np.uint8]:
    """Scale each pixel so that the background becomes white.

    Computes ``round(gray * 255 / background)``; pixels within one level of the
    background, or above it, become 255.

    Args:
        gray (NDArray[np.uint8]): Image to normalise. Not modified.
        background (NDArray[np.uint8]): Background estimate of the same shape.

    Returns:
        NDArray[np.uint8]: Normalised image.
    """
    out = np.empty(gray.shape, dtype=np.uint8)
    _divide_by_background(
        np.ascontiguousarray(gray), np.ascontiguousarray(background), out
    )
    return out


def _render_band(
    left_factor: NDArray[np.float64],
    hor_basis_t: NDArray[np.float64],
    start: int,
    stop: int,
) -> NDArray[np.uint8]:
    """Evaluate rows ``start:stop`` of a surface, rounded as ``render`` rounds."""
    values = left_factor[start:stop] @ hor_basis_t
    values *= 255.0
    values += 0.5
    np.floor(values, out=values)
    np.clip(values, 0, 255, out=values)
    return values.astype(np.uint8)


def normalize_illumination_gray(
    gray: NDArray[np.uint8],
    area_to_consider: Sequence[tuple[float, float]] | None = None,
    dbg: DebugImages | None = None,
) -> NDArray[np.uint8]:
    """Flatten uneven lighting in a grayscale page.

    Args:
        gray (NDArray[np.uint8]): 2-D grayscale page. Not modified.
        area_to_consider (Sequence[tuple[float, float]] | None): Polygon limiting
            where the background is sampled; see ``estimate_background``.
        dbg (DebugImages | None): Receives the page, the estimation's images,
            the background rendered at full size and the normalised page.

    Returns:
        NDArray[np.uint8]: New array of the same shape in which the estimated
        paper colour is mapped to 255.
    """
    if dbg is not None:
        dbg.add(gray, "toBeNormalized")
    surface = estimate_background(gray, area_to_consider, dbg)
    if dbg is not None:
        dbg.add(surface.render(gray.shape), "background")
    height, width = gray.shape
    # Bands give the same values as ``surface.render`` without its full-page
    # float64 temporaries.
    left_factor = _axis_basis(height, surface.vert_degree) @ surface.coeffs
    hor_basis_t = np.ascontiguousarray(_axis_basis(width, surface.hor_degree).T)
    gray = np.ascontiguousarray(gray)
    out = np.empty(gray.shape, dtype=np.uint8)
    for start in range(0, height, _BAND_ROWS):
        stop = min(start + _BAND_ROWS, height)
        _divide_by_background(
            gray[start:stop],
            _render_band(left_factor, hor_basis_t, start, stop),
            out[start:stop],
        )
    if dbg is not None:
        dbg.add(out, "normalized_illumination")
    return out
