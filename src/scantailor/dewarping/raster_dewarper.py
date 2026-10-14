"""Raster dewarping - applying dewarping to images.

This module provides functions to apply the cylindrical surface dewarping
model to actual images, producing corrected (flattened) output.

The C++ implementation uses custom area-mapping interpolation for high quality.
This Python version uses OpenCV's cv2.remap() with configurable interpolation,
which is simpler and leverages optimized implementations.
"""

from enum import Enum

import cv2
import numpy as np
from numba import njit
from numpy.typing import NDArray

from scantailor.dewarping.dewarper import CylindricalSurfaceDewarper


class InterpolationMethod(Enum):
    """Interpolation method for dewarping.

    NEAREST: Nearest-neighbor, fast but blocky.
    BILINEAR: Bilinear interpolation, good balance of quality/speed.
    BICUBIC: Bicubic interpolation, higher quality but slower.
    LANCZOS: Lanczos interpolation, best quality but slowest.
    """

    NEAREST = cv2.INTER_NEAREST
    BILINEAR = cv2.INTER_LINEAR
    BICUBIC = cv2.INTER_CUBIC
    LANCZOS = cv2.INTER_LANCZOS4


def dewarp_image(
    src: NDArray[np.uint8],
    dewarper: CylindricalSurfaceDewarper,
    dst_size: tuple[int, int],
    model_domain: tuple[float, float, float, float],
    background_color: int | tuple[int, int, int] | tuple[int, int, int, int] = 255,
    interpolation: InterpolationMethod = InterpolationMethod.BILINEAR,
) -> NDArray[np.uint8]:
    """Apply dewarping to an image.

    This function maps each pixel in the output image back to its corresponding
    location in the input image using the dewarping model, then samples the
    input using the specified interpolation method.

    Args:
        src: Source image (grayscale, RGB, or RGBA).
        dewarper: The cylindrical surface dewarper with the distortion model.
        dst_size: Output size as (width, height).
        model_domain: The region in dewarped space that maps to the output,
            as (left, top, right, bottom) in normalized [0, 1] coordinates.
        background_color: Color for pixels outside the source image.
            For grayscale: int. For RGB: (r, g, b). For RGBA: (r, g, b, a).
        interpolation: Interpolation method to use.

    Returns:
        Dewarped image with the specified size and type.

    Example:
        >>> import numpy as np
        >>> from scantailor.dewarping import CylindricalSurfaceDewarper
        >>> from scantailor.dewarping.raster_dewarper import dewarp_image
        >>>
        >>> # Create a test image and dewarper
        >>> image = numpy.zeros((300, 200), dtype=numpy.uint8)
        >>> top = numpy.array([[0, 10], [100, 5], [199, 12]])
        >>> bottom = numpy.array([[0, 290], [100, 295], [199, 288]])
        >>> dewarper = CylindricalSurfaceDewarper.from_directrices(top, bottom)
        >>>
        >>> # Dewarp the image
        >>> result = dewarp_image(
        ...     image, dewarper,
        ...     dst_size=(200, 300),
        ...     model_domain=(0, 0, 1, 1)
        ... )
    """
    maps = compute_dewarp_maps(dewarper, dst_size, model_domain)
    return remap_with_maps(src, maps, background_color, interpolation)


def remap_with_maps(
    src: NDArray[np.uint8],
    maps: tuple[NDArray[np.float32], NDArray[np.float32]],
    background_color: int | tuple[int, int, int] | tuple[int, int, int, int] = 255,
    interpolation: InterpolationMethod = InterpolationMethod.BILINEAR,
) -> NDArray[np.uint8]:
    """Sample an image through precomputed dewarp maps.

    Computing the maps costs far more than sampling, so a page that needs
    several images dewarped (pixels, illumination-normalised gray, masks)
    builds the maps once with ``compute_dewarp_maps`` and reuses them.

    Args:
        src (numpy.ndarray of numpy.uint8): Source image, gray or 3 or 4 channels.
        maps (tuple[numpy.ndarray of numpy.float32, numpy.ndarray of numpy.float32]):
            ``(map_x, map_y)``
            of shape (height, width) in source pixel coordinates.
        background_color (int | tuple[int, ...]): Colour for output pixels whose
            source position falls outside ``src``.
        interpolation (InterpolationMethod): Sampling filter.

    Returns:
        numpy.ndarray of numpy.uint8:
            New image of the maps' size with ``src``'s channel layout.

    Raises:
        ValueError: If ``src`` has an unsupported shape.
    """
    map_x, map_y = maps
    dst_height, dst_width = map_x.shape

    # Prepare background
    if src.ndim == 2:
        # Grayscale
        bg = (
            int(background_color)
            if isinstance(background_color, int | float)
            else background_color[0]
        )
        dst = np.full((dst_height, dst_width), bg, dtype=np.uint8)
    elif src.ndim == 3 and src.shape[2] == 3:
        # RGB/BGR
        if isinstance(background_color, int):
            bg = (background_color, background_color, background_color)
        else:
            bg = background_color[:3]
        dst = np.full((dst_height, dst_width, 3), bg, dtype=np.uint8)
    elif src.ndim == 3 and src.shape[2] == 4:
        # RGBA/BGRA
        if isinstance(background_color, int):
            bg = (background_color, background_color, background_color, 255)
        elif len(background_color) == 3:
            bg = (*background_color, 255)
        else:
            bg = background_color
        dst = np.full((dst_height, dst_width, 4), bg, dtype=np.uint8)
    else:
        raise ValueError(f"Unsupported image format with shape {src.shape}")

    # Apply the remapping
    cv2.remap(
        src,
        map_x,
        map_y,
        interpolation=interpolation.value,
        dst=dst,
        borderMode=cv2.BORDER_TRANSPARENT,
    )

    return dst


def compute_dewarp_maps(
    dewarper: CylindricalSurfaceDewarper,
    dst_size: tuple[int, int],
    model_domain: tuple[float, float, float, float],
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Compute the source coordinates of every output pixel.

    Args:
        dewarper (CylindricalSurfaceDewarper): The dewarping model.
        dst_size (tuple[int, int]): Output (width, height).
        model_domain (tuple[float, float, float, float]): (left, top, right,
            bottom) in model coordinates that the output spans. Values outside
            [0, 1] extrapolate the model.

    Returns:
        tuple[numpy.ndarray of numpy.float32, numpy.ndarray of numpy.float32]:
            ``(map_x, map_y)``
            for ``remap_with_maps`` or ``cv2.remap``.

    Raises:
        ValueError: If the domain has no extent.
    """
    left, top, right, bottom = model_domain
    if right <= left or bottom <= top:
        raise ValueError("model_domain must have positive width and height")
    # The grids address pixel corners; cv2.remap addresses pixel centres.
    return _compute_dewarp_maps(
        dewarper, dst_size[0], dst_size[1], left, top, right, bottom, shift=0.5
    )


def output_model_domain(
    dewarper: CylindricalSurfaceDewarper,
    dst_size: tuple[int, int],
) -> tuple[float, float, float, float]:
    """Place the model's bounding box inside an output of a given size.

    The model's bounding box is where the flattened page content lands, so
    margins around it stay margins. Unrolling a curved page stretches the lines,
    so the box is shrunk vertically, about the output's centre, by the arc
    length factor; stretching it horizontally instead would push content out of
    the page.

    Args:
        dewarper (CylindricalSurfaceDewarper): The dewarping model.
        dst_size (tuple[int, int]): Output (width, height) in the pixel space the
            dewarper's directrices are in.

    Returns:
        tuple[float, float, float, float]: Model-space (left, top, right, bottom)
            covered by the output, for ``compute_dewarp_maps``.

    Raises:
        ValueError: If the model's bounding box has no extent.
    """
    width, height = dst_size
    box_left, box_top, box_right, box_bottom = dewarper.directrix_bounding_box()
    box_width = box_right - box_left
    box_height = box_bottom - box_top
    if box_width <= 0 or box_height <= 0:
        raise ValueError("The distortion model has no extent")

    vertical_scale = 1.0 / dewarper.directrix_arc_length
    centre_y = height / 2.0
    domain_top = centre_y - (centre_y - box_top) * vertical_scale
    domain_height = box_height * vertical_scale
    return (
        -box_left / box_width,
        -domain_top / domain_height,
        (width - box_left) / box_width,
        (height - domain_top) / domain_height,
    )


def _compute_dewarp_maps(
    dewarper: CylindricalSurfaceDewarper,
    dst_width: int,
    dst_height: int,
    left: float,
    top: float,
    right: float,
    bottom: float,
    shift: float = 0.0,
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Compute the coordinate mapping grids for cv2.remap.

    For each pixel (x, y) in the destination image, computes the corresponding
    source coordinates.

    Args:
        dewarper: The dewarping model.
        dst_width: Destination image width.
        dst_height: Destination image height.
        left: Left model domain bound.
        top: Top model domain bound.
        right: Right model domain bound.
        bottom: Bottom model domain bound.
        shift: Subtracted from every coordinate after conversion to float32.

    Returns:
        Tuple of (map_x, map_y) arrays for cv2.remap.
    """
    x_scale = (right - left) / dst_width
    y_scale = (bottom - top) / dst_height

    generatrices = dewarper.map_generatrices(
        left + (np.arange(dst_width) + 0.5) * x_scale
    )
    starts = generatrices.start
    directions = generatrices.end - generatrices.start
    homographies = generatrices.pln2img_1d

    columns = np.ascontiguousarray(
        np.column_stack(
            [
                starts[:, 0],
                starts[:, 1],
                directions[:, 0],
                directions[:, 1],
                homographies[:, 0, 0],
                homographies[:, 0, 1],
                homographies[:, 1, 0],
                homographies[:, 1, 1],
            ]
        ).T,
        dtype=np.float64,
    )
    map_x = np.empty((dst_height, dst_width), dtype=np.float32)
    map_y = np.empty((dst_height, dst_width), dtype=np.float32)
    _fill_maps(columns, top, y_scale, np.float32(shift), map_x, map_y)
    return map_x, map_y


@njit(cache=True)
def _fill_maps(
    columns: NDArray[np.float64],
    top: float,
    y_scale: float,
    shift: np.float32,
    map_x: NDArray[np.float32],
    map_y: NDArray[np.float32],
) -> None:
    """Fill the maps row by row from the per-column generatrix constants.

    ``columns`` rows are the generatrix start x and y, its direction x and y
    and the four entries of its 1-D homography. A degenerate homography
    (denominator near zero) maps to the middle of the generatrix.
    """
    height, width = map_x.shape
    for row in range(height):
        model_y = top + (row + 0.5) * y_scale
        for col in range(width):
            numerator = columns[4, col] * model_y + columns[5, col]
            denominator = columns[6, col] * model_y + columns[7, col]
            if abs(denominator) < 1e-10:
                along = 0.5
            else:
                along = numerator / denominator
            map_x[row, col] = (
                np.float32(columns[0, col] + along * columns[2, col]) - shift
            )
            map_y[row, col] = (
                np.float32(columns[1, col] + along * columns[3, col]) - shift
            )


def compute_dewarped_size(
    src_size: tuple[int, int],
    dewarper: CylindricalSurfaceDewarper,
    dpi_scale: float = 1.0,
) -> tuple[int, int]:
    """Compute appropriate output size for dewarped image.

    The dewarped image should have approximately the same pixel density
    as the source, accounting for the arc length of the directrix.

    Args:
        src_size: Source image size as (width, height).
        dewarper: The dewarping model.
        dpi_scale: Optional scale factor (>1 for higher resolution output).

    Returns:
        Recommended (width, height) for the dewarped output.
    """
    src_width, src_height = src_size
    arc_length = dewarper.directrix_arc_length

    # The directrix arc length tells us how much wider the unfolded
    # surface is compared to the chord (straight-line width).
    dst_width = int(src_width * arc_length * dpi_scale)
    dst_height = int(src_height * dpi_scale)

    return (max(1, dst_width), max(1, dst_height))
