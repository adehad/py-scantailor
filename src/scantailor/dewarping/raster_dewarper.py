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
        >>> image = np.zeros((300, 200), dtype=np.uint8)
        >>> top = np.array([[0, 10], [100, 5], [199, 12]])
        >>> bottom = np.array([[0, 290], [100, 295], [199, 288]])
        >>> dewarper = CylindricalSurfaceDewarper.from_directrices(top, bottom)
        >>>
        >>> # Dewarp the image
        >>> result = dewarp_image(
        ...     image, dewarper,
        ...     dst_size=(200, 300),
        ...     model_domain=(0, 0, 1, 1)
        ... )
    """
    dst_width, dst_height = dst_size
    left, top, right, bottom = model_domain

    if right <= left or bottom <= top:
        raise ValueError("model_domain must have positive width and height")

    # Compute the mapping grid
    map_x, map_y = _compute_dewarp_maps(
        dewarper, dst_width, dst_height, left, top, right, bottom
    )

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


def _compute_dewarp_maps(
    dewarper: CylindricalSurfaceDewarper,
    dst_width: int,
    dst_height: int,
    left: float,
    top: float,
    right: float,
    bottom: float,
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

    Returns:
        Tuple of (map_x, map_y) arrays for cv2.remap.
    """
    # Scale factors from destination pixels to normalized model coordinates
    x_scale = (right - left) / dst_width
    y_scale = (bottom - top) / dst_height

    # Initialize output arrays
    map_x = np.zeros((dst_height, dst_width), dtype=np.float32)
    map_y = np.zeros((dst_height, dst_width), dtype=np.float32)

    # Pre-compute generatrices for each column (more efficient than per-pixel)
    for dst_x in range(dst_width):
        # Convert destination x to normalized model x
        model_x = left + (dst_x + 0.5) * x_scale

        # Get the generatrix for this x coordinate
        generatrix = dewarper.map_generatrix(model_x)
        img_p1, img_p2 = generatrix.img_line
        H = generatrix.pln2img_1d

        for dst_y in range(dst_height):
            # Convert destination y to normalized model y
            model_y = top + (dst_y + 0.5) * y_scale

            # Apply 1D homography to get image y parameter
            num = H[0, 0] * model_y + H[0, 1]
            denom = H[1, 0] * model_y + H[1, 1]
            if abs(denom) < 1e-10:
                img_t = 0.5
            else:
                img_t = num / denom

            # Interpolate along the generatrix line
            src_x = img_p1[0] + img_t * (img_p2[0] - img_p1[0])
            src_y = img_p1[1] + img_t * (img_p2[1] - img_p1[1])

            map_x[dst_y, dst_x] = src_x
            map_y[dst_y, dst_x] = src_y

    return map_x, map_y


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
