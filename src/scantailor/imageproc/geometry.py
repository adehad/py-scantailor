"""Quarter-turn rotation of images."""

import cv2
import numpy as np
from numpy.typing import NDArray

from scantailor.core.models import OrthogonalDegrees


def rotate_orthogonal(
    image: NDArray[np.uint8],
    degrees: OrthogonalDegrees,
) -> NDArray[np.uint8]:
    """Rotate an image by 0, 90, 180, or 270 degrees clockwise.

    Args:
        image (NDArray[np.uint8]): Input image.
        degrees (OrthogonalDegrees): Rotation angle (0, 90, 180, or 270).

    Returns:
        NDArray[np.uint8]: Rotated image.
    """
    if degrees == 0:
        return image.copy()
    if degrees == 90:
        return np.asarray(cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE), dtype=np.uint8)
    if degrees == 180:
        return np.asarray(cv2.rotate(image, cv2.ROTATE_180), dtype=np.uint8)
    # 270 degrees
    return np.asarray(cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE), dtype=np.uint8)
