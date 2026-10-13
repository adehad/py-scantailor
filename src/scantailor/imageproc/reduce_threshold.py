"""Halve a black-and-white image, keeping a block black by a threshold on its pixels."""

import numpy as np
from numpy.typing import NDArray

_MAX_THRESHOLD = 4


def reduce_threshold(image: NDArray[np.bool_], threshold: int) -> NDArray[np.bool_]:
    """Downscale by 2 in both directions.

    An output pixel is black when at least ``threshold`` of the four pixels of
    its aligned 2x2 block are. A trailing odd row or column is dropped. A single
    row or column reduces along its length only, where thresholds 1 and 2 mean
    "either" and 3 and 4 mean "both".

    Args:
        image (NDArray[np.bool_]): (H, W) image, True for black.
        threshold (int): Black pixels needed in a block, from 1 to 4.

    Returns:
        NDArray[np.bool_]: New image of size (H // 2, W // 2), or the input
        unchanged when it is a single pixel.

    Raises:
        ValueError: If ``threshold`` is outside 1..4.
    """
    if not 1 <= threshold <= _MAX_THRESHOLD:
        raise ValueError(f"threshold must be 1..4, got {threshold}")
    height, width = image.shape
    half_h, half_w = height // 2, width // 2

    if half_h == 0 or half_w == 0:
        if height == 1 and width == 1:
            return image
        either = threshold <= 2
        if half_h == 0:
            pairs = image[:, : half_w * 2].reshape(1, half_w, 2)
            return np.asarray(
                pairs.any(axis=2) if either else pairs.all(axis=2), dtype=np.bool_
            )
        pairs = image[: half_h * 2].reshape(half_h, 2, 1)
        return np.asarray(
            pairs.any(axis=1) if either else pairs.all(axis=1), dtype=np.bool_
        )

    blocks = image[: half_h * 2, : half_w * 2].reshape(half_h, 2, half_w, 2)
    return blocks.sum(axis=(1, 3), dtype=np.uint8) >= threshold
