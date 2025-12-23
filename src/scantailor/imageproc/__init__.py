"""Image processing utilities for ScanTailor.

This module provides image processing functions using OpenCV, NumPy,
scikit-image, and SciPy. Functions operate directly on numpy arrays
without wrapper classes.
"""

from __future__ import annotations

from scantailor.imageproc.binarize import (
    binarize_otsu,
    binarize_sauvola,
)
from scantailor.imageproc.geometry import (
    rotate_orthogonal,
    scale,
)
from scantailor.imageproc.morphology import (
    close_morph,
    dilate,
    erode,
    open_morph,
)

__all__ = [
    # Binarization
    "binarize_otsu",
    "binarize_sauvola",
    # Geometry
    "rotate_orthogonal",
    "scale",
    # Morphology
    "close_morph",
    "dilate",
    "erode",
    "open_morph",
]
