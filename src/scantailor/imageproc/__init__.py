"""Image processing utilities for ScanTailor.

This module provides image processing functions using OpenCV, NumPy,
scikit-image, and SciPy. Functions operate directly on numpy arrays
without wrapper classes.
"""

from __future__ import annotations

from scantailor.imageproc.analysis import (
    SkewResult,
    connected_components,
    distance_transform,
    find_skew,
)
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
    # Analysis
    SkewResult.__name__,
    connected_components.__name__,
    distance_transform.__name__,
    find_skew.__name__,
    # Binarization
    binarize_otsu.__name__,
    binarize_sauvola.__name__,
    # Geometry
    rotate_orthogonal.__name__,
    scale.__name__,
    # Morphology
    close_morph.__name__,
    dilate.__name__,
    erode.__name__,
    open_morph.__name__,
]
