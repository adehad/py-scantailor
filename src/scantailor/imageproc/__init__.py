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
    binarize_bradley,
    binarize_edge_div,
    binarize_mokji,
    binarize_otsu,
    binarize_peak,
    binarize_sauvola,
    binarize_wolf,
    peak_threshold,
)
from scantailor.imageproc.geometry import (
    rotate_orthogonal,
    scale,
)
from scantailor.imageproc.morphology import (
    black_top_hat,
    close_morph,
    dilate,
    erode,
    hit_miss,
    open_morph,
    remove_small_components,
    white_top_hat,
)

__all__ = [
    # Analysis
    SkewResult.__name__,
    connected_components.__name__,
    distance_transform.__name__,
    find_skew.__name__,
    # Binarization
    binarize_bradley.__name__,
    binarize_edge_div.__name__,
    binarize_mokji.__name__,
    binarize_otsu.__name__,
    binarize_peak.__name__,
    binarize_sauvola.__name__,
    binarize_wolf.__name__,
    peak_threshold.__name__,
    # Geometry
    rotate_orthogonal.__name__,
    scale.__name__,
    # Morphology
    black_top_hat.__name__,
    close_morph.__name__,
    dilate.__name__,
    erode.__name__,
    hit_miss.__name__,
    open_morph.__name__,
    remove_small_components.__name__,
    white_top_hat.__name__,
]
