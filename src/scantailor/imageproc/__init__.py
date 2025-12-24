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
    find_contours,
    find_skew,
    hough_lines,
    hough_lines_p,
    max_whitespace_rect,
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
from scantailor.imageproc.filters import (
    gaussian_blur,
    savgol_filter_2d,
    sobel,
    sobel_magnitude,
    wiener_filter,
)
from scantailor.imageproc.geometry import (
    rotate,
    rotate_orthogonal,
    scale,
    shear,
    transform_affine,
    transform_perspective,
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
from scantailor.imageproc.utils import (
    blend,
    bounding_rect,
    color_interpolate,
    contour_area,
    contour_perimeter,
    convex_hull,
    draw_over,
    draw_polygon,
    fill_polygon,
    min_area_rect,
    to_color,
    to_grayscale,
)

__all__ = [
    # Analysis
    SkewResult.__name__,
    connected_components.__name__,
    distance_transform.__name__,
    find_contours.__name__,
    find_skew.__name__,
    hough_lines.__name__,
    hough_lines_p.__name__,
    max_whitespace_rect.__name__,
    # Binarization
    binarize_bradley.__name__,
    binarize_edge_div.__name__,
    binarize_mokji.__name__,
    binarize_otsu.__name__,
    binarize_peak.__name__,
    binarize_sauvola.__name__,
    binarize_wolf.__name__,
    peak_threshold.__name__,
    # Filters
    gaussian_blur.__name__,
    savgol_filter_2d.__name__,
    sobel.__name__,
    sobel_magnitude.__name__,
    wiener_filter.__name__,
    # Geometry
    rotate.__name__,
    rotate_orthogonal.__name__,
    scale.__name__,
    shear.__name__,
    transform_affine.__name__,
    transform_perspective.__name__,
    # Morphology
    black_top_hat.__name__,
    close_morph.__name__,
    dilate.__name__,
    erode.__name__,
    hit_miss.__name__,
    open_morph.__name__,
    remove_small_components.__name__,
    white_top_hat.__name__,
    # Utilities
    blend.__name__,
    bounding_rect.__name__,
    color_interpolate.__name__,
    contour_area.__name__,
    contour_perimeter.__name__,
    convex_hull.__name__,
    draw_over.__name__,
    draw_polygon.__name__,
    fill_polygon.__name__,
    min_area_rect.__name__,
    to_color.__name__,
    to_grayscale.__name__,
]
