"""Image processing utilities for ScanTailor.

This module provides image processing functions using OpenCV, NumPy,
scikit-image, and SciPy. Functions operate directly on numpy arrays
without wrapper classes.

Image Type Conventions:
    - BinaryImage: uint8 array with values 0 (black) or 255 (white)
    - GrayImage: uint8 array with grayscale values 0-255
    - ColorImage: uint8 array with shape (H, W, 3) for BGR color images
    - FloatImage: floating-point array for intermediate computations

See image_types module for type aliases and conversion utilities.
"""
