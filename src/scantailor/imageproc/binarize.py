"""Binarization algorithms for converting grayscale images to binary.

All functions take grayscale images as numpy arrays and return binary images.
Binary images use 0 for black (background) and 255 for white (foreground).
"""

import cv2
import numpy as np
from numpy.typing import NDArray
from skimage.filters import threshold_sauvola


def peak_threshold(image: NDArray[np.uint8]) -> int:
    """Find a threshold value using peak detection in the histogram.

    This algorithm finds the dominant peaks from both ends of the histogram,
    then sets the threshold at 75% between the left and right peaks.
    Good for documents where foreground and background are clearly separated.

    Args:
        image: Grayscale input image.

    Returns:
        Threshold value (0-255).
    """
    # Compute histogram
    hist = cv2.calcHist([image], [0], None, [256], [0, 256]).flatten().astype(int)

    # Find right peak (starting from bright end)
    ri = 255
    right_peak = hist[ri]
    for i in range(254, -1, -1):
        if hist[i] <= right_peak:
            if float(hist[i]) < float(right_peak) * 0.66:
                break
            continue
        if hist[i] > right_peak:
            right_peak = hist[i]
            ri = i

    # Find left peak (starting from dark end)
    li = 0
    left_peak = hist[li]
    for i in range(1, 256):
        if hist[i] <= left_peak:
            if float(hist[i]) < float(left_peak) * 0.66:
                break
            continue
        if hist[i] > left_peak:
            left_peak = hist[i]
            li = i

    # Set threshold at 75% between peaks
    threshold = int(li + (ri - li) * 0.75)
    return threshold


def binarize_peak(image: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Binarize an image using peak threshold detection.

    Args:
        image: Grayscale input image.

    Returns:
        Binary image (0 or 255).
    """
    threshold = peak_threshold(image)
    _, binary = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
    return np.asarray(binary, dtype=np.uint8)


def binarize_mokji(
    image: NDArray[np.uint8],
    max_edge_width: int = 3,
    min_edge_magnitude: int = 20,
) -> NDArray[np.uint8]:
    """Binarize an image using Mokji's edge-based thresholding.

    Mokji's method computes a threshold based on edge transitions. It dilates
    the image to find the darkest neighbor within a window, then builds a
    histogram of (pixel, darkest_neighbor) pairs. The threshold is computed
    as the average midpoint of pairs with sufficient edge magnitude.

    Args:
        image: Grayscale input image.
        max_edge_width: Maximum edge width to consider (pixels).
        min_edge_magnitude: Minimum brightness difference to consider as an edge.

    Returns:
        Binary image (0 or 255).
    """
    if max_edge_width < 1:
        raise ValueError("max_edge_width must be >= 1")
    if min_edge_magnitude < 1:
        raise ValueError("min_edge_magnitude must be >= 1")

    # Dilate the image (max filter) to find brightest neighbor
    # Note: C++ code uses dilateGray which finds MINIMUM (darkest),
    # but calls it "dilated". We need the minimum/erosion for correct behavior.
    dilate_size = (max_edge_width + 1) * 2 - 1
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (dilate_size, dilate_size))
    # Use erode to find minimum (darkest neighbor in each window)
    eroded = cv2.erode(image, kernel)

    # Crop to avoid boundary effects
    h, w = image.shape
    y_start, y_end = max_edge_width, h - max_edge_width
    x_start, x_end = max_edge_width, w - max_edge_width

    if y_end <= y_start or x_end <= x_start:
        # Image too small, use default threshold
        _, binary = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY)
        return np.asarray(binary, dtype=np.uint8)

    src_region = image[y_start:y_end, x_start:x_end]
    eroded_region = eroded[y_start:y_end, x_start:x_end]

    # Build matrix of (darkest_neighbor, pixel) pairs
    # Use vectorized operations for efficiency
    pixels = src_region.flatten().astype(np.int32)
    darkest = eroded_region.flatten().astype(np.int32)

    # Only consider pixels where there's sufficient edge magnitude
    edge_magnitude = pixels - darkest
    valid_mask = edge_magnitude >= min_edge_magnitude

    if not np.any(valid_mask):
        # No edges found, use default threshold
        _, binary = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY)
        return np.asarray(binary, dtype=np.uint8)

    # Compute threshold as average of midpoints weighted by occurrence
    valid_pixels = pixels[valid_mask]
    valid_darkest = darkest[valid_mask]
    midpoints = (valid_pixels + valid_darkest) / 2.0
    threshold = int(np.mean(midpoints) + 0.5)

    _, binary = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
    return np.asarray(binary, dtype=np.uint8)


def binarize_otsu(image: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Binarize an image using Otsu's method.

    Otsu's method automatically determines the optimal threshold by minimizing
    intra-class variance. Works well for images with bimodal histograms.

    Args:
        image (NDArray[np.uint8]): Grayscale input image.

    Returns:
        NDArray[np.uint8]: Binary image (0 or 255).
    """
    _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return np.asarray(binary, dtype=np.uint8)


def binarize_sauvola(
    image: NDArray[np.uint8],
    window_size: int = 25,
    k: float = 0.2,
) -> NDArray[np.uint8]:
    """Binarize an image using Sauvola's adaptive thresholding.

    Sauvola's method computes a local threshold based on the mean and standard
    deviation within a window. Good for documents with uneven illumination.

    Args:
        image (NDArray[np.uint8]): Grayscale input image.
        window_size (int): Size of the local window (must be odd).
        k (float): Sensitivity parameter (typically 0.1-0.5).

    Returns:
        NDArray[np.uint8]: Binary image (0 or 255).
    """
    # Ensure window_size is odd
    if window_size % 2 == 0:
        window_size += 1

    threshold = threshold_sauvola(image, window_size=window_size, k=k)
    binary = (image > threshold).astype(np.uint8) * 255
    return binary


def binarize_wolf(
    image: NDArray[np.uint8],
    window_size: int = 25,
    k: float = 0.5,
) -> NDArray[np.uint8]:
    """Binarize an image using Wolf's adaptive thresholding.

    Wolf's method is a modification of Sauvola's that uses the minimum
    and maximum grayscale values in addition to mean and standard deviation.
    It's more robust for low-contrast images.

    Args:
        image (NDArray[np.uint8]): Grayscale input image.
        window_size (int): Size of the local window (must be odd).
        k (float): Sensitivity parameter (typically 0.3-0.5).

    Returns:
        NDArray[np.uint8]: Binary image (0 or 255).
    """
    # Ensure window_size is odd
    if window_size % 2 == 0:
        window_size += 1

    # Compute local mean using box filter
    mean = cv2.blur(image.astype(np.float64), (window_size, window_size))

    # Compute local standard deviation
    sq_mean = cv2.blur((image.astype(np.float64)) ** 2, (window_size, window_size))
    std = np.sqrt(np.maximum(sq_mean - mean**2, 0))

    # Get min value of image
    min_val = float(np.min(image))

    # Get max standard deviation
    max_std = float(np.max(std))
    if max_std < 1e-6:
        max_std = 1.0

    # Wolf's formula: T = (1-k)*mean + k*min + k*(std/max_std)*(mean - min)
    threshold = (1 - k) * mean + k * min_val + k * (std / max_std) * (mean - min_val)

    binary = (image > threshold).astype(np.uint8) * 255
    return binary


def binarize_edge_div(
    image: NDArray[np.uint8],
    window_size: int = 25,
    kep: float = 0.5,
    kbd: float = 0.5,
) -> NDArray[np.uint8]:
    """Binarize an image using edge division preprocessing with Otsu.

    This method enhances edges by dividing each pixel by its local mean,
    then applies Otsu thresholding. The result is a combination of:
    - EdgePlus: Enhances pixels that differ from local mean
    - BlurDiv: Normalizes by the local mean to handle uneven illumination

    Args:
        image: Grayscale input image.
        window_size: Size of the local window (must be odd).
        kep: Edge enhancement weight (0-1). Higher = more edge enhancement.
        kbd: Blur division weight (0-1). Higher = more illumination normalization.

    Returns:
        Binary image (0 or 255).
    """
    # Ensure window_size is odd
    if window_size % 2 == 0:
        window_size += 1

    img_float = image.astype(np.float64)

    # Compute local mean using box filter
    local_mean = cv2.blur(img_float, (window_size, window_size))

    # Start with original values
    result = img_float.copy()

    if kep > 0.0:
        # EdgePlus transformation
        # edge = I / blur (shift = -0.5), mean value = 0.5
        edge = (result + 1) / (local_mean + 1) - 0.5
        # edgeplus = I * edge, mean value = 0.5 * mean(I)
        edgeplus = img_float * edge
        # Blend: k * edgeplus + (1 - k) * I
        result = kep * edgeplus + (1.0 - kep) * img_float

    if kbd > 0.0:
        # BlurDiv transformation
        # edgeinv = blur / I (shift = -0.5)
        edgeinv = (local_mean + 1) / (result + 1) - 0.5
        # edgenorm = edge * k + 1 * (1 - k)
        edgenorm = kbd * edgeinv + (1.0 - kbd)
        # Normalize: I / edgenorm
        result = np.where(edgenorm > 0, img_float / edgenorm, img_float)

    # Clip to valid range and convert back to uint8
    result = np.clip(result, 0, 255).astype(np.uint8)

    # Apply Otsu thresholding
    return binarize_otsu(result)


def binarize_bradley(
    image: NDArray[np.uint8],
    window_size: int = 25,
    k: float = 0.15,
) -> NDArray[np.uint8]:
    """Binarize an image using Bradley's adaptive thresholding.

    Bradley's method computes a local threshold based only on the mean
    within a window. A pixel is set to black if it's sufficiently darker
    than the local mean. Simple and fast, works well for clean documents.

    The threshold at each pixel is: mean * (1 - k)
    Pixels below this threshold are classified as foreground (black).

    Args:
        image: Grayscale input image.
        window_size: Size of the local window (must be odd).
        k: Sensitivity parameter (0-1). Higher values make it
            easier for pixels to be classified as foreground (black).
            Typically 0.1-0.2.

    Returns:
        Binary image (0 or 255).
    """
    # Ensure window_size is odd
    if window_size % 2 == 0:
        window_size += 1

    # Compute local mean using box filter (efficient via integral image internally)
    local_mean = cv2.blur(image.astype(np.float64), (window_size, window_size))

    # Bradley threshold: pixel < mean * (1 - k) -> black
    # For k >= 1, threshold is 0 (everything white except pure black)
    if k >= 1.0:
        threshold = np.zeros_like(local_mean)
    else:
        threshold = local_mean * (1.0 - k)

    binary = (image >= threshold).astype(np.uint8) * 255
    return binary
