"""Least-squares polynomial fits along the rows or columns of an image."""

from typing import Literal

import numpy as np
from numpy.typing import NDArray


def _vandermonde(num_values: int, degree: int) -> NDArray[np.float64]:
    """Build a Vandermonde matrix over positions normalised to [0, 1].

    Args:
        num_values (int): Number of sample positions.
        degree (int): Polynomial degree.

    Returns:
        NDArray[np.float64]: Array of shape (num_values, degree + 1).
    """
    scale = 1.0 / (num_values - 1) if num_values > 1 else 0.0
    positions = np.arange(num_values, dtype=np.float64) * scale
    return np.vander(positions, degree + 1, increasing=True)


def fit_polynomial_lines(
    data: NDArray[np.uint8],
    degree: int,
    axis: Literal[0, 1],
) -> NDArray[np.uint8]:
    """Replace every line of an image by its least-squares polynomial fit.

    All lines are fitted in one batched solve. Positions are normalised to
    [0, 1] and a 1e-5 ridge term keeps the normal equations solvable when the
    line is shorter than the polynomial has coefficients.

    Args:
        data (NDArray[np.uint8]): 2-D image. Not modified.
        degree (int): Polynomial degree. Silently reduced to ``length - 1`` for
            lines with fewer samples than coefficients.
        axis (Literal[0, 1]): 0 fits each column (along y), 1 fits each row
            (along x).

    Returns:
        NDArray[np.uint8]: Array with the shape of ``data`` holding the rounded
        and clipped fitted values.

    Raises:
        ValueError: If ``degree`` is negative, ``data`` is not 2-D or empty.
    """
    if degree < 0:
        raise ValueError("degree must be non-negative")
    if data.ndim != 2 or data.size == 0:
        raise ValueError("data must be a non-empty 2-D array")

    lines = data.astype(np.float64) if axis == 0 else data.T.astype(np.float64)
    length = lines.shape[0]
    basis = _vandermonde(length, min(degree, length - 1))
    normal = basis.T @ basis + 1e-5 * np.eye(basis.shape[1])
    coeffs = np.linalg.solve(normal, basis.T @ lines)
    fitted = np.clip(np.floor(basis @ coeffs + 0.5), 0, 255).astype(np.uint8)
    return fitted if axis == 0 else np.ascontiguousarray(fitted.T)
