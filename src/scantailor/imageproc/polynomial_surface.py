"""Two-dimensional polynomial surface fitted to a grayscale image."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

# Added to the diagonal of the normal equations so that rank-deficient fits
# (for example a mask covering a single column) still have a solution.
_RIDGE = 1e-5


def _axis_basis(num_values: int, degree: int) -> NDArray[np.float64]:
    """Return powers of positions in [0, 1], shape (num_values, degree + 1)."""
    scale = 1.0 / (num_values - 1) if num_values > 1 else 0.0
    positions = np.arange(num_values, dtype=np.float64) * scale
    return np.vander(positions, degree + 1, increasing=True)


def _separable_normal_matrix(
    vert_basis: NDArray[np.float64],
    hor_basis: NDArray[np.float64],
    weights: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Return ``D.T @ diag(w) @ D`` for the design matrix ``D`` of a surface fit.

    Row ``y * width + x`` of ``D`` is ``outer(V[y], H[x]).ravel()``, so entry
    ``((i, j), (k, l))`` is ``sum over y, x of V[y,i] V[y,k] w[y,x] H[x,j] H[x,l]``.
    That sum factors into two small matrix products, and ``D`` itself, which
    has one row per pixel, is never built.

    Args:
        vert_basis (NDArray[np.float64]): ``V``, shape (height, p).
        hor_basis (NDArray[np.float64]): ``H``, shape (width, q).
        weights (NDArray[np.float64]): ``w``, shape (height, width).

    Returns:
        NDArray[np.float64]: Shape (p * q, p * q), indexed row-major over
        (y power, x power) like ``PolynomialSurface.coeffs.reshape(-1)``.
    """
    height, p = vert_basis.shape
    width, q = hor_basis.shape
    vert_pairs = (vert_basis[:, :, None] * vert_basis[:, None, :]).reshape(height, -1)
    hor_pairs = (hor_basis[:, :, None] * hor_basis[:, None, :]).reshape(width, -1)
    by_pairs = (vert_pairs.T @ weights @ hor_pairs).reshape(p, p, q, q)
    return by_pairs.transpose(0, 2, 1, 3).reshape(p * q, p * q)


@dataclass(frozen=True)
class PolynomialSurface:
    """Polynomial of two variables over the unit square.

    Pixel coordinates are mapped to [0, 1] on each axis, so a surface fitted on
    one raster size can be rendered at any other size.

    Attributes:
        coeffs: Array of shape (vert_degree + 1, hor_degree + 1);
            ``coeffs[i, j]`` multiplies ``y**i * x**j``. Values are in units of
            gray level divided by 255.
    """

    coeffs: NDArray[np.float64]

    @property
    def vert_degree(self) -> int:
        """Degree along y."""
        return self.coeffs.shape[0] - 1

    @property
    def hor_degree(self) -> int:
        """Degree along x."""
        return self.coeffs.shape[1] - 1

    @classmethod
    def fit(
        cls,
        image: NDArray[np.uint8],
        hor_degree: int,
        vert_degree: int,
        mask: NDArray[np.bool_] | None = None,
    ) -> PolynomialSurface:
        """Least-squares fit of a surface to an image.

        Degrees are reduced until the number of coefficients does not exceed
        the number of data points.

        Args:
            image (NDArray[np.uint8]): 2-D grayscale image. Not modified.
            hor_degree (int): Polynomial degree along x.
            vert_degree (int): Polynomial degree along y.
            mask (NDArray[np.bool_] | None): Same shape as ``image``; True marks
                pixels that take part in the fit. None uses every pixel.

        Returns:
            PolynomialSurface: The fitted surface. A constant zero surface if
            there are no data points.

        Raises:
            ValueError: If a degree is negative, ``image`` is not 2-D or the
                mask shape differs from the image shape.
        """
        if hor_degree < 0 or vert_degree < 0:
            raise ValueError("polynomial degrees must be non-negative")
        if image.ndim != 2:
            raise ValueError("image must be 2-D")
        if mask is not None and mask.shape != image.shape:
            raise ValueError("image and mask have different shapes")

        height, width = image.shape
        num_points = image.size if mask is None else int(np.count_nonzero(mask))
        if num_points == 0:
            return cls(coeffs=np.zeros((1, 1)))

        while num_points < (hor_degree + 1) * (vert_degree + 1):
            if hor_degree > vert_degree:
                hor_degree -= 1
            else:
                vert_degree -= 1

        vert_basis = _axis_basis(height, vert_degree)
        hor_basis = _axis_basis(width, hor_degree)
        weights = (
            np.ones((height, width), np.float64)
            if mask is None
            else mask.astype(np.float64)
        )
        normal = _separable_normal_matrix(vert_basis, hor_basis, weights)
        # In place, so the page holds one float64 array: the weights become
        # the masked target once the normal matrix no longer needs them.
        weighted_target = weights
        weighted_target *= image
        weighted_target /= 255.0
        rhs = (vert_basis.T @ weighted_target @ hor_basis).reshape(-1)
        normal += _RIDGE * np.eye(normal.shape[0])
        solution = np.linalg.solve(normal, rhs)
        grid = np.asarray(solution, dtype=np.float64).reshape(
            vert_degree + 1, hor_degree + 1
        )
        return cls(coeffs=grid)

    def render(self, shape: tuple[int, int]) -> NDArray[np.uint8]:
        """Evaluate the surface on a pixel grid.

        Args:
            shape (tuple[int, int]): Output (height, width).

        Returns:
            NDArray[np.uint8]: Surface values scaled to 0-255, rounded and clipped.
        """
        height, width = shape
        if height <= 0 or width <= 0:
            return np.zeros((max(height, 0), max(width, 0)), dtype=np.uint8)
        vert_basis = _axis_basis(height, self.vert_degree)
        hor_basis = _axis_basis(width, self.hor_degree)
        values = vert_basis @ self.coeffs @ hor_basis.T
        return np.clip(np.floor(values * 255.0 + 0.5), 0, 255).astype(np.uint8)
