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
    ) -> "PolynomialSurface":
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
        # Row-major over (y power, x power): matches coeffs.reshape(-1).
        design = (vert_basis[:, None, :, None] * hor_basis[None, :, None, :]).reshape(
            height * width, -1
        )
        target = image.reshape(-1).astype(np.float64) / 255.0
        if mask is not None:
            selected = mask.reshape(-1)
            design = design[selected]
            target = target[selected]

        normal = design.T @ design + _RIDGE * np.eye(design.shape[1])
        solution = np.linalg.solve(normal, design.T @ target)
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
