"""Mathematical functions for optimization and geometric operations.

This module provides classes for linear and quadratic functions
that can be used in optimization algorithms and geometric computations.
"""

from dataclasses import dataclass
from typing import Self

import numpy as np
from numpy.typing import NDArray


@dataclass
class LinearFunction:
    """A linear function from N variables: F(x) = a^T * x + b.

    This represents a general linear function where:
    - a: Vector of N coefficients
    - b: Constant term
    - F(x) = a[0]*x[0] + a[1]*x[1] + ... + a[N-1]*x[N-1] + b

    Attributes:
        a: Coefficient vector (NumPy array).
        b: Constant term.
    """

    a: NDArray[np.float64]
    b: float = 0.0

    def __post_init__(self) -> None:
        """Convert a to numpy array if needed."""
        self.a = np.asarray(self.a, dtype=np.float64)

    @classmethod
    def zeros(cls, num_vars: int) -> Self:
        """Create a zero linear function with the given number of variables.

        Args:
            num_vars: Number of variables.

        Returns:
            LinearFunction with all coefficients set to zero.
        """
        return cls(a=np.zeros(num_vars, dtype=np.float64), b=0.0)

    @property
    def num_vars(self) -> int:
        """Return the number of variables."""
        return len(self.a)

    def evaluate(self, x: NDArray[np.float64] | list[float]) -> float:
        """Evaluate a^T * x + b.

        Args:
            x: Vector of variable values.

        Returns:
            Function value at x.
        """
        x = np.asarray(x, dtype=np.float64)
        return float(np.dot(self.a, x) + self.b)

    def reset(self) -> None:
        """Reset all coefficients to zero."""
        self.a[:] = 0.0
        self.b = 0.0

    def __iadd__(self, other: Self) -> Self:
        """Add another linear function in place."""
        if self.num_vars != other.num_vars:
            msg = "Cannot add functions with different number of variables"
            raise ValueError(msg)
        self.a += other.a
        self.b += other.b
        return self

    def __add__(self, other: Self) -> Self:
        """Add two linear functions."""
        if self.num_vars != other.num_vars:
            msg = "Cannot add functions with different number of variables"
            raise ValueError(msg)
        return LinearFunction(a=self.a + other.a, b=self.b + other.b)

    def __imul__(self, scalar: float) -> Self:
        """Multiply by a scalar in place."""
        self.a *= scalar
        self.b *= scalar
        return self

    def __mul__(self, scalar: float) -> Self:
        """Multiply by a scalar."""
        return LinearFunction(a=self.a * scalar, b=self.b * scalar)

    def __rmul__(self, scalar: float) -> Self:
        """Right multiply by a scalar."""
        return self * scalar


@dataclass
class QuadraticGradient:
    """Gradient of a quadratic function: nabla F(x) = A * x + b.

    Attributes:
        A: Coefficient matrix (N x N).
        b: Constant vector (N).
    """

    A: NDArray[np.float64]
    b: NDArray[np.float64]

    def evaluate(self, x: NDArray[np.float64] | list[float]) -> NDArray[np.float64]:
        """Evaluate A * x + b.

        Args:
            x: Vector of variable values.

        Returns:
            Gradient vector at x.
        """
        x = np.asarray(x, dtype=np.float64)
        return np.dot(self.A, x) + self.b


@dataclass
class QuadraticFunction:
    """A quadratic function from N variables: F(x) = x^T * A * x + b^T * x + c.

    This represents a general quadratic function where:
    - A: N x N matrix of quadratic coefficients
    - b: Vector of N linear coefficients
    - c: Constant term

    Attributes:
        A: Quadratic coefficient matrix (N x N NumPy array).
        b: Linear coefficient vector (N NumPy array).
        c: Constant term.
    """

    A: NDArray[np.float64]
    b: NDArray[np.float64]
    c: float = 0.0

    def __post_init__(self) -> None:
        """Convert arrays to numpy if needed."""
        self.A = np.asarray(self.A, dtype=np.float64)
        self.b = np.asarray(self.b, dtype=np.float64)

    @classmethod
    def zeros(cls, num_vars: int) -> Self:
        """Create a zero quadratic function with the given number of variables.

        Args:
            num_vars: Number of variables.

        Returns:
            QuadraticFunction with all coefficients set to zero.
        """
        return cls(
            A=np.zeros((num_vars, num_vars), dtype=np.float64),
            b=np.zeros(num_vars, dtype=np.float64),
            c=0.0,
        )

    @property
    def num_vars(self) -> int:
        """Return the number of variables."""
        return len(self.b)

    def evaluate(self, x: NDArray[np.float64] | list[float]) -> float:
        """Evaluate x^T * A * x + b^T * x + c.

        Args:
            x: Vector of variable values.

        Returns:
            Function value at x.
        """
        x = np.asarray(x, dtype=np.float64)
        return float(np.dot(x, np.dot(self.A, x)) + np.dot(self.b, x) + self.c)

    def gradient(self) -> QuadraticGradient:
        """Compute the gradient function.

        The gradient of F(x) = x^T * A * x + b^T * x + c is:
        nabla F(x) = (A + A^T) * x + b

        Returns:
            QuadraticGradient representing the gradient function.
        """
        return QuadraticGradient(A=self.A + self.A.T, b=self.b.copy())

    def gradient_at(self, x: NDArray[np.float64] | list[float]) -> NDArray[np.float64]:
        """Evaluate the gradient at a specific point.

        Args:
            x: Vector of variable values.

        Returns:
            Gradient vector at x.
        """
        x = np.asarray(x, dtype=np.float64)
        return np.dot(self.A + self.A.T, x) + self.b

    def reset(self) -> None:
        """Reset all coefficients to zero."""
        self.A[:] = 0.0
        self.b[:] = 0.0
        self.c = 0.0

    def recalc_for_translated_arguments(
        self, translation: NDArray[np.float64] | list[float]
    ) -> None:
        """Recalculate so that g(x) = f(x + translation).

        After this operation, evaluating the function at x will give
        the same result as the original function evaluated at x + translation.

        Args:
            translation: Translation vector.
        """
        t = np.asarray(translation, dtype=np.float64)

        # g(x) = f(x + t) = (x+t)^T * A * (x+t) + b^T * (x+t) + c
        #      = x^T*A*x + x^T*A*t + t^T*A*x + t^T*A*t + b^T*x + b^T*t + c
        #      = x^T*A*x + (A + A^T)^T*t * x + (b^T*x) + (t^T*A*t + b^T*t + c)
        #      = x^T*A*x + ((A + A^T)*t + b)^T * x + (t^T*A*t + b^T*t + c)

        # New constant: t^T * A * t + b^T * t + c
        new_c = float(np.dot(t, np.dot(self.A, t)) + np.dot(self.b, t) + self.c)

        # New linear coefficients: (A + A^T) * t + b
        new_b = np.dot(self.A + self.A.T, t) + self.b

        # A matrix stays the same
        self.b = new_b
        self.c = new_c

    def __iadd__(self, other: Self) -> Self:
        """Add another quadratic function in place."""
        if self.num_vars != other.num_vars:
            msg = "Cannot add functions with different number of variables"
            raise ValueError(msg)
        self.A += other.A
        self.b += other.b
        self.c += other.c
        return self

    def __add__(self, other: Self) -> Self:
        """Add two quadratic functions."""
        if self.num_vars != other.num_vars:
            msg = "Cannot add functions with different number of variables"
            raise ValueError(msg)
        return QuadraticFunction(
            A=self.A + other.A, b=self.b + other.b, c=self.c + other.c
        )

    def __imul__(self, scalar: float) -> Self:
        """Multiply by a scalar in place."""
        self.A *= scalar
        self.b *= scalar
        self.c *= scalar
        return self

    def __mul__(self, scalar: float) -> Self:
        """Multiply by a scalar."""
        return QuadraticFunction(
            A=self.A * scalar, b=self.b * scalar, c=self.c * scalar
        )

    def __rmul__(self, scalar: float) -> Self:
        """Right multiply by a scalar."""
        return self * scalar
