"""Tests for math functions module."""

import numpy as np
import pytest

from scantailor.math import LinearFunction, QuadraticFunction, QuadraticGradient


class TestLinearFunction:
    """Tests for LinearFunction class."""

    def test_zeros_construction(self) -> None:
        """Test creating zero function."""
        f = LinearFunction.zeros(3)
        assert f.num_vars == 3
        assert np.allclose(f.a, [0, 0, 0])
        assert f.b == 0.0

    def test_evaluate_zero(self) -> None:
        """Test evaluating zero function."""
        f = LinearFunction.zeros(2)
        assert f.evaluate([1.0, 2.0]) == 0.0

    def test_evaluate_simple(self) -> None:
        """Test simple function evaluation."""
        # f(x) = 2*x1 + 3*x2 + 1
        f = LinearFunction(a=np.array([2.0, 3.0]), b=1.0)
        assert f.evaluate([1.0, 1.0]) == 6.0  # 2 + 3 + 1
        assert f.evaluate([0.0, 0.0]) == 1.0  # just the constant
        assert f.evaluate([2.0, 3.0]) == 14.0  # 4 + 9 + 1

    def test_reset(self) -> None:
        """Test resetting function to zero."""
        f = LinearFunction(a=np.array([1.0, 2.0, 3.0]), b=4.0)
        f.reset()
        assert np.allclose(f.a, [0, 0, 0])
        assert f.b == 0.0

    def test_addition(self) -> None:
        """Test adding two functions."""
        f1 = LinearFunction(a=np.array([1.0, 2.0]), b=3.0)
        f2 = LinearFunction(a=np.array([4.0, 5.0]), b=6.0)
        result = f1 + f2
        assert np.allclose(result.a, [5.0, 7.0])
        assert result.b == 9.0

    def test_inplace_addition(self) -> None:
        """Test in-place addition."""
        f1 = LinearFunction(a=np.array([1.0, 2.0]), b=3.0)
        f2 = LinearFunction(a=np.array([4.0, 5.0]), b=6.0)
        f1 += f2
        assert np.allclose(f1.a, [5.0, 7.0])
        assert f1.b == 9.0

    def test_scalar_multiplication(self) -> None:
        """Test multiplying by scalar."""
        f = LinearFunction(a=np.array([1.0, 2.0]), b=3.0)
        result = f * 2.0
        assert np.allclose(result.a, [2.0, 4.0])
        assert result.b == 6.0

    def test_right_scalar_multiplication(self) -> None:
        """Test right multiplying by scalar."""
        f = LinearFunction(a=np.array([1.0, 2.0]), b=3.0)
        result = 2.0 * f
        assert np.allclose(result.a, [2.0, 4.0])
        assert result.b == 6.0

    def test_inplace_scalar_multiplication(self) -> None:
        """Test in-place scalar multiplication."""
        f = LinearFunction(a=np.array([1.0, 2.0]), b=3.0)
        f *= 2.0
        assert np.allclose(f.a, [2.0, 4.0])
        assert f.b == 6.0

    def test_mismatched_vars_raises(self) -> None:
        """Test adding functions with different variable counts."""
        f1 = LinearFunction(a=np.array([1.0, 2.0]), b=1.0)
        f2 = LinearFunction(a=np.array([1.0, 2.0, 3.0]), b=1.0)
        with pytest.raises(ValueError, match="different number of variables"):
            _ = f1 + f2


class TestQuadraticGradient:
    """Tests for QuadraticGradient class."""

    def test_evaluate(self) -> None:
        """Test evaluating gradient."""
        # nabla F(x) = A*x + b
        A = np.array([[2.0, 0.0], [0.0, 2.0]])
        b = np.array([1.0, 1.0])
        grad = QuadraticGradient(A=A, b=b)
        result = grad.evaluate([1.0, 2.0])
        # [2, 0; 0, 2] * [1; 2] + [1; 1] = [2; 4] + [1; 1] = [3; 5]
        assert np.allclose(result, [3.0, 5.0])


class TestQuadraticFunction:
    """Tests for QuadraticFunction class."""

    def test_zeros_construction(self) -> None:
        """Test creating zero function."""
        f = QuadraticFunction.zeros(3)
        assert f.num_vars == 3
        assert f.A.shape == (3, 3)
        assert np.allclose(f.A, 0)
        assert np.allclose(f.b, 0)
        assert f.c == 0.0

    def test_evaluate_constant(self) -> None:
        """Test evaluating constant function."""
        f = QuadraticFunction.zeros(2)
        f.c = 5.0
        assert f.evaluate([1.0, 2.0]) == 5.0

    def test_evaluate_linear(self) -> None:
        """Test evaluating with linear terms only."""
        f = QuadraticFunction.zeros(2)
        f.b = np.array([2.0, 3.0])
        # f(x) = 2*x1 + 3*x2
        assert f.evaluate([1.0, 1.0]) == 5.0

    def test_evaluate_quadratic(self) -> None:
        """Test evaluating full quadratic function."""
        # f(x) = x1^2 + x2^2
        f = QuadraticFunction(
            A=np.array([[1.0, 0.0], [0.0, 1.0]]),
            b=np.array([0.0, 0.0]),
            c=0.0,
        )
        assert f.evaluate([1.0, 1.0]) == 2.0
        assert f.evaluate([2.0, 3.0]) == 13.0

    def test_evaluate_mixed(self) -> None:
        """Test quadratic with cross term."""
        # f(x) = x1*x2
        f = QuadraticFunction(
            A=np.array([[0.0, 0.5], [0.5, 0.0]]),  # symmetric for x1*x2
            b=np.array([0.0, 0.0]),
            c=0.0,
        )
        assert f.evaluate([2.0, 3.0]) == 6.0

    def test_reset(self) -> None:
        """Test resetting function to zero."""
        f = QuadraticFunction(
            A=np.array([[1.0, 2.0], [3.0, 4.0]]),
            b=np.array([5.0, 6.0]),
            c=7.0,
        )
        f.reset()
        assert np.allclose(f.A, 0)
        assert np.allclose(f.b, 0)
        assert f.c == 0.0

    def test_gradient(self) -> None:
        """Test computing gradient function."""
        # f(x) = x1^2 + x2^2 + x1 + 2*x2 + 3
        # nabla f = [2*x1 + 1, 2*x2 + 2]
        f = QuadraticFunction(
            A=np.array([[1.0, 0.0], [0.0, 1.0]]),
            b=np.array([1.0, 2.0]),
            c=3.0,
        )
        grad = f.gradient()
        # A + A^T = 2*I
        assert np.allclose(grad.A, [[2.0, 0.0], [0.0, 2.0]])
        assert np.allclose(grad.b, [1.0, 2.0])

    def test_gradient_at(self) -> None:
        """Test evaluating gradient at a point."""
        # f(x) = x1^2 + x2^2
        # nabla f(x) = [2*x1, 2*x2]
        f = QuadraticFunction(
            A=np.array([[1.0, 0.0], [0.0, 1.0]]),
            b=np.array([0.0, 0.0]),
            c=0.0,
        )
        grad = f.gradient_at([3.0, 4.0])
        assert np.allclose(grad, [6.0, 8.0])

    def test_translation(self) -> None:
        """Test recalculating for translated arguments."""
        # f(x) = x1^2 + x2^2
        f = QuadraticFunction(
            A=np.array([[1.0, 0.0], [0.0, 1.0]]),
            b=np.array([0.0, 0.0]),
            c=0.0,
        )
        original_value = f.evaluate([3.0, 4.0])  # = 25

        # After translation by [1, 1], g(x) = f(x + [1, 1])
        # So g([2, 3]) should equal f([3, 4]) = 25
        f.recalc_for_translated_arguments([1.0, 1.0])
        translated_value = f.evaluate([2.0, 3.0])
        assert abs(translated_value - original_value) < 1e-10

    def test_addition(self) -> None:
        """Test adding two functions."""
        f1 = QuadraticFunction(
            A=np.array([[1.0, 0.0], [0.0, 1.0]]),
            b=np.array([1.0, 2.0]),
            c=3.0,
        )
        f2 = QuadraticFunction(
            A=np.array([[2.0, 1.0], [1.0, 2.0]]),
            b=np.array([3.0, 4.0]),
            c=5.0,
        )
        result = f1 + f2
        assert np.allclose(result.A, [[3.0, 1.0], [1.0, 3.0]])
        assert np.allclose(result.b, [4.0, 6.0])
        assert result.c == 8.0

    def test_scalar_multiplication(self) -> None:
        """Test multiplying by scalar."""
        f = QuadraticFunction(
            A=np.array([[1.0, 0.0], [0.0, 1.0]]),
            b=np.array([2.0, 3.0]),
            c=4.0,
        )
        result = f * 2.0
        assert np.allclose(result.A, [[2.0, 0.0], [0.0, 2.0]])
        assert np.allclose(result.b, [4.0, 6.0])
        assert result.c == 8.0

    def test_mismatched_vars_raises(self) -> None:
        """Test adding functions with different variable counts."""
        f1 = QuadraticFunction.zeros(2)
        f2 = QuadraticFunction.zeros(3)
        with pytest.raises(ValueError, match="different number of variables"):
            _ = f1 + f2
