"""Recorded outputs of the dewarping kernels on fixed inputs.

The inputs are stored next to the outputs or built from integer formulas, so
every step is plain IEEE arithmetic that comes out the same on any platform,
and the outputs must match bit for bit. A change of behaviour, however small,
shows up as a mismatch.
"""

import functools
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

_REFERENCE = (
    Path(__file__).parents[1] / "fixtures" / "dewarping" / "kernel_reference.npz"
)


@functools.cache
def _load() -> dict[str, NDArray]:
    with np.load(_REFERENCE) as archive:
        return {name: archive[name] for name in archive.files}


def recorded(name: str) -> NDArray:
    """A copy of one recorded array, safe to change.

    Args:
        name (str): Name of the array in the recording.

    Returns:
        numpy.ndarray: The array.
    """
    return _load()[name].copy()


def formula_grid(
    height: int, width: int, period_y: int, period_x: int
) -> NDArray[np.float32]:
    """A float32 grid of products of two sawtooth waves, exact on any platform.

    Args:
        height (int): Rows.
        width (int): Columns.
        period_y (int): Period of the wave down the rows.
        period_x (int): Period of the wave along the columns.

    Returns:
        numpy.ndarray of numpy.float32: Values that are multiples of 1/64.
    """
    yy, xx = np.mgrid[0:height, 0:width]
    waves = ((yy % period_y) - period_y // 2) * ((xx % period_x) - period_x // 2)
    return waves.astype(np.float32) / np.float32(64.0)
