"""Greyscale morphological reconstruction.

The marker is raised by directional sweeps: each row takes the maximum of
itself and its neighbours in the row before it, capped by the mask, first down
the image and back up, then across it and back on a transposed copy. A row is
only visited again once the row before it has risen since it was last passed
on, so after the first sweeps the work shrinks to the rows where values are
still moving. Every sweep moves values arbitrarily far in its direction, so
natural images settle in a few rounds; only paths that wind back and forth need
more. The marker is final when no row is left to pass on in either orientation.
The result is the unique reconstruction, so it equals
``skimage.morphology.reconstruction`` with a 4 or 8-connected footprint.
"""

from typing import Literal

import cv2
import numpy as np
from numpy.typing import NDArray

Connectivity = Literal[4, 8]
Method = Literal["dilation", "erosion"]

# Element types cv2.transpose accepts; it is about three times faster than a
# NumPy transposed copy at page size.
_CV2_TRANSPOSE_DTYPES = frozenset(
    np.dtype(kind)
    for kind in (np.uint8, np.uint16, np.int16, np.int32, np.float32, np.float64)
)


def reconstruction(
    seed: NDArray,
    mask: NDArray,
    method: Method = "dilation",
    connectivity: Connectivity = 8,
) -> NDArray:
    """Reconstruct a mask from a seed by geodesic dilation or erosion.

    Dilation spreads the seed's bright values as far as the mask allows, so the
    result lies between the seed and the mask (seed <= result <= mask).
    Erosion is the dual: dark values spread and mask <= result <= seed.

    Args:
        seed (NDArray): 2-D integer or float array, the starting values. Not
            modified. Must not exceed ``mask`` for dilation, nor fall below it
            for erosion.
        mask (NDArray): 2-D array of the same shape and dtype as ``seed``; the
            limit the seed can grow or shrink to. Not modified.
        method (Method): ``"dilation"`` or ``"erosion"``.
        connectivity (Connectivity): 4 or 8 neighbours per pixel.

    Returns:
        NDArray: New array with the dtype and shape of ``seed``.

    Raises:
        ValueError: If the arguments are not 2-D arrays of equal shape and
            dtype, if ``method`` or ``connectivity`` is unknown, or if the
            seed is on the wrong side of the mask.
    """
    if method not in ("dilation", "erosion"):
        raise ValueError(f"unknown reconstruction method: {method!r}")
    if connectivity not in (4, 8):
        raise ValueError(f"connectivity must be 4 or 8, got {connectivity!r}")
    if seed.ndim != 2 or seed.shape != mask.shape:
        raise ValueError("seed and mask must be 2-D arrays of the same shape")
    if seed.dtype != mask.dtype:
        raise ValueError("seed and mask must have the same dtype")
    if seed.dtype.kind not in "uif":
        raise ValueError(f"unsupported dtype {seed.dtype}")

    if method == "dilation":
        if (seed > mask).any():
            raise ValueError("seed must not exceed mask for reconstruction by dilation")
        return _dilate(seed, mask, connectivity == 8)
    if (seed < mask).any():
        raise ValueError("seed must not be below mask for reconstruction by erosion")
    # Erosion is dilation of the order-reversed image.
    return _reverse_order(
        _dilate(_reverse_order(seed), _reverse_order(mask), connectivity == 8)
    )


def _reverse_order(image: NDArray) -> NDArray:
    """Map values so that their order is reversed; applying it twice restores them."""
    return -image if image.dtype.kind == "f" else ~image


def _lowest_value(dtype: np.dtype) -> float | int:
    """The value no other value of ``dtype`` is below, so it never wins a maximum."""
    return -np.inf if dtype.kind == "f" else int(np.iinfo(dtype).min)


def _copy_transposed(source: NDArray, target: NDArray) -> None:
    """Write the transpose of ``source`` into ``target``, which may be a view."""
    if source.dtype in _CV2_TRANSPOSE_DTYPES:
        cv2.transpose(source, target)
    else:
        target[...] = source.T


class _Rows:
    """The marker in one orientation, and which rows still have to pass on a rise.

    Attributes:
        mask (NDArray): C-contiguous upper limit, shape (rows, columns).
        framed (NDArray): Marker with a column of the lowest value on each
            side, so the diagonal neighbours of the edge pixels need no special
            case. Shape (rows, columns + 2).
        marker (NDArray): View of ``framed`` without the side columns.
        unspread_down (list[bool]): Per row, whether it has risen since the
            row below was last raised from it.
        unspread_up (list[bool]): Per row, whether it has risen since the row
            above was last raised from it.
    """

    def __init__(self, mask: NDArray, fill: float | int) -> None:
        """Set up an orientation in which every row still has to be passed on.

        Args:
            mask (NDArray): C-contiguous upper limit in this orientation.
            fill (float | int): The lowest value of the dtype.
        """
        rows, columns = mask.shape
        self.mask = mask
        self.framed = np.empty((rows, columns + 2), dtype=mask.dtype)
        self.framed[:, 0] = fill
        self.framed[:, -1] = fill
        self.marker = self.framed[:, 1:-1]
        self.unspread_down = [True] * rows
        self.unspread_up = [True] * rows

    def has_unspread_rows(self) -> bool:
        """Whether any row may still raise its neighbour row."""
        # The last row has no row below it and the first none above.
        return any(self.unspread_down[:-1]) or any(self.unspread_up[1:])

    def mark_unspread(self, rows: NDArray) -> None:
        """Record that ``rows`` rose while the other orientation was swept.

        Args:
            rows (NDArray): Indices of rows of this orientation.
        """
        for row in rows.tolist():
            self.unspread_down[row] = True
            self.unspread_up[row] = True

    def sweep_down_and_up(self, eight_connected: bool) -> NDArray:
        """Raise each row from the row above, top down, then from the row below.

        Args:
            eight_connected (bool): Also take the diagonal neighbours.

        Returns:
            NDArray: Bool per column, whether any pixel in it rose.
        """
        risen_columns = np.zeros(self.marker.shape[1], dtype=bool)
        rows = self.marker.shape[0]
        self._sweep(range(1, rows), -1, eight_connected, risen_columns)
        self._sweep(range(rows - 2, -1, -1), 1, eight_connected, risen_columns)
        return risen_columns

    def _sweep(
        self,
        order: range,
        step: int,
        eight_connected: bool,
        risen_columns: NDArray,
    ) -> None:
        """Raise the rows in ``order`` from their neighbour row ``step`` away.

        Args:
            order (range): Rows to raise, in sweep order.
            step (int): -1 to raise each row from the row above, 1 from below.
            eight_connected (bool): Also take the diagonal neighbours.
            risen_columns (NDArray): Bool per column; set where a pixel rose.
        """
        unspread, unspread_back = (
            (self.unspread_down, self.unspread_up)
            if step == -1
            else (self.unspread_up, self.unspread_down)
        )
        lefts = self.framed[:, :-2]
        rights = self.framed[:, 2:]
        marker = self.marker
        mask = self.mask
        raised = np.empty(marker.shape[1], dtype=marker.dtype)
        rose = np.empty(marker.shape[1], dtype=bool)
        # A Python loop over rows; every step is a vectorised operation on a whole row.
        for row in order:
            source = row + step
            if not unspread[source]:
                continue
            unspread[source] = False
            if eight_connected:
                np.maximum(lefts[source], rights[source], out=raised)
                np.maximum(raised, marker[source], out=raised)
                np.minimum(raised, mask[row], out=raised)
            else:
                np.minimum(marker[source], mask[row], out=raised)
            line = marker[row]
            np.maximum(raised, line, out=raised)
            # Most visited rows do not change, and comparing their bytes is
            # several times cheaper than np.array_equal on one row.
            if raised.tobytes() == line.tobytes():
                continue
            np.not_equal(raised, line, out=rose)
            np.logical_or(risen_columns, rose, out=risen_columns)
            line[...] = raised
            unspread[row] = True
            unspread_back[row] = True


def _dilate(seed: NDArray, mask: NDArray, eight_connected: bool) -> NDArray:
    """Reconstruction by dilation; returns a new array."""
    if seed.size == 0:
        return seed.copy()
    fill = _lowest_value(seed.dtype)
    upright_mask = np.ascontiguousarray(mask)
    lying_mask = np.empty(upright_mask.shape[::-1], dtype=mask.dtype)
    _copy_transposed(upright_mask, lying_mask)
    current = _Rows(upright_mask, fill)
    current.marker[...] = seed
    other = _Rows(lying_mask, fill)
    while True:
        # The columns of one orientation are the rows of the other.
        risen_columns = current.sweep_down_and_up(eight_connected)
        other.mark_unspread(np.flatnonzero(risen_columns))
        if not (current.has_unspread_rows() or other.has_unspread_rows()):
            break
        _copy_transposed(current.marker, other.marker)
        current, other = other, current
    if current.mask is upright_mask:
        return current.marker.copy()
    result = np.empty_like(upright_mask)
    _copy_transposed(current.marker, result)
    return result
