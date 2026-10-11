"""Distortion model for marginal dewarping.

Marginal dewarping corrects only the strip of page next to the binding. The top
and bottom curves follow the content rectangle, except near the spine side
where their control points move out to the page's top and bottom edges.
"""

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import PchipInterpolator

from scantailor.dewarping.distortion_model import Curve, DistortionModel

# Gray levels below this are page background or ink; paper is brighter.
_PAPER_THRESHOLD = 64
# A page edge counts once this many pixels in a row, going inwards, are paper.
_EDGE_RUN = 16
# The content width is split into this many parts; the spine strip is the
# outermost `_SPINE_POINTS` of them.
_WIDTH_PARTS = 29
_SPINE_POINTS = 5
_CURVE_SAMPLES = 30


def build_marginal_model(
    gray: NDArray[np.uint8],
    content_rect: tuple[float, float, float, float],
    spine_on_right: bool,
) -> DistortionModel | None:
    """Build a model whose curves bend toward the page edges near the spine.

    Args:
        gray (NDArray[np.uint8]): 2-D gray page, the coordinates of the model.
        content_rect (tuple[float, float, float, float]): (x, y, width, height)
            of the content.
        spine_on_right (bool): True for a single or left page, whose binding is
            on the right.

    Returns:
        DistortionModel | None: The model, or None if the content rectangle is
            empty or the result is not a valid model.
    """
    x, y, width, height = content_rect
    if width <= 0 or height <= 0:
        return None
    paper = gray >= _PAPER_THRESHOLD

    if spine_on_right:
        fractions = range(_WIDTH_PARTS - _SPINE_POINTS, _WIDTH_PARTS)
    else:
        fractions = range(1, _SPINE_POINTS + 1)
    xs = np.array(
        [x]
        + [x + width * fraction / _WIDTH_PARTS for fraction in fractions]
        + [x + width]
    )
    columns = np.clip(xs.astype(int), 0, gray.shape[1] - 1)

    top_ys = _move_to_edge(paper, columns, np.full(len(xs), y), from_top=True)
    bottom_ys = _move_to_edge(
        paper, columns, np.full(len(xs), y + height), from_top=False
    )
    model = DistortionModel(
        top_curve=Curve.from_polyline(_smooth_curve(xs, top_ys)),
        bottom_curve=Curve.from_polyline(_smooth_curve(xs, bottom_ys)),
    )
    return model if model.is_valid() else None


def _move_to_edge(
    paper: NDArray[np.bool_],
    columns: NDArray[np.int_],
    ys: NDArray[np.float64],
    from_top: bool,
) -> NDArray[np.float64]:
    """Move each point to the first page edge between the image border and it.

    Args:
        paper (NDArray[np.bool_]): True for paper pixels.
        columns (NDArray[np.int_]): Column of every point.
        ys (NDArray[np.float64]): Starting y of every point.
        from_top (bool): True to search downwards from row 0 up to the point,
            False to search upwards from the last row down to the point.

    Returns:
        NDArray[np.float64]: New y of every point; unchanged where no edge is found.
    """
    height = paper.shape[0]
    strip = paper[:, columns] if from_top else paper[::-1, columns]
    padded = np.vstack(
        [np.zeros((1, strip.shape[1]), dtype=np.int32), strip.cumsum(axis=0)]
    )
    runs = padded[_EDGE_RUN:] - padded[:-_EDGE_RUN]
    # runs[j] counts the paper pixels in rows j .. j + _EDGE_RUN - 1.
    full_run = np.zeros(strip.shape, dtype=bool)
    full_run[: len(runs)] = runs == _EDGE_RUN

    rows = np.arange(height)[:, None]
    limit = ys[None, :] if from_top else (height - 1 - ys)[None, :]
    candidates = full_run & (rows < limit)
    found = candidates.any(axis=0)
    first = candidates.argmax(axis=0)
    edge = first if from_top else height - 1 - first
    return np.where(found, edge, ys).astype(np.float64)


def _smooth_curve(
    xs: NDArray[np.float64], ys: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Interpolate control points with a curve that does not overshoot them."""
    samples = np.linspace(xs[0], xs[-1], _CURVE_SAMPLES)
    return np.column_stack([samples, PchipInterpolator(xs, ys)(samples)])
