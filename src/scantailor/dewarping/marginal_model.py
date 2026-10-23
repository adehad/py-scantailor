"""Distortion model for marginal dewarping.

Marginal dewarping corrects only the strip of page next to the binding. The top
and bottom curves are X-splines along the content rectangle's top and bottom
edges. All seven control points of each curve, the two ends and the five near
the binding, are moved out to the page's top and bottom margins, found as the
first run of 16 paper pixels in the column when searching in from the image edge.
"""

import numpy as np
from numpy.typing import NDArray

from scantailor.core.debug_images import DebugImages
from scantailor.dewarping.distortion_model import Curve, DistortionModel
from scantailor.dewarping.spline_fit import sample_spline
from scantailor.imageproc import debug_drawing as draw

# Gray levels below this are page background or ink; paper is brighter.
_PAPER_THRESHOLD = 64
# A margin starts where this many pixels in a row, going inwards, are paper.
_MARGIN_RUN = 16
# Side of the square that marks a control point in the debug image.
_DEBUG_MARK_SIZE = 20
# The content width is split into this many parts; five control points sit
# at the parts nearest the binding.
_WIDTH_PARTS = 29
_BINDING_POINTS = 5
_END_TENSION = 0.0
_INNER_TENSION = 1.0


def build_marginal_model(
    gray: NDArray[np.uint8],
    content_rect: tuple[float, float, float, float],
    spine_on_right: bool,
    dbg: DebugImages | None = None,
) -> DistortionModel | None:
    """Build a model whose curves bend towards the page edges near the binding.

    Args:
        gray (NDArray[np.uint8]): 2-D gray page, the coordinates of the model.
        content_rect (tuple[float, float, float, float]): (x, y, width, height)
            of the content.
        spine_on_right (bool): True for a single or left page, whose binding is
            on the right.
        dbg (DebugImages | None): Receives the paper mask with the control
            points of both curves marked.

    Returns:
        DistortionModel | None: The model, or None when the content rectangle
            is empty or the result is not a valid model (which the C++ code
            replaces with the model that leaves the content as it is).
    """
    x, y, width, height = content_rect
    if width <= 0 or height <= 0:
        return None
    paper = gray >= _PAPER_THRESHOLD

    xs = _control_point_xs(x, width, spine_on_right)
    top_ys = _top_margin(paper, xs, np.full(len(xs), float(y)))
    bottom_ys = _bottom_margin(paper, xs, np.full(len(xs), float(y + height)))
    if dbg is not None:
        dbg.add(
            _visualize_control_points(paper, xs, top_ys, bottom_ys),
            "marginal dewarping",
        )

    model = DistortionModel(
        top_curve=_curve(xs, top_ys),
        bottom_curve=_curve(xs, bottom_ys),
    )
    return model if model.is_valid() else None


def _visualize_control_points(
    paper: NDArray[np.bool_],
    xs: NDArray[np.float64],
    top_ys: NDArray[np.float64],
    bottom_ys: NDArray[np.float64],
) -> NDArray[np.uint8]:
    """Mark the spline control points of both page edges on the paper mask.

    Args:
        paper (numpy.ndarray of numpy.bool): True on the paper.
        xs (numpy.ndarray of numpy.float64): x of each control point.
        top_ys (numpy.ndarray of numpy.float64): y of each top edge point.
        bottom_ys (numpy.ndarray of numpy.float64): y of each bottom edge point.

    Returns:
        numpy.ndarray of numpy.uint8: New ``(H, W, 3)`` BGR image.
    """
    canvas = draw.to_bgr(~paper)
    half = _DEBUG_MARK_SIZE // 2
    for px, py in zip(
        np.concatenate([xs, xs]), np.concatenate([top_ys, bottom_ys]), strict=True
    ):
        cx, cy = round(px), round(py)
        canvas[
            max(0, cy - half) : max(0, cy + half), max(0, cx - half) : max(0, cx + half)
        ] = draw.RED
    return canvas


def _control_point_xs(
    x: float, width: float, spine_on_right: bool
) -> NDArray[np.float64]:
    """X of the seven control points of a curve along the content's width.

    Args:
        x (float): Left side of the content.
        width (float): Width of the content.
        spine_on_right (bool): True when the binding is on the right.

    Returns:
        NDArray[np.float64]: The two ends and, between them, the five points
            nearest the binding. The inner points are interpolated the way a
            point on a line is, so a column that falls on a whole number is
            reached from below, as in the C++ code, and picks the pixel before it.
    """
    if spine_on_right:
        fractions = range(_WIDTH_PARTS - _BINDING_POINTS, _WIDTH_PARTS)
    else:
        fractions = range(1, _BINDING_POINTS + 1)
    right = x + width
    return np.array(
        [x]
        + [x + (right - x) * (fraction / _WIDTH_PARTS) for fraction in fractions]
        + [right]
    )


def _curve(xs: NDArray[np.float64], ys: NDArray[np.float64]) -> Curve:
    """The X-spline through control points, with tension 0 at the ends and 1 inside."""
    positions = np.column_stack([xs, ys])
    tensions = np.full(len(xs), _INNER_TENSION)
    tensions[0] = tensions[-1] = _END_TENSION
    return Curve.from_polyline(sample_spline(positions, tensions)[1])


def _top_margin(
    paper: NDArray[np.bool_], xs: NDArray[np.float64], ys: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Move each point up to the first paper run above it, searching down from row 0."""
    moved = ys.copy()
    height, width = paper.shape
    for index, (px, py) in enumerate(zip(xs, ys, strict=True)):
        column = int(px)
        if not 0 <= column < width:
            continue
        for row in range(0, int(np.ceil(py))):
            if row >= height:
                break
            if paper[row, column] and _is_run(paper[row : row + _MARGIN_RUN, column]):
                moved[index] = row
                break
    return moved


def _bottom_margin(
    paper: NDArray[np.bool_], xs: NDArray[np.float64], ys: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Move each point down to the first paper run below it, searching up."""
    moved = ys.copy()
    height, width = paper.shape
    for index, (px, py) in enumerate(zip(xs, ys, strict=True)):
        column = int(px)
        if not 0 <= column < width:
            continue
        for row in range(height - 1, int(np.floor(py)), -1):
            if paper[row, column] and _is_run(
                paper[max(0, row - _MARGIN_RUN + 1) : row + 1, column]
            ):
                moved[index] = row
                break
    return moved


def _is_run(pixels: NDArray[np.bool_]) -> bool:
    """True when the slice is a full run of paper; one cut off by the image is not."""
    return len(pixels) == _MARGIN_RUN and bool(pixels.all())
