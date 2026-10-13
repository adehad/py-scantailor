"""Automatic distortion model of a page, as the output stage builds it.

Text lines are traced on the page as laid out for output. The page's top and
bottom edges are traced on the image the model is expressed in, between the
vertical bounds the text lines gave. A model is built from all of those curves;
when its outer vertical border is slanted by more than a few degrees, the
shorter curve is extended to make the border vertical.

The tracers and the model builder are the C++ ones. A page whose curves give no
valid model gets none, where the C++ output stage substitutes the model of the
content rectangle, which leaves the content as it is. ``trivial_distortion_model``
builds that model for callers that need it.
"""

import math

import cv2
import numpy as np
from numpy.typing import NDArray

from scantailor.core.models import Dpi
from scantailor.dewarping.distortion_model import Curve, DistortionModel
from scantailor.dewarping.distortion_model_builder import DistortionModelBuilder
from scantailor.dewarping.line_geometry import q_round
from scantailor.dewarping.text_line_tracer import trace_text_lines
from scantailor.dewarping.top_bottom_edge_tracer import trace_top_bottom_edges

ContentRect = tuple[float, float, float, float]

# The outer border may lean this many degrees before a curve is extended.
_MAX_BORDER_SKEW_DEG = 2.75
_WARM_UP_SIZE = (720, 520)


def warm_up_kernels() -> None:
    """Run the automatic model builder once on a small page to compile its kernels.

    Each numba kernel compiles on its first call, which takes seconds, and the
    result is cached on disk. A page with a few lines of text between dark page
    edges reaches every kernel of the tracers and the spline fit.
    """
    height, width = _WARM_UP_SIZE
    columns = np.arange(width)
    sag = 14 * np.sin(np.pi * columns / width)
    top = np.linspace(80, 60, width) + sag
    bottom = np.linspace(660, 640, width) - 0.7 * sag
    rows = np.arange(height)[:, None]
    page = np.full((height, width), 45, dtype=np.uint8)
    page[(rows > top[None, :]) & (rows < bottom[None, :])] = 235
    rng = np.random.default_rng(0)
    words = "the quick brown fox jumps over lazy dog and runs far".split()
    font = cv2.FONT_HERSHEY_SIMPLEX
    for baseline in range(130, 600, 28):
        x = 70
        while x < 420:
            word = words[int(rng.integers(len(words)))]
            cv2.putText(page, word, (x, baseline), font, 0.55, 25, 1, cv2.LINE_AA)
            x += cv2.getTextSize(word, font, 0.55, 1)[0][0] + 14
    build_auto_distortion_model(page, 200, (60.0, 100.0, 320.0, 420.0))


def trivial_distortion_model(content_rect: ContentRect) -> DistortionModel:
    """The model that maps the content rectangle onto itself.

    Args:
        content_rect (ContentRect): ``(x, y, width, height)``.

    Returns:
        DistortionModel: Straight top and bottom curves along the rectangle.
    """
    x, y, width, height = content_rect
    if width > 0 and height > 0:
        left, top, right, bottom = x, y, x + width, y + height
    else:
        left, top, right, bottom = x - 0.5, y - 0.5, x + 0.5, y + 0.5
    return DistortionModel(
        top_curve=Curve.from_polyline(np.array([[left, top], [right, top]])),
        bottom_curve=Curve.from_polyline(np.array([[left, bottom], [right, bottom]])),
    )


def build_auto_distortion_model(
    gray: NDArray[np.uint8],
    dpi: Dpi | tuple[float, float] | float,
    content_rect: ContentRect,
    spine_on_right: bool = True,
    edge_gray: NDArray[np.uint8] | None = None,
    to_edge_image: NDArray[np.float64] | None = None,
) -> DistortionModel | None:
    """Detect the page's curvature from its text lines and page edges.

    Args:
        gray (NDArray[np.uint8]): 2-D gray page as laid out for output, dark text
            on a light background.
        dpi (Dpi | tuple[float, float] | float): Resolution of ``gray``.
        content_rect (ContentRect): ``(x, y, width, height)`` of the content in
            ``gray`` pixels.
        spine_on_right (bool): True for a single or left page, False for a right
            page. Decides which outer border is straightened.
        edge_gray (NDArray[np.uint8] | None): The image the page edges are
            traced on and the model is expressed in, when that is not ``gray``;
            the C++ code uses the unprocessed input image. None uses ``gray``.
        to_edge_image (NDArray[np.float64] | None): Affine ``(2, 3)`` or
            ``(3, 3)`` matrix from ``gray`` pixels to ``edge_gray`` pixels.
            Required with ``edge_gray``.

    Returns:
        DistortionModel | None: The model in the pixels of ``edge_gray`` (or
            ``gray``), or None when the curves give no valid model.
    """
    if (edge_gray is None) != (to_edge_image is None):
        raise ValueError("edge_gray and to_edge_image go together")

    rect = (
        q_round(content_rect[0]),
        q_round(content_rect[1]),
        q_round(content_rect[2]),
        q_round(content_rect[3]),
    )
    traced = trace_text_lines(gray, dpi, rect)
    bounds = traced.vertical_bounds
    if bounds is None:
        return None

    builder = DistortionModelBuilder((0.0, 1.0))
    builder.set_vertical_bounds(bounds.first, bounds.second)
    for line in traced.lines:
        builder.add_horizontal_curve(line)
    if to_edge_image is not None:
        builder.transform(to_edge_image)

    edge_image = gray if edge_gray is None else edge_gray
    for edge in trace_top_bottom_edges(edge_image, builder.vertical_bounds()):
        builder.add_horizontal_curve(edge)

    model = builder.try_build_model()
    if not model.is_valid():
        return None
    model = straighten_outer_border(model, spine_on_right)
    return model if model.is_valid() else None


def straighten_outer_border(
    model: DistortionModel, spine_on_right: bool
) -> DistortionModel:
    """Extend a curve so that a strongly slanted outer border becomes vertical.

    The outer border is the left one of a single or left page and the right one
    of a right page. When it leans more than 2.75 degrees from the vertical, the
    curve whose end lies nearer the page's inside gets one more point, straight
    above or below the other curve's end.

    Args:
        model (DistortionModel): A valid model.
        spine_on_right (bool): True for a single or left page.

    Returns:
        DistortionModel: The same model, or one with one extended curve.
    """
    assert model.top_curve is not None and model.bottom_curve is not None
    top = model.top_curve.polyline
    bottom = model.bottom_curve.polyline
    end = 0 if spine_on_right else -1
    if _border_skew_degrees(top[end], bottom[end]) <= _MAX_BORDER_SKEW_DEG:
        return model

    # The C++ code keeps these coordinates in single precision.
    top_x = float(np.float32(top[end][0]))
    bottom_x = float(np.float32(bottom[end][0]))
    top_end_is_outer = top_x < bottom_x if spine_on_right else top_x > bottom_x
    if top_end_is_outer:
        extension = np.array([[top_x, bottom[end][1]]])
        extended = _with_point(bottom, extension, spine_on_right)
        return DistortionModel(
            top_curve=model.top_curve, bottom_curve=Curve.from_polyline(extended)
        )
    extension = np.array([[bottom_x, top[end][1]]])
    extended = _with_point(top, extension, spine_on_right)
    return DistortionModel(
        top_curve=Curve.from_polyline(extended), bottom_curve=model.bottom_curve
    )


def _with_point(
    polyline: NDArray[np.float64], point: NDArray[np.float64], at_front: bool
) -> NDArray[np.float64]:
    """The polyline with one point added at its start or its end."""
    return np.vstack([point, polyline] if at_front else [polyline, point])


def _border_skew_degrees(
    top: NDArray[np.float64], bottom: NDArray[np.float64]
) -> float:
    """Angle in degrees between the border ``top`` to ``bottom`` and the vertical."""
    dx = abs(float(bottom[0] - top[0]))
    dy = abs(float(bottom[1] - top[1]))
    if dx == 0.0 and dy == 0.0:
        return 0.0
    return math.degrees(math.atan2(dx, dy))
