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

Unlike upstream, a model whose two curves lie close together is refused: the
builder extends the pair over the whole page, so two neighbouring lines turn a
wobble of a pixel into an error of dozens of pixels at the far end of the page.
``DewarpMetrics`` records what the tracing found and why a page got no model.
"""

import math
from dataclasses import dataclass
from enum import Enum

import numpy as np
from numpy.typing import NDArray

from scantailor.core.debug_images import DebugImages
from scantailor.core.models import Dpi
from scantailor.dewarping import DEFAULT_MIN_SPAN_PERCENT
from scantailor.dewarping.distortion_model import Curve, DistortionModel
from scantailor.dewarping.distortion_model_builder import DistortionModelBuilder
from scantailor.dewarping.line_geometry import q_round
from scantailor.dewarping.text_line_tracer import trace_text_lines
from scantailor.dewarping.top_bottom_edge_tracer import trace_top_bottom_edges

ContentRect = tuple[float, float, float, float]

# The outer border may lean this many degrees before a curve is extended.
_MAX_BORDER_SKEW_DEG = 2.75


class ModelOutcome(Enum):
    """Whether a page got an automatic model, and if not, why.

    NO_TEXT: neither text lines nor page edges were found.
    TOO_FEW_CURVES: fewer than two curves survived the tracers' filters.
    NO_PAIR: no pair of curves gives a workable dewarper.
    SPAN_TOO_SMALL: the chosen curves lie closer together than the minimum.
    """

    BUILT = "built"
    NO_TEXT = "no_text"
    TOO_FEW_CURVES = "too_few_curves"
    NO_PAIR = "no_pair"
    SPAN_TOO_SMALL = "span_too_small"


@dataclass(frozen=True)
class DewarpMetrics:
    """What automatic dewarping found on a page.

    Attributes:
        lines_traced (int): Text lines the tracer found before its filters.
        lines_kept (int): Text lines left after the filters.
        edges_traced (int): Page edges found, 0 to 2.
        outcome (ModelOutcome): Whether a model was built, or why not.
        min_span_percent (float): The minimum span the page was held to.
        span_percent (float | None): Distance between the chosen curves as a
            percentage of the content height; None when no pair was chosen.
        top_from_edge (bool | None): True when the top curve is a page edge,
            False when it is a text line; None when no pair was chosen.
        bottom_from_edge (bool | None): The same for the bottom curve.
    """

    lines_traced: int
    lines_kept: int
    edges_traced: int
    outcome: ModelOutcome
    min_span_percent: float
    span_percent: float | None = None
    top_from_edge: bool | None = None
    bottom_from_edge: bool | None = None

    def describe(self) -> str:
        """Say in a sentence or two what was found and what became of it.

        Returns:
            str: Plain words for the Output stage's dewarping controls.
        """
        if self.outcome is ModelOutcome.NO_TEXT:
            return "No text lines or page edges found, so the page is left as is."
        found = (
            f"{self.lines_kept} of {self.lines_traced} lines kept, "
            f"{_edges(self.edges_traced)}"
        )
        if self.outcome is ModelOutcome.TOO_FEW_CURVES:
            return f"{found}: fewer than two curves, so the page is left as is."
        if self.outcome is ModelOutcome.NO_PAIR:
            return (
                f"{found}, but no pair of curves gives a usable model, "
                "so the page is left as is."
            )
        assert self.span_percent is not None
        span = f"Curves span {self.span_percent:.0f}% of the text"
        if self.outcome is ModelOutcome.SPAN_TOO_SMALL:
            return (
                f"{found}. {span}. Below the {self.min_span_percent:.0f}% minimum, "
                f"so the page is left as is; a minimum of "
                f"{math.floor(self.span_percent):.0f}% or less would use them."
            )
        return (
            f"{found}. {span}, top from {_source(self.top_from_edge)}, "
            f"bottom from {_source(self.bottom_from_edge)}."
        )


def _edges(count: int) -> str:
    if count == 0:
        return "no page edge"
    return f"{count} page edge" if count == 1 else f"{count} page edges"


def _source(from_edge: bool | None) -> str:
    return "the page edge" if from_edge else "a text line"


@dataclass(frozen=True)
class AutoModelResult:
    """A detected model, or none, and the metrics behind it.

    Attributes:
        model (DistortionModel | None): The model, None when the page gets none.
        metrics (DewarpMetrics): What was found and why the model was or was
            not built.
    """

    model: DistortionModel | None
    metrics: DewarpMetrics


def trivial_distortion_model(content_rect: ContentRect) -> DistortionModel:
    """The model that maps the content rectangle onto itself.

    Args:
        content_rect (tuple[float, float, float, float]): ``(x, y, width, height)``.

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
    min_span_percent: float = DEFAULT_MIN_SPAN_PERCENT,
) -> DistortionModel | None:
    """Detect the page's curvature from its text lines and page edges.

    ``detect_distortion_model`` does the same and also tells what it found.

    Args:
        gray (numpy.ndarray of numpy.uint8):
            2-D gray page as laid out for output, dark text
            on a light background.
        dpi (Dpi | tuple[float, float] | float): Resolution of ``gray``.
        content_rect (tuple[float, float, float, float]):
            ``(x, y, width, height)`` of the content in
            ``gray`` pixels.
        spine_on_right (bool): True for a single or left page, False for a right
            page. Decides which outer border is straightened.
        edge_gray (numpy.ndarray of numpy.uint8 | None): The image the page edges are
            traced on and the model is expressed in, when that is not ``gray``;
            the C++ code uses the unprocessed input image. None uses ``gray``.
        to_edge_image (numpy.ndarray of numpy.float64 | None): Affine ``(2, 3)`` or
            ``(3, 3)`` matrix from ``gray`` pixels to ``edge_gray`` pixels.
            Required with ``edge_gray``.
        min_span_percent (float): A model whose curves lie closer together
            than this share of the content height, in percent, is refused.

    Returns:
        DistortionModel | None: The model in the pixels of ``edge_gray`` (or
            ``gray``), or None when the curves give no valid model or the model
            is refused.
    """
    return detect_distortion_model(
        gray,
        dpi,
        content_rect,
        spine_on_right,
        edge_gray,
        to_edge_image,
        min_span_percent,
    ).model


def detect_distortion_model(
    gray: NDArray[np.uint8],
    dpi: Dpi | tuple[float, float] | float,
    content_rect: ContentRect,
    spine_on_right: bool = True,
    edge_gray: NDArray[np.uint8] | None = None,
    to_edge_image: NDArray[np.float64] | None = None,
    min_span_percent: float = DEFAULT_MIN_SPAN_PERCENT,
    dbg: DebugImages | None = None,
) -> AutoModelResult:
    """Detect the page's curvature and report what the tracing found.

    The arguments are those of ``build_auto_distortion_model``, and ``dbg``
    receives the tracers' and the model builder's intermediate images.

    Returns:
        AutoModelResult: The model, or None with the reason in the metrics.
    """
    if (edge_gray is None) != (to_edge_image is None):
        raise ValueError("edge_gray and to_edge_image go together")

    rect = (
        q_round(content_rect[0]),
        q_round(content_rect[1]),
        q_round(content_rect[2]),
        q_round(content_rect[3]),
    )
    traced = trace_text_lines(gray, dpi, rect, dbg)
    lines_traced = traced.traced_count
    lines_kept = len(traced.lines)
    bounds = traced.vertical_bounds
    edges: list[NDArray[np.float64]] = []
    builder = DistortionModelBuilder((0.0, 1.0))
    edge_image = gray if edge_gray is None else edge_gray
    if bounds is not None:
        builder.set_vertical_bounds(bounds.first, bounds.second)
        for line in traced.lines:
            builder.add_horizontal_curve(line)
        if to_edge_image is not None:
            builder.transform(to_edge_image)
        edges = trace_top_bottom_edges(edge_image, builder.vertical_bounds(), dbg)
        for edge in edges:
            builder.add_horizontal_curve(edge)

    def result(
        outcome: ModelOutcome,
        model: DistortionModel | None = None,
        span_percent: float | None = None,
        top_from_edge: bool | None = None,
        bottom_from_edge: bool | None = None,
    ) -> AutoModelResult:
        return AutoModelResult(
            model,
            DewarpMetrics(
                lines_traced=lines_traced,
                lines_kept=lines_kept,
                edges_traced=len(edges),
                outcome=outcome,
                min_span_percent=min_span_percent,
                span_percent=span_percent,
                top_from_edge=top_from_edge,
                bottom_from_edge=bottom_from_edge,
            ),
        )

    if lines_traced == 0 and not edges:
        return result(ModelOutcome.NO_TEXT)
    built = builder.build(dbg, edge_image)
    if built.top_index is None or built.bottom_index is None:
        if lines_kept + len(edges) < 2:
            return result(ModelOutcome.TOO_FEW_CURVES)
        return result(ModelOutcome.NO_PAIR)
    span_percent = 100.0 * built.span / max(1, rect[3])
    # Edges were added after the text lines.
    top_from_edge = built.top_index >= lines_kept
    bottom_from_edge = built.bottom_index >= lines_kept
    if span_percent < min_span_percent:
        return result(
            ModelOutcome.SPAN_TOO_SMALL,
            None,
            span_percent,
            top_from_edge,
            bottom_from_edge,
        )
    model = straighten_outer_border(built.model, spine_on_right)
    if not model.is_valid():
        return result(
            ModelOutcome.NO_PAIR, None, span_percent, top_from_edge, bottom_from_edge
        )
    return result(
        ModelOutcome.BUILT, model, span_percent, top_from_edge, bottom_from_edge
    )


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
