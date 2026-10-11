"""Carry a source PDF page's text layer onto the processed output page.

The text-showing content of a source page (every BT..ET block and the fonts it
references) is re-wrapped so that, placed on an output page, each glyph lands
over the same part of the picture it sat over in the source. Rendering mode 3
makes it invisible: selectable and searchable, never painted.

The mapping from source to output is an affine matrix, so pages that were
dewarped (a non-linear mapping) have no text layer.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

import numpy as np
import pikepdf
from loguru import logger
from numpy.typing import NDArray

from scantailor.core.models import Dpi
from scantailor.core.transformation import ImageTransformation

_POINTS_PER_INCH: Final = 72.0
_INVISIBLE_RENDER_MODE: Final = 3
# Text state operators that persist in the graphics state outside BT..ET.
_TEXT_STATE_OPERATORS: Final = frozenset({"Tc", "Tw", "Tz", "TL", "Tf", "Ts"})
_MIN_DETERMINANT: Final = 1e-12


@dataclass
class TextLayer:
    """Invisible text for one output page.

    Attributes:
        content (bytes): Content stream operators to append to the output page,
            in output PDF user space (points, origin bottom-left).
        resources (dict[str, pikepdf.Object]): Page resources the content needs,
            fonts under ``"/Font"``. The objects live in a private PDF kept alive
            by this layer; a writer must ``copy_foreign`` them into its document.
    """

    content: bytes
    resources: dict[str, Any] = field(default_factory=dict)
    _owner: pikepdf.Pdf | None = field(default=None, repr=False, compare=False)


def _inherited(node: pikepdf.Object | None, key: str) -> pikepdf.Object | None:
    while node is not None:
        if key in node:
            return node[key]
        node = node.get("/Parent")
    return None


def _matrix_of(operands: list[pikepdf.Object]) -> NDArray[np.float64]:
    a, b, c, d, e, f = (float(v) for v in operands)
    return np.array([[a, c, e], [b, d, f], [0, 0, 1]], dtype=np.float64)


def _cm_operands(matrix: NDArray[np.float64]) -> list[pikepdf.Object]:
    values = (
        matrix[0, 0],
        matrix[1, 0],
        matrix[0, 1],
        matrix[1, 1],
        matrix[0, 2],
        matrix[1, 2],
    )
    return [pikepdf.Object.parse(f"{v:.6f}".encode()) for v in values]


def _instruction(
    operator: str, *operands: pikepdf.Object | float
) -> pikepdf.ContentStreamInstruction:
    return pikepdf.ContentStreamInstruction(list(operands), pikepdf.Operator(operator))


def _text_blocks(
    page: pikepdf.Page,
) -> tuple[list[pikepdf.ContentStreamInstruction], set[str]]:
    """Collect each BT..ET block with the CTM and text state in force at its start.

    Returns:
        The instructions, each block wrapped in q/cm/Q with its text state
        restated and its render mode forced to invisible; and the names of the
        fonts selected by them.
    """
    ctm = np.eye(3)
    ctm_stack: list[NDArray[np.float64]] = []
    state: dict[str, pikepdf.ContentStreamInstruction] = {}
    state_stack: list[dict[str, pikepdf.ContentStreamInstruction]] = []
    output: list[pikepdf.ContentStreamInstruction] = []
    fonts: set[str] = set()
    in_text = False

    for item in pikepdf.parse_content_stream(page):
        if isinstance(item, pikepdf.ContentStreamInlineImage):
            continue
        operator = str(item.operator)
        operands = list(item.operands)

        if in_text:
            if operator == "Tr":
                continue
            if operator in _TEXT_STATE_OPERATORS:
                state[operator] = item
            if operator == "Tf":
                fonts.add(str(operands[0]))
            output.append(item)
            if operator == "ET":
                output.append(_instruction("Q"))
                in_text = False
            continue

        if operator == "q":
            ctm_stack.append(ctm)
            state_stack.append(dict(state))
        elif operator == "Q" and ctm_stack:
            ctm = ctm_stack.pop()
            state = state_stack.pop()
        elif operator == "cm":
            ctm = ctm @ _matrix_of(operands)
        elif operator in _TEXT_STATE_OPERATORS:
            state[operator] = item
        elif operator == "BT":
            in_text = True
            output.append(_instruction("q"))
            output.append(_instruction("cm", *_cm_operands(ctm)))
            for restated in state.values():
                output.append(restated)
                if str(restated.operator) == "Tf":
                    fonts.add(str(restated.operands[0]))
            output.append(item)
            output.append(_instruction("Tr", _INVISIBLE_RENDER_MODE))
    return output, fonts


def _page_orientation_matrix(page: pikepdf.Page) -> NDArray[np.float64]:
    """Map PDF user space (points, y up) to the rendered page (points, y down)."""
    left, bottom, right, top = (float(v) for v in page.cropbox)
    x0, y0 = min(left, right), min(bottom, top)
    width, height = abs(right - left), abs(top - bottom)
    unrotated = np.array([[1, 0, -x0], [0, -1, y0 + height], [0, 0, 1]], dtype=float)
    rotation = int(_inherited(page.obj, "/Rotate") or 0) % 360
    if rotation == 90:
        turn = np.array([[0, -1, height], [1, 0, 0], [0, 0, 1]], dtype=float)
    elif rotation == 180:
        turn = np.array([[-1, 0, width], [0, -1, height], [0, 0, 1]], dtype=float)
    elif rotation == 270:
        turn = np.array([[0, 1, 0], [-1, 0, width], [0, 0, 1]], dtype=float)
    else:
        turn = np.eye(3)
    return turn @ unrotated


def source_to_output_points(
    page: pikepdf.Page,
    source_to_output_px: NDArray[np.float64],
    source_dpi: Dpi,
    output_dpi: Dpi,
    output_height_px: float,
) -> NDArray[np.float64]:
    """Compose the matrix taking source user space to output user space.

    Args:
        page (pikepdf.Page): Source page; its CropBox (MediaBox when absent) and
            /Rotate fix the pixel grid the page was rendered to.
        source_to_output_px (NDArray[np.float64]): 3x3 matrix from source pixels
            (y down, at ``source_dpi``) to output pixels (y down, origin at the
            output's top-left corner).
        source_dpi (Dpi): Resolution the source page was rendered at.
        output_dpi (Dpi): Resolution of the output image.
        output_height_px (float): Height of the output image in pixels.

    Returns:
        NDArray[np.float64]: 3x3 matrix in column-vector form, source points to
        output points (y up in both).
    """
    to_source_px = np.diag(
        np.array(
            [
                source_dpi.horizontal / _POINTS_PER_INCH,
                source_dpi.vertical / _POINTS_PER_INCH,
                1.0,
            ],
            dtype=np.float64,
        )
    )
    to_output_points: NDArray[np.float64] = np.array(
        [
            [_POINTS_PER_INCH / output_dpi.horizontal, 0, 0],
            [
                0,
                -_POINTS_PER_INCH / output_dpi.vertical,
                output_height_px * _POINTS_PER_INCH / output_dpi.vertical,
            ],
            [0, 0, 1],
        ]
    )
    return (
        to_output_points
        @ source_to_output_px
        @ to_source_px
        @ _page_orientation_matrix(page)
    )


def build_text_layer(
    pdf_path: Path,
    page_index: int,
    transformation: ImageTransformation,
    source_dpi: Dpi,
    output_dpi: Dpi,
    *,
    dewarped: bool = False,
) -> TextLayer | None:
    """Build the invisible text layer of one output page.

    Text drawn by form XObjects and Type 3 glyph procedures is not carried over;
    only text operators in the page's own content stream are.

    Args:
        pdf_path (Path): The source PDF.
        page_index (int): Zero-based page in that PDF the output was made from.
        transformation (ImageTransformation): Final transformation of the page;
            maps source pixels to output pixels relative to its resulting rect.
        source_dpi (Dpi): Resolution the source page was imported at.
        output_dpi (Dpi): Resolution of the output image.
        dewarped (bool): True when output was dewarped, which no affine matrix
            describes.

    Returns:
        TextLayer | None: The layer, or None when the page has no text or the
        mapping is not affine and invertible.
    """
    if dewarped:
        logger.info("Page {}: no text layer for a dewarped page", page_index)
        return None

    matrix = transformation.transform
    if not np.allclose(matrix[2], [0, 0, 1], atol=1e-9):
        logger.info("Page {}: no text layer, transformation is projective", page_index)
        return None
    if abs(np.linalg.det(matrix[:2, :2])) < _MIN_DETERMINANT:
        logger.info("Page {}: no text layer, transformation is singular", page_index)
        return None

    rect = transformation.resulting_rect
    to_output_origin = np.array([[1, 0, -rect.x], [0, 1, -rect.y], [0, 0, 1]])

    with pikepdf.open(pdf_path) as source:
        page = source.pages[page_index]
        instructions, font_names = _text_blocks(page)
        if not instructions:
            logger.info("Page {}: no text to carry over", page_index)
            return None

        cm = source_to_output_points(
            page, to_output_origin @ matrix, source_dpi, output_dpi, rect.height
        )
        owner = pikepdf.Pdf.new()
        fonts = pikepdf.Dictionary()
        source_resources = _inherited(page.obj, "/Resources")
        source_fonts = (
            None if source_resources is None else source_resources.get("/Font")
        )
        for name in sorted(font_names):
            if source_fonts is not None and name in source_fonts:
                fonts[name] = owner.copy_foreign(source_fonts[name])
            else:
                logger.warning("Page {}: font {} not in resources", page_index, name)

        content = pikepdf.unparse_content_stream(
            [
                _instruction("q"),
                _instruction("cm", *_cm_operands(cm)),
                *instructions,
                _instruction("Q"),
            ]
        )

    return TextLayer(content=content, resources={"/Font": fonts}, _owner=owner)
