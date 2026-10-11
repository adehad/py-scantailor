"""Carry a source PDF page's text layer onto the processed output page.

The text-showing content of a source page (every BT..ET block and the fonts it
references) is re-wrapped so that, placed on an output page, each glyph lands
over the same part of the picture it sat over in the source. Rendering mode 3
makes it invisible: selectable and searchable, never painted.

The mapping from source to output is an affine matrix, so pages that were
dewarped (a non-linear mapping) have no text layer.
"""

from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

import numpy as np
import pikepdf
from loguru import logger
from numpy.typing import NDArray

from scantailor.core.batch import map_pool
from scantailor.core.models import Dpi
from scantailor.core.transformation import ImageTransformation

_POINTS_PER_INCH: Final = 72.0
_INVISIBLE_RENDER_MODE: Final = 3
# Text state operators that persist in the graphics state outside BT..ET.
_TEXT_STATE_OPERATORS: Final = frozenset({"Tc", "Tw", "Tz", "TL", "Tf", "Ts"})
_MIN_DETERMINANT: Final = 1e-12
_DIRECT_OBJECT: Final = (0, 0)


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


@dataclass(frozen=True)
class TextLayerJob:
    """Plain-data description of one output page's text layer, safe to pickle.

    Attributes:
        pdf_path (str): The source PDF.
        page_index (int): Zero-based page in that PDF.
        matrix (tuple[float, ...]): The 3x3 source-pixel to output-pixel matrix
            of the page's transformation, row-major.
        rect_x (float): Left edge of the transformation's resulting rect.
        rect_y (float): Top edge of the transformation's resulting rect.
        rect_height (float): Height of that rect, which is the output height.
        source_dpi (tuple[float, float]): Horizontal and vertical resolution the
            source page was imported at.
        output_dpi (tuple[float, float]): Horizontal and vertical resolution of
            the output image.
        dewarped (bool): True when the output was dewarped.
    """

    pdf_path: str
    page_index: int
    matrix: tuple[float, ...]
    rect_x: float
    rect_y: float
    rect_height: float
    source_dpi: tuple[float, float]
    output_dpi: tuple[float, float]
    dewarped: bool = False

    @staticmethod
    def of(
        pdf_path: Path,
        page_index: int,
        transformation: ImageTransformation,
        source_dpi: Dpi,
        output_dpi: Dpi,
        *,
        dewarped: bool = False,
    ) -> "TextLayerJob":
        """Describe the text layer of a page from its transformation."""
        rect = transformation.resulting_rect
        return TextLayerJob(
            pdf_path=str(pdf_path),
            page_index=page_index,
            matrix=tuple(float(v) for v in transformation.transform.ravel()),
            rect_x=float(rect.x),
            rect_y=float(rect.y),
            rect_height=float(rect.height),
            source_dpi=(source_dpi.horizontal, source_dpi.vertical),
            output_dpi=(output_dpi.horizontal, output_dpi.vertical),
            dewarped=dewarped,
        )


@dataclass(frozen=True)
class TextLayerData:
    """What a worker returns for one text layer.

    Attributes:
        content (bytes): Content stream of the layer.
        fonts (dict[str, tuple[int, int]]): Object number and generation, in the
            source PDF, of each font the content selects; ``(0, 0)`` for a font
            written directly in the page's resources.
    """

    content: bytes
    fonts: dict[str, tuple[int, int]]


def _source_font(
    source: pikepdf.Pdf, page_index: int, name: str, reference: tuple[int, int]
) -> pikepdf.Object | None:
    """Find a font by reference, or by name in the page's resources when direct."""
    if reference != _DIRECT_OBJECT:
        return source.get_object(reference)
    resources = _inherited(source.pages[page_index].obj, "/Resources")
    fonts = None if resources is None else resources.get("/Font")
    if fonts is None or name not in fonts:
        return None
    return source.make_indirect(fonts[name])


def _extract_layer(source: pikepdf.Pdf, job: TextLayerJob) -> TextLayerData | None:
    if job.dewarped:
        logger.info("Page {}: no text layer for a dewarped page", job.page_index)
        return None

    matrix = np.array(job.matrix, dtype=np.float64).reshape(3, 3)
    if not np.allclose(matrix[2], [0, 0, 1], atol=1e-9):
        logger.info(
            "Page {}: no text layer, transformation is projective", job.page_index
        )
        return None
    if abs(np.linalg.det(matrix[:2, :2])) < _MIN_DETERMINANT:
        logger.info(
            "Page {}: no text layer, transformation is singular", job.page_index
        )
        return None

    to_output_origin = np.array(
        [[1, 0, -job.rect_x], [0, 1, -job.rect_y], [0, 0, 1]], dtype=np.float64
    )
    page = source.pages[job.page_index]
    instructions, font_names = _text_blocks(page)
    if not instructions:
        logger.info("Page {}: no text to carry over", job.page_index)
        return None

    cm = source_to_output_points(
        page,
        to_output_origin @ matrix,
        Dpi(horizontal=job.source_dpi[0], vertical=job.source_dpi[1]),
        Dpi(horizontal=job.output_dpi[0], vertical=job.output_dpi[1]),
        job.rect_height,
    )
    source_resources = _inherited(page.obj, "/Resources")
    source_fonts = None if source_resources is None else source_resources.get("/Font")
    fonts: dict[str, tuple[int, int]] = {}
    for name in sorted(font_names):
        if source_fonts is not None and name in source_fonts:
            fonts[name] = source_fonts[name].objgen
        else:
            logger.warning("Page {}: font {} not in resources", job.page_index, name)

    content = pikepdf.unparse_content_stream(
        [
            _instruction("q"),
            _instruction("cm", *_cm_operands(cm)),
            *instructions,
            _instruction("Q"),
        ]
    )
    return TextLayerData(content, fonts)


def _attach_layer(
    data: TextLayerData, source: pikepdf.Pdf, job: TextLayerJob, owner: pikepdf.Pdf
) -> TextLayer:
    """Make a layer whose fonts are objects of ``owner``, copied from ``source``."""
    fonts = pikepdf.Dictionary()
    for name, reference in data.fonts.items():
        font = _source_font(source, job.page_index, name, reference)
        if font is not None:
            fonts[name] = owner.copy_foreign(font)
    return TextLayer(content=data.content, resources={"/Font": fonts}, _owner=owner)


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
    job = TextLayerJob.of(
        pdf_path, page_index, transformation, source_dpi, output_dpi, dewarped=dewarped
    )
    return build_text_layers([job])[0]


_worker_source: tuple[str, pikepdf.Pdf] | None = None


def _extract_in_worker(job: TextLayerJob) -> TextLayerData | None:
    """Extract one layer in a worker, keeping the source open between pages."""
    global _worker_source
    if _worker_source is None or _worker_source[0] != job.pdf_path:
        if _worker_source is not None:
            _worker_source[1].close()
        _worker_source = (job.pdf_path, pikepdf.open(job.pdf_path))
    return _extract_layer(_worker_source[1], job)


def build_text_layers(
    jobs: Sequence[TextLayerJob],
    *,
    executor: ProcessPoolExecutor | None = None,
    max_workers: int | None = None,
) -> list[TextLayer | None]:
    """Build the text layers of many pages, on a pool when one is given.

    Workers return the content and font references as plain data. The fonts are
    then copied from one open copy of each source PDF into one private PDF, so a
    font shared by many pages is one object rather than one per page.

    Args:
        jobs (Sequence[TextLayerJob]): One per output page.
        executor (ProcessPoolExecutor | None): A pool from ``spawn_pool``; the
            layers are built in this process when None.
        max_workers (int | None): Size of ``executor``.

    Returns:
        list[TextLayer | None]: One entry per job, None where a page has no layer.
    """
    sources: dict[str, pikepdf.Pdf] = {}

    def source_of(path: str) -> pikepdf.Pdf:
        if path not in sources:
            sources[path] = pikepdf.open(path)
        return sources[path]

    try:
        if executor is None:
            extracted = [_extract_layer(source_of(job.pdf_path), job) for job in jobs]
        else:
            extracted = map_pool(_extract_in_worker, jobs, executor, max_workers)

        owner = pikepdf.Pdf.new()
        layers: list[TextLayer | None] = []
        for job, data in zip(jobs, extracted, strict=True):
            if data is None:
                layers.append(None)
            else:
                layers.append(_attach_layer(data, source_of(job.pdf_path), job, owner))
        return layers
    finally:
        for source in sources.values():
            source.close()
