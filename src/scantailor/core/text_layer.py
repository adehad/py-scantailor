"""Carry a source PDF page's text layer onto the processed output page.

The text-showing content of a source page (every BT..ET block, including those
drawn by form XObjects, and the fonts it references) is re-wrapped so that,
placed on an output page, each glyph lands over the same part of the picture it
sat over in the source. Rendering mode 3 makes it invisible: selectable and
searchable, never painted. Type 3 fonts ignore the rendering mode, so they are
replaced by stand-ins with the same metrics whose glyphs draw nothing.

An affine source-to-output mapping moves the original text operators as one
block. A dewarped page has no such mapping, so its text is read back through
pdfium and re-placed character by character at the dewarped position, in a
plain font whose codes are mapped to the characters by a ToUnicode table.
"""

from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

import numpy as np
import pikepdf
import pypdfium2 as pdfium
from loguru import logger
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict

from scantailor.core.dewarp_spec import DewarpSpec
from scantailor.core.models import Dpi
from scantailor.core.pdf_pool import map_pool_progress
from scantailor.core.transformation import ImageTransformation
from scantailor.dewarping.point_mapper import DewarpingPointMapper

_POINTS_PER_INCH: Final = 72.0
_INVISIBLE_RENDER_MODE: Final = 3
# Text state operators that persist in the graphics state outside BT..ET.
_TEXT_STATE_OPERATORS: Final = frozenset({"Tc", "Tw", "Tz", "TL", "Tf", "Ts"})
_MIN_DETERMINANT: Final = 1e-12
_DIRECT_OBJECT: Final = (0, 0)
_MAX_FORM_DEPTH: Final = 12
# pdfium extracts no text from codes below 0x21, so the codes of a generated
# font start there.
_FIRST_CODE: Final = 0x21
_CODES_PER_FONT: Final = 256 - _FIRST_CODE
_GENERATED_FONT_PREFIX: Final = "/STU"

ObjectGen = tuple[int, int]
# An object number, and where to find the object from the page's resources when
# it is written directly in a resource dictionary and has no number.
FontRef = tuple[ObjectGen, tuple[str, ...]]


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


def _matrix_of(operands: Sequence[pikepdf.Object | float]) -> NDArray[np.float64]:
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


_Instruction = pikepdf.ContentStreamInstruction
_Resources = pikepdf.Object | None


class _TextCollector:
    """Collects the text blocks of a page, descending into form XObjects.

    Each BT..ET block is wrapped in q/cm/Q with the CTM and the text state in
    force at its start, and its render mode forced to invisible. Fonts are
    renamed on collection, so two resource dictionaries that use one name for
    different fonts do not clash on the output page.
    """

    def __init__(self) -> None:
        self.output: list[_Instruction] = []
        self.fonts: dict[str, FontRef] = {}
        self._aliases: dict[ObjectGen | tuple[str, ...], str] = {}

    def collect(self, page: pikepdf.Page) -> None:
        resources = _inherited(page.obj, "/Resources")
        self._run(
            pikepdf.parse_content_stream(page), resources, (), np.eye(3), {}, 0, ()
        )
        used = {str(i.operands[0]) for i in self.output if str(i.operator) == "Tf"}
        self.fonts = {name: ref for name, ref in self.fonts.items() if name in used}

    def _font_alias(
        self, resources: _Resources, path: tuple[str, ...], name: str
    ) -> str | None:
        fonts = None if resources is None else resources.get("/Font")
        if fonts is None or name not in fonts:
            logger.warning("Font {} not in resources", name)
            return None
        objgen = fonts[name].objgen
        key = objgen if objgen != _DIRECT_OBJECT else (*path, "/Font", name)
        alias = self._aliases.get(key)
        if alias is None:
            alias, number = name, 1
            while alias in self.fonts:
                number += 1
                alias = f"{name}_{number}"
            self._aliases[key] = alias
            self.fonts[alias] = (objgen, (*path, "/Font", name))
        return alias

    def _with_font_alias(
        self, item: _Instruction, resources: _Resources, path: tuple[str, ...]
    ) -> _Instruction:
        operands = list(item.operands)
        alias = self._font_alias(resources, path, str(operands[0]))
        if alias is None or alias == str(operands[0]):
            return item
        return _instruction("Tf", pikepdf.Name(alias), operands[1])

    def _run(
        self,
        items: Sequence[Any],
        resources: _Resources,
        path: tuple[str, ...],
        ctm: NDArray[np.float64],
        state: dict[str, _Instruction],
        depth: int,
        active_forms: tuple[ObjectGen, ...],
    ) -> None:
        ctm_stack: list[NDArray[np.float64]] = []
        state_stack: list[dict[str, _Instruction]] = []
        state = dict(state)
        in_text = False

        for item in items:
            if isinstance(item, pikepdf.ContentStreamInlineImage):
                continue
            operator = str(item.operator)
            operands = list(item.operands)

            if in_text:
                if operator == "Tr":
                    continue
                if operator == "Tf":
                    item = self._with_font_alias(item, resources, path)
                if operator in _TEXT_STATE_OPERATORS:
                    state[operator] = item
                self.output.append(item)
                if operator == "ET":
                    self.output.append(_instruction("Q"))
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
            elif operator == "Tf":
                state["Tf"] = self._with_font_alias(item, resources, path)
            elif operator in _TEXT_STATE_OPERATORS:
                state[operator] = item
            elif operator == "Do":
                self._enter_form(
                    str(operands[0]), resources, path, ctm, state, depth, active_forms
                )
            elif operator == "BT":
                in_text = True
                self.output.append(_instruction("q"))
                self.output.append(_instruction("cm", *_cm_operands(ctm)))
                self.output.extend(state.values())
                self.output.append(item)
                self.output.append(_instruction("Tr", _INVISIBLE_RENDER_MODE))

    def _enter_form(
        self,
        name: str,
        resources: _Resources,
        path: tuple[str, ...],
        ctm: NDArray[np.float64],
        state: dict[str, _Instruction],
        depth: int,
        active_forms: tuple[ObjectGen, ...],
    ) -> None:
        xobjects = None if resources is None else resources.get("/XObject")
        form = None if xobjects is None else xobjects.get(name)
        if (
            form is None
            or form.get("/Subtype") != "/Form"
            or depth >= _MAX_FORM_DEPTH
            or (form.objgen != _DIRECT_OBJECT and form.objgen in active_forms)
        ):
            return
        matrix = form.get("/Matrix")
        form_ctm = ctm if matrix is None else ctm @ _matrix_of(list(matrix))
        own = form.get("/Resources")
        form_resources = resources if own is None else own
        form_path = path if own is None else (*path, "/XObject", name, "/Resources")
        self._run(
            pikepdf.parse_content_stream(form),
            form_resources,
            form_path,
            form_ctm,
            state,
            depth + 1,
            (*active_forms, form.objgen),
        )


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


def _source_pixels_matrix(source_dpi: Dpi) -> NDArray[np.float64]:
    return np.diag(
        np.array(
            [
                source_dpi.horizontal / _POINTS_PER_INCH,
                source_dpi.vertical / _POINTS_PER_INCH,
                1.0,
            ],
            dtype=np.float64,
        )
    )


def _output_points_matrix(
    output_dpi: Dpi, output_height_px: float
) -> NDArray[np.float64]:
    return np.array(
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
        source_to_output_px (numpy.ndarray of numpy.float64):
            3x3 matrix from source pixels
            (y down, at ``source_dpi``) to output pixels (y down, origin at the
            output's top-left corner).
        source_dpi (Dpi): Resolution the source page was rendered at.
        output_dpi (Dpi): Resolution of the output image.
        output_height_px (float): Height of the output image in pixels.

    Returns:
        numpy.ndarray of numpy.float64:
            3x3 matrix in column-vector form, source points to
        output points (y up in both).
    """
    return (
        _output_points_matrix(output_dpi, output_height_px)
        @ source_to_output_px
        @ _source_pixels_matrix(source_dpi)
        @ _page_orientation_matrix(page)
    )


class TextGeometry(BaseModel):
    """How an output page relates to its source page, as storable plain data.

    This is everything a text layer needs beyond the source PDF itself, so it can
    be saved with a project and still map text after the pipeline has not run in
    the current session.

    Attributes:
        matrix: The 3x3 source-pixel to output-pixel matrix of the page's
            transformation, row-major.
        rect_x: Left edge of the transformation's resulting rect.
        rect_y: Top edge of that rect.
        rect_width: Width of that rect, which is the output width.
        rect_height: Height of that rect, which is the output height.
        output_dpi: Horizontal and vertical resolution of the output image.
        dewarped: True when the output was dewarped.
        dewarp: How it was dewarped; a dewarped page has text only when given.
    """

    model_config = ConfigDict(frozen=True)

    matrix: tuple[float, ...]
    rect_x: float
    rect_y: float
    rect_width: float
    rect_height: float
    output_dpi: tuple[float, float]
    dewarped: bool = False
    dewarp: DewarpSpec | None = None

    @classmethod
    def of(
        cls,
        transformation: ImageTransformation,
        output_dpi: Dpi,
        *,
        dewarped: bool = False,
        dewarp: DewarpSpec | None = None,
    ) -> TextGeometry:
        """Capture the geometry of a page from its final transformation."""
        rect = transformation.resulting_rect
        return cls(
            matrix=tuple(float(v) for v in transformation.transform.ravel()),
            rect_x=float(rect.x),
            rect_y=float(rect.y),
            rect_width=float(rect.width),
            rect_height=float(rect.height),
            output_dpi=(output_dpi.horizontal, output_dpi.vertical),
            dewarped=dewarped,
            dewarp=dewarp,
        )

    def job(self, pdf_path: Path, page_index: int, source_dpi: Dpi) -> TextLayerJob:
        """Describe the text layer of source page ``page_index`` of ``pdf_path``."""
        return TextLayerJob(
            pdf_path=str(pdf_path),
            page_index=page_index,
            matrix=self.matrix,
            rect_x=self.rect_x,
            rect_y=self.rect_y,
            rect_height=self.rect_height,
            source_dpi=(source_dpi.horizontal, source_dpi.vertical),
            output_dpi=self.output_dpi,
            dewarped=self.dewarped,
            dewarp=self.dewarp,
            rect_width=self.rect_width,
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
        dewarp (DewarpSpec | None): How it was dewarped; without it a dewarped
            page gets no text layer.
        rect_width (float): Width of the resulting rect, which is the output width.
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
    dewarp: DewarpSpec | None = None
    rect_width: float = 0.0

    @staticmethod
    def of(
        pdf_path: Path,
        page_index: int,
        transformation: ImageTransformation,
        source_dpi: Dpi,
        output_dpi: Dpi,
        *,
        dewarped: bool = False,
        dewarp: DewarpSpec | None = None,
    ) -> TextLayerJob:
        """Describe the text layer of a page from its transformation."""
        geometry = TextGeometry.of(
            transformation, output_dpi, dewarped=dewarped, dewarp=dewarp
        )
        return geometry.job(pdf_path, page_index, source_dpi)


@dataclass(frozen=True)
class TextLayerData:
    """What a worker returns for one text layer.

    Attributes:
        content (bytes): Content stream of the layer.
        fonts (dict[str, tuple[tuple[int, int], tuple[str, ...]]]):
            Object number and generation, in the source
            PDF, of each font the content selects, and the path to it from the
            page's resources for a font written directly in a dictionary.
        generated_fonts (dict[str, tuple[int, ...]]): Fonts the layer defines
            itself, by name: the Unicode code point of each code, from the first.
    """

    content: bytes
    fonts: dict[str, FontRef]
    generated_fonts: dict[str, tuple[int, ...]] = field(default_factory=dict)


def _font_at(source: pikepdf.Pdf, page_index: int, reference: FontRef) -> Any:
    """Find a font by object number, or by its path through the page's resources."""
    objgen, path = reference
    if objgen != _DIRECT_OBJECT:
        return source.get_object(objgen)
    node: Any = _inherited(source.pages[page_index].obj, "/Resources")
    try:
        for key in path:
            node = node[key]
    except KeyError, TypeError, AttributeError:
        return None
    return node


def _affine_checks(job: TextLayerJob) -> NDArray[np.float64] | None:
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
    return matrix


def _extract_layer(source: pikepdf.Pdf, job: TextLayerJob) -> TextLayerData | None:
    matrix = _affine_checks(job)
    if matrix is None:
        return None
    if job.dewarped:
        if job.dewarp is None:
            logger.info("Page {}: no text layer for a dewarped page", job.page_index)
            return None
        return _extract_dewarped_layer(source, job, matrix, job.dewarp)

    to_output_origin = np.array(
        [[1, 0, -job.rect_x], [0, 1, -job.rect_y], [0, 0, 1]], dtype=np.float64
    )
    page = source.pages[job.page_index]
    collector = _TextCollector()
    collector.collect(page)
    if not collector.output:
        logger.info("Page {}: no text to carry over", job.page_index)
        return None

    cm = source_to_output_points(
        page,
        to_output_origin @ matrix,
        Dpi(horizontal=job.source_dpi[0], vertical=job.source_dpi[1]),
        Dpi(horizontal=job.output_dpi[0], vertical=job.output_dpi[1]),
        job.rect_height,
    )
    content = pikepdf.unparse_content_stream(
        [
            _instruction("q"),
            _instruction("cm", *_cm_operands(cm)),
            *collector.output,
            _instruction("Q"),
        ]
    )
    return TextLayerData(content, collector.fonts)


def _dewarp_mapper(
    job: TextLayerJob, matrix: NDArray[np.float64], spec: DewarpSpec
) -> Callable[[NDArray[np.float64]], NDArray[np.float64]] | None:
    """Return a function taking source pixels to dewarped output pixels, or None.

    The output generator dewarps the page in its view, the source image after
    the page's transformation, so source pixels are carried into the view first
    and then through the same model the generator built.
    """
    to_view = (
        np.array(
            [[1, 0, -job.rect_x], [0, 1, -job.rect_y], [0, 0, 1]], dtype=np.float64
        )
        @ matrix
    )
    width, height = round(job.rect_width), round(job.rect_height)
    if width <= 0 or height <= 0:
        return None

    def in_view(points: NDArray[np.float64]) -> NDArray[np.float64]:
        homogeneous = np.column_stack([points, np.ones(len(points))])
        return (homogeneous @ to_view.T)[:, :2]

    def edge_in_view(
        polyline: tuple[tuple[float, float], ...],
    ) -> list[tuple[float, float]]:
        mapped = in_view(np.asarray(polyline, dtype=np.float64))
        return [(float(x), float(y)) for x, y in mapped]

    try:
        mapper = DewarpingPointMapper(
            edge_in_view(spec.top),
            edge_in_view(spec.bottom),
            spec.depth_perception,
            (width, height),
            spec.post_deskew_angle,
        )
    except ValueError, ArithmeticError, np.linalg.LinAlgError, IndexError:
        return None

    def map_points(source_px: NDArray[np.float64]) -> NDArray[np.float64]:
        return mapper.to_dewarped(in_view(source_px))

    return map_points


@dataclass(frozen=True)
class _SourceChar:
    codepoint: int
    left: float
    bottom: float
    right: float
    top: float


def _source_characters(pdf_path: str, page_index: int) -> list[_SourceChar]:
    """Read every positioned character of a page, in reading order, via pdfium."""
    document = pdfium.PdfDocument(pdf_path)
    try:
        textpage = document[page_index].get_textpage()
        count = textpage.count_chars()
        text = textpage.get_text_range(0, count) if count else ""
        characters: list[_SourceChar] = []
        for index in range(count):
            character = (
                text[index] if len(text) == count else textpage.get_text_range(index, 1)
            )
            if not character or character in "\r\n\x00":
                continue
            left, bottom, right, top = textpage.get_charbox(index, loose=True)
            if right <= left or top <= bottom:
                continue
            characters.append(_SourceChar(ord(character[0]), left, bottom, right, top))
        return characters
    finally:
        document.close()


def _extract_dewarped_layer(
    source: pikepdf.Pdf,
    job: TextLayerJob,
    matrix: NDArray[np.float64],
    spec: DewarpSpec,
) -> TextLayerData | None:
    mapper = _dewarp_mapper(job, matrix, spec)
    if mapper is None:
        logger.info("Page {}: no text layer, dewarp model unusable", job.page_index)
        return None
    characters = _source_characters(job.pdf_path, job.page_index)
    if not characters:
        logger.info("Page {}: no text to carry over", job.page_index)
        return None

    page = source.pages[job.page_index]
    source_dpi = Dpi(horizontal=job.source_dpi[0], vertical=job.source_dpi[1])
    output_dpi = Dpi(horizontal=job.output_dpi[0], vertical=job.output_dpi[1])
    user_to_pixels = _source_pixels_matrix(source_dpi) @ _page_orientation_matrix(page)
    to_points = _output_points_matrix(output_dpi, job.rect_height)

    # Origin, end of the baseline and top of the left edge of every character box.
    user = np.array(
        [
            point
            for c in characters
            for point in (
                (c.left, c.bottom, 1),
                (c.right, c.bottom, 1),
                (c.left, c.top, 1),
            )
        ],
        dtype=np.float64,
    )
    pixels = (user @ user_to_pixels.T)[:, :2]
    placed = mapper(pixels)
    points = (np.column_stack([placed, np.ones(len(placed))]) @ to_points.T)[:, :2]
    corners = points.reshape(-1, 3, 2)

    codes: dict[int, tuple[int, int]] = {}
    codepoints: list[int] = []
    lines = [f"BT {_INVISIBLE_RENDER_MODE} Tr"]
    current_font = -1
    for character, (origin, end, top) in zip(characters, corners, strict=True):
        if not np.isfinite([origin, end, top]).all():
            continue
        if character.codepoint not in codes:
            codes[character.codepoint] = (
                len(codepoints) // _CODES_PER_FONT,
                len(codepoints) % _CODES_PER_FONT + _FIRST_CODE,
            )
            codepoints.append(character.codepoint)
        font, code = codes[character.codepoint]
        if font != current_font:
            lines.append(f"{_GENERATED_FONT_PREFIX}{font} 1 Tf")
            current_font = font
        along, up = end - origin, top - origin
        lines.append(
            f"{along[0]:.4f} {along[1]:.4f} {up[0]:.4f} {up[1]:.4f} "
            f"{origin[0]:.4f} {origin[1]:.4f} Tm <{code:02x}> Tj"
        )
    lines.append("ET")
    if len(lines) <= 2:
        return None
    generated = {
        f"{_GENERATED_FONT_PREFIX}{font}": tuple(
            codepoints[font * _CODES_PER_FONT : (font + 1) * _CODES_PER_FONT]
        )
        for font in range(current_font + 1 if current_font >= 0 else 0)
    }
    return TextLayerData("\n".join(lines).encode("ascii"), {}, generated)


def _copy_object(owner: pikepdf.Pdf, source: pikepdf.Pdf, value: Any) -> Any:
    """Copy an object, direct or not, from ``source`` into ``owner``."""
    if not value.is_indirect:
        value = source.make_indirect(value)
    return owner.copy_foreign(value)


def _generated_font(owner: pikepdf.Pdf, codepoints: tuple[int, ...]) -> pikepdf.Object:
    """Build a font of ``len(codepoints)`` one-em glyphs that extract as given."""
    entries = "\n".join(
        f"<{code:02X}> <{_utf16_hex(codepoint)}>"
        for code, codepoint in enumerate(codepoints, start=_FIRST_CODE)
    )
    cmap = (
        "/CIDInit /ProcSet findresource begin 12 dict begin begincmap\n"
        "/CMapName /Adobe-Identity-UCS def /CMapType 2 def\n"
        "1 begincodespacerange <00> <FF> endcodespacerange\n"
        f"{len(codepoints)} beginbfchar\n{entries}\nendbfchar\n"
        "endcmap CMapName currentdict /CMap defineresource pop end end"
    )
    to_unicode = owner.make_stream(cmap.encode("ascii"))
    return owner.make_indirect(
        pikepdf.Dictionary(
            Type=pikepdf.Name.Font,
            Subtype=pikepdf.Name.Type1,
            BaseFont=pikepdf.Name.Helvetica,
            FirstChar=_FIRST_CODE,
            LastChar=_FIRST_CODE + len(codepoints) - 1,
            Widths=pikepdf.Array([1000] * len(codepoints)),
            ToUnicode=to_unicode,
        )
    )


def _utf16_hex(codepoint: int) -> str:
    return chr(codepoint).encode("utf-16-be").hex().upper()


def _is_type3(font: Any) -> bool:
    return font.get("/Subtype") == "/Type3"


def _invisible_type3(owner: pikepdf.Pdf, source: pikepdf.Pdf, font: Any) -> Any:
    """Copy a Type 3 font with glyph procedures that paint nothing.

    Widths, encoding and ToUnicode are kept, so the text still advances and
    extracts as it did.
    """
    widths = [float(w) for w in font.get("/Widths", [])]
    first = int(font.get("/FirstChar", 0))
    code_of: dict[str, int] = {}
    encoding = font.get("/Encoding")
    if isinstance(encoding, pikepdf.Dictionary) and "/Differences" in encoding:
        code = 0
        for entry in encoding["/Differences"]:
            if isinstance(entry, pikepdf.Name):
                code_of[str(entry)] = code
                code += 1
            else:
                code = int(entry)
    procs = pikepdf.Dictionary()
    char_procs = font.get("/CharProcs")
    for name in () if char_procs is None else char_procs.keys():
        index = code_of.get(name, -1) - first
        width = widths[index] if 0 <= index < len(widths) else 0.0
        procs[name] = owner.make_stream(f"{width:g} 0 d0".encode("ascii"))
    stand_in = pikepdf.Dictionary(
        Type=pikepdf.Name.Font,
        Subtype=pikepdf.Name.Type3,
        FontBBox=pikepdf.Array([float(v) for v in font.get("/FontBBox", [0, 0, 0, 0])]),
        FontMatrix=pikepdf.Array(
            [float(v) for v in font.get("/FontMatrix", [0.001, 0, 0, 0.001, 0, 0])]
        ),
        CharProcs=owner.make_indirect(procs),
        FirstChar=first,
        LastChar=int(font.get("/LastChar", first + len(widths) - 1)),
        Widths=pikepdf.Array(widths),
    )
    if encoding is not None:
        stand_in["/Encoding"] = _copy_object(owner, source, encoding)
    if "/ToUnicode" in font:
        stand_in["/ToUnicode"] = _copy_object(owner, source, font["/ToUnicode"])
    return owner.make_indirect(stand_in)


def _attach_layer(
    data: TextLayerData, source: pikepdf.Pdf, job: TextLayerJob, owner: pikepdf.Pdf
) -> TextLayer:
    """Make a layer whose fonts are objects of ``owner``, copied from ``source``."""
    fonts = pikepdf.Dictionary()
    for name, reference in data.fonts.items():
        font = _font_at(source, job.page_index, reference)
        if font is None:
            continue
        if _is_type3(font):
            fonts[name] = _invisible_type3(owner, source, font)
        else:
            fonts[name] = _copy_object(owner, source, font)
    for name, codepoints in data.generated_fonts.items():
        fonts[name] = _generated_font(owner, codepoints)
    return TextLayer(content=data.content, resources={"/Font": fonts}, _owner=owner)


def build_text_layer(
    pdf_path: Path,
    page_index: int,
    transformation: ImageTransformation,
    source_dpi: Dpi,
    output_dpi: Dpi,
    *,
    dewarped: bool = False,
    dewarp: DewarpSpec | None = None,
) -> TextLayer | None:
    """Build the invisible text layer of one output page.

    Text drawn by form XObjects is carried like the page's own. Type 3 fonts are
    replaced by invisible stand-ins with the same metrics.

    Args:
        pdf_path (pathlib.Path): The source PDF.
        page_index (int): Zero-based page in that PDF the output was made from.
        transformation (ImageTransformation): Final transformation of the page;
            maps source pixels to output pixels relative to its resulting rect.
        source_dpi (Dpi): Resolution the source page was imported at.
        output_dpi (Dpi): Resolution of the output image.
        dewarped (bool): True when output was dewarped, which no affine matrix
            describes.
        dewarp (DewarpSpec | None): How the page was dewarped. A dewarped page
            has a text layer only when this is given.

    Returns:
        scantailor.core.text_layer.TextLayer | None:
            The layer, or None when the page has no text or the
        mapping cannot be inverted.
    """
    job = TextLayerJob.of(
        pdf_path,
        page_index,
        transformation,
        source_dpi,
        output_dpi,
        dewarped=dewarped,
        dewarp=dewarp,
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
        executor (concurrent.futures.ProcessPoolExecutor | None):
            A pool from ``spawn_pool``; the
            layers are built in this process when None.
        max_workers (int | None): Size of ``executor``.

    Returns:
        list[scantailor.core.text_layer.TextLayer | None]:
            One entry per job, None where a page has no layer.
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
            extracted = map_pool_progress(
                _extract_in_worker, jobs, executor, max_workers
            )

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
