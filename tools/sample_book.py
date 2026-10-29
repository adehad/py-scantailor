"""Generate the sample book behind the documentation screenshots and their tests.

Every page is synthetic. Text is set with OpenCV's Hershey fonts, pictures are
noise-textured gradients, and each defect (a skew, a shear, a curl, specks, a
rotation) is applied in code. Pages are A5 at 150 dpi on a dark scanner bed, and
every run writes the same bytes.

Usage:
    python tools/sample_book.py --out DIR
"""

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import cv2
import numpy as np
import pikepdf
from numpy.typing import NDArray
from PIL import Image

SAMPLE_DPI: Final = 150
BED_MARGIN: Final = 40
PAPER_WIDTH: Final = 794
PAPER_HEIGHT: Final = 1160
_PAPER_RGB: Final = (238, 233, 222)
_BED_RGB: Final = (206, 204, 198)
_INK_RGB: Final = (34, 31, 29)
_BODY_FONT: Final = cv2.FONT_HERSHEY_COMPLEX_SMALL
_HEADING_FONT: Final = cv2.FONT_HERSHEY_COMPLEX
_BODY_SCALE: Final = 0.95
_HEADING_SCALE: Final = 1.45
_CAPTION_SCALE: Final = 0.8
_LINE_PITCH: Final = 36
_PARAGRAPH_GAP: Final = 14
_INDENT: Final = 34
_TEXT_LEFT: Final = 92
_TEXT_TOP: Final = 150
_TEXT_WIDTH: Final = 610
_FOOT_CLEARANCE: Final = 100
_SEED: Final = 20260613
_SKEW_DEGREES: Final = 3.0
# Inside the 5 degree limit of the oblique correction, so it can be undone in full.
_OBLIQUE_DEGREES: Final = 4.0
_RULE_INSET: Final = 24
_RULE_REACH: Final = 110
_CURL_DIP: Final = 26.0
_CURL_DECAY: Final = 0.42
_SPECK_COUNT: Final = 420
_OFFCUT_MAIN_WIDTH: Final = 720
_OFFCUT_GAP: Final = 22
_OFFCUT_STRIP_WIDTH: Final = 52
_OFFCUT_TEXT_REACH: Final = 10
_PDF_FONT_POINTS: Final = 9.0
SPREAD_SPINE_X: Final = BED_MARGIN + PAPER_WIDTH
OFFCUT_PAGE_RIGHT: Final = BED_MARGIN + _OFFCUT_MAIN_WIDTH
OFFCUT_STRIP_LEFT: Final = OFFCUT_PAGE_RIGHT + _OFFCUT_GAP
SHADOW_WIDTH: Final = 60
_SHADOW_DEPTH: Final = 0.13

_SENTENCES: Final = (
    "The mill stood where the river bent north, and its wheel turned through every "
    "season the valley could remember.",
    "Farmers brought grain down the lane in the early hours, before the frost had "
    "left the hedges.",
    "Inside the long shed the stones ground slowly, and the dust settled on every "
    "beam and sill.",
    "A clerk kept the accounts in a ledger bound in green cloth, entry by entry, "
    "without a single erasure.",
    "Some years the water ran high and the race flooded the lower yard, and the "
    "work stopped for a week.",
    "Visitors often asked why the building had two doors, and the answer was that "
    "carts and people never shared one.",
    "The sluice gates were older than the mill itself, and the millwright oiled "
    "their hinges each autumn.",
    "Nobody recorded when the first bridge was built, but its stone piers still "
    "carried the old road across.",
    "Children climbed the stacked sacks in the loft, and were sent down again before "
    "the miller came back from market.",
    "Prices for flour rose and fell with the harvest, and the ledger shows each "
    "change in the same neat hand.",
    "On still evenings the sound of the wheel carried up the valley, past the church "
    "and the last cottages.",
    "A later owner replaced the wooden gearing with iron, though he kept the old "
    "wheel for the sake of its look.",
    "The archive holds a plan of the site drawn in brown ink, with each sluice "
    "numbered and each field named.",
    "Walking the bank today, one can still find the shallow steps cut for the "
    "lightermen who loaded the barges.",
    "The road now carries lorries past the gate, and the mill is a house with a "
    "garden where the wheel once stood.",
    "Everything the valley produced once passed through that building on its way to "
    "the coast and the markets beyond.",
)
_HEADINGS: Final = (
    "Chapter One",
    "The River and the Wheel",
    "Work in the Mill",
    "A Season of Floods",
    "The Last Owner",
)


@dataclass(frozen=True)
class TextLine:
    """One line of text as it was set on a page.

    Attributes:
        text (str): The characters of the line, ASCII only.
        x (int): Left edge of the first character, in pixels from the image's left.
        baseline (int): Baseline of the line, in pixels from the image's top.
        scale (float): OpenCV font scale of the line.
        thickness (int): OpenCV stroke thickness of the line.
    """

    text: str
    x: int
    baseline: int
    scale: float
    thickness: int


@dataclass(frozen=True)
class SamplePage:
    """A rendered sample page and the text it carries.

    Attributes:
        name (str): File name stem; the prefix number gives the page order.
        image (NDArray[np.uint8]): (H, W, 3) RGB scan of the page.
        text_lines (tuple[TextLine, ...]): Text of the page in image pixels, used
            for the invisible text layer of the sample PDF.
    """

    name: str
    image: NDArray[np.uint8]
    text_lines: tuple[TextLine, ...]


@dataclass(frozen=True)
class SampleBook:
    """Files written by ``write_sample_book``.

    Attributes:
        pages (tuple[Path, ...]): The page images, in project order.
        pdf (Path): A PDF with a text layer built from ``PDF_PAGE_NAMES``.
    """

    pages: tuple[Path, ...]
    pdf: Path


@dataclass(frozen=True)
class _Sheet:
    """A sheet of paper drawn flat, before it is placed on the scanner bed.

    Attributes:
        pixels (NDArray[np.uint8]): (H, W, 3) RGB paper with its text and pictures.
        lines (tuple[TextLine, ...]): Text positions in sheet pixels.
    """

    pixels: NDArray[np.uint8]
    lines: tuple[TextLine, ...]


_PAGE_SIZES: Final[dict[str, tuple[int, int]]] = {
    "01_single_page": (874, 1240),
    "02_spread_light_gutter": (1668, 1240),
    "03_spread_black_gutter": (1668, 1240),
    "04_offcut": (874, 1240),
    "05_skewed": (874, 1240),
    "06_oblique": (874, 1240),
    "07_curved": (874, 1240),
    "08_pictures": (874, 1240),
    "09_specks": (874, 1240),
    "10_sideways": (1240, 874),
}
SAMPLE_PAGE_NAMES: Final = tuple(_PAGE_SIZES)
PDF_PAGE_NAMES: Final = ("01_single_page", "02_spread_light_gutter", "08_pictures")


def expected_size(name: str) -> tuple[int, int]:
    """Return the width and height in pixels of a sample page.

    Args:
        name (str): One of ``SAMPLE_PAGE_NAMES``.

    Returns:
        tuple[int, int]: Width and height of the rendered page image.
    """
    return _PAGE_SIZES[name]


def _rng_for(name: str, stage: int = 0) -> np.random.Generator:
    """Return a generator seeded from the page's place in the book.

    Args:
        name (str): One of ``SAMPLE_PAGE_NAMES``.
        stage (int): Separates the draws for text, specks and grain.

    Returns:
        np.random.Generator: A generator that gives the same numbers every run.
    """
    return np.random.default_rng([_SEED, SAMPLE_PAGE_NAMES.index(name), stage])


def _wrap(text: str, width: int) -> list[str]:
    """Split a paragraph into lines that fit ``width`` pixels in the body font.

    Args:
        text (str): Words separated by single spaces.
        width (int): The line width in pixels.

    Returns:
        list[str]: The wrapped lines, in order.
    """
    lines: list[str] = []
    current = ""
    for word in text.split():
        candidate = f"{current} {word}" if current else word
        ((candidate_width, _), _) = cv2.getTextSize(
            candidate, _BODY_FONT, _BODY_SCALE, 1
        )
        if candidate_width <= width or not current:
            current = candidate
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def _paragraph(rng: np.random.Generator, sentences: int) -> str:
    """Join randomly chosen sentences into one paragraph.

    Args:
        rng (np.random.Generator): Source of the choice.
        sentences (int): How many sentences to join; at most the number available.

    Returns:
        str: The paragraph.
    """
    picks = rng.choice(len(_SENTENCES), size=sentences, replace=False)
    return " ".join(_SENTENCES[int(index)] for index in picks)


def _draw_line(
    paper: NDArray[np.uint8],
    text: str,
    x: int,
    baseline: int,
    scale: float,
    thickness: int,
    font: int,
) -> TextLine:
    """Set one line of text on the paper and return where it landed."""
    cv2.putText(
        paper, text, (x, baseline), font, scale, _INK_RGB, thickness, cv2.LINE_AA
    )
    return TextLine(text, x, baseline, scale, thickness)


def _set_heading(
    paper: NDArray[np.uint8], text: str, top: int, lines: list[TextLine]
) -> int:
    """Set a chapter heading, centred on the paper, and return the next top."""
    ((width, _), _) = cv2.getTextSize(text, _HEADING_FONT, _HEADING_SCALE, 2)
    baseline = top + 52
    lines.append(
        _draw_line(
            paper,
            text,
            (PAPER_WIDTH - width) // 2,
            baseline,
            _HEADING_SCALE,
            2,
            _HEADING_FONT,
        )
    )
    return baseline + 40


def _set_body(
    paper: NDArray[np.uint8], paragraph: str, top: int, lines: list[TextLine]
) -> int:
    """Set a paragraph with an indented first line and return the next top."""
    wrapped = _wrap(paragraph, _TEXT_WIDTH - _INDENT)
    for number, text in enumerate(wrapped):
        indent = _INDENT if number == 0 else 0
        lines.append(
            _draw_line(
                paper,
                text,
                _TEXT_LEFT + indent,
                top + _LINE_PITCH * number,
                _BODY_SCALE,
                1,
                _BODY_FONT,
            )
        )
    return top + _LINE_PITCH * len(wrapped) + _PARAGRAPH_GAP


def _set_caption(
    paper: NDArray[np.uint8], text: str, top: int, lines: list[TextLine]
) -> int:
    """Set a centred caption whose box starts at ``top`` and return the next top."""
    ((width, _), _) = cv2.getTextSize(text, _BODY_FONT, _CAPTION_SCALE, 1)
    lines.append(
        _draw_line(
            paper,
            text,
            _TEXT_LEFT + (_TEXT_WIDTH - width) // 2,
            top + 24,
            _CAPTION_SCALE,
            1,
            _BODY_FONT,
        )
    )
    return top + 24 + _PARAGRAPH_GAP + 10


def _set_folio(paper: NDArray[np.uint8], number: int, lines: list[TextLine]) -> None:
    """Set the page number centred at the foot of the paper."""
    text = str(number)
    ((width, _), _) = cv2.getTextSize(text, _BODY_FONT, _CAPTION_SCALE, 1)
    lines.append(
        _draw_line(
            paper,
            text,
            (PAPER_WIDTH - width) // 2,
            PAPER_HEIGHT - 80,
            _CAPTION_SCALE,
            1,
            _BODY_FONT,
        )
    )


def _blank_paper() -> NDArray[np.uint8]:
    """Return an empty sheet of paper."""
    return np.full((PAPER_HEIGHT, PAPER_WIDTH, 3), _PAPER_RGB, np.uint8)


def _text_sheet(
    rng: np.random.Generator,
    heading: str | None,
    paragraphs: int,
    folio: int,
    sentences: int = 2,
) -> _Sheet:
    """Draw a sheet of running text, with an optional heading and a folio.

    Args:
        rng (np.random.Generator): Source of the text.
        heading (str | None): Chapter heading, or None for a page that continues.
        paragraphs (int): Number of paragraphs of body text.
        folio (int): Page number printed at the foot.
        sentences (int): Sentences per paragraph.

    Returns:
        _Sheet: The paper and its text.

    Raises:
        ValueError: When the text would run into the foot of the page.
    """
    paper = _blank_paper()
    lines: list[TextLine] = []
    top = _TEXT_TOP
    if heading is not None:
        top = _set_heading(paper, heading, top, lines)
    for _ in range(paragraphs):
        top = _set_body(paper, _paragraph(rng, sentences), top, lines)
    if top > PAPER_HEIGHT - _FOOT_CLEARANCE:
        raise ValueError(f"text ends at {top}px, below the foot of the page")
    _set_folio(paper, folio, lines)
    return _Sheet(paper, tuple(lines))


def _paint_landscape(
    paper: NDArray[np.uint8],
    left: int,
    top: int,
    width: int,
    height: int,
    rng: np.random.Generator,
) -> None:
    """Paint a photograph-like landscape: a sky gradient and a grained green hill."""
    fade = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None, None]
    sky = np.array([120, 160, 205], np.float32)
    horizon = np.array([225, 210, 180], np.float32)
    block = np.broadcast_to(sky * (1 - fade) + horizon * fade, (height, width, 3))
    block = block.copy()
    ridge = height * 0.55 + 40 * np.sin(np.linspace(0.0, 3.1, width))
    below_ridge = np.arange(height, dtype=np.float32)[:, None] > ridge[None, :]
    grain = rng.random((max(2, height // 10), max(2, width // 10))).astype(np.float32)
    grain = cv2.resize(grain, (width, height), interpolation=cv2.INTER_CUBIC)
    hill = np.array([72, 112, 62], np.float32) * (0.75 + 0.5 * grain[..., None])
    block[below_ridge] = hill[below_ridge]
    paper[top : top + height, left : left + width] = np.clip(block, 0, 255).astype(
        np.uint8
    )


def _place(
    canvas: NDArray[np.uint8], sheet: _Sheet, left: int, top: int
) -> list[TextLine]:
    """Paste a sheet onto a canvas and return its text in canvas pixels."""
    height, width = sheet.pixels.shape[:2]
    canvas[top : top + height, left : left + width] = sheet.pixels
    return [
        TextLine(
            line.text, line.x + left, line.baseline + top, line.scale, line.thickness
        )
        for line in sheet.lines
    ]


def _bed(width: int, height: int) -> NDArray[np.uint8]:
    """Return a scanner bed canvas of the given size."""
    return np.full((height, width, 3), _BED_RGB, np.uint8)


def _single_canvas(sheet: _Sheet) -> tuple[NDArray[np.uint8], tuple[TextLine, ...]]:
    """Centre one sheet on a scanner bed.

    Args:
        sheet (_Sheet): The paper to scan.

    Returns:
        tuple[NDArray[np.uint8], tuple[TextLine, ...]]: The scan and its text.
    """
    canvas = _bed(PAPER_WIDTH + 2 * BED_MARGIN, PAPER_HEIGHT + 2 * BED_MARGIN)
    return canvas, tuple(_place(canvas, sheet, BED_MARGIN, BED_MARGIN))


def _shade_spine(canvas: NDArray[np.uint8], seam: int, ramp: int, depth: float) -> None:
    """Darken the columns either side of the spine, in place.

    Args:
        canvas (NDArray[np.uint8]): The scan; changed in place.
        seam (int): Column of the spine.
        ramp (int): Half-width of the shadow, in pixels.
        depth (float): Brightness kept at the spine, from 0 (black) to 1 (none).
    """
    for offset in range(-ramp, ramp + 1):
        factor = depth + (1.0 - depth) * (abs(offset) / ramp) ** 1.5
        column = seam + offset
        canvas[:, column] = np.clip(
            canvas[:, column].astype(np.float32) * factor, 0, 255
        ).astype(np.uint8)


def _render_spread(
    name: str, black_gutter: bool
) -> tuple[NDArray[np.uint8], tuple[TextLine, ...]]:
    """Two facing pages on one bed with the spine between them.

    Args:
        name (str): The page name, which seeds its text.
        black_gutter (bool): True for a dark spine shadow, False for a light gutter.

    Returns:
        tuple[NDArray[np.uint8], tuple[TextLine, ...]]: The scan and its text.
    """
    rng = _rng_for(name)
    left = _text_sheet(rng, None, 4, 2)
    right = _text_sheet(rng, _HEADINGS[1], 3, 3)
    canvas = _bed(2 * PAPER_WIDTH + 2 * BED_MARGIN, PAPER_HEIGHT + 2 * BED_MARGIN)
    lines = _place(canvas, left, BED_MARGIN, BED_MARGIN)
    lines += _place(canvas, right, BED_MARGIN + PAPER_WIDTH, BED_MARGIN)
    seam = SPREAD_SPINE_X
    if black_gutter:
        _shade_spine(canvas, seam, ramp=70, depth=0.10)
    else:
        _shade_spine(canvas, seam, ramp=46, depth=0.86)
    return canvas, tuple(lines)


def _render_offcut() -> tuple[NDArray[np.uint8], tuple[TextLine, ...]]:
    """A page with a strip of the neighbouring page's text along its right edge.

    A dark gap separates the strip from the page, as when the neighbouring page is
    in view beside it on the scanner. The strip has no text layer of its own.

    Returns:
        tuple[NDArray[np.uint8], tuple[TextLine, ...]]: The scan and the text of the
        page itself.
    """
    name = "04_offcut"
    width, height = _PAGE_SIZES[name]
    main = _text_sheet(_rng_for(name), _HEADINGS[2], 3, 4)
    neighbour = _text_sheet(_rng_for(name, stage=1), None, 4, 5)
    canvas = _bed(width, height)
    lines = _place(canvas, main, BED_MARGIN, BED_MARGIN)
    main_right = OFFCUT_PAGE_RIGHT
    strip_left = OFFCUT_STRIP_LEFT
    strip_right = strip_left + _OFFCUT_STRIP_WIDTH
    canvas[:, main_right:strip_left] = _BED_RGB
    neighbour_left = strip_left + _OFFCUT_TEXT_REACH - _TEXT_LEFT
    strip = neighbour.pixels[
        :, strip_left - neighbour_left : strip_right - neighbour_left
    ]
    canvas[BED_MARGIN : BED_MARGIN + PAPER_HEIGHT, strip_left:strip_right] = strip
    canvas[:, strip_right:] = _BED_RGB
    return canvas, tuple(line for line in lines if line.x < main_right)


def _render_pictures() -> tuple[NDArray[np.uint8], tuple[TextLine, ...]]:
    """Running text with two landscape pictures, each with a caption.

    Returns:
        tuple[NDArray[np.uint8], tuple[TextLine, ...]]: The scan and its text.
    """
    rng = _rng_for("08_pictures")
    paper = _blank_paper()
    lines: list[TextLine] = []
    top = _set_heading(paper, _HEADINGS[3], _TEXT_TOP, lines)
    top = _set_body(paper, _paragraph(rng, 2), top, lines)
    _paint_landscape(paper, _TEXT_LEFT, top, _TEXT_WIDTH, 200, rng)
    top = _set_caption(paper, "The wheel pit in spring", top + 212, lines)
    top = _set_body(paper, _paragraph(rng, 2), top, lines)
    _paint_landscape(paper, _TEXT_LEFT, top, _TEXT_WIDTH, 130, rng)
    _set_caption(paper, "The race below the sluice", top + 142, lines)
    _set_folio(paper, 6, lines)
    return _single_canvas(_Sheet(paper, tuple(lines)))


def _add_specks(canvas: NDArray[np.uint8], rng: np.random.Generator) -> None:
    """Scatter dust specks over the scan, in place.

    Args:
        canvas (NDArray[np.uint8]): The scan; changed in place.
        rng (np.random.Generator): Source of the speck positions and sizes.
    """
    height, width = canvas.shape[:2]
    xs = rng.integers(BED_MARGIN, width - BED_MARGIN, _SPECK_COUNT)
    ys = rng.integers(BED_MARGIN, height - BED_MARGIN, _SPECK_COUNT)
    radii = rng.choice([1, 1, 1, 2, 2, 3], _SPECK_COUNT)
    for x, y, radius in zip(xs, ys, radii, strict=True):
        cv2.circle(canvas, (int(x), int(y)), int(radius), (30, 28, 26), -1)


def _curl(
    canvas: NDArray[np.uint8], lines: tuple[TextLine, ...]
) -> tuple[NDArray[np.uint8], tuple[TextLine, ...]]:
    """Bend the page as if photographed open, so its lines dip at the spine.

    The spine is the left edge of the image. Lines dip most there and flatten out
    towards the outer edge, and the page darkens as it turns away from the light.

    Args:
        canvas (NDArray[np.uint8]): The scan.
        lines (tuple[TextLine, ...]): The text of the scan.

    Returns:
        tuple[NDArray[np.uint8], tuple[TextLine, ...]]: The bent scan and its text.
    """
    height, width = canvas.shape[:2]
    scale = _CURL_DECAY * width
    columns = np.arange(width, dtype=np.float32)
    rows = np.arange(height, dtype=np.float32)[:, None]
    dip = (_CURL_DIP * np.exp(-columns / scale)).astype(np.float32)
    map_x = np.broadcast_to(columns[None, :], (height, width)).astype(np.float32)
    map_y = np.broadcast_to(rows - dip[None, :], (height, width)).astype(np.float32)
    bent = cv2.remap(
        canvas,
        map_x,
        map_y,
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=_BED_RGB,
    )
    light = 0.80 + 0.20 * np.clip(columns / (0.35 * width), 0.0, 1.0)
    shaded = np.clip(bent.astype(np.float32) * light[None, :, None], 0, 255)
    moved = tuple(
        TextLine(
            line.text,
            line.x,
            round(float(line.baseline + _CURL_DIP * np.exp(-line.x / scale))),
            line.scale,
            line.thickness,
        )
        for line in lines
    )
    return shaded.astype(np.uint8), moved


def _affine_line(line: TextLine, matrix: NDArray[np.float64]) -> TextLine:
    """Map the anchor point of a line through a 2x3 or 3x3 affine matrix."""
    point = matrix[:2, :2] @ np.array([line.x, line.baseline], float) + matrix[:2, 2]
    return TextLine(
        line.text,
        round(float(point[0])),
        round(float(point[1])),
        line.scale,
        line.thickness,
    )


def _skew(
    canvas: NDArray[np.uint8], lines: tuple[TextLine, ...], degrees: float
) -> tuple[NDArray[np.uint8], tuple[TextLine, ...]]:
    """Rotate the sheet about its centre, counter-clockwise for a positive angle.

    Args:
        canvas (NDArray[np.uint8]): The scan.
        lines (tuple[TextLine, ...]): The text of the scan.
        degrees (float): Rotation angle.

    Returns:
        tuple[NDArray[np.uint8], tuple[TextLine, ...]]: The rotated scan and its text.
    """
    height, width = canvas.shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), degrees, 1.0)
    rotated = cv2.warpAffine(canvas, matrix, (width, height), borderValue=_BED_RGB)
    return rotated, tuple(_affine_line(line, matrix) for line in lines)


def _rule_text_block(paper: NDArray[np.uint8]) -> None:
    """Draw a vertical rule down each side of the text block.

    Text alone gives the oblique detector too weak a vertical edge to find the
    shear: ragged line ends and the blank rows between lines drown it.

    Args:
        paper (NDArray[np.uint8]): The sheet to draw on, changed in place.
    """
    for x in (_TEXT_LEFT - _RULE_INSET, _TEXT_LEFT + _TEXT_WIDTH + _RULE_INSET):
        cv2.line(paper, (x, _RULE_REACH), (x, PAPER_HEIGHT - _RULE_REACH), _INK_RGB, 2)


def _oblique(
    canvas: NDArray[np.uint8], lines: tuple[TextLine, ...], degrees: float
) -> tuple[NDArray[np.uint8], tuple[TextLine, ...]]:
    """Shear the sheet so vertical strokes lean while text lines stay level.

    Args:
        canvas (NDArray[np.uint8]): The scan.
        lines (tuple[TextLine, ...]): The text of the scan.
        degrees (float): Lean of the vertical strokes, in degrees.

    Returns:
        tuple[NDArray[np.uint8], tuple[TextLine, ...]]: The sheared scan and its text.
    """
    height, width = canvas.shape[:2]
    slope = float(np.tan(np.radians(degrees)))
    matrix = np.array([[1.0, slope, -slope * height / 2], [0.0, 1.0, 0.0]])
    sheared = cv2.warpAffine(canvas, matrix, (width, height), borderValue=_BED_RGB)
    return sheared, tuple(_affine_line(line, matrix) for line in lines)


def _sideways(
    canvas: NDArray[np.uint8], lines: tuple[TextLine, ...]
) -> tuple[NDArray[np.uint8], tuple[TextLine, ...]]:
    """Turn the sheet a quarter turn clockwise, so the text runs down the page.

    Args:
        canvas (NDArray[np.uint8]): The scan.
        lines (tuple[TextLine, ...]): The text of the scan.

    Returns:
        tuple[NDArray[np.uint8], tuple[TextLine, ...]]: The turned scan and its text.
    """
    height = canvas.shape[0]
    turned = cv2.rotate(canvas, cv2.ROTATE_90_CLOCKWISE)
    moved = tuple(
        TextLine(line.text, height - line.baseline, line.x, line.scale, line.thickness)
        for line in lines
    )
    return turned, moved


def _render_scan(name: str) -> tuple[NDArray[np.uint8], tuple[TextLine, ...]]:
    """Render one page of the book before grain is added.

    Args:
        name (str): One of ``SAMPLE_PAGE_NAMES``.

    Returns:
        tuple[NDArray[np.uint8], tuple[TextLine, ...]]: The scan and its text.

    Raises:
        KeyError: When ``name`` is not a sample page.
    """
    if name == "02_spread_light_gutter":
        return _render_spread(name, black_gutter=False)
    if name == "03_spread_black_gutter":
        return _render_spread(name, black_gutter=True)
    if name == "04_offcut":
        return _render_offcut()
    if name == "08_pictures":
        return _render_pictures()
    headings = {
        "01_single_page": (_HEADINGS[0], 1),
        "05_skewed": (_HEADINGS[0], 5),
        "06_oblique": (_HEADINGS[0], 6),
        "07_curved": (_HEADINGS[4], 7),
        "09_specks": (_HEADINGS[2], 9),
        "10_sideways": (_HEADINGS[1], 10),
    }
    heading, folio = headings[name]
    sheet = _text_sheet(_rng_for(name), heading, 4, folio)
    if name == "06_oblique":
        _rule_text_block(sheet.pixels)
    canvas, lines = _single_canvas(sheet)
    if name == "05_skewed":
        return _skew(canvas, lines, _SKEW_DEGREES)
    if name == "06_oblique":
        return _oblique(canvas, lines, _OBLIQUE_DEGREES)
    if name == "07_curved":
        return _curl(canvas, lines)
    if name == "09_specks":
        _add_specks(canvas, _rng_for(name, stage=1))
        return canvas, lines
    if name == "10_sideways":
        return _sideways(canvas, lines)
    if name == "01_single_page":
        return canvas, lines
    raise KeyError(name)


def render_sample_page(name: str) -> SamplePage:
    """Render one page of the sample book.

    Args:
        name (str): One of ``SAMPLE_PAGE_NAMES``.

    Returns:
        SamplePage: The page image, RGB uint8 of ``expected_size(name)``.

    Raises:
        KeyError: When ``name`` is not a sample page.
    """
    canvas, lines = _render_scan(name)
    return SamplePage(name, canvas, lines)


_UNMOVED_SHEET_LEFTS: Final[dict[str, tuple[int, ...]]] = {
    "01_single_page": (BED_MARGIN,),
    "02_spread_light_gutter": (BED_MARGIN, SPREAD_SPINE_X),
    "03_spread_black_gutter": (BED_MARGIN, SPREAD_SPINE_X),
    "04_offcut": (BED_MARGIN,),
}


def text_block_columns(name: str) -> tuple[tuple[int, int], ...]:
    """Return the image columns each sheet's text block may reach, left to right.

    The neighbour's strip on ``04_offcut`` is not a sheet of its own and is left
    out.

    Args:
        name (str): A page whose sheets are placed flat and upright:
            ``01_single_page``, ``02_spread_light_gutter``,
            ``03_spread_black_gutter`` or ``04_offcut``.

    Returns:
        tuple[tuple[int, int], ...]: ``(left, right)`` of each text block. Lines
        end ragged, so ink stays inside these columns without filling them.

    Raises:
        KeyError: For any other page.
    """
    return tuple(
        (left + _TEXT_LEFT, left + _TEXT_LEFT + _TEXT_WIDTH)
        for left in _UNMOVED_SHEET_LEFTS[name]
    )


def render_spread_at(dpi: float, black_gutter: bool) -> tuple[NDArray[np.uint8], float]:
    """Render a sample spread resampled from 150 dpi to another resolution.

    Not part of the book: tests use it to check that results do not depend on the
    scan resolution.

    Args:
        dpi (float): Resolution of the returned image.
        black_gutter (bool): True for ``03_spread_black_gutter``, False for
            ``02_spread_light_gutter``.

    Returns:
        tuple[NDArray[np.uint8], float]: The (H, W, 3) RGB scan, and the column
        of its spine in the returned image's pixels.
    """
    name = "03_spread_black_gutter" if black_gutter else "02_spread_light_gutter"
    canvas = render_sample_page(name).image
    factor = dpi / SAMPLE_DPI
    if factor == 1.0:
        return canvas, float(SPREAD_SPINE_X)
    height, width = canvas.shape[:2]
    size = (round(width * factor), round(height * factor))
    interpolation = cv2.INTER_AREA if factor < 1.0 else cv2.INTER_CUBIC
    resized = np.asarray(cv2.resize(canvas, size, interpolation=interpolation))
    return resized.astype(np.uint8), SPREAD_SPINE_X * factor


def _plain_single_canvas() -> tuple[NDArray[np.uint8], tuple[TextLine, ...]]:
    """The scan of ``01_single_page``, ready for a defect to be applied.

    Returns:
        tuple[NDArray[np.uint8], tuple[TextLine, ...]]: The scan and its text.
    """
    sheet = _text_sheet(_rng_for("01_single_page"), _HEADINGS[0], 4, 1)
    return _single_canvas(sheet)


def render_skewed_page(degrees: float) -> NDArray[np.uint8]:
    """Render ``01_single_page`` turned about its centre by a chosen angle.

    Not part of the book: tests use it to check skew detection over a range of
    angles.

    Args:
        degrees (float): Rotation, counter-clockwise on screen for a positive angle.

    Returns:
        NDArray[np.uint8]: The (H, W, 3) RGB scan.
    """
    canvas, lines = _plain_single_canvas()
    return _skew(canvas, lines, degrees)[0]


def render_folio_only_page(folio: int) -> tuple[NDArray[np.uint8], TextLine]:
    """Render a sheet that holds nothing but its page number, at the usual place.

    Not part of the book: tests use it to check that Select Content keeps a lone
    page number.

    Args:
        folio (int): Page number printed at the foot.

    Returns:
        tuple[NDArray[np.uint8], TextLine]: The (H, W, 3) RGB scan and the page
        number in image pixels.
    """
    paper = _blank_paper()
    lines: list[TextLine] = []
    _set_folio(paper, folio, lines)
    canvas, placed = _single_canvas(_Sheet(paper, tuple(lines)))
    return canvas, placed[0]


def render_shadowed_edge_page(side: str) -> NDArray[np.uint8]:
    """Render ``01_single_page`` with a dark shadow along one side edge.

    The shadow is ``SHADOW_WIDTH`` pixels wide from the image edge, covering the
    scanner bed and the outer part of the paper margin but none of the text, as
    where a book's binding lifts the page off the glass.

    Not part of the book: tests use it to check how a shadowed edge is split.

    Args:
        side (str): ``"left"`` or ``"right"``.

    Returns:
        NDArray[np.uint8]: The (H, W, 3) RGB scan.

    Raises:
        ValueError: When ``side`` is neither.
    """
    canvas, _lines = _plain_single_canvas()
    if side == "left":
        band = canvas[:, :SHADOW_WIDTH]
    elif side == "right":
        band = canvas[:, -SHADOW_WIDTH:]
    else:
        raise ValueError(f"side must be 'left' or 'right', got {side!r}")
    band[...] = (band.astype(np.float32) * _SHADOW_DEPTH).astype(np.uint8)
    return canvas


def _pdf_string(text: str) -> str:
    """Escape text for a PDF literal string."""
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def write_sample_pdf(path: Path, pages: list[SamplePage]) -> Path:
    """Write sample pages as a PDF with an invisible text layer.

    Each page is its scan, drawn at 150 dpi, with the text of the page set in
    Helvetica in render mode 3 (invisible). Text search and extraction find it
    while the page looks like the scan.

    Args:
        path (Path): Where to write the PDF.
        pages (list[SamplePage]): The pages, in order.

    Returns:
        Path: The path written.
    """
    pdf = pikepdf.new()
    font = pdf.make_indirect(
        pikepdf.Dictionary(
            Type=pikepdf.Name.Font,
            Subtype=pikepdf.Name.Type1,
            BaseFont=pikepdf.Name.Helvetica,
            Encoding=pikepdf.Name.WinAnsiEncoding,
        )
    )
    to_points = 72.0 / SAMPLE_DPI
    for page in pages:
        height, width = page.image.shape[:2]
        width_pt, height_pt = width * to_points, height * to_points
        image = pikepdf.Stream(pdf, np.ascontiguousarray(page.image).tobytes())
        image["/Type"] = pikepdf.Name.XObject
        image["/Subtype"] = pikepdf.Name.Image
        image["/Width"] = width
        image["/Height"] = height
        image["/ColorSpace"] = pikepdf.Name.DeviceRGB
        image["/BitsPerComponent"] = 8
        operators = [f"q {width_pt:.3f} 0 0 {height_pt:.3f} 0 0 cm /Im0 Do Q"]
        for line in page.text_lines:
            x_pt = line.x * to_points
            y_pt = (height - line.baseline) * to_points
            operators.append(
                f"BT 3 Tr /F1 {_PDF_FONT_POINTS} Tf 1 0 0 1 {x_pt:.3f} {y_pt:.3f} Tm "
                f"({_pdf_string(line.text)}) Tj ET"
            )
        pdf_page = pdf.add_blank_page(page_size=(width_pt, height_pt))
        pdf_page.obj["/Resources"] = pikepdf.Dictionary(
            XObject=pikepdf.Dictionary(Im0=image),
            Font=pikepdf.Dictionary(F1=font),
        )
        pdf_page.obj["/Contents"] = pdf.make_stream(
            "\n".join(operators).encode("latin-1")
        )
    pdf.save(path, deterministic_id=True)
    return path


def write_sample_book(directory: Path) -> SampleBook:
    """Render every sample page and the sample PDF into a folder.

    Args:
        directory (Path): Folder to write into; created when missing.

    Returns:
        SampleBook: The page images in project order and the PDF.
    """
    directory.mkdir(parents=True, exist_ok=True)
    rendered = {name: render_sample_page(name) for name in SAMPLE_PAGE_NAMES}
    pages = []
    for name in SAMPLE_PAGE_NAMES:
        path = directory / f"{name}.png"
        Image.fromarray(rendered[name].image).save(path, dpi=(SAMPLE_DPI, SAMPLE_DPI))
        pages.append(path)
    pdf = write_sample_pdf(
        directory / "sample_book.pdf",
        [rendered[name] for name in PDF_PAGE_NAMES],
    )
    return SampleBook(tuple(pages), pdf)


def main(argv: list[str] | None = None) -> int:
    """Write the sample book from the command line.

    Args:
        argv (list[str] | None): Command line without the program name.

    Returns:
        int: Process exit code.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    book = write_sample_book(args.out)
    for path in (*book.pages, book.pdf):
        sys.stdout.write(f"{path}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
