"""Generated pages of printed text in both polarities."""

import cv2
import numpy as np
from numpy.typing import NDArray

_WORDS = (
    "the",
    "page",
    "light",
    "dark",
    "scanner",
    "content",
    "margin",
    "text",
    "line",
    "paper",
    "binding",
    "chapter",
)
_LINE_PITCH = 44
_FONT_SCALE = 1.0
_STROKE = 2
TEXT_MARGIN = 120


def printed_page(
    width: int = 1100, height: int = 1500, angle: float = 0.0, seed: int = 0
) -> NDArray[np.uint8]:
    """Return a gray page of printed lines, black on white, about a tenth ink.

    Args:
        width (int): Page width in pixels.
        height (int): Page height in pixels.
        angle (float): Counter-clockwise rotation of the text in degrees.
        seed (int): Picks the words.

    Returns:
        NDArray[np.uint8]: (height, width) page.
    """
    rng = np.random.default_rng(seed)
    page = np.full((height, width), 255, dtype=np.uint8)
    for baseline in range(TEXT_MARGIN + 30, height - TEXT_MARGIN, _LINE_PITCH):
        cv2.putText(
            page,
            _line_of_words(rng, width - 2 * TEXT_MARGIN),
            (TEXT_MARGIN, baseline),
            cv2.FONT_HERSHEY_SIMPLEX,
            _FONT_SCALE,
            0,
            _STROKE,
            cv2.LINE_AA,
        )
    if angle:
        turn = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
        page = np.asarray(
            cv2.warpAffine(page, turn, (width, height), borderValue=255),
            dtype=np.uint8,
        )
    return page


def _line_of_words(rng: np.random.Generator, room: int) -> str:
    """Return random words that fit in ``room`` pixels when printed."""
    line = str(rng.choice(_WORDS))
    while True:
        longer = f"{line} {rng.choice(_WORDS)}"
        (width, _), _ = cv2.getTextSize(
            longer, cv2.FONT_HERSHEY_SIMPLEX, _FONT_SCALE, _STROKE
        )
        if width > room:
            return line
        line = longer


def dark_page(
    width: int = 1100, height: int = 1500, angle: float = 0.0
) -> NDArray[np.uint8]:
    """Return ``printed_page`` inverted: light text on a black page."""
    return 255 - printed_page(width, height, angle)


def dark_sheet_on_light_bed(
    angle: float = 0.0, bed_margin: int = 150, bed_level: int = 225
) -> NDArray[np.uint8]:
    """Return a ``dark_page`` lying on a light scanner bed.

    Args:
        angle (float): Counter-clockwise rotation of the text in degrees.
        bed_margin (int): Width of the bed showing around the sheet.
        bed_level (int): Gray level of the bed.

    Returns:
        NDArray[np.uint8]: The scan; the sheet starts at ``bed_margin`` on
            both axes and is 1100 by 1500 pixels.
    """
    sheet = dark_page(angle=angle)
    height, width = sheet.shape
    scan = np.full(
        (height + 2 * bed_margin, width + 2 * bed_margin), bed_level, dtype=np.uint8
    )
    scan[bed_margin : bed_margin + height, bed_margin : bed_margin + width] = sheet
    return scan
