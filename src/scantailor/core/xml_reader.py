"""Readers for the building blocks of the C++ ScanTailor project format.

The C++ reader is lenient: a missing attribute reads as zero or false. These
helpers keep that behaviour for absent data, but treat a value that is present
and unreadable as malformed, so one broken element can be skipped and reported
without losing the rest of the project.
"""

import base64
import codecs
import locale
import math
import re
import struct
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from xml.etree import ElementTree as ET

from loguru import logger

from scantailor.core.models import ImageId, PageId, SubPage
from scantailor.core.project import StageParamsEntry

Point = tuple[float, float]
Line = tuple[Point, Point]
Rect = tuple[float, float, float, float]

_FLOAT32_PAIR = struct.Struct("<ff")
_DECLARED_ENCODING = re.compile(
    rb"""\A\s*<\?xml[^>]*?\sencoding\s*=\s*["']([A-Za-z0-9._:-]+)["']"""
)


class XmlFormatError(ValueError):
    """An element of a C++ project file holds a value that cannot be read."""


def _ansi_codepage() -> str:
    """Name the codepage a Qt5 build on this machine writes non-Unicode text in."""
    if sys.platform == "win32":
        return locale.getencoding()
    return "cp1252"


def _declared_encoding(raw: bytes) -> str | None:
    """Return the codec named by an XML declaration, or None if absent or unknown."""
    match = _DECLARED_ENCODING.match(raw)
    if match is None:
        return None
    name = match.group(1).decode("ascii")
    try:
        return codecs.lookup(name).name
    except LookupError:
        logger.warning("The project file declares an unknown encoding {!r}", name)
        return None


def decode_project_bytes(raw: bytes) -> str:
    """Decode the bytes of a project file without ever failing.

    The encoding named in an XML declaration is used when there is one,
    otherwise UTF-8. Qt5 on Windows wrote paths in the locale codepage, so bytes
    that are not valid in that encoding are read as the Windows codepage
    instead, and as Windows-1252 with replacement characters as a last resort.

    Args:
        raw (bytes): The file content.

    Returns:
        str: The text, without a byte order mark.
    """
    if raw.startswith(codecs.BOM_UTF8):
        raw = raw[len(codecs.BOM_UTF8) :]
    expected = _declared_encoding(raw) or "utf-8"
    try:
        return raw.decode(expected)
    except UnicodeDecodeError:
        pass
    fallback = _ansi_codepage()
    logger.warning(
        "The project file is not valid {}; reading it as {}", expected, fallback
    )
    try:
        return raw.decode(fallback)
    except UnicodeDecodeError, LookupError:
        return raw.decode("cp1252", errors="replace")


@dataclass
class XmlProjectContext:
    """Lookup tables from the numeric ids a C++ project file uses.

    Attributes:
        image_ids (dict[int, ImageId]): Image id of each ``<image>`` element.
        page_ids (dict[int, PageId]): Page id of each ``<page>`` element.
    """

    image_ids: dict[int, ImageId] = field(default_factory=dict)
    page_ids: dict[int, PageId] = field(default_factory=dict)


def child(element: ET.Element | None, name: str) -> ET.Element | None:
    """Return the first direct child called ``name``, or None.

    Args:
        element (ET.Element | None): The parent; None is allowed and has no children.
        name (str): Tag name to look for.

    Returns:
        ET.Element | None: The child element.
    """
    if element is None:
        return None
    return element.find(name)


def children(element: ET.Element | None, name: str) -> Iterator[ET.Element]:
    """Yield the direct children called ``name``.

    Args:
        element (ET.Element | None): The parent; None yields nothing.
        name (str): Tag name to look for.
    """
    if element is not None:
        yield from element.findall(name)


def attr_text(element: ET.Element | None, name: str) -> str:
    """Return an attribute as text, empty when the attribute or element is absent."""
    if element is None:
        return ""
    return element.get(name, "")


def attr_int(element: ET.Element | None, name: str, default: int = 0) -> int:
    """Read an integer attribute.

    Args:
        element (ET.Element | None): The element; None reads as ``default``.
        name (str): Attribute name.
        default (int): Value for an absent attribute.

    Returns:
        int: The value.

    Raises:
        XmlFormatError: If the attribute is present but not an integer.
    """
    text = attr_text(element, name).strip()
    if not text:
        return default
    try:
        return int(text)
    except ValueError as error:
        msg = f"attribute {name}={text!r} is not an integer"
        raise XmlFormatError(msg) from error


def attr_float(element: ET.Element | None, name: str, default: float = 0.0) -> float:
    """Read a floating point attribute.

    Args:
        element (ET.Element | None): The element; None reads as ``default``.
        name (str): Attribute name.
        default (float): Value for an absent attribute.

    Returns:
        float: The value.

    Raises:
        XmlFormatError: If the attribute is present but not a finite number.
    """
    text = attr_text(element, name).strip()
    if not text:
        return default
    return parse_float(text, f"attribute {name}")


def parse_float(text: str, what: str) -> float:
    """Parse a finite floating point number.

    Args:
        text (str): The text to parse.
        what (str): Description of the value, for the error message.

    Returns:
        float: The number.

    Raises:
        XmlFormatError: If the text is not a finite number.
    """
    try:
        value = float(text)
    except ValueError as error:
        msg = f"{what}={text!r} is not a number"
        raise XmlFormatError(msg) from error
    if not math.isfinite(value):
        msg = f"{what}={text!r} is not finite"
        raise XmlFormatError(msg)
    return value


def attr_flag(element: ET.Element | None, name: str, *, default: bool = False) -> bool:
    """Read a ``"1"``/``"0"`` attribute the way the C++ reader does.

    Args:
        element (ET.Element | None): The element.
        name (str): Attribute name.
        default (bool): Value when the attribute is absent.

    Returns:
        bool: True only for the exact text ``1`` (or ``default`` when absent).
    """
    if element is None or name not in element.attrib:
        return default
    return element.attrib[name] == "1"


def read_text(element: ET.Element | None) -> str:
    """Return the text content of an element, stripped; empty when absent."""
    if element is None or element.text is None:
        return ""
    return element.text.strip()


def read_size(element: ET.Element | None) -> tuple[int, int]:
    """Read a ``width``/``height`` integer pair; (0, 0) when absent."""
    return attr_int(element, "width"), attr_int(element, "height")


def read_size_f(element: ET.Element | None) -> tuple[float, float]:
    """Read a ``width``/``height`` pair of floats; (0, 0) when absent."""
    return attr_float(element, "width"), attr_float(element, "height")


def read_point(element: ET.Element | None) -> Point:
    """Read an ``x``/``y`` pair of floats; (0, 0) when absent."""
    return attr_float(element, "x"), attr_float(element, "y")


def read_line(element: ET.Element | None) -> Line:
    """Read a line from its ``p1`` and ``p2`` children."""
    return read_point(child(element, "p1")), read_point(child(element, "p2"))


def read_rect_f(element: ET.Element | None) -> Rect:
    """Read an ``x``, ``y``, ``width``, ``height`` rectangle of floats."""
    x, y = read_point(element)
    width, height = read_size_f(element)
    return x, y, width, height


def read_polygon(element: ET.Element | None) -> tuple[Point, ...]:
    """Read the ``<point>`` children of an element as a polygon."""
    return tuple(read_point(point) for point in children(element, "point"))


def read_float32_polyline(element: ET.Element | None) -> tuple[Point, ...]:
    """Read a polyline stored as base64 of little-endian float32 pairs.

    Args:
        element (ET.Element | None): Element whose text holds the data.

    Returns:
        tuple[Point, ...]: The points; empty when absent. A trailing partial
            pair is ignored, as in the C++ reader.

    Raises:
        XmlFormatError: If the text is not valid base64.
    """
    text = read_text(element)
    if not text:
        return ()
    try:
        raw = base64.b64decode(text, validate=True)
    except ValueError as error:
        msg = "polyline is not valid base64"
        raise XmlFormatError(msg) from error
    count = len(raw) // _FLOAT32_PAIR.size
    return tuple(
        _FLOAT32_PAIR.unpack_from(raw, index * _FLOAT32_PAIR.size)
        for index in range(count)
    )


@contextmanager
def skip_malformed(what: str, element: ET.Element | None = None) -> Iterator[None]:
    """Log and swallow a :class:`ValueError` raised while reading one element.

    Args:
        what (str): Description of what is being read, for the log message.
        element (ET.Element | None): The element being read, shown by its id.
    """
    try:
        yield
    except ValueError as error:
        element_id = "" if element is None else element.get("id", "")
        logger.warning(
            "Skipping malformed {} (id {!r}) in the project file: {}",
            what,
            element_id,
            error,
        )


def page_elements(
    filter_element: ET.Element | None, context: XmlProjectContext
) -> Iterator[tuple[PageId, ET.Element]]:
    """Yield the ``<page>`` children of a filter element with their page ids.

    Elements whose id is not a number, or names no known page, are passed over
    as the C++ reader does.

    Args:
        filter_element (ET.Element | None): The filter's element.
        context (XmlProjectContext): Page id lookup of the project.
    """
    for element in children(filter_element, "page"):
        try:
            numeric_id = int(element.get("id", ""))
        except ValueError:
            continue
        page_id = context.page_ids.get(numeric_id)
        if page_id is None:
            logger.debug("Ignoring settings for unknown page id {}", numeric_id)
            continue
        yield page_id, element


def image_elements(
    filter_element: ET.Element | None, context: XmlProjectContext
) -> Iterator[tuple[ImageId, ET.Element]]:
    """Yield the ``<image>`` children of a filter element with their image ids.

    Args:
        filter_element (ET.Element | None): The filter's element.
        context (XmlProjectContext): Image id lookup of the project.
    """
    for element in children(filter_element, "image"):
        try:
            numeric_id = int(element.get("id", ""))
        except ValueError:
            continue
        image_id = context.image_ids.get(numeric_id)
        if image_id is None:
            logger.debug("Ignoring settings for unknown image id {}", numeric_id)
            continue
        yield image_id, element


def sub_page_from_string(text: str) -> SubPage:
    """Parse the ``subPage`` attribute of a ``<page>`` element.

    Args:
        text (str): ``single``, ``left`` or ``right``.

    Returns:
        SubPage: The sub page.

    Raises:
        XmlFormatError: For any other text, which the C++ reader skips too.
    """
    match text:
        case "single":
            return SubPage.SINGLE_PAGE
        case "left":
            return SubPage.LEFT_PAGE
        case "right":
            return SubPage.RIGHT_PAGE
    msg = f"unknown subPage {text!r}"
    raise XmlFormatError(msg)


def entry_for_image(image_id: ImageId, params: dict[str, object]) -> StageParamsEntry:
    """Build a stage-params entry for a stage that stores parameters per image."""
    return StageParamsEntry(page=PageId(image_id=image_id), params=params)
