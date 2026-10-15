"""The bundled UI typeface, sized from the platform UI font so user text scaling counts.

Every size is a ratio of the application font, so raising it in the operating
system or on the application grows the whole interface together.
"""

import pathlib

from loguru import logger
from PySide6 import QtCore, QtGui, QtWidgets

from scantailor.app._ui import resources_rc

assert resources_rc, "The import ensures the Inter files are registered with Qt."

FAMILIES = ["Inter", "Noto Sans KR"]
SEMIBOLD = 600
# Inter sets visibly smaller than Segoe UI or San Francisco at the same nominal
# size, so the body text is a step above the platform size. The small size is
# body / BODY_RATIO, which puts it back at the platform size.
BODY_RATIO = 13 / 12
SMALL_RATIO = 1 / BODY_RATIO
PANEL_HEADING_RATIO = 1.25
DISPLAY_RATIO = 1.4

_INTER_FILES = (
    ":/fonts/inter/Inter-Regular.ttf",
    ":/fonts/inter/Inter-Medium.ttf",
    ":/fonts/inter/Inter-SemiBold.ttf",
)
# Hangul needs several MB per weight, too large to embed in the resource module,
# so these two files ship next to it as plain package data.
_KOREAN_FOLDER = pathlib.Path(__file__).resolve().parents[3] / "resources" / "fonts"
_KOREAN_FILES = (
    "noto-sans-kr/NotoSansKR-Regular.otf",
    "noto-sans-kr/NotoSansKR-Bold.otf",
)
_TABULAR_FIGURES = "tnum"


def load_fonts() -> list[str]:
    """Register the bundled fonts with Qt.

    Returns:
        list[str]: Families made available, in FAMILIES order.
    """
    sources = [*_INTER_FILES, *(str(_KOREAN_FOLDER / name) for name in _KOREAN_FILES)]
    loaded: list[str] = []
    for source in sources:
        font_id = QtGui.QFontDatabase.addApplicationFont(source)
        if font_id < 0:
            logger.warning("Could not load font {}", source)
            continue
        for family in QtGui.QFontDatabase.applicationFontFamilies(font_id):
            if family not in loaded:
                loaded.append(family)
    return loaded


def scaled(font: QtGui.QFont, ratio: float) -> QtGui.QFont:
    """Return a copy of a font at a multiple of its size, in the unit it was set in.

    Args:
        font (QtGui.QFont): Font to copy; usually the widget or application font.
        ratio (float): Size multiplier, for example SMALL_RATIO.

    Returns:
        QtGui.QFont: The resized copy.
    """
    copy = QtGui.QFont(font)
    if font.pointSizeF() > 0:
        copy.setPointSizeF(font.pointSizeF() * ratio)
    else:
        copy.setPixelSize(max(1, round(font.pixelSize() * ratio)))
    return copy


def small(font: QtGui.QFont) -> QtGui.QFont:
    """Return the small text size for captions, hints and overlay labels.

    Args:
        font (QtGui.QFont): Body font to derive from, usually ``widget.font()``.

    Returns:
        QtGui.QFont: A copy at the small size.
    """
    return scaled(font, SMALL_RATIO)


def tabular(font: QtGui.QFont) -> QtGui.QFont:
    """Return a copy of a font with fixed-width digits, for numbers that line up.

    Args:
        font (QtGui.QFont): Font to copy.

    Returns:
        QtGui.QFont: The copy with the OpenType tnum feature on.
    """
    copy = QtGui.QFont(font)
    copy.setFeature(QtGui.QFont.Tag(_TABULAR_FIGURES), 1)
    return copy


def pixel_size(font: QtGui.QFont) -> float:
    """Return a font's size in logical pixels, whichever unit it was set in.

    Args:
        font (QtGui.QFont): Font to measure.

    Returns:
        float: Size in logical pixels.
    """
    if font.pixelSize() > 0:
        return float(font.pixelSize())
    screen = QtGui.QGuiApplication.primaryScreen()
    dpi = screen.logicalDotsPerInchY() if screen is not None else 96.0
    return font.pointSizeF() * dpi / 72


def apply_fonts(app: QtWidgets.QApplication) -> None:
    """Load the bundled fonts and make them the application font.

    The size follows the platform UI font, so the operating system text size
    setting carries over.

    Args:
        app (QtWidgets.QApplication): The running application.
    """
    load_fonts()
    system = QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.SystemFont.GeneralFont)
    font = scaled(system, BODY_RATIO)
    font.setFamilies(FAMILIES)
    font.setStyleHint(QtGui.QFont.StyleHint.SansSerif)
    app.setFont(font)


class TabularFigures(QtCore.QObject):
    """Event filter that gives spin boxes fixed-width digits so values do not shift."""

    def eventFilter(self, watched: QtCore.QObject, event: QtCore.QEvent) -> bool:
        """Enable tabular figures on a spin box the first time it is polished.

        Args:
            watched (QtCore.QObject): Object receiving the event.
            event (QtCore.QEvent): The event.

        Returns:
            bool: Always False; the event is never consumed.
        """
        if event.type() == QtCore.QEvent.Type.Polish and isinstance(
            watched, QtWidgets.QAbstractSpinBox
        ):
            font = watched.font()
            tag = QtGui.QFont.Tag(_TABULAR_FIGURES)
            if not font.isFeatureSet(tag):
                font.setFeature(tag, 1)
                watched.setFont(font)
        return False
