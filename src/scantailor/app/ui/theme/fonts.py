"""The bundled UI typeface, applied identically on every platform."""

import pathlib

from loguru import logger
from PySide6 import QtCore, QtGui, QtWidgets

from scantailor.app._ui import resources_rc

assert resources_rc, "The import ensures the Inter files are registered with Qt."

FAMILIES = ["Inter", "Noto Sans KR"]
BASE_PX = 13
CAPTION_PX = 11
HEADING_PX = 15
SEMIBOLD = 600

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


def apply_fonts(app: QtWidgets.QApplication) -> None:
    """Load the bundled fonts and make them the application font.

    Sizes are pixels because point sizes scale differently per platform.

    Args:
        app (QtWidgets.QApplication): The running application.
    """
    load_fonts()
    font = app.font()
    font.setFamilies(FAMILIES)
    font.setStyleHint(QtGui.QFont.StyleHint.SansSerif)
    font.setPixelSize(BASE_PX)
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
