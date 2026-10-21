"""Remember the main window's size, position and maximised state across runs."""

from collections.abc import Iterable
from typing import cast

from PySide6 import QtCore, QtGui, QtWidgets

MAXIMIZED_KEY = "mainWindow/maximized"
GEOMETRY_KEY = "mainWindow/nonMaximizedGeometry"
DEFAULT_SIZE = QtCore.QSize(1014, 689)
# The band at the top of the window that must stay on a screen, so the title
# bar can still be grabbed to drag the window back.
_GRAB_BAND_HEIGHT = 30
_MIN_GRAB_WIDTH = 100


def is_reachable(geometry: QtCore.QRect, screens: Iterable[QtCore.QRect]) -> bool:
    """Tell whether enough of the window's top edge lies on a screen to grab it.

    Args:
        geometry (QtCore.QRect): Window frame in global coordinates.
        screens (Iterable[QtCore.QRect]): Available geometry of each screen.

    Returns:
        bool: True when some screen shows a stretch of the top band at least
            ``_MIN_GRAB_WIDTH`` pixels wide.
    """
    band = QtCore.QRect(
        geometry.topLeft(), QtCore.QSize(geometry.width(), _GRAB_BAND_HEIGHT)
    )
    minimum_width = min(_MIN_GRAB_WIDTH, geometry.width())
    for screen in screens:
        shown = band.intersected(screen)
        if not shown.isEmpty() and shown.width() >= minimum_width:
            return True
    return False


def _available_screens() -> list[QtCore.QRect]:
    return [screen.availableGeometry() for screen in QtGui.QGuiApplication.screens()]


def _place_at_default(window: QtWidgets.QWidget) -> None:
    window.resize(DEFAULT_SIZE)
    primary = QtGui.QGuiApplication.primaryScreen()
    if primary is None:
        return
    available = primary.availableGeometry()
    frame = window.frameGeometry()
    frame.moveCenter(available.center())
    # A window larger than the screen keeps its top left corner, and so its
    # title bar and menus, on the screen.
    window.move(max(frame.left(), available.left()), max(frame.top(), available.top()))


def restore_window_geometry(
    window: QtWidgets.QWidget, settings: QtCore.QSettings
) -> bool:
    """Give the window the size and position it had when it last closed.

    A window with nothing saved, or whose saved place no current screen shows,
    gets the default size centred on the primary screen.

    Args:
        window (QtWidgets.QWidget): The main window, not shown yet.
        settings (QtCore.QSettings): Where the geometry was saved.

    Returns:
        bool: Whether the window was maximised when it closed, so the caller
            shows it maximised.
    """
    saved = cast(
        QtCore.QByteArray,
        settings.value(GEOMETRY_KEY, QtCore.QByteArray(), type=QtCore.QByteArray),
    )
    restored = not saved.isEmpty() and window.restoreGeometry(saved)
    if not restored or not is_reachable(window.frameGeometry(), _available_screens()):
        _place_at_default(window)
    return cast(bool, settings.value(MAXIMIZED_KEY, False, type=bool))


def save_window_geometry(window: QtWidgets.QWidget, settings: QtCore.QSettings) -> None:
    """Record the window's geometry and maximised state for the next start.

    ``saveGeometry`` keeps the normal geometry of a maximised window, so the
    window un-maximises to the size it had before. A window that was never
    shown has no geometry worth keeping and is skipped.

    Args:
        window (QtWidgets.QWidget): The main window, about to close.
        settings (QtCore.QSettings): Where to save the geometry.
    """
    if not window.isVisible():
        return
    settings.setValue(MAXIMIZED_KEY, window.isMaximized())
    settings.setValue(GEOMETRY_KEY, window.saveGeometry())
