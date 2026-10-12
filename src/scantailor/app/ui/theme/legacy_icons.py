"""Recolours old black bitmap-style icons so they stay legible on dark panels.

Icons made with ``scantailor.app.ui.icons.icon`` recolour themselves and are left
alone; this only covers resource icons that are still plain black artwork.
"""

from PySide6 import QtCore, QtGui, QtWidgets

from scantailor.app.ui.icons import is_palette_icon

_SOURCE = "_scantailor_icon_source"
_DISABLED_ALPHA = 0.4


def tint_icon(
    icon: QtGui.QIcon, color: QtGui.QColor, size: QtCore.QSize
) -> QtGui.QIcon:
    """Return a copy of an icon with every opaque pixel painted in one colour.

    Args:
        icon (QtGui.QIcon): Source icon.
        color (QtGui.QColor): Replacement colour; alpha of the source is kept.
        size (QtCore.QSize): Pixmap size to render.

    Returns:
        QtGui.QIcon: Tinted icon with normal and disabled pixmaps.
    """
    tinted = QtGui.QIcon()
    for mode in (QtGui.QIcon.Mode.Normal, QtGui.QIcon.Mode.Disabled):
        for state in (QtGui.QIcon.State.Off, QtGui.QIcon.State.On):
            pixmap = icon.pixmap(size, mode, state)
            if pixmap.isNull():
                continue
            painter = QtGui.QPainter(pixmap)
            painter.setCompositionMode(
                QtGui.QPainter.CompositionMode.CompositionMode_SourceIn
            )
            fill = QtGui.QColor(color)
            if mode == QtGui.QIcon.Mode.Disabled:
                fill.setAlphaF(_DISABLED_ALPHA)
            painter.fillRect(pixmap.rect(), fill)
            painter.end()
            tinted.addPixmap(pixmap, mode, state)
    return tinted


def _retint(button: QtWidgets.QToolButton, source: QtGui.QIcon) -> None:
    size = button.iconSize()
    if not size.isValid() or size.isEmpty():
        size = QtCore.QSize(16, 16)
    color = QtGui.QGuiApplication.palette().color(QtGui.QPalette.ColorRole.ButtonText)
    button.setIcon(tint_icon(source, color, size))


class IconTinter(QtCore.QObject):
    """Event filter that tints tool button icons the first time they are polished."""

    def eventFilter(self, watched: QtCore.QObject, event: QtCore.QEvent) -> bool:
        """Tint a QToolButton's icon on its first Polish event.

        Args:
            watched (QtCore.QObject): Object receiving the event.
            event (QtCore.QEvent): The event.

        Returns:
            bool: Always False; the event is never consumed.
        """
        if (
            event.type() == QtCore.QEvent.Type.Polish
            and isinstance(watched, QtWidgets.QToolButton)
            and watched.property(_SOURCE) is None
        ):
            icon = watched.icon()
            if not icon.isNull() and not is_palette_icon(icon):
                watched.setProperty(_SOURCE, icon)
                _retint(watched, icon)
        return False

    @staticmethod
    def retint_all(app: QtWidgets.QApplication) -> None:
        """Repaint every tinted button after the palette changed.

        Args:
            app (QtWidgets.QApplication): The running application.
        """
        for widget in app.allWidgets():
            if isinstance(widget, QtWidgets.QToolButton):
                source = widget.property(_SOURCE)
                if isinstance(source, QtGui.QIcon):
                    _retint(widget, source)
