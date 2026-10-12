"""Keeps the options dock wide enough for the control sizes."""

from PySide6 import QtCore, QtWidgets

OPTIONS_DOCK_MIN_WIDTH = 340
_SIZED = "_scantailor_dock_sized"


class DockSizer(QtCore.QObject):
    """Event filter that gives non-thumbnail docks a minimum width on first Show."""

    def eventFilter(self, watched: QtCore.QObject, event: QtCore.QEvent) -> bool:
        """Raise the minimum width of an options dock the first time it shows.

        Args:
            watched (QtCore.QObject): Object receiving the event.
            event (QtCore.QEvent): The event.

        Returns:
            bool: Always False; the event is never consumed.
        """
        if (
            event.type() == QtCore.QEvent.Type.Show
            and isinstance(watched, QtWidgets.QDockWidget)
            and not watched.property(_SIZED)
        ):
            watched.setProperty(_SIZED, True)
            if watched.objectName() != "thumbnailsDockWidget":
                watched.setMinimumWidth(OPTIONS_DOCK_MIN_WIDTH)
        return False
