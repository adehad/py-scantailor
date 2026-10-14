"""Keeps spin box text clear of the field border."""

from PySide6 import QtCore, QtWidgets

from .stylesheet import FIELD_BORDER, FIELD_PADDING_X

_TEXT_INSET = FIELD_BORDER + FIELD_PADDING_X


class SpinBoxTextInset(QtCore.QObject):
    """Event filter that indents spin box text when Qt drops the QSS padding.

    Qt 6.12 lays out the editor of a style-sheeted spin box at the left edge of
    the field, ignoring its padding and border, so the digits touch the frame.
    Earlier Qt versions place the editor inside the padding. The margin is worked
    out from where the editor actually sits, so it adds nothing on a Qt that
    honours the padding.
    """

    def eventFilter(self, watched: QtCore.QObject, event: QtCore.QEvent) -> bool:
        """Watch each spin box editor and re-indent its text when it moves.

        Args:
            watched (QtCore.QObject): Object receiving the event.
            event (QtCore.QEvent): The event.

        Returns:
            bool: Always False; the event is never consumed.
        """
        kind = event.type()
        if kind == QtCore.QEvent.Type.Polish and isinstance(
            watched, QtWidgets.QAbstractSpinBox
        ):
            editor = watched.lineEdit()
            editor.installEventFilter(self)
            _indent(editor)
        elif kind in (QtCore.QEvent.Type.Move, QtCore.QEvent.Type.Resize) and (
            isinstance(watched, QtWidgets.QLineEdit)
            and isinstance(watched.parent(), QtWidgets.QAbstractSpinBox)
        ):
            _indent(watched)
        return False


def _indent(editor: QtWidgets.QLineEdit) -> None:
    """Set the editor's left text margin so the text starts at the field padding.

    Args:
        editor (QtWidgets.QLineEdit): The editor inside a spin box.
    """
    left = max(0, _TEXT_INSET - editor.x())
    margins = editor.textMargins()
    if margins.left() != left:
        margins.setLeft(left)
        editor.setTextMargins(margins)
