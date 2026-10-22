"""Panel that takes the place of the page view when a page cannot be shown."""

import html
import pathlib

from PySide6 import QtCore, QtWidgets

_ICON_SIZE = 48


class ErrorWidget(QtWidgets.QWidget):
    """A warning icon and a message, centred in a box.

    Signals:
        link_activated: The user clicked a link in the message. Args: link (str),
            the link's href.
    """

    link_activated = QtCore.Signal(str)

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        """Build the panel with no message.

        Args:
            parent (QtWidgets.QWidget | None): Owner of the widget.
        """
        super().__init__(parent)
        self.text_label = QtWidgets.QLabel()
        self.text_label.setWordWrap(True)
        self.text_label.setTextInteractionFlags(
            QtCore.Qt.TextInteractionFlag.LinksAccessibleByMouse
        )
        self.text_label.linkActivated.connect(self.link_activated)
        icon_label = QtWidgets.QLabel()
        icon = self.style().standardIcon(
            QtWidgets.QStyle.StandardPixmap.SP_MessageBoxWarning
        )
        icon_label.setPixmap(icon.pixmap(_ICON_SIZE, _ICON_SIZE))

        box = QtWidgets.QGroupBox()
        box_layout = QtWidgets.QHBoxLayout(box)
        box_layout.addWidget(icon_label)
        box_layout.addWidget(self.text_label)

        layout = QtWidgets.QGridLayout(self)
        layout.setRowStretch(0, 1)
        layout.setRowStretch(2, 1)
        layout.setColumnStretch(0, 1)
        layout.setColumnStretch(2, 1)
        layout.addWidget(box, 1, 1)

    def set_message(self, text: str, text_format: QtCore.Qt.TextFormat) -> None:
        """Show a message.

        Args:
            text (str): The message, plain or rich text as ``text_format`` says.
            text_format (QtCore.Qt.TextFormat): How to read ``text``.
        """
        self.text_label.setTextFormat(text_format)
        self.text_label.setText(text)


def load_failure_message(file_path: pathlib.Path) -> tuple[str, QtCore.Qt.TextFormat]:
    """Describe why a page's source image cannot be shown, as upstream words it.

    A file that is gone gets a link to the Relinking tool; a file that exists
    but cannot be decoded does not, since relinking cannot help.

    Args:
        file_path (pathlib.Path): The source image that failed to load.

    Returns:
        tuple[str, QtCore.Qt.TextFormat]: The message and its text format.
    """
    native = QtCore.QDir.toNativeSeparators(str(file_path))
    if file_path.exists():
        text = QtCore.QCoreApplication.translate(
            "LoadFileTask", "The following file could not be loaded:\n%1"
        )
        return text.replace("%1", native), QtCore.Qt.TextFormat.PlainText
    text = QtCore.QCoreApplication.translate(
        "LoadFileTask",
        "The following file doesn't exist:<br>%1<br>"
        "<br>"
        'Use the <a href="#relink">Relinking Tool</a> to locate it.',
    )
    return text.replace("%1", html.escape(native)), QtCore.Qt.TextFormat.RichText
