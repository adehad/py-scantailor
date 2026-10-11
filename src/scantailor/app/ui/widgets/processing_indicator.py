"""Busy indicator for the status bar."""

from PySide6 import QtWidgets


class ProcessingIndicator(QtWidgets.QWidget):
    """A label with an endless progress bar, visible only while work runs."""

    def __init__(
        self, text: str = "Processing...", parent: QtWidgets.QWidget | None = None
    ) -> None:
        """Build the indicator, initially inactive.

        Args:
            text (str): Label shown beside the bar.
            parent (QtWidgets.QWidget | None): Owner of the widget.
        """
        super().__init__(parent)
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(QtWidgets.QLabel(text, self))
        bar = QtWidgets.QProgressBar(self)
        bar.setRange(0, 0)
        bar.setTextVisible(False)
        bar.setFixedWidth(90)
        layout.addWidget(bar)
        self._active = False
        self.hide()

    @property
    def is_active(self) -> bool:
        """Whether work is being indicated."""
        return self._active

    def set_active(self, active: bool) -> None:
        """Show or hide the indicator.

        Args:
            active (bool): True while work runs.
        """
        self._active = active
        self.setVisible(active)
