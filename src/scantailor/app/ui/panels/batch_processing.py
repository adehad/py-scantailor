"""Batch processing lower panel for batch operation controls."""

from pathlib import Path

from PySide6 import QtWidgets
from PySide6.QtCore import Signal

from scantailor.app.ui.panels.system_load import SystemLoadWidget

_UI_FOLDER = Path(__file__).parent


class BatchProcessingLowerPanel(QtWidgets.QWidget):
    """Panel displayed during batch processing operations.

    Contains:
    - System load control (CPU core usage slider)
    - "Beep when finished" checkbox

    Signals:
        load_changed: Emitted when system load changes.
            Args: load_value (int)
        beep_setting_changed: Emitted when beep checkbox changes.
            Args: enabled (bool)
    """

    load_changed = Signal(int)
    beep_setting_changed = Signal(bool)

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        """Initialize the batch processing panel.

        Args:
            parent: Parent widget.
        """
        super().__init__(parent)

        # Load UI - note: the UI references SystemLoadWidget as a custom widget
        # We need to manually create it since QUiLoader won't know about it
        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self) -> None:
        """Set up the UI manually since we have a custom widget."""
        # Create main layout
        layout = QtWidgets.QGridLayout(self)

        # Create system load widget
        self._system_load_widget = SystemLoadWidget(parent=self)
        layout.addWidget(self._system_load_widget, 0, 0, 1, 4)

        # Add spacers and checkbox
        layout.addItem(
            QtWidgets.QSpacerItem(
                1,
                1,
                QtWidgets.QSizePolicy.Policy.Expanding,
                QtWidgets.QSizePolicy.Policy.Minimum,
            ),
            1,
            0,
        )

        self._beep_checkbox = QtWidgets.QCheckBox("Beep when finished", self)
        layout.addWidget(self._beep_checkbox, 1, 1)

        layout.addItem(
            QtWidgets.QSpacerItem(
                1,
                1,
                QtWidgets.QSizePolicy.Policy.Expanding,
                QtWidgets.QSizePolicy.Policy.Minimum,
            ),
            1,
            2,
        )

    def _connect_signals(self) -> None:
        """Connect widget signals."""
        self._system_load_widget.load_changed.connect(self.load_changed.emit)
        self._beep_checkbox.toggled.connect(self.beep_setting_changed.emit)

    def get_load(self) -> int:
        """Get the current system load value.

        Returns:
            Current load value (number of threads to use).
        """
        return self._system_load_widget.get_load()

    def set_load(self, value: int) -> None:
        """Set the system load value.

        Args:
            value: Load value to set.
        """
        self._system_load_widget.set_load(value)

    def is_beep_enabled(self) -> bool:
        """Check if beep on finish is enabled.

        Returns:
            True if beep is enabled.
        """
        return self._beep_checkbox.isChecked()

    def set_beep_enabled(self, enabled: bool) -> None:
        """Set the beep on finish setting.

        Args:
            enabled: Whether to beep when finished.
        """
        self._beep_checkbox.setChecked(enabled)
