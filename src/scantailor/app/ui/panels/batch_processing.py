"""Batch processing lower panel for batch operation controls."""

from PySide6 import QtWidgets
from PySide6.QtCore import Signal

from scantailor.app.ui.panels.system_load import SystemLoadWidget


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
        layout = QtWidgets.QGridLayout(self)

        # SystemLoadWidget loads its own form, and a QUiLoader that instantiates it
        # invalidates the Python wrappers it holds, so it is created here instead.
        self._system_load_widget = SystemLoadWidget(parent=self)
        self._system_load_widget.setObjectName("systemLoadWidget")
        layout.addWidget(self._system_load_widget, 0, 0, 1, 4)

        # Spacers on both sides centre the checkbox.
        self._add_horizontal_spacer(layout, column=0)
        self._beep_checkbox = QtWidgets.QCheckBox(self.tr("Beep when finished"), self)
        self._beep_checkbox.setObjectName("beepWhenFinished")
        layout.addWidget(self._beep_checkbox, 1, 1)
        self._add_horizontal_spacer(layout, column=2)

        self._system_load_widget.load_changed.connect(self.load_changed.emit)
        self._beep_checkbox.toggled.connect(self.beep_setting_changed.emit)

    @staticmethod
    def _add_horizontal_spacer(layout: QtWidgets.QGridLayout, column: int) -> None:
        """Add an expanding spacer to the checkbox row.

        Args:
            layout (QtWidgets.QGridLayout): The panel layout.
            column (int): Grid column of the spacer.
        """
        layout.addItem(
            QtWidgets.QSpacerItem(
                1,
                1,
                QtWidgets.QSizePolicy.Policy.Expanding,
                QtWidgets.QSizePolicy.Policy.Minimum,
            ),
            1,
            column,
        )

    def get_load(self) -> int:
        """Get the current system load value.

        Returns:
            int: Number of pages processed at once.
        """
        return self._system_load_widget.get_load()

    def set_load(self, value: int) -> None:
        """Set the system load value.

        Args:
            value (int): Number of pages to process at once.
        """
        self._system_load_widget.set_load(value)

    def is_beep_enabled(self) -> bool:
        """Check if beep on finish is enabled.

        Returns:
            bool: True if the panel will beep when the batch finishes.
        """
        return self._beep_checkbox.isChecked()

    def set_beep_enabled(self, enabled: bool) -> None:
        """Set the beep on finish setting.

        Args:
            enabled (bool): Whether to beep when finished.
        """
        self._beep_checkbox.setChecked(enabled)
