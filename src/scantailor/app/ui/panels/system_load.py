"""System load widget for controlling parallel processing load."""

from __future__ import annotations

from pathlib import Path

from PySide6 import QtWidgets
from PySide6.QtCore import Signal

from scantailor.app.ui import load_ui_widget
from scantailor.app.ui.utils import get_cwidget

_UI_FOLDER = Path(__file__).parent


class SystemLoadWidget(QtWidgets.QWidget):
    """Widget for controlling system load during batch processing.

    Displays a slider to control how many CPU cores/threads to use
    for parallel processing, with +/- buttons for fine control.

    Signals:
        load_changed: Emitted when the load value changes.
            Args: load_value (int) - the new load value
    """

    load_changed = Signal(int)

    def __init__(
        self,
        min_value: int = 1,
        max_value: int | None = None,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        """Initialize the system load widget.

        Args:
            min_value: Minimum load value (default 1).
            max_value: Maximum load value (default is CPU count).
            parent: Parent widget.
        """
        import os

        super().__init__(parent)

        self._min_value = min_value
        self._max_value = max_value or os.cpu_count() or 4

        # Load UI
        self.ui = load_ui_widget(_UI_FOLDER / "SystemLoadWidget.ui", self)
        self._setup_widgets()
        self._connect_signals()

    def _setup_widgets(self) -> None:
        """Set up widget references."""
        self._slider = get_cwidget(self.ui, QtWidgets.QSlider, "slider")
        self._minus_btn = get_cwidget(self.ui, QtWidgets.QToolButton, "minusBtn")
        self._plus_btn = get_cwidget(self.ui, QtWidgets.QToolButton, "plusBtn")

        # Configure slider
        self._slider.setMinimum(self._min_value)
        self._slider.setMaximum(self._max_value)
        self._slider.setValue(self._max_value)  # Default to max

        self._update_button_states()

    def _connect_signals(self) -> None:
        """Connect widget signals."""
        self._slider.valueChanged.connect(self._on_slider_changed)
        self._minus_btn.clicked.connect(self._decrease_load)
        self._plus_btn.clicked.connect(self._increase_load)

    def _on_slider_changed(self, value: int) -> None:
        """Handle slider value change."""
        self._update_button_states()
        self.load_changed.emit(value)

    def _decrease_load(self) -> None:
        """Decrease the load value by 1."""
        current = self._slider.value()
        if current > self._min_value:
            self._slider.setValue(current - 1)

    def _increase_load(self) -> None:
        """Increase the load value by 1."""
        current = self._slider.value()
        if current < self._max_value:
            self._slider.setValue(current + 1)

    def _update_button_states(self) -> None:
        """Update enabled state of +/- buttons based on current value."""
        current = self._slider.value()
        self._minus_btn.setEnabled(current > self._min_value)
        self._plus_btn.setEnabled(current < self._max_value)

    def get_load(self) -> int:
        """Get the current load value.

        Returns:
            Current load value.
        """
        return self._slider.value()

    def set_load(self, value: int) -> None:
        """Set the load value.

        Args:
            value: New load value (clamped to min/max range).
        """
        clamped = max(self._min_value, min(self._max_value, value))
        self._slider.setValue(clamped)

    def show(self) -> None:
        """Show the widget."""
        self.ui.show()
