"""System load widget for controlling parallel processing load."""

from __future__ import annotations

import sys
from pathlib import Path

from PySide6 import QtWidgets
from PySide6.QtCore import QSettings, Signal
from PySide6.QtGui import QCursor

from scantailor.app.ui import load_ui_widget
from scantailor.app.ui.utils import get_cwidget

_UI_FOLDER = Path(__file__).parent
_SETTINGS_KEY = "settings/batch_processing_threads"


class SystemLoadWidget(QtWidgets.QWidget):
    """Widget for controlling system load during batch processing.

    Displays a slider to control how many CPU cores/threads to use
    for parallel processing, with +/- buttons for fine control.

    The load value is persisted in QSettings.

    Signals:
        load_changed: Emitted when the load value changes.
            Args: load_value (int) - the new load value
    """

    load_changed = Signal(int)

    def __init__(
        self,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        """Initialize the system load widget.

        Args:
            parent: Parent widget.
        """
        import os

        super().__init__(parent)

        # Determine max threads
        self._max_threads = os.cpu_count() or 4

        # Restrict threads for 32-bit systems due to address space constraints
        if sys.maxsize <= 2**32:  # 32-bit system
            self._max_threads = min(self._max_threads, 2)

        # Load UI
        self.ui = load_ui_widget(_UI_FOLDER / "SystemLoadWidget.ui", self)
        self._setup_widgets()
        self._connect_signals()
        self._load_state()

    def _setup_widgets(self) -> None:
        """Set up widget references."""
        self._slider = get_cwidget(self.ui, QtWidgets.QSlider, "slider")
        self._minus_btn = get_cwidget(self.ui, QtWidgets.QToolButton, "minusBtn")
        self._plus_btn = get_cwidget(self.ui, QtWidgets.QToolButton, "plusBtn")

        # Configure slider
        self._slider.setMinimum(1)
        self._slider.setMaximum(self._max_threads)
        self._slider.setTracking(True)

        # Set icons (using standard icons as fallback)
        style = self.style()
        if style:
            minus_icon = style.standardIcon(QtWidgets.QStyle.StandardPixmap.SP_ArrowLeft)
            plus_icon = style.standardIcon(QtWidgets.QStyle.StandardPixmap.SP_ArrowRight)
            self._minus_btn.setIcon(minus_icon)
            self._plus_btn.setIcon(plus_icon)

        self._update_button_states()

    def _connect_signals(self) -> None:
        """Connect widget signals."""
        self._slider.sliderPressed.connect(self._on_slider_pressed)
        self._slider.sliderMoved.connect(self._show_tooltip)
        self._slider.valueChanged.connect(self._on_value_changed)
        self._minus_btn.clicked.connect(self._decrease_load)
        self._plus_btn.clicked.connect(self._increase_load)

    def _on_slider_pressed(self) -> None:
        """Handle slider press to show tooltip."""
        self._show_tooltip(self._slider.value())

    def _on_value_changed(self, value: int) -> None:
        """Handle slider value change.

        Args:
            value: New slider value.
        """
        self._update_button_states()
        self._save_state()
        self.load_changed.emit(value)

    def _decrease_load(self) -> None:
        """Decrease the load value by 1."""
        current = self._slider.value()
        if current > 1:
            self._slider.setValue(current - 1)
            self._show_tooltip(current - 1)

    def _increase_load(self) -> None:
        """Increase the load value by 1."""
        current = self._slider.value()
        if current < self._max_threads:
            self._slider.setValue(current + 1)
            self._show_tooltip(current + 1)

    def _show_tooltip(self, threads: int) -> None:
        """Show tooltip with current thread count.

        Args:
            threads: Number of threads to display.
        """
        # Calculate tooltip position near cursor on slider
        center = self._slider.rect().center()
        tooltip_pos = self._slider.mapFromGlobal(QCursor.pos())

        # Clamp to slider bounds
        if tooltip_pos.x() < 0 or tooltip_pos.x() >= self._slider.width():
            tooltip_pos.setX(center.x())
        if tooltip_pos.y() < 0 or tooltip_pos.y() >= self._slider.height():
            tooltip_pos.setY(center.y())

        tooltip_pos = self._slider.mapToGlobal(tooltip_pos)

        # Show tooltip
        QtWidgets.QToolTip.showText(
            tooltip_pos, f"{threads}/{self._max_threads}", self._slider
        )

    def _update_button_states(self) -> None:
        """Update enabled state of +/- buttons based on current value."""
        current = self._slider.value()
        self._minus_btn.setEnabled(current > 1)
        self._plus_btn.setEnabled(current < self._max_threads)

    def _load_state(self) -> None:
        """Load the saved thread count from QSettings."""
        settings = QSettings()
        saved_value = settings.value(_SETTINGS_KEY, self._max_threads)

        # Clamp to valid range
        if isinstance(saved_value, (int, str)):
            try:
                value = int(saved_value)
                value = min(self._max_threads, max(1, value))
                self._slider.setValue(value)
            except (ValueError, TypeError):
                self._slider.setValue(self._max_threads)
        else:
            self._slider.setValue(self._max_threads)

    def _save_state(self) -> None:
        """Save the thread count to QSettings."""
        settings = QSettings()
        value = self._slider.value()

        # Only save if different from default (max threads)
        if value == self._max_threads:
            settings.remove(_SETTINGS_KEY)
        else:
            settings.setValue(_SETTINGS_KEY, value)

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
        clamped = max(1, min(self._max_threads, value))
        self._slider.setValue(clamped)

    def show(self) -> None:
        """Show the widget."""
        self.ui.show()
