"""Status bar panel for displaying page and position information."""

from pathlib import Path

from PySide6 import QtWidgets

from scantailor.app.ui import load_ui_widget
from scantailor.app.ui.utils import get_cwidget

_UI_FOLDER = Path(__file__).parent


class StatusBarPanel(QtWidgets.QWidget):
    """Status bar panel showing page information and mouse position.

    Displays:
    - Zone creation mode indicator
    - Mouse position relative to page
    - Physical size of page
    - Current page number
    - Page information (name and type)
    """

    def __init__(
        self,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        """Initialize the status bar panel.

        Args:
            parent: Parent widget.
        """
        super().__init__(parent)

        # Load UI
        self.ui = load_ui_widget(_UI_FOLDER / "StatusBarPanel.ui", self)
        self._setup_widgets()

    def _setup_widgets(self) -> None:
        """Set up widget references."""
        self._zone_mode_label = get_cwidget(self.ui, QtWidgets.QLabel, "zoneModeLabel")
        self._mouse_pos_label = get_cwidget(self.ui, QtWidgets.QLabel, "mousePosLabel")
        self._phys_size_label = get_cwidget(self.ui, QtWidgets.QLabel, "physSizeLabel")
        self._page_no_label = get_cwidget(self.ui, QtWidgets.QLabel, "pageNoLabel")
        self._page_info_label = get_cwidget(self.ui, QtWidgets.QLabel, "pageInfoLabel")

        # Initialize with empty values
        self.clear()

    def clear(self) -> None:
        """Clear all status bar fields."""
        self._zone_mode_label.clear()
        self._mouse_pos_label.clear()
        self._phys_size_label.clear()
        self._page_no_label.clear()
        self._page_info_label.clear()

    def set_zone_mode(self, mode: str) -> None:
        """Set the zone creation mode indicator.

        Args:
            mode: Zone mode string (e.g., "Picture", "Fill", "Auto").
        """
        self._zone_mode_label.setText(mode)

    def set_mouse_position(self, x: float, y: float, unit: str = "px") -> None:
        """Set the mouse position display.

        Args:
            x: X coordinate.
            y: Y coordinate.
            unit: Unit string (e.g., "px", "mm", "in").
        """
        self._mouse_pos_label.setText(f"{x:.1f}, {y:.1f} {unit}")

    def clear_mouse_position(self) -> None:
        """Clear the mouse position display."""
        self._mouse_pos_label.clear()

    def set_physical_size(
        self,
        width: float,
        height: float,
        unit: str = "mm",
    ) -> None:
        """Set the physical size display.

        Args:
            width: Width in given units.
            height: Height in given units.
            unit: Unit string (e.g., "mm", "in").
        """
        self._phys_size_label.setText(f"{width:.1f} x {height:.1f} {unit}")

    def set_page_number(self, current: int, total: int) -> None:
        """Set the page number display.

        Args:
            current: Current page number (1-based).
            total: Total number of pages.
        """
        self._page_no_label.setText(f"{current} / {total}")

    def set_page_info(self, name: str, page_type: str = "") -> None:
        """Set the page information display.

        Args:
            name: Page name (usually filename).
            page_type: Page type (e.g., "Left", "Right", "Single").
        """
        if page_type:
            self._page_info_label.setText(f"{name} ({page_type})")
        else:
            self._page_info_label.setText(name)
