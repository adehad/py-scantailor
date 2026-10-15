"""Collapsible group box widget.

A QGroupBox whose whole header row folds its contents away. The collapsed state
is persisted across sessions using QSettings.
"""

from PySide6.QtCore import QEvent, QSettings, Qt, Signal
from PySide6.QtGui import QResizeEvent, QShowEvent
from PySide6.QtWidgets import QGroupBox, QToolButton, QWidget

from scantailor.app.ui.icons import icon
from scantailor.app.ui.theme.stylesheet import SPACE_XS, header_height

# QWIDGETSIZE_MAX, which PySide6 does not export.
_UNBOUNDED = 16777215


class CollapsibleGroupBox(QGroupBox):
    """A group box that folds to its header row when the header is clicked.

    The title is the text of a tool button that spans the header row, with the
    chevron on its left. The button takes Tab focus and toggles with Space, so
    the group folds from the keyboard as well as the mouse.

    Signals:
        collapsed_state_changed: Emitted when the collapsed state changes.
    """

    collapsed_state_changed = Signal(bool)

    def __init__(self, title: str = "", parent: QWidget | None = None) -> None:
        """Initialize the collapsible group box.

        Args:
            title (str): The group title, shown in the header row.
            parent (QWidget | None): The parent widget.
        """
        super().__init__(parent)
        self._collapsed = False
        self._shown = False
        self._collapsed_widgets: set[QWidget] = set()

        self._header = QToolButton(self)
        self._header.setObjectName("groupHeader")
        self._header.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self._header.setCursor(Qt.CursorShape.PointingHandCursor)
        self._header.clicked.connect(self.toggle_collapsed)
        self.setTitle(title)
        self._update_widgets()

    def title(self) -> str:
        """Return the group title.

        Returns:
            str: The text of the header row.
        """
        return self._header.text()

    def setTitle(self, title: str) -> None:
        """Show a title in the header row instead of the group box frame.

        Args:
            title (str): The group title.
        """
        self._header.setText(title)

    def header(self) -> QToolButton:
        """Return the button that spans the header row and toggles the group.

        Returns:
            QToolButton: The header button.
        """
        return self._header

    def is_collapsed(self) -> bool:
        """Return the current collapsed state.

        Returns:
            bool: True if collapsed, False if expanded.
        """
        return self._collapsed

    def set_collapsed(self, collapse: bool) -> None:
        """Set the collapsed state.

        Args:
            collapse (bool): True to collapse, False to expand.
        """
        if self._collapsed == collapse:
            return
        self._collapsed = collapse
        self._update_widgets()
        self._save_state()
        self.collapsed_state_changed.emit(self._collapsed)

    def toggle_collapsed(self) -> None:
        """Toggle between collapsed and expanded states."""
        self.set_collapsed(not self._collapsed)

    def _nested(self) -> bool:
        return isinstance(self.parentWidget(), QGroupBox)

    def _place_header(self) -> None:
        nested = self._nested()
        height = header_height(self.font(), nested=nested)
        inset = 0 if nested else SPACE_XS
        self._header.setGeometry(inset, inset, self.width() - 2 * inset, height)
        if self._collapsed:
            self.setMaximumHeight(self._header.geometry().bottom() + 1 + inset)

    def _update_widgets(self) -> None:
        self._header.setIcon(
            icon("chevron-right" if self._collapsed else "chevron-down")
        )
        self._header.setAccessibleDescription(
            self.tr("Collapsed") if self._collapsed else self.tr("Expanded")
        )
        if self._collapsed:
            self._collapsed_widgets.clear()
            for child in self.findChildren(QWidget):
                if child is self._header or child.parent() is not self:
                    continue
                if child.isVisible():
                    self._collapsed_widgets.add(child)
                    child.hide()
        else:
            for child in self._collapsed_widgets:
                child.show()
            self._collapsed_widgets.clear()
            self.setMaximumHeight(_UNBOUNDED)
        self._place_header()

    def resizeEvent(self, event: QResizeEvent) -> None:
        """Stretch the header row across the new width.

        Args:
            event (QResizeEvent): The resize event.
        """
        super().resizeEvent(event)
        self._place_header()

    def changeEvent(self, event: QEvent) -> None:
        """Resize the header row when the font or style changes.

        Args:
            event (QEvent): The change event.
        """
        super().changeEvent(event)
        if event.type() in (QEvent.Type.FontChange, QEvent.Type.StyleChange):
            self._place_header()

    def showEvent(self, event: QShowEvent) -> None:
        """Restore the saved state the first time the group shows.

        Args:
            event (QShowEvent): The show event.
        """
        super().showEvent(event)
        if not self._shown:
            self._shown = True
            self._load_state()
            self._update_widgets()
            self.collapsed_state_changed.emit(self._collapsed)

    def _get_settings_key(self) -> str:
        """Get the settings key for persisting collapsed state.

        Returns:
            str: A unique key based on the widget hierarchy.
        """
        parts = []
        widget: QWidget | None = self
        while widget is not None:
            name = widget.objectName()
            if name:
                parts.insert(0, name)
            widget = widget.parentWidget()
        if not parts:
            parts = ["CollapsibleGroupBox"]
        return "/".join(["CollapsibleGroupBox", *parts, "collapsed"])

    def _load_state(self) -> None:
        # QSettings returns the strings "true" and "false" on some platforms.
        value = QSettings().value(self._get_settings_key())
        if isinstance(value, bool):
            self._collapsed = value
        elif isinstance(value, str):
            self._collapsed = value.lower() == "true"

    def _save_state(self) -> None:
        QSettings().setValue(self._get_settings_key(), self._collapsed)
