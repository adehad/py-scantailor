"""Collapsible group box widget.

A QGroupBox that can be collapsed to hide its contents, saving screen space.
The collapsed state is persisted across sessions using QSettings.
"""

from PySide6.QtCore import QEvent, QSettings, Signal
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QGroupBox, QStyle, QToolButton, QWidget


class CollapsibleGroupBox(QGroupBox):
    """A group box that can be collapsed to hide its contents.

    The collapsed state is persisted using QSettings based on the widget's
    object name and parent hierarchy.

    Signals:
        collapsed_state_changed: Emitted when the collapsed state changes.
    """

    collapsed_state_changed = Signal(bool)

    def __init__(
        self,
        title: str = "",
        parent: QWidget | None = None,
    ) -> None:
        """Initialize the collapsible group box.

        Args:
            title: The group box title.
            parent: The parent widget.
        """
        super().__init__(title, parent)

        self._collapsed = False
        self._shown = False
        self._collapse_button: QToolButton | None = None
        self._ignore_visibility_events = 0
        self._collapsed_widgets: set[QWidget] = set()

        # Icons for collapse/expand states
        style = self.style()
        if style is not None:
            self._collapse_icon = style.standardIcon(
                QStyle.StandardPixmap.SP_TitleBarMinButton
            )
            self._expand_icon = style.standardIcon(
                QStyle.StandardPixmap.SP_TitleBarNormalButton
            )
        else:
            self._collapse_icon = QIcon()
            self._expand_icon = QIcon()

        self._initialize()

    def _initialize(self) -> None:
        """Set up the collapse button and connections."""
        # Create collapse button
        self._collapse_button = QToolButton(self)
        self._collapse_button.setObjectName("collapseButton")
        self._collapse_button.setAutoRaise(True)
        self._collapse_button.setIcon(self._collapse_icon)
        self._collapse_button.clicked.connect(self.toggle_collapsed)

        # Connect checkbox if checkable
        if self.isCheckable():
            self.toggled.connect(self._on_check_toggled)
            self.clicked.connect(self._on_check_clicked)

        self._update_widgets()

    def is_collapsed(self) -> bool:
        """Return the current collapsed state.

        Returns:
            True if collapsed, False if expanded.
        """
        return self._collapsed

    def set_collapsed(self, collapse: bool) -> None:
        """Set the collapsed state.

        Args:
            collapse: True to collapse, False to expand.
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

    def _update_widgets(self) -> None:
        """Update visibility of child widgets based on collapsed state."""
        if self._collapse_button is None:
            return

        # Update button icon
        if self._collapsed:
            self._collapse_button.setIcon(self._expand_icon)
        else:
            self._collapse_button.setIcon(self._collapse_icon)

        # Position the button in the title area
        self._position_collapse_button()

        # Show/hide children
        self._ignore_visibility_events += 1
        try:
            if self._collapsed:
                # Hide all visible children and remember them
                self._collapsed_widgets.clear()
                for child in self.findChildren(QWidget):
                    if child is self._collapse_button:
                        continue
                    if child.parent() is not self:
                        continue
                    if child.isVisible():
                        self._collapsed_widgets.add(child)
                        child.hide()
            else:
                # Restore previously hidden children
                for child in self._collapsed_widgets:
                    child.show()
                self._collapsed_widgets.clear()
        finally:
            self._ignore_visibility_events -= 1

    def _position_collapse_button(self) -> None:
        """Position the collapse button in the title bar area."""
        if self._collapse_button is None:
            return

        # Get the title bar height
        fm = self.fontMetrics()
        title_height = fm.height() + 4  # Add some padding

        # Position button at right side of title
        button_size = title_height - 4
        self._collapse_button.setFixedSize(button_size, button_size)

        # Position at right edge, vertically centered in title
        x = self.width() - button_size - 4
        y = 2
        self._collapse_button.move(x, y)

    def _on_check_toggled(self, checked: bool) -> None:
        """Handle checkbox toggle.

        Args:
            checked: Whether the checkbox is checked.
        """
        # If unchecked, collapse; if checked, expand
        if not checked and not self._collapsed:
            self.set_collapsed(True)

    def _on_check_clicked(self, checked: bool) -> None:
        """Handle checkbox click.

        Args:
            checked: Whether the checkbox is checked.
        """
        if checked and self._collapsed:
            self.set_collapsed(False)

    def showEvent(self, event: QEvent) -> None:
        """Handle show event.

        Args:
            event: The show event.
        """
        super().showEvent(event)

        if not self._shown:
            self._shown = True
            self._load_state()
            self._update_widgets()
            self.collapsed_state_changed.emit(self._collapsed)

    def resizeEvent(self, event: QEvent) -> None:
        """Handle resize event.

        Args:
            event: The resize event.
        """
        super().resizeEvent(event)
        self._position_collapse_button()

    def changeEvent(self, event: QEvent) -> None:
        """Handle change event.

        Args:
            event: The change event.
        """
        super().changeEvent(event)

        if event.type() == QEvent.Type.EnabledChange:
            if self._collapse_button is not None:
                self._collapse_button.setEnabled(self.isEnabled())

    def _get_settings_key(self) -> str:
        """Get the settings key for persisting collapsed state.

        Returns:
            A unique key based on the widget hierarchy.
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
        """Load the collapsed state from settings."""
        settings = QSettings()
        key = self._get_settings_key()
        value = settings.value(key)

        if value is not None:
            # QSettings returns string "true"/"false" on some platforms
            if isinstance(value, bool):
                self._collapsed = value
            elif isinstance(value, str):
                self._collapsed = value.lower() == "true"

    def _save_state(self) -> None:
        """Save the collapsed state to settings."""
        settings = QSettings()
        key = self._get_settings_key()
        settings.setValue(key, self._collapsed)
