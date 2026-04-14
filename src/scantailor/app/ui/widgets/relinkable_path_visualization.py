"""Widget for visualizing and interacting with relinkable file paths.

This widget displays a file path as a series of clickable buttons, one for each
path component. Components are styled green if they exist and red if missing.
Hovering over a component highlights it and all components before it.
"""

from dataclasses import dataclass
from pathlib import Path

from PySide6.QtCore import QEvent, QObject, Qt, Signal
from PySide6.QtGui import QCursor, QPaintEvent
from PySide6.QtWidgets import QHBoxLayout, QPushButton, QWidget


# Path types matching C++ RelinkablePath::Type
class PathType:
    """Type of path component."""

    FILE = 0
    DIR = 1


@dataclass
class PathComponent:
    """A component of a relinkable path."""

    label: str  # Display text
    prefix_path: str  # Path up to and including this component
    suffix_path: str  # Rest of the path after this component
    path_type: int  # PathType.FILE or PathType.DIR
    exists: bool = False  # Whether this component exists on filesystem


class ComponentButton(QPushButton):
    """Button representing a single path component."""

    def __init__(self, parent: QWidget | None = None) -> None:
        """Initialize the component button.

        Args:
            parent: Parent widget.
        """
        super().__init__(parent)
        self._force_highlight = False
        self._stick_highlight = False

    def paintEvent(self, arg__1: QPaintEvent) -> None:
        """Custom paint to support forced highlighting.

        Args:
            arg__1: Paint event.
        """
        # Check if we should force hover appearance
        if self.property("forceHighlight") or self.property("stickHighlight"):
            # Temporarily enable hover state
            self.setAttribute(Qt.WidgetAttribute.WA_UnderMouse, True)
            super().paintEvent(arg__1)
            # Restore original state
            actual_under_mouse = self.rect().contains(self.mapFromGlobal(QCursor.pos()))
            self.setAttribute(Qt.WidgetAttribute.WA_UnderMouse, actual_under_mouse)
        else:
            super().paintEvent(arg__1)


class RelinkablePathVisualization(QWidget):
    """Widget for visualizing relinkable file paths.

    Displays a file path as clickable buttons for each component, showing
    which parts of the path exist (green) and which are missing (red).

    Signals:
        clicked: Emitted when a path component is clicked.

    Args:
                prefix_path (str): Path up to and including clicked component
                suffix_path (str): Rest of path after clicked component
                path_type (int): PathType.FILE or PathType.DIR
    """

    clicked = Signal(str, str, int)

    def __init__(self, parent: QWidget | None = None) -> None:
        """Initialize the path visualization widget.

        Args:
            parent: Parent widget.
        """
        super().__init__(parent)
        self._layout = QHBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(0)
        self._components: list[PathComponent] = []

    def clear(self) -> None:
        """Clear all path components."""
        while self._layout.count():
            item = self._layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        self._components.clear()

    def set_path(self, path_str: str, path_type: int, clickable: bool = True) -> None:
        """Set the path to display.

        Args:
            path_str: The path string to display.
            path_type: PathType.FILE or PathType.DIR for the last component.
            clickable: Whether the components should be clickable.
        """
        self.clear()

        # Normalize and split path
        path = Path(path_str)
        parts = path.parts

        if not parts:
            return

        # Build path components
        prefix_path = ""
        components: list[PathComponent] = []

        for i, part in enumerate(parts):
            # Build prefix path (path up to and including this component)
            if prefix_path and not prefix_path.endswith(("/", "\\")):
                prefix_path += "/"
            prefix_path += part

            # Build suffix path (rest of path after this component)
            suffix_path = "/".join(parts[i + 1 :])

            # All components except the last are directories
            comp_type = PathType.DIR if i < len(parts) - 1 else path_type

            components.append(
                PathComponent(part, prefix_path, suffix_path, comp_type, False)
            )

        # Check which components exist using binary search
        self._check_for_existence(components)
        self._components = components

        # Create buttons for each component
        for idx, component in enumerate(components):
            btn = ComponentButton(self)
            btn.setText(component.label.replace("/", "\\"))
            btn.setEnabled(clickable)

            if clickable:
                btn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
                btn.clicked.connect(
                    lambda checked=False, i=idx: self._on_button_clicked(i)
                )

            self._style_button(btn, component.exists)
            btn.installEventFilter(self)
            self._layout.addWidget(btn)

        # Add stretch to push buttons to the left
        self._layout.addStretch()

    def _check_for_existence(self, components: list[PathComponent]) -> None:
        """Check which path components exist using binary search.

        Args:
            components: List of path components to check.
        """
        if not components:
            return

        # Check if full path exists
        if Path(components[-1].prefix_path).exists():
            for comp in components:
                comp.exists = True
            return

        # Binary search to find where path stops existing
        left = -1  # Existing component (or -1 if none)
        right = len(components) - 1  # Non-existing component

        while right - left > 1:
            mid = (left + right + 1) // 2
            if Path(components[mid].prefix_path).exists():
                left = mid
            else:
                right = mid

        # Mark components as existing or not
        for i, comp in enumerate(components):
            comp.exists = i < right

    def _style_button(self, btn: QPushButton, exists: bool) -> None:
        """Style a path component button.

        Args:
            btn: The button to style.
            exists: Whether the path component exists.
        """
        # Use border color from palette
        border_color = self.palette().window().color().darker(150).name()

        # Base style
        style = f"""
            QPushButton {{
                border: 2px solid {border_color};
                border-radius: 0.5em;
                padding: 0.2em;
                margin-left: 1px;
                margin-right: 1px;
                min-width: 2em;
                font-weight: bold;
        """

        # Add color based on existence
        if exists:
            style += """
                color: #3a5827;
                background: qradialgradient(cx: 0.3, cy: -0.4, fx: 0.3, fy: -0.4,
                                           radius: 1.35, stop: 0 #fff, stop: 1 #89e74a);
            """
        else:
            style += """
                color: #6f2719;
                background: qradialgradient(cx: 0.3, cy: -0.4, fx: 0.3, fy: -0.4,
                                           radius: 1.35, stop: 0 #fff, stop: 1 #ff674b);
            """

        # Add hover and pressed states
        style += """
            }
            QPushButton:hover {
                color: #333;
                background: qradialgradient(cx: 0.3, cy: -0.4, fx: 0.3, fy: -0.4,
                                           radius: 1.35, stop: 0 #fff, stop: 1 #bbb);
            }
            QPushButton:pressed {
                color: #333;
                background: qradialgradient(cx: 0.4, cy: -0.1, fx: 0.4, fy: -0.1,
                                           radius: 1.35, stop: 0 #fff, stop: 1 #ddd);
            }
        """

        btn.setStyleSheet(style)

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:
        """Filter events to implement hover highlighting.

        Args:
            watched: The watched widget.
            event: The event.

        Returns:
            True if event was handled.
        """
        if event.type() == QEvent.Type.Enter:
            # Find which button was entered
            for i in range(self._layout.count()):
                widget = self._layout.itemAt(i).widget()
                if widget is watched:
                    # Highlight this button and all before it
                    for j in range(i + 1):
                        btn = self._layout.itemAt(j).widget()
                        if btn:
                            btn.setProperty("forceHighlight", True)
                            btn.update()
                    break

        elif event.type() == QEvent.Type.Leave:
            # Clear highlight from all buttons
            for i in range(self._layout.count()):
                widget = self._layout.itemAt(i).widget()
                if widget:
                    widget.setProperty("forceHighlight", False)
                    widget.update()

        return super().eventFilter(watched, event)

    def _on_button_clicked(self, component_idx: int) -> None:
        """Handle button click.

        Args:
            component_idx: Index of the clicked component.
        """
        if component_idx >= len(self._components):
            return

        component = self._components[component_idx]

        # Stick highlighting during signal emission
        for i in range(component_idx + 1):
            widget = self._layout.itemAt(i).widget()
            if widget:
                widget.setProperty("stickHighlight", True)

        # Emit signal
        self.clicked.emit(
            component.prefix_path, component.suffix_path, component.path_type
        )

        # Clear stick highlighting
        for i in range(component_idx + 1):
            widget = self._layout.itemAt(i).widget()
            if widget:
                widget.setProperty("stickHighlight", False)
