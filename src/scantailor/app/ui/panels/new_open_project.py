"""New/Open Project panel for the welcome screen."""

from collections.abc import Callable
from pathlib import Path

from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt, Signal

from scantailor.app.ui import load_ui_widget
from scantailor.app.ui.utils import get_cwidget

_UI_FOLDER = Path(__file__).parent
_LINK_STYLE = (
    "QLabel { color: palette(link); } QLabel:hover { text-decoration: underline; }"
)


def _make_clickable(label: QtWidgets.QLabel, on_click: Callable[[], None]) -> None:
    """Style ``label`` as a link and call ``on_click`` when it is clicked.

    Args:
        label (QtWidgets.QLabel): The label to turn into a link.
        on_click (Callable[[], None]): Called on a left click.
    """
    label.setCursor(Qt.CursorShape.PointingHandCursor)
    label.setStyleSheet(_LINK_STYLE)
    label.installEventFilter(_ClickFilter(label, on_click))


class _ClickFilter(QtCore.QObject):
    """Event filter that reports left clicks on the label it is installed on."""

    def __init__(self, label: QtWidgets.QLabel, on_click: Callable[[], None]) -> None:
        super().__init__(label)
        self._on_click = on_click

    def eventFilter(self, watched: QtCore.QObject, event: QtCore.QEvent) -> bool:
        """Handle a left button release over the label."""
        if (
            event.type() == QtCore.QEvent.Type.MouseButtonRelease
            and isinstance(event, QtGui.QMouseEvent)
            and event.button() == Qt.MouseButton.LeftButton
        ):
            self._on_click()
            return True
        return super().eventFilter(watched, event)


class NewOpenProjectPanel(QtWidgets.QWidget):
    """Welcome panel with New and Open links and the recent projects.

    Signals:
        new_project_requested: The user clicked "New Project".
        open_project_requested: The user clicked "Open Project".
        recent_project_selected: The user clicked a recent project (its path).
    """

    new_project_requested = Signal()
    open_project_requested = Signal()
    recent_project_selected = Signal(Path)

    def __init__(
        self,
        recent_projects: list[Path] | None = None,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        """Initialize the panel.

        Args:
            recent_projects (list[Path] | None): Project files to list, newest first.
            parent (QtWidgets.QWidget | None): Parent widget.
        """
        super().__init__(parent)
        self.ui = load_ui_widget(_UI_FOLDER / "NewOpenProjectPanel.ui", self)
        self.ui.setWindowFlags(Qt.WindowType.Widget)
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(self.ui)

        self._recent_group = get_cwidget(
            self.ui, QtWidgets.QGroupBox, "recentProjectsGroup"
        )
        self._new_label = get_cwidget(self.ui, QtWidgets.QLabel, "newProjectLabel")
        self._open_label = get_cwidget(self.ui, QtWidgets.QLabel, "openProjectLabel")
        _make_clickable(self._new_label, self.new_project_requested.emit)
        _make_clickable(self._open_label, self.open_project_requested.emit)
        self._recent_labels: list[QtWidgets.QLabel] = []
        self.set_recent_projects(recent_projects or [])

    def set_recent_projects(self, projects: list[Path]) -> None:
        """Show ``projects`` in the recent list; the group hides when it is empty.

        Args:
            projects (list[Path]): Project files, newest first.
        """
        layout = self._recent_group.layout()
        assert layout is not None
        for label in self._recent_labels:
            layout.removeWidget(label)
            label.deleteLater()
        self._recent_labels.clear()
        for project in projects:
            label = QtWidgets.QLabel(project.stem or "_", self._recent_group)
            label.setToolTip(str(project))
            _make_clickable(
                label, lambda chosen=project: self.recent_project_selected.emit(chosen)
            )
            layout.addWidget(label)
            self._recent_labels.append(label)
        self._recent_group.setVisible(bool(projects))
