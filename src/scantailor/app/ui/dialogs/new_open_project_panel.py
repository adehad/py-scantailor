"""New/Open Project panel for the welcome screen."""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

from PySide6 import QtWidgets
from PySide6.QtCore import Signal

from scantailor.app.ui import UI_FOLDER, load_ui_widget
from scantailor.app.ui.utils import get_cwidget


class NewOpenProjectPanel(QtWidgets.QWidget):
    """Panel for creating new or opening existing projects.

    Displayed on the welcome screen with options to:
    - Create a new project
    - Open an existing project
    - Open recent projects

    Signals:
        new_project_requested: Emitted when user clicks "New Project"
        open_project_requested: Emitted when user clicks "Open Project"
        recent_project_selected: Emitted when user selects a recent project.
            Args: project_path (Path)
    """

    new_project_requested = Signal()
    open_project_requested = Signal()
    recent_project_selected = Signal(Path)

    MAX_RECENT_PROJECTS: ClassVar[int] = 10

    def __init__(
        self,
        recent_projects: list[Path] | None = None,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        """Initialize the panel.

        Args:
            recent_projects: List of recent project paths.
            parent: Parent widget.
        """
        super().__init__(parent)
        self._recent_projects = recent_projects or []
        self._recent_buttons: list[QtWidgets.QPushButton] = []

        # Load UI
        self.ui = load_ui_widget(UI_FOLDER / "NewOpenProjectPanel.ui", self)
        self._setup_widgets()
        self._populate_recent_projects()
        self._connect_signals()

    def _setup_widgets(self) -> None:
        """Set up widget references."""
        self._new_project_label = get_cwidget(
            self.ui, QtWidgets.QLabel, "newProjectLabel"
        )
        self._open_project_label = get_cwidget(
            self.ui, QtWidgets.QLabel, "openProjectLabel"
        )
        self._recent_group = get_cwidget(
            self.ui, QtWidgets.QGroupBox, "recentProjectsGroup"
        )

        # Make labels clickable by setting cursor and installing event filter
        self._new_project_label.setCursor(QtWidgets.QApplication.overrideCursor())
        self._open_project_label.setCursor(QtWidgets.QApplication.overrideCursor())

        # Style the labels to look clickable
        link_style = (
            "QLabel { color: palette(link); } "
            "QLabel:hover { color: palette(highlight); text-decoration: underline; }"
        )
        self._new_project_label.setStyleSheet(link_style)
        self._open_project_label.setStyleSheet(link_style)

    def _populate_recent_projects(self) -> None:
        """Populate the recent projects list."""
        # Get the layout of the recent projects group
        layout = self._recent_group.layout()
        if not layout:
            return

        # Remove the spacer temporarily
        spacer_item = layout.takeAt(0)

        # Clear existing buttons
        for btn in self._recent_buttons:
            layout.removeWidget(btn)
            btn.deleteLater()
        self._recent_buttons.clear()

        # Add buttons for recent projects
        for project_path in self._recent_projects[: self.MAX_RECENT_PROJECTS]:
            btn = QtWidgets.QPushButton(project_path.name)
            btn.setToolTip(str(project_path))
            btn.setFlat(True)
            btn.setStyleSheet(
                "QPushButton { text-align: left; padding: 4px; } "
                "QPushButton:hover { background: palette(highlight); "
                "color: palette(highlighted-text); }"
            )
            btn.clicked.connect(
                lambda checked, p=project_path: self.recent_project_selected.emit(p)
            )
            layout.addWidget(btn)
            self._recent_buttons.append(btn)

        # Re-add spacer at the end
        if spacer_item:
            layout.addItem(spacer_item)

        # Show/hide based on whether there are recent projects
        self._recent_group.setVisible(len(self._recent_buttons) > 0)

    def _connect_signals(self) -> None:
        """Connect widget signals."""
        # Use mouse press events on labels
        self._new_project_label.mousePressEvent = (
            lambda ev: self.new_project_requested.emit()
        )
        self._open_project_label.mousePressEvent = (
            lambda ev: self.open_project_requested.emit()
        )

    def set_recent_projects(self, projects: list[Path]) -> None:
        """Update the recent projects list.

        Args:
            projects: List of recent project paths.
        """
        self._recent_projects = projects
        self._populate_recent_projects()

    def add_recent_project(self, project_path: Path) -> None:
        """Add a project to the recent projects list.

        Args:
            project_path: Path to the project.
        """
        # Remove if already in list
        if project_path in self._recent_projects:
            self._recent_projects.remove(project_path)

        # Add to front
        self._recent_projects.insert(0, project_path)

        # Trim to max
        self._recent_projects = self._recent_projects[: self.MAX_RECENT_PROJECTS]

        self._populate_recent_projects()
