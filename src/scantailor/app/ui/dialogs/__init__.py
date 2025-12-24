"""UI dialogs for the ScanTailor application."""

from __future__ import annotations

from scantailor.app.ui.dialogs.about import AboutDialog
from scantailor.app.ui.dialogs.fix_dpi import FixDpiDialog
from scantailor.app.ui.dialogs.load_files_status import LoadFilesStatusDialog
from scantailor.app.ui.dialogs.new_open_project_panel import NewOpenProjectPanel
from scantailor.app.ui.dialogs.project_files import ProjectFilesDialog
from scantailor.app.ui.dialogs.remove_pages import RemovePagesDialog
from scantailor.app.ui.dialogs.settings import SettingsDialog
from scantailor.app.ui.dialogs.status_bar_panel import StatusBarPanel

__all__ = [
    "AboutDialog",
    "FixDpiDialog",
    "LoadFilesStatusDialog",
    "NewOpenProjectPanel",
    "ProjectFilesDialog",
    "RemovePagesDialog",
    "SettingsDialog",
    "StatusBarPanel",
]
