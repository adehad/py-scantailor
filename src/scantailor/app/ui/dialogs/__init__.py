"""UI dialogs for the ScanTailor application."""

from __future__ import annotations

from scantailor.app.ui.dialogs.fix_dpi_dialog import FixDpiDialog
from scantailor.app.ui.dialogs.load_files_status_dialog import LoadFilesStatusDialog
from scantailor.app.ui.dialogs.new_open_project_panel import NewOpenProjectPanel
from scantailor.app.ui.dialogs.project_files_dialog import ProjectFilesDialog
from scantailor.app.ui.dialogs.remove_pages_dialog import RemovePagesDialog
from scantailor.app.ui.dialogs.settings_dialog import SettingsDialog
from scantailor.app.ui.dialogs.status_bar_panel import StatusBarPanel

__all__ = [
    "FixDpiDialog",
    "LoadFilesStatusDialog",
    "NewOpenProjectPanel",
    "ProjectFilesDialog",
    "RemovePagesDialog",
    "SettingsDialog",
    "StatusBarPanel",
]
