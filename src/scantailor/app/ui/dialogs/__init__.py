"""UI dialogs for the ScanTailor application."""

from scantailor.app.ui.dialogs.about import AboutDialog
from scantailor.app.ui.dialogs.default_params import DefaultParamsDialog
from scantailor.app.ui.dialogs.fix_dpi import FixDpiDialog
from scantailor.app.ui.dialogs.load_files_status import LoadFilesStatusDialog
from scantailor.app.ui.dialogs.out_of_memory import OutOfMemoryDialog
from scantailor.app.ui.dialogs.project_files import ProjectFilesDialog
from scantailor.app.ui.dialogs.relinking import RelinkingDialog
from scantailor.app.ui.dialogs.remove_pages import RemovePagesDialog
from scantailor.app.ui.dialogs.settings import SettingsDialog
from scantailor.app.ui.dialogs.split_mode import ApplyScope, SplitModeDialog

__all__ = [
    "AboutDialog",
    "ApplyScope",
    "DefaultParamsDialog",
    "FixDpiDialog",
    "LoadFilesStatusDialog",
    "OutOfMemoryDialog",
    "ProjectFilesDialog",
    "RelinkingDialog",
    "RemovePagesDialog",
    "SettingsDialog",
    "SplitModeDialog",
]
