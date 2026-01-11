"""UI panels for the ScanTailor application."""

from scantailor.app.ui.panels.batch_processing import BatchProcessingLowerPanel
from scantailor.app.ui.panels.new_open_project import NewOpenProjectPanel
from scantailor.app.ui.panels.status_bar import StatusBarPanel
from scantailor.app.ui.panels.system_load import SystemLoadWidget

__all__ = [
    "BatchProcessingLowerPanel",
    "NewOpenProjectPanel",
    "StatusBarPanel",
    "SystemLoadWidget",
]
