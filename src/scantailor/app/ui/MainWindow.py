"""Main Window."""

import pathlib

from loguru import logger as _LOG
from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QGraphicsView,
    QScrollArea,
    QStackedLayout,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from scantailor.app.ui import UI_FOLDER, load_ui_widget
from scantailor.app.ui.dialogs.about import AboutDialog
from scantailor.app.ui.utils import get_cwidget


class MainWindowManager:
    """Controller Class for the Main Application window for ScanTailor.

    Note: This class is not a QMainWindow.
    When using QtUiLoader to load a QMainWindow based .ui file, and setting the parent
    as a QMainWindow, we get weird behavior.
    e.g. https://stackoverflow.com/questions/53828666/pyside2-qmainwindow-loaded-from-ui-file-not-triggering-window-events
    """

    def __init__(self):
        """Application Initialize."""
        super().__init__()
        self.ui = load_ui_widget(UI_FOLDER / "MainWindow.ui")

        self.pages = None  # std::make_shared<ProjectPages>()
        self.stages = (
            None  # std::make_shared<StageSequence>(m_pages, newPageSelectionAccessor())
        )
        self.workerThreadPool = None  # std::make_unique<WorkerThreadPool>()
        self.interactiveQueue = None  # std::make_unique<ProcessingTaskQueue>()
        self.outOfMemoryDialog = None  # std::make_unique<OutOfMemoryDialog>()
        self.curFilter = 0
        self.ignoreSelectionChanges = 0
        self.ignorePageOrderingChanges = 0
        self.debug = False
        self.closing = False
        self.projectFile = ""
        self.optionsWidget = None

        self.settings = QtCore.QSettings()
        # self.units = self.unitsFromString(self.settings.value("units", "PIXELS"))

        # UI element assignments from .ui file
        self.actionFirstPage = get_cwidget(self.ui, QtGui.QAction, "actionFirstPage")
        self.actionLastPage = get_cwidget(self.ui, QtGui.QAction, "actionLastPage")
        self.actionPrevPage = get_cwidget(self.ui, QtGui.QAction, "actionPrevPage")
        self.actionNextPage = get_cwidget(self.ui, QtGui.QAction, "actionNextPage")
        self.actionPrevPageQ = get_cwidget(self.ui, QtGui.QAction, "actionPrevPageQ")
        self.actionNextPageW = get_cwidget(self.ui, QtGui.QAction, "actionNextPageW")
        self.actionPrevSelectedPage = get_cwidget(
            self.ui, QtGui.QAction, "actionPrevSelectedPage"
        )
        self.actionNextSelectedPage = get_cwidget(
            self.ui, QtGui.QAction, "actionNextSelectedPage"
        )
        self.actionPrevSelectedPageQ = get_cwidget(
            self.ui, QtGui.QAction, "actionPrevSelectedPageQ"
        )
        self.actionNextSelectedPageW = get_cwidget(
            self.ui, QtGui.QAction, "actionNextSelectedPageW"
        )
        self.actionGotoPage = get_cwidget(self.ui, QtGui.QAction, "actionGotoPage")
        self.actionAbout = get_cwidget(self.ui, QtGui.QAction, "actionAbout")
        self.actionReloadPage = get_cwidget(self.ui, QtGui.QAction, "actionReloadPage")
        self.actionSwitchFilter1 = get_cwidget(
            self.ui, QtGui.QAction, "actionSwitchFilter1"
        )
        self.actionSwitchFilter2 = get_cwidget(
            self.ui, QtGui.QAction, "actionSwitchFilter2"
        )
        self.actionSwitchFilter3 = get_cwidget(
            self.ui, QtGui.QAction, "actionSwitchFilter3"
        )
        self.actionSwitchFilter4 = get_cwidget(
            self.ui, QtGui.QAction, "actionSwitchFilter4"
        )
        self.actionSwitchFilter5 = get_cwidget(
            self.ui, QtGui.QAction, "actionSwitchFilter5"
        )
        self.actionSwitchFilter6 = get_cwidget(
            self.ui, QtGui.QAction, "actionSwitchFilter6"
        )
        self.selectionModeBtn = get_cwidget(self.ui, QToolButton, "selectionModeBtn")
        self.focusButton = get_cwidget(self.ui, QToolButton, "focusButton")
        self.sortOptions = get_cwidget(self.ui, QComboBox, "sortOptions")
        self.actionFixDpi = get_cwidget(self.ui, QtGui.QAction, "actionFixDpi")
        self.actionRelinking = get_cwidget(self.ui, QtGui.QAction, "actionRelinking")
        self.actionDebug = get_cwidget(self.ui, QtGui.QAction, "actionDebug")
        self.actionSettings = get_cwidget(self.ui, QtGui.QAction, "actionSettings")
        self.actionDefaults = get_cwidget(self.ui, QtGui.QAction, "actionDefaults")
        self.actionNewProject = get_cwidget(self.ui, QtGui.QAction, "actionNewProject")
        self.actionOpenProject = get_cwidget(
            self.ui, QtGui.QAction, "actionOpenProject"
        )
        self.actionSaveProject = get_cwidget(
            self.ui, QtGui.QAction, "actionSaveProject"
        )
        self.actionSaveProjectAs = get_cwidget(
            self.ui, QtGui.QAction, "actionSaveProjectAs"
        )
        self.actionCloseProject = get_cwidget(
            self.ui, QtGui.QAction, "actionCloseProject"
        )
        self.actionQuit = get_cwidget(self.ui, QtGui.QAction, "actionQuit")
        self.thumbView = get_cwidget(self.ui, QGraphicsView, "thumbView")
        self.filterOptions = get_cwidget(self.ui, QWidget, "filterOptions")
        self.imageViewFrame = get_cwidget(self.ui, QWidget, "imageViewFrame")
        self.scrollAreaWidgetContents = get_cwidget(
            self.ui, QWidget, "scrollAreaWidgetContents"
        )
        self.scrollArea = get_cwidget(self.ui, QScrollArea, "scrollArea")
        self.magnifyThumbnailsBtn = get_cwidget(
            self.ui, QToolButton, "magnifyThumbnailsBtn"
        )
        self.sortingOrderBtn = get_cwidget(self.ui, QToolButton, "sortingOrderBtn")

        # Layouts
        self.imageFrameLayout = QStackedLayout(self.imageViewFrame)
        self.optionsFrameLayout = QVBoxLayout(self.scrollAreaWidgetContents)

        # Connections
        self.actionFirstPage.triggered.connect(self.goFirstPage)
        self.actionLastPage.triggered.connect(self.goLastPage)
        self.actionPrevPage.triggered.connect(self.goPrevPage)
        self.actionNextPage.triggered.connect(self.goNextPage)
        self.actionPrevPageQ.triggered.connect(self.goPrevPage)
        self.actionNextPageW.triggered.connect(self.goNextPage)
        self.actionPrevSelectedPage.triggered.connect(self.goPrevSelectedPage)
        self.actionNextSelectedPage.triggered.connect(self.goNextSelectedPage)
        self.actionPrevSelectedPageQ.triggered.connect(self.goPrevSelectedPage)
        self.actionNextSelectedPageW.triggered.connect(self.goNextSelectedPage)
        self.actionGotoPage.triggered.connect(self.execGotoPageDialog)
        self.actionAbout.triggered.connect(self.showAboutDialog)
        self.actionReloadPage.triggered.connect(self.reloadCurrentPage)
        self.actionSwitchFilter1.triggered.connect(self.switchFilter1)
        self.actionSwitchFilter2.triggered.connect(self.switchFilter2)
        self.actionSwitchFilter3.triggered.connect(self.switchFilter3)
        self.actionSwitchFilter4.triggered.connect(self.switchFilter4)
        self.actionSwitchFilter5.triggered.connect(self.switchFilter5)
        self.actionSwitchFilter6.triggered.connect(self.switchFilter6)
        self.selectionModeBtn.clicked.connect(
            self.setSelectionModeEnabled
        )  # m_thumbSequence.get().setSelectionModeEnabled
        self.focusButton.clicked.connect(self.thumbViewFocusToggled)
        self.sortOptions.currentIndexChanged.connect(self.pageOrderingChanged)
        self.actionFixDpi.triggered.connect(self.fixDpiDialogRequested)
        self.actionRelinking.triggered.connect(self.showRelinkingDialog)
        self.actionDebug.toggled.connect(self.debugToggled)
        self.actionSettings.triggered.connect(self.openSettingsDialog)
        self.actionDefaults.triggered.connect(self.openDefaultParamsDialog)
        self.actionNewProject.triggered.connect(self.newProject)
        self.actionOpenProject.triggered.connect(self.openProject)
        self.actionSaveProject.triggered.connect(self.saveProjectTriggered)
        self.actionSaveProjectAs.triggered.connect(self.saveProjectAsTriggered)
        self.actionCloseProject.triggered.connect(self.closeProject)
        self.actionQuit.triggered.connect(self.quit)

        # if self.settings.value("mainWindow/maximized") == False:
        #    if not self.restoreGeometry(geom.toByteArray()):
        #        pass

        # Additional setup (from C++ constructor)
        # self.setupThumbView()
        # self.updateThumbViewMinWidth()
        # self.updateSortOptions()
        # self.updateProjectActions()
        # self.updateWindowTitle()
        # self.updateAutoSaveTimer()
        # self.setupIcons()

    def allPages(self) -> None:  # PageSequence
        """Return all pages."""
        pass

    def selectedPages(self) -> None:  # std::set<PageId>
        """Return selected pages."""
        pass

    def selectedRanges(self) -> None:  # std::vector<PageRange>
        """Return selected ranges."""
        pass

    def openProject(self, _checked: bool) -> None:
        """Open project."""
        project_file, _project_filter = QFileDialog.getOpenFileName(
            self.ui,
            self.ui.tr("Open Project"),
            str(pathlib.Path().cwd()),
            f"{self.ui.tr('Scan Tailor Projects')} (*.ScanTailor)",
        )
        if any(project_file):
            _LOG.info(f"Opening project: {project_file}")

    def goFirstPage(self) -> None:
        """Go to first page."""
        _LOG.info("Go to first page")

    def goLastPage(self) -> None:
        """Go to last page."""
        _LOG.info("Go to last page")

    def goNextPage(self) -> None:
        """Go to next page."""
        _LOG.info("Go to next page")

    def goPrevPage(self) -> None:
        """Go to previous page."""
        _LOG.info("Go to previous page")

    def goNextSelectedPage(self) -> None:
        """Go to next selected page."""
        _LOG.info("Go to next selected page")

    def goPrevSelectedPage(self) -> None:
        """Go to previous selected page."""
        _LOG.info("Go to previous selected page")

    def execGotoPageDialog(self) -> None:
        """Execute go to page dialog."""
        _LOG.info("Execute go to page dialog")

    def showAboutDialog(self) -> None:
        """Show about dialog."""
        _LOG.info("Show About Dialog")
        AboutDialog(self.ui).show()

    def reloadCurrentPage(self) -> None:
        """Reload current page."""
        _LOG.info("Reload current page")

    def switchFilter1(self) -> None:
        """Switch filter 1."""
        _LOG.info("Switch filter 1")

    def switchFilter2(self) -> None:
        """Switch filter 2."""
        _LOG.info("Switch filter 2")

    def switchFilter3(self) -> None:
        """Switch filter 3."""
        _LOG.info("Switch filter 3")

    def switchFilter4(self) -> None:
        """Switch filter 4."""
        _LOG.info("Switch filter 4")

    def switchFilter5(self) -> None:
        """Switch filter 5."""
        _LOG.info("Switch filter 5")

    def switchFilter6(self) -> None:
        """Switch filter 6."""
        _LOG.info("Switch filter 6")

    def setSelectionModeEnabled(self, enabled: bool) -> None:
        """Set selection mode enabled."""
        _LOG.info(f"Set selection mode enabled: {enabled}")

    def thumbViewFocusToggled(self, checked: bool) -> None:
        """Toggle thumb view focus."""
        _LOG.info(f"Thumb view focus toggled: {checked}")

    def pageOrderingChanged(self, index: int) -> None:
        """Page ordering changed."""
        _LOG.info(f"Page ordering changed: {index}")

    def fixDpiDialogRequested(self) -> None:
        """Fix DPI dialog requested."""
        _LOG.info("Fix DPI dialog requested")

    def showRelinkingDialog(self) -> None:
        """Show relinking dialog."""
        _LOG.info("Show relinking dialog")

    def debugToggled(self, enabled: bool) -> None:
        """Debug toggled."""
        _LOG.info(f"Debug toggled: {enabled}")

    def openSettingsDialog(self) -> None:
        """Open settings dialog."""
        _LOG.info("Open settings dialog")

    def openDefaultParamsDialog(self) -> None:
        """Open default params dialog."""
        _LOG.info("Open default params dialog")

    def newProject(self) -> None:
        """New project."""
        _LOG.info("New project")

    def saveProjectTriggered(self) -> None:
        """Save project triggered."""
        _LOG.info("Save project triggered")

    def saveProjectAsTriggered(self) -> None:
        """Save project as triggered."""
        _LOG.info("Save project as triggered")

    def closeProject(self) -> None:
        """Close project."""
        _LOG.info("Close project")

    def quit(self, _clicked: bool) -> None:
        """Handle close event."""
        _LOG.info("Quit")
        QtWidgets.QApplication.instance().quit()  # type: ignore[union-attr]

    def show(self):
        """Show the window."""
        self.ui.show()
