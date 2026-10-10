"""Main Window."""

import enum
import pathlib
from collections.abc import Callable

from loguru import logger as _LOG
from PySide6 import QtCore, QtGui, QtUiTools, QtWidgets
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QGraphicsView,
    QMessageBox,
    QScrollArea,
    QStackedLayout,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from scantailor.app.ui import UI_FOLDER
from scantailor.app.ui.dialogs.about import AboutDialog
from scantailor.app.ui.dialogs.load_files_status import LoadFilesStatusDialog
from scantailor.app.ui.dialogs.project_files import ProjectFilesDialog
from scantailor.app.ui.utils import get_cwidget
from scantailor.app.ui.widgets.stage_list_view import StageListView
from scantailor.core.image_io import ImageLoadError, list_image_pages
from scantailor.core.models import ImageId, PageId
from scantailor.core.project import ImageMetadata, LayoutDirection, Project
from scantailor.core.stage_params import apply_stage_params, collect_stage_params
from scantailor.core.stage_sequence import StageSequence

PROJECT_SUFFIX = ".ScanTailor"
APP_TITLE = "ScanTailor Advanced"
_LAST_DIR_KEY = "project/lastDir"
_LAST_INPUT_DIR_KEY = "project/lastInputDir"


class SaveChoice(enum.Enum):
    """Answer to the unsaved-changes prompt."""

    SAVE = enum.auto()
    DISCARD = enum.auto()
    CANCEL = enum.auto()


ProjectFilesDialogFactory = Callable[[QWidget], ProjectFilesDialog]
LoadStatusPresenter = Callable[
    [list[pathlib.Path], list[tuple[pathlib.Path, str]], QWidget], bool
]
SavePrompt = Callable[[], SaveChoice]


def _load_main_window_ui() -> QWidget:
    """Load ``MainWindow.ui`` with the promoted ``StageListView`` registered.

    Without registration QUiLoader silently substitutes the plain base widget
    for promoted classes.

    Returns:
        QWidget: The main window widget.
    """
    loader = QtUiTools.QUiLoader()
    loader.registerCustomWidget(StageListView)
    ui_file = QtCore.QFile(str(UI_FOLDER / "MainWindow.ui"))
    ui_file.open(QtCore.QFile.OpenModeFlag.ReadOnly)
    try:
        ui = loader.load(ui_file)
    finally:
        ui_file.close()
    return ui


class _CloseGuard(QtCore.QObject):
    """Event filter that lets the manager veto closing the window."""

    def __init__(
        self, can_close: Callable[[], bool], parent: QtCore.QObject | None = None
    ) -> None:
        """Create the filter.

        Args:
            can_close (Callable[[], bool]): Returns False to keep the window open.
            parent (QtCore.QObject | None): Owner of the filter.
        """
        super().__init__(parent)
        self._can_close = can_close

    def eventFilter(self, watched: QtCore.QObject, event: QtCore.QEvent) -> bool:
        """Ignore close events the manager vetoes.

        Args:
            watched (QtCore.QObject): The filtered object.
            event (QtCore.QEvent): The event.

        Returns:
            bool: True when the event was consumed.
        """
        if event.type() == QtCore.QEvent.Type.Close and not self._can_close():
            event.ignore()
            return True
        return False


class MainWindowManager:
    """Controller Class for the Main Application window for ScanTailor.

    Note: This class is not a QMainWindow.
    When using QtUiLoader to load a QMainWindow based .ui file, and setting the parent
    as a QMainWindow, we get weird behavior.
    e.g. https://stackoverflow.com/questions/53828666/pyside2-qmainwindow-loaded-from-ui-file-not-triggering-window-events
    """

    def __init__(
        self,
        *,
        project_files_dialog_factory: ProjectFilesDialogFactory | None = None,
        load_status_presenter: LoadStatusPresenter | None = None,
        save_prompt: SavePrompt | None = None,
    ):
        """Application Initialize.

        The keyword arguments replace the modal dialogs so they can be driven
        without user interaction.

        Args:
            project_files_dialog_factory (ProjectFilesDialogFactory | None): Builds
                the dialog used to pick the files of a new project or to add pages.
            load_status_presenter (LoadStatusPresenter | None): Shows which files
                failed to load; returns False when the user abandons the import.
            save_prompt (SavePrompt | None): Asks what to do with unsaved changes.
        """
        super().__init__()
        self.ui = _load_main_window_ui()

        self.project_files_dialog_factory: ProjectFilesDialogFactory = (
            project_files_dialog_factory or self._default_project_files_dialog
        )
        self.load_status_presenter: LoadStatusPresenter = (
            load_status_presenter or self._default_load_status_presenter
        )
        self.save_prompt: SavePrompt = save_prompt or self._ask_save_changes

        self.project: Project | None = None
        self.stages = StageSequence()
        self.current_page: PageId | None = None
        self._modified = False
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
        self.filterList = get_cwidget(self.ui, StageListView, "filterList")
        self.actionAddFiles = QtGui.QAction(self.ui.tr("Add Files..."), self.ui)
        self._insert_add_files_action()

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
        self.actionAddFiles.triggered.connect(self.addFiles)

        # Shortcuts of actions that are not in any menu only fire when attached
        # to a widget.
        self.ui.addActions(self._switch_filter_actions())

        self._close_guard = _CloseGuard(self._close_project_interactive, self.ui)
        self.ui.installEventFilter(self._close_guard)

        # if self.settings.value("mainWindow/maximized") == False:
        #    if not self.restoreGeometry(geom.toByteArray()):
        #        pass

        # Additional setup (from C++ constructor)
        # self.setupThumbView()
        # self.updateThumbViewMinWidth()
        # self.updateSortOptions()
        # self.updateAutoSaveTimer()
        # self.setupIcons()
        self._attach_stages()
        self._update_project_actions()
        self._update_window_title()

    def _insert_add_files_action(self) -> None:
        """Place the add-files action in the File menu, above the save entries."""
        file_menu = get_cwidget(self.ui, QtWidgets.QMenu, "menuFile")
        actions = file_menu.actions()
        save_index = actions.index(self.actionSaveProject)
        file_menu.insertAction(actions[save_index - 1], self.actionAddFiles)

    def _switch_filter_actions(self) -> list[QtGui.QAction]:
        """Return the stage switching actions in stage order."""
        return [
            self.actionSwitchFilter1,
            self.actionSwitchFilter2,
            self.actionSwitchFilter3,
            self.actionSwitchFilter4,
            self.actionSwitchFilter5,
            self.actionSwitchFilter6,
        ]

    @property
    def modified(self) -> bool:
        """Whether the project has changes that are not on disk."""
        return self._modified

    @modified.setter
    def modified(self, value: bool) -> None:
        self._modified = value
        self._update_window_title()

    @property
    def is_project_loaded(self) -> bool:
        """Whether a project is currently open."""
        return self.project is not None

    def pages(self) -> list[PageId]:
        """Return the pages of the open project in display order."""
        return self.project.get_pages() if self.project else []

    def _attach_stages(self) -> None:
        """Show the current stage sequence in the filter list."""
        self.filterList.set_stages(self.stages)
        self.filterList.selectionModel().currentRowChanged.connect(
            self._on_stage_row_changed
        )
        self._select_stage_row(self.curFilter)

    def _on_stage_row_changed(
        self, current: QtCore.QModelIndex, _previous: QtCore.QModelIndex
    ) -> None:
        """Track stage changes made by clicking the filter list."""
        if current.isValid():
            self.curFilter = current.row()

    def _select_stage_row(self, row: int) -> None:
        """Select a stage in the filter list.

        Args:
            row (int): Zero-based stage index.
        """
        self.curFilter = row
        self.filterList.setCurrentIndex(self.filterList.model().index(row, 0))

    def _update_project_actions(self) -> None:
        """Enable the actions that need an open project."""
        loaded = self.is_project_loaded
        for action in (
            self.actionSaveProject,
            self.actionSaveProjectAs,
            self.actionCloseProject,
            self.actionAddFiles,
            self.actionFixDpi,
            self.actionRelinking,
        ):
            action.setEnabled(loaded)

    def _update_window_title(self) -> None:
        """Show the project name and, via ``[*]``, the unsaved-changes marker."""
        if not self.is_project_loaded:
            self.ui.setWindowTitle(APP_TITLE)
            self.ui.setWindowModified(False)
            return
        name = (
            pathlib.Path(self.projectFile).stem
            if self.projectFile
            else self.ui.tr("Unnamed")
        )
        self.ui.setWindowTitle(f"{name}[*] - {APP_TITLE}")
        self.ui.setWindowModified(self._modified)

    def allPages(self) -> list[PageId]:
        """Return all pages."""
        return self.pages()

    def selectedPages(self) -> None:  # std::set<PageId>
        """Return selected pages."""
        pass

    def selectedRanges(self) -> None:  # std::vector<PageRange>
        """Return selected ranges."""
        pass

    def openProject(self, _checked: bool = False) -> None:
        """Ask for a project file and open it, replacing the current project."""
        project_file, _project_filter = QFileDialog.getOpenFileName(
            self.ui,
            self.ui.tr("Open Project"),
            self._last_project_dir(),
            f"{self.ui.tr('Scan Tailor Projects')} (*{PROJECT_SUFFIX})",
        )
        if not project_file:
            return
        _LOG.info(f"Opening project: {project_file}")
        path = pathlib.Path(project_file)
        try:
            project = Project.load(path)
            stages = StageSequence()
            apply_stage_params(stages, project.stage_params)
        except (OSError, ValueError, SyntaxError) as error:
            _LOG.warning(f"Cannot open project {path}: {error}")
            QMessageBox.warning(
                self.ui,
                self.ui.tr("Error"),
                self.ui.tr("Unable to open the project file.") + f"\n{error}",
            )
            return
        if not self._close_project_interactive():
            return
        self._install_project(project, stages, project_file=path, modified=False)
        self.settings.setValue(_LAST_DIR_KEY, str(path.parent))

    def _go_to_page_index(self, index: int) -> None:
        """Make the page at ``index`` current, clamping to the valid range.

        Args:
            index (int): Zero-based position in the page sequence.
        """
        pages = self.pages()
        if pages:
            self.current_page = pages[max(0, min(index, len(pages) - 1))]

    def _current_page_index(self) -> int:
        """Return the position of the current page, or -1 when there is none."""
        pages = self.pages()
        if self.current_page is None or self.current_page not in pages:
            return -1
        return pages.index(self.current_page)

    def goFirstPage(self) -> None:
        """Go to first page."""
        self._go_to_page_index(0)

    def goLastPage(self) -> None:
        """Go to last page."""
        self._go_to_page_index(len(self.pages()) - 1)

    def goNextPage(self) -> None:
        """Go to next page."""
        self._go_to_page_index(self._current_page_index() + 1)

    def goPrevPage(self) -> None:
        """Go to previous page."""
        self._go_to_page_index(max(self._current_page_index() - 1, 0))

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
        """Switch to stage 1."""
        self._select_stage_row(0)

    def switchFilter2(self) -> None:
        """Switch to stage 2."""
        self._select_stage_row(1)

    def switchFilter3(self) -> None:
        """Switch to stage 3."""
        self._select_stage_row(2)

    def switchFilter4(self) -> None:
        """Switch to stage 4."""
        self._select_stage_row(3)

    def switchFilter5(self) -> None:
        """Switch to stage 5."""
        self._select_stage_row(4)

    def switchFilter6(self) -> None:
        """Switch to stage 6."""
        self._select_stage_row(5)

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
        """Create a project from files chosen in the project files dialog."""
        if not self._close_project_interactive():
            return
        dialog = self.project_files_dialog_factory(self.ui)
        if not self._run_dialog(dialog):
            return
        # The dialog is deleted once the event loop runs again, e.g. in a nested
        # dialog, so read everything it holds first.
        input_dir = dialog.get_input_directory()
        names = dialog.get_project_files()
        output_dir = dialog.get_output_directory()
        layout_direction: LayoutDirection = "RTL" if dialog.is_rtl_layout() else "LTR"
        if input_dir is None or not names:
            QMessageBox.warning(
                self.ui,
                self.ui.tr("Error"),
                self.ui.tr("No files were added to the project."),
            )
            return
        pages = self._read_pages([input_dir / name for name in names])
        if pages is None:
            return
        project = Project(
            output_directory=output_dir or input_dir / "out",
            layout_direction=layout_direction,
        )
        for image_id, metadata in pages:
            project.add_image(image_id, metadata)
        self._install_project(
            project, StageSequence(), project_file=None, modified=True
        )
        self.settings.setValue(_LAST_INPUT_DIR_KEY, str(input_dir))

    def addFiles(self) -> None:
        """Append images chosen in the project files dialog to the open project."""
        if self.project is None:
            return
        dialog = self.project_files_dialog_factory(self.ui)
        if not self._run_dialog(dialog):
            return
        input_dir = dialog.get_input_directory()
        if input_dir is None:
            return
        present = {info.id.file_path for info in self.project.images}
        paths = [
            input_dir / name
            for name in dialog.get_project_files()
            if input_dir / name not in present
        ]
        pages = self._read_pages(paths)
        if not pages:
            return
        for image_id, metadata in pages:
            self.project.add_image(image_id, metadata)
        if self.current_page is None:
            self.current_page = self.pages()[0]
        self.modified = True

    @staticmethod
    def _run_dialog(dialog: ProjectFilesDialog) -> bool:
        """Run a modal dialog and release it, as it is parented to the window.

        Args:
            dialog (ProjectFilesDialog): The dialog to run.

        Returns:
            bool: True if the user accepted it.
        """
        accepted = dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted
        dialog.deleteLater()
        return accepted

    def _read_pages(
        self, paths: list[pathlib.Path]
    ) -> list[tuple[ImageId, ImageMetadata]] | None:
        """Read the page metadata of image files, reporting files that fail.

        Args:
            paths (list[pathlib.Path]): Image files in the order they are added.

        Returns:
            list[tuple[ImageId, ImageMetadata]] | None: Pages of every readable
                file, or None when the user abandons the import after failures.
        """
        pages: list[tuple[ImageId, ImageMetadata]] = []
        loaded: list[pathlib.Path] = []
        failed: list[tuple[pathlib.Path, str]] = []
        for path in paths:
            try:
                pages.extend(list_image_pages(path))
            except (ImageLoadError, OSError) as error:
                failed.append((path, str(error)))
            else:
                loaded.append(path)
        if failed and not self.load_status_presenter(loaded, failed, self.ui):
            return None
        return pages

    def _install_project(
        self,
        project: Project,
        stages: StageSequence,
        *,
        project_file: pathlib.Path | None,
        modified: bool,
    ) -> None:
        """Make ``project`` the open project and refresh everything derived from it.

        Args:
            project (Project): The project to show.
            stages (StageSequence): Stages holding the project's parameters.
            project_file (pathlib.Path | None): Where the project is stored, or
                None for one that was never saved.
            modified (bool): Whether the project has unsaved changes.
        """
        self.project = project
        self.stages = stages
        self.projectFile = str(project_file) if project_file else ""
        self._modified = modified
        pages = project.get_pages()
        if project.selected_page in pages:
            self.current_page = project.selected_page
        else:
            self.current_page = pages[0] if pages else None
        self._attach_stages()
        self._update_project_actions()
        self._update_window_title()

    def _default_project_files_dialog(self, parent: QWidget) -> ProjectFilesDialog:
        """Build the file chooser, starting from the open project's folder.

        Args:
            parent (QWidget): Owner of the dialog.

        Returns:
            ProjectFilesDialog: The dialog, not yet shown.
        """
        last_input_dir = str(self.settings.value(_LAST_INPUT_DIR_KEY, "", type=str))
        input_dir = pathlib.Path(last_input_dir) if last_input_dir else None
        output_dir = None
        if self.project and self.project.images:
            input_dir = self.project.images[0].id.file_path.parent
            output_dir = self.project.output_directory
        return ProjectFilesDialog(
            input_dir=input_dir,
            output_dir=output_dir,
            parent=parent,
        )

    @staticmethod
    def _default_load_status_presenter(
        loaded: list[pathlib.Path],
        failed: list[tuple[pathlib.Path, str]],
        parent: QWidget,
    ) -> bool:
        """Show the load status dialog; see ``LoadStatusPresenter``."""
        return LoadFilesStatusDialog.show_status(loaded, failed, parent)

    def saveProjectTriggered(self) -> None:
        """Save the project, asking for a file name if it has none yet."""
        self.save_project()

    def save_project(self) -> bool:
        """Save the project to its file, or ask for one when it has none.

        Returns:
            bool: True if the project was written.
        """
        if self.project is None:
            return False
        if not self.projectFile:
            return self.save_project_as()
        return self._write_project(pathlib.Path(self.projectFile))

    def save_project_as(self) -> bool:
        """Ask for a file name and save the project there.

        Returns:
            bool: True if the project was written; False if the user cancelled
                or writing failed.
        """
        if self.project is None:
            return False
        chosen, _selected_filter = QFileDialog.getSaveFileName(
            self.ui,
            "",
            self._last_project_dir(),
            f"{self.ui.tr('Scan Tailor Projects')} (*{PROJECT_SUFFIX})",
        )
        if not chosen:
            return False
        path = pathlib.Path(chosen)
        if path.suffix.lower() != PROJECT_SUFFIX.lower():
            path = path.with_name(path.name + PROJECT_SUFFIX)
        if not self._write_project(path):
            return False
        self.projectFile = str(path)
        self._update_window_title()
        self.settings.setValue(_LAST_DIR_KEY, str(path.parent))
        return True

    def _write_project(self, path: pathlib.Path) -> bool:
        """Write the project, including stage parameters, to ``path``.

        Args:
            path (pathlib.Path): Destination file.

        Returns:
            bool: True on success; on failure the user is told and False returned.
        """
        project = self.project
        if project is None:
            return False
        project.stage_params = collect_stage_params(self.stages, project.get_pages())
        project.selected_page = self.current_page
        try:
            project.save(path)
        except OSError as error:
            _LOG.warning(f"Cannot save project {path}: {error}")
            QMessageBox.warning(
                self.ui,
                self.ui.tr("Error"),
                self.ui.tr("Error saving the project file!") + f"\n{error}",
            )
            return False
        self.modified = False
        return True

    def _last_project_dir(self) -> str:
        """Return the folder project file dialogs should start in."""
        if self.projectFile:
            return str(pathlib.Path(self.projectFile).parent)
        return str(
            self.settings.value(_LAST_DIR_KEY, str(pathlib.Path.cwd()), type=str)
        )

    def saveProjectAsTriggered(self) -> None:
        """Save the project under a new file name."""
        self.save_project_as()

    def closeProject(self) -> None:
        """Close the project, offering to save unsaved changes first."""
        self._close_project_interactive()

    def _close_project_interactive(self) -> bool:
        """Close the project unless the user cancels.

        Returns:
            bool: True if no project remains open; False if the user cancelled or
                saving failed.
        """
        if self.project is None:
            return True
        if self._modified:
            choice = self.save_prompt()
            if choice is SaveChoice.CANCEL:
                return False
            if choice is SaveChoice.SAVE and not self.save_project():
                return False
        self._close_project_without_saving()
        return True

    def _close_project_without_saving(self) -> None:
        """Drop the project and reset the window to its empty state."""
        self.project = None
        self.stages = StageSequence()
        self.projectFile = ""
        self.current_page = None
        self._modified = False
        self._attach_stages()
        self._update_project_actions()
        self._update_window_title()

    def _ask_save_changes(self) -> SaveChoice:
        """Ask whether to save the unsaved changes.

        Returns:
            SaveChoice: The user's decision; closing the box counts as cancel.
        """
        box = QMessageBox(
            QMessageBox.Icon.Question,
            self.ui.tr("Save Project"),
            self.ui.tr("Save this project?"),
            buttons=QMessageBox.StandardButton.Save
            | QMessageBox.StandardButton.Discard
            | QMessageBox.StandardButton.Cancel,
            parent=self.ui,
        )
        box.setDefaultButton(QMessageBox.StandardButton.Save)
        box.setEscapeButton(QMessageBox.StandardButton.Cancel)
        box.exec()
        match box.standardButton(box.clickedButton()):
            case QMessageBox.StandardButton.Save:
                return SaveChoice.SAVE
            case QMessageBox.StandardButton.Discard:
                return SaveChoice.DISCARD
            case _:
                return SaveChoice.CANCEL

    def quit(self, _clicked: bool = False) -> None:
        """Quit the application after offering to save unsaved changes."""
        if self._close_project_interactive():
            self.closing = True
            QtWidgets.QApplication.instance().quit()  # type: ignore[union-attr]

    def show(self):
        """Show the window."""
        self.ui.show()
