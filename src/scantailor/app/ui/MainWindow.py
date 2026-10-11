"""Main Window."""

import enum
import pathlib
from collections.abc import Callable, Hashable
from typing import Any, cast

from loguru import logger as _LOG
from PySide6 import QtCore, QtGui, QtWidgets
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

from scantailor.app.ui import UI_FOLDER, load_ui_widget
from scantailor.app.ui.batch_runner import BatchRunner
from scantailor.app.ui.dialogs.about import AboutDialog
from scantailor.app.ui.dialogs.apply_to import ApplyToDialog
from scantailor.app.ui.dialogs.batch_progress import BatchProgressDialog
from scantailor.app.ui.dialogs.fix_dpi import FixDpiDialog
from scantailor.app.ui.dialogs.load_files_status import LoadFilesStatusDialog
from scantailor.app.ui.dialogs.project_files import ProjectFilesDialog
from scantailor.app.ui.dialogs.relinking import RelinkingDialog
from scantailor.app.ui.page_renderer import PageRenderer, render_thumbnail
from scantailor.app.ui.stage_panels import StagePanels
from scantailor.app.ui.thumbnails import ThumbnailListWidget
from scantailor.app.ui.utils import get_cwidget
from scantailor.app.ui.widgets.non_owning_widget import NonOwningWidget
from scantailor.app.ui.widgets.processing_indicator import ProcessingIndicator
from scantailor.app.ui.widgets.stage_list_view import StageListView
from scantailor.core import propagation
from scantailor.core.batch import BatchResult, ProgressEvent
from scantailor.core.image_io import ImageLoadError, list_image_pages
from scantailor.core.models import ImageId, PageId
from scantailor.core.project import ImageInfo, ImageMetadata, LayoutDirection, Project
from scantailor.core.recent_projects import RecentProjects
from scantailor.core.relinker import Relinker, image_directories
from scantailor.core.stage_params import apply_stage_params, collect_stage_params
from scantailor.core.stage_sequence import FilterStage, StageSequence

PROJECT_SUFFIX = ".ScanTailor"
APP_TITLE = "ScanTailor Advanced"
_LAST_DIR_KEY = "project/lastDir"
_LAST_INPUT_DIR_KEY = "project/lastInputDir"
THUMBNAIL_SIZE = QtCore.QSize(160, 160)
# Written by SystemLoadWidget: how many pages a batch processes at once.
_BATCH_THREADS_KEY = "settings/batch_processing_threads"
_SELECT_CONTENT_OPTIONS = {
    "content_box": "Apply content box",
    "page_box": "Apply page box",
}


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
FixDpiDialogFactory = Callable[[list[ImageInfo], QWidget], FixDpiDialog]
RelinkingDialogFactory = Callable[[list[pathlib.Path], QWidget], RelinkingDialog]
PageMetadata = list[tuple[ImageId, ImageMetadata]]


def _message_widget(text: str) -> QtWidgets.QLabel:
    """Build a centred, wrapping label that fills the main area with a message."""
    label = QtWidgets.QLabel(text)
    label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
    label.setWordWrap(True)
    return label


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
        fix_dpi_dialog_factory: FixDpiDialogFactory | None = None,
        relinking_dialog_factory: RelinkingDialogFactory | None = None,
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
            fix_dpi_dialog_factory (FixDpiDialogFactory | None): Builds the dialog
                that corrects undefined or implausible DPI.
            relinking_dialog_factory (RelinkingDialogFactory | None): Builds the
                dialog that points moved image folders at their new location.
        """
        super().__init__()
        self.ui = load_ui_widget(
            UI_FOLDER / "MainWindow.ui",
            extra_widgets=[StageListView, NonOwningWidget],
        )

        self.project_files_dialog_factory: ProjectFilesDialogFactory = (
            project_files_dialog_factory or self._default_project_files_dialog
        )
        self.load_status_presenter: LoadStatusPresenter = (
            load_status_presenter or self._default_load_status_presenter
        )
        self.save_prompt: SavePrompt = save_prompt or self._ask_save_changes
        self.fix_dpi_dialog_factory: FixDpiDialogFactory = (
            fix_dpi_dialog_factory or FixDpiDialog
        )
        self.relinking_dialog_factory: RelinkingDialogFactory = (
            relinking_dialog_factory or RelinkingDialog
        )

        self.project: Project | None = None
        self.stages = StageSequence()
        self._current_page: PageId | None = None
        self._panels: StagePanels | None = None
        self._thumbnail_widget: QWidget | None = None
        self.renderer = PageRenderer(self.ui)
        self.renderer.render_done.connect(self._on_render_done)
        self.renderer.render_failed.connect(self._on_render_failed)
        self._render_pending = False
        self._thumbnail_epoch = 0
        self._thumbnail_versions: dict[PageId, int] = {}
        self.batch_runner = BatchRunner(self.ui)
        self.batch_runner.progress.connect(self._on_batch_progress)
        self.batch_runner.finished.connect(self._on_batch_finished)
        self.batch_runner.failed.connect(self._on_batch_failed)
        self.batch_dialog: BatchProgressDialog | None = None
        self.last_batch_result: BatchResult | None = None
        self._modified = False
        # Serialised project as last saved or opened; None while never saved.
        self._saved_state: str | None = None
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
        self.menuOpenRecent = QtWidgets.QMenu(self.ui.tr("Open Recent"), self.ui)
        self._insert_recent_menu()

        # Layouts
        self.imageFrameLayout = QStackedLayout(self.imageViewFrame)
        self.not_ready_widget = _message_widget(
            self.ui.tr(
                "Output is not yet possible, as the final size of pages is not "
                "yet known.\nTo determine it, run batch processing at "
                '"Select Content" or "Margins".'
            )
        )
        self.imageFrameLayout.addWidget(self.not_ready_widget)
        self.processing_indicator = ProcessingIndicator(self.ui.tr("Processing..."))
        get_cwidget(self.ui, QtWidgets.QStatusBar, "statusBar").addPermanentWidget(
            self.processing_indicator
        )
        self.optionsFrameLayout = QVBoxLayout(self.filterOptions)
        self.optionsFrameLayout.setContentsMargins(0, 0, 0, 0)

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
        self.thumbnails = ThumbnailListWidget(
            self._render_page_thumbnail,
            params_hash=self._thumbnail_digest,
            thumbnail_size=THUMBNAIL_SIZE,
        )
        self.thumbnails.page_selected.connect(self.page_selected)
        self.set_thumbnail_widget(self.thumbnails)
        self._wire_thumbnail_controls()
        self.filterList.launch_batch_processing.connect(self.startBatchProcessing)
        self._attach_stages()
        self._update_project_actions()
        self._update_window_title()
        self._rebuild_recent_menu()

    def _render_page_thumbnail(self, page: PageId, stage: str) -> QtGui.QImage:
        """Render a thumbnail on a worker thread; settings are only read."""
        return render_thumbnail(
            page, self.stages, FilterStage(int(stage)), THUMBNAIL_SIZE
        )

    def _thumbnail_digest(self, page: PageId, _stage: str) -> Hashable:
        """Return what identifies the parameters a page's thumbnail was made with."""
        return (self._thumbnail_epoch, self._thumbnail_versions.get(page, 0))

    def _invalidate_thumbnails(self, page: PageId | None = None) -> None:
        """Make thumbnails render again after parameters changed.

        Args:
            page (PageId | None): The page to refresh, or None for all pages.
        """
        if page is None:
            self._thumbnail_epoch += 1
        else:
            self._thumbnail_versions[page] = self._thumbnail_versions.get(page, 0) + 1
        self.thumbnails.invalidate(page)

    def _wire_thumbnail_controls(self) -> None:
        """Connect the focus, selection mode and sort controls to the thumbnails."""
        self.thumbnails.order_combo.hide()
        self.sortOptions.clear()
        self.sortOptions.addItems(
            [
                self.thumbnails.order_combo.itemText(row)
                for row in range(self.thumbnails.order_combo.count())
            ]
        )
        self.sortOptions.setCurrentIndex(self.thumbnails.order_combo.currentIndex())
        self.sortOptions.currentIndexChanged.connect(self.thumbnails.set_order)
        self.focusButton.toggled.connect(self.thumbnails.set_follow_current)
        self.selectionModeBtn.toggled.connect(
            self.thumbnails.set_multi_selection_enabled
        )
        self.thumbnails.set_follow_current(self.focusButton.isChecked())
        self.thumbnails.set_multi_selection_enabled(self.selectionModeBtn.isChecked())

    def _insert_add_files_action(self) -> None:
        """Place the add-files action in the File menu, above the save entries."""
        file_menu = get_cwidget(self.ui, QtWidgets.QMenu, "menuFile")
        actions = file_menu.actions()
        save_index = actions.index(self.actionSaveProject)
        file_menu.insertAction(actions[save_index - 1], self.actionAddFiles)

    def _insert_recent_menu(self) -> None:
        """Place the recent projects submenu right below Open Project."""
        file_menu = get_cwidget(self.ui, QtWidgets.QMenu, "menuFile")
        actions = file_menu.actions()
        following = actions[actions.index(self.actionOpenProject) + 1]
        file_menu.insertMenu(following, self.menuOpenRecent)
        self.menuOpenRecent.aboutToShow.connect(self._rebuild_recent_menu)

    def _rebuild_recent_menu(self) -> None:
        """List the recent projects that still exist, dropping the others."""
        recent = RecentProjects.read(self.settings)
        if not recent.validate():
            recent.write(self.settings)
        self.menuOpenRecent.clear()
        for path in recent.enumerate():
            action = self.menuOpenRecent.addAction(path.replace("&", "&&"))
            action.setToolTip(path)
            action.triggered.connect(
                lambda _checked=False, chosen=path: self.open_project_file(
                    pathlib.Path(chosen)
                )
            )
        self.menuOpenRecent.setEnabled(not self.menuOpenRecent.isEmpty())

    def _remember_project(self, path: pathlib.Path) -> None:
        """Record ``path`` as the most recently used project and the last folder.

        Args:
            path (pathlib.Path): A project file that was opened or saved.
        """
        recent = RecentProjects.read(self.settings)
        recent.set_most_recent(path)
        recent.write(self.settings)
        self.settings.setValue(_LAST_DIR_KEY, str(path.parent))
        self._rebuild_recent_menu()

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
    def current_page(self) -> PageId | None:
        """The page shown in the main area."""
        return self._current_page

    @current_page.setter
    def current_page(self, page: PageId | None) -> None:
        if page == self._current_page:
            return
        self._current_page = page
        if page is not None and page != self.thumbnails.current_page():
            self.thumbnails.set_current_page(page)
        self._show_current_page()

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
        self._show_stage()

    def _on_stage_row_changed(
        self, current: QtCore.QModelIndex, _previous: QtCore.QModelIndex
    ) -> None:
        """Track stage changes made by clicking the filter list."""
        if current.isValid():
            self.curFilter = current.row()
            self._show_stage()

    def _show_stage(self) -> None:
        """Show the options widget and image view of the current stage.

        Nothing changes when they already belong to the current stage and stage
        sequence.
        """
        stage = FilterStage(self.curFilter)
        panels = self._panels
        if panels and panels.stage == stage and panels.sequence is self.stages:
            return
        previous = panels.stage if panels and panels.sequence is self.stages else None
        self._discard_panels()
        if previous is not None:
            self._propagate_between_stages(previous, stage)
        self.thumbnails.set_stage(str(int(stage)))
        panels = StagePanels(stage, self.stages, self.ui)
        panels.settings_changed.connect(self._on_settings_edited)
        self._connect_apply_requests(panels)
        self.optionsFrameLayout.addWidget(panels.options)
        self.imageFrameLayout.addWidget(panels.view)
        self.imageFrameLayout.setCurrentWidget(panels.view)
        self._panels = panels
        self._show_current_page()

    def _propagate_between_stages(
        self, previous: FilterStage, new: FilterStage
    ) -> None:
        """Pass results on to the stages below when the user moves down the list.

        Args:
            previous (FilterStage): The stage being left.
            new (FilterStage): The stage being entered.
        """
        project = self.project
        if project is None:
            return
        if previous == FilterStage.FIX_ORIENTATION < new:
            changed = propagation.propagate_page_orientations(project, self.stages)
            if changed:
                self.modified = True
                self._refresh_page_list()
        if previous <= FilterStage.SELECT_CONTENT < new:
            propagation.propagate_content_boxes(project, self.stages)
            self._invalidate_thumbnails()

    def _refresh_page_list(self) -> None:
        """List the project's pages again after the page count of an image changed."""
        pages = self.pages()
        previous = self._current_page
        if previous in pages or previous is None:
            target = previous
        else:
            same_image = [page for page in pages if page.image_id == previous.image_id]
            target = (same_image or pages or [None])[0]
        self.thumbnails.set_project(self.project)
        self._current_page = target
        if target is not None:
            self.thumbnails.set_current_page(target)
        if target != previous:
            self._show_current_page()

    def _discard_panels(self) -> None:
        """Remove the widgets of the stage that is being left."""
        panels, self._panels = self._panels, None
        if panels is None:
            return
        self._cancel_render()
        for widget, layout in (
            (panels.options, self.optionsFrameLayout),
            (panels.view, self.imageFrameLayout),
        ):
            layout.removeWidget(widget)
            widget.setParent(None)
            widget.deleteLater()
        panels.deleteLater()

    def _show_current_page(self) -> None:
        """Point the stage panels at the current page and render it."""
        panels = self._panels
        if panels is None:
            return
        panels.set_page(self._current_page)
        self._request_render()

    def _request_render(self) -> None:
        """Render the current page up to the current stage in the background."""
        panels = self._panels
        if panels is None or self._current_page is None:
            self._cancel_render()
            return
        if self.batch_runner.is_running:
            return
        if not self._is_output_ready():
            self._cancel_render()
            self._show_not_ready(True)
            return
        self._show_not_ready(False)
        self._set_render_pending(True)
        self.renderer.request(self._current_page, self.stages, panels.stage)

    def _cancel_render(self) -> None:
        self.renderer.cancel()
        self._set_render_pending(False)

    def _set_render_pending(self, pending: bool) -> None:
        self._render_pending = pending
        self.processing_indicator.set_active(pending or self.batch_runner.is_running)

    def _is_output_ready(self) -> bool:
        """Whether the Output stage may render the current page.

        Output sizes depend on every page's content size, which Select Content
        and Page Layout record as pages are processed.
        """
        panels = self._panels
        if panels is None or panels.stage != FilterStage.OUTPUT:
            return True
        return self.stages.page_layout_filter.settings.check_everything_defined(
            self.pages(), ignore=self._current_page
        )

    def _show_not_ready(self, not_ready: bool) -> None:
        """Swap the stage's options and view for the not-ready message, or undo it."""
        panels = self._panels
        if panels is None:
            return
        panels.options.setVisible(not not_ready)
        self.imageFrameLayout.setCurrentWidget(
            self.not_ready_widget if not_ready else panels.view
        )

    def _on_settings_edited(self) -> None:
        """React to an edit made in the options widget."""
        self.modified = True
        self._invalidate_thumbnails(self._current_page)
        self._request_render()

    def _on_render_done(self, page_id: PageId, image: object) -> None:
        self._set_render_pending(False)
        if self._panels is not None and page_id == self._current_page:
            self._panels.show_image(image)  # type: ignore[arg-type]

    def _on_render_failed(self, page_id: PageId, message: str) -> None:
        self._set_render_pending(False)
        _LOG.warning(f"Page {page_id} could not be shown: {message}")

    def set_thumbnail_widget(self, widget: QWidget) -> None:
        """Show ``widget`` in the thumbnails dock instead of the placeholder view.

        Args:
            widget (QWidget): The thumbnail list. Replacing it later unparents it.
        """
        previous = self._thumbnail_widget or self.thumbView
        layout = self.ui.findChild(QVBoxLayout, "verticalLayout")
        assert layout is not None
        layout.replaceWidget(previous, widget)
        previous.hide()
        if previous is not self.thumbView:
            previous.setParent(None)
        widget.show()
        self._thumbnail_widget = widget

    def page_selected(self, page: PageId) -> None:
        """Make ``page`` current, e.g. when picked in the thumbnails.

        Args:
            page (PageId): A page of the open project; others are ignored.
        """
        if page in self.pages():
            self.current_page = page

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
        self.filterList.set_batch_processing_possible(loaded)
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

    def selectedPages(self) -> list[PageId]:
        """Return the pages selected in the thumbnail list, in display order."""
        return self.thumbnails.selected_pages()

    def _connect_apply_requests(self, panels: StagePanels) -> None:
        """Route the options widget's "Apply to..." buttons to the apply dialog.

        Args:
            panels (StagePanels): The panels of the stage being shown.
        """
        options = cast(Any, panels.options)
        requests: dict[FilterStage, list[tuple[str, str]]] = {
            FilterStage.FIX_ORIENTATION: [("apply_to_requested", "")],
            FilterStage.DESKEW: [("apply_to_requested", "")],
            FilterStage.SELECT_CONTENT: [("apply_to_requested", "")],
            FilterStage.PAGE_LAYOUT: [
                ("apply_margins_requested", "margins"),
                ("apply_alignment_requested", "alignment"),
            ],
            FilterStage.OUTPUT: [("apply_to_requested", "")],
        }
        for signal_name, part in requests.get(panels.stage, []):
            getattr(options, signal_name).connect(
                lambda *_args, stage=panels.stage, part=part: self.show_apply_dialog(
                    stage, part
                )
            )

    def show_apply_dialog(self, stage: FilterStage, part: str = "") -> bool:
        """Ask which pages should get the current page's settings, then copy them.

        Args:
            stage (FilterStage): The stage whose settings are copied.
            part (str): For Page Layout, ``"margins"`` or ``"alignment"``.

        Returns:
            bool: True if settings were copied to at least one other page.
        """
        page = self._current_page
        if page is None or self.batch_runner.is_running:
            return False
        titles = {
            "margins": self.ui.tr("Apply Margins"),
            "alignment": self.ui.tr("Apply Alignment"),
        }
        option_labels = (
            {key: self.ui.tr(label) for key, label in _SELECT_CONTENT_OPTIONS.items()}
            if stage == FilterStage.SELECT_CONTENT
            else {}
        )
        dialog = ApplyToDialog(
            page,
            self.pages(),
            self.selectedPages(),
            self.ui,
            title=titles.get(part, self.ui.tr("Apply to")),
            options=option_labels,
        )
        accepted = dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted
        targets = dialog.pages() if accepted else []
        flags = {key: dialog.option_checked(key) for key in option_labels}
        dialog.deleteLater()
        others = [target for target in targets if target != page]
        if not others:
            return False
        self.apply_to_pages(
            stage,
            others,
            part=part,
            content_box=flags.get("content_box", True),
            page_box=flags.get("page_box", True),
        )
        return True

    def apply_to_pages(
        self,
        stage: FilterStage,
        targets: list[PageId],
        *,
        part: str = "",
        content_box: bool = True,
        page_box: bool = True,
    ) -> None:
        """Copy the current page's settings of ``stage`` to other pages.

        Args:
            stage (FilterStage): The stage whose settings are copied.
            targets (list[PageId]): Pages that receive the settings.
            part (str): For Page Layout, ``"margins"`` or ``"alignment"``.
            content_box (bool): Select Content: copy a manual content box.
            page_box (bool): Select Content: copy a manual page box.
        """
        source = self._current_page
        if source is None or not targets:
            return
        stages = self.stages
        match stage:
            case FilterStage.FIX_ORIENTATION:
                propagation.apply_orientation(stages, source, targets)
            case FilterStage.DESKEW:
                propagation.apply_deskew(stages, source, targets)
            case FilterStage.SELECT_CONTENT:
                propagation.apply_select_content(
                    stages,
                    source,
                    targets,
                    apply_content_box=content_box,
                    apply_page_box=page_box,
                )
            case FilterStage.PAGE_LAYOUT if part == "alignment":
                propagation.apply_alignment(stages, source, targets)
            case FilterStage.PAGE_LAYOUT:
                propagation.apply_margins(stages, source, targets)
            case FilterStage.OUTPUT:
                propagation.apply_output(stages, source, targets)
            case _:
                return
        self.modified = True
        self._invalidate_thumbnails(None if len(targets) > 1 else targets[0])
        self._request_render()

    def _batch_pages(self) -> list[PageId]:
        """Return the pages a batch processes: the current page and those after it."""
        return self.thumbnails.pages()[max(self._display_position(), 0) :]

    def _batch_worker_count(self) -> int | None:
        """Return the page parallelism chosen in the system load slider, if any."""
        saved = self.settings.value(_BATCH_THREADS_KEY, 0, type=int)
        return saved if isinstance(saved, int) and saved > 0 else None

    def startBatchProcessing(self) -> None:
        """Process the current page and the following ones in the background."""
        project = self.project
        if project is None or self.batch_runner.is_running:
            return
        pages = self._batch_pages()
        if not pages:
            return
        self._cancel_render()
        self._set_batch_ui(True)
        if self.batch_dialog is not None:
            self.batch_dialog.deleteLater()
        dialog = BatchProgressDialog(len(pages), self.ui)
        dialog.stop_requested.connect(self.stopBatchProcessing)
        self.batch_dialog = dialog
        dialog.show()
        self.batch_runner.start(
            self.stages,
            pages,
            project.output_directory,
            layout_direction=project.layout_direction,
            max_workers=self._batch_worker_count(),
        )

    def stopBatchProcessing(self) -> None:
        """Stop starting new pages; pages already running finish."""
        self.batch_runner.cancel()

    def _abort_batch(self) -> None:
        """Stop the batch and wait for its thread, so settings are free to change."""
        if self.batch_runner.is_running:
            self.batch_runner.cancel()
            self.batch_runner.wait()

    def _set_batch_ui(self, in_progress: bool) -> None:
        """Lock the stage controls while the worker owns the settings.

        Args:
            in_progress (bool): True when a batch starts, False when it ends.
        """
        self.filterList.set_batch_processing_in_progress(in_progress)
        self.filterList.setEnabled(not in_progress)
        self.filterOptions.setEnabled(not in_progress)
        self.processing_indicator.set_active(in_progress)

    def _on_batch_progress(self, event: ProgressEvent) -> None:
        if self.batch_dialog is not None:
            self.batch_dialog.add_result(event)
        self._invalidate_thumbnails(event.page)

    def _on_batch_finished(self, result: BatchResult) -> None:
        self.last_batch_result = result
        self._set_batch_ui(False)
        if self.batch_dialog is not None:
            self.batch_dialog.finish(result)
        if self.project is None:
            return
        if result.written or result.failed:
            self.modified = True
        self._invalidate_thumbnails()
        self._request_render()

    def _on_batch_failed(self, message: str) -> None:
        self._set_batch_ui(False)
        _LOG.warning(f"Batch processing failed: {message}")
        if self.batch_dialog is not None:
            self.batch_dialog.deleteLater()
            self.batch_dialog = None
        QMessageBox.warning(
            self.ui,
            self.ui.tr("Batch processing"),
            self.ui.tr("Batch processing failed.") + f"\n{message}",
        )

    def openProject(self, _checked: bool = False) -> None:
        """Ask for a project file and open it, replacing the current project."""
        project_file, _project_filter = QFileDialog.getOpenFileName(
            self.ui,
            self.ui.tr("Open Project"),
            self._last_project_dir(),
            f"{self.ui.tr('Scan Tailor Projects')} (*{PROJECT_SUFFIX})",
        )
        if project_file:
            self.open_project_file(pathlib.Path(project_file))

    def open_project_file(self, path: pathlib.Path) -> bool:
        """Open a project file, replacing the current project.

        Image folders that no longer exist are offered for relinking first.

        Args:
            path (pathlib.Path): The project file.

        Returns:
            bool: True if the project is now open; False if loading failed or the
                user cancelled relinking or the save prompt.
        """
        _LOG.info(f"Opening project: {path}")
        try:
            project = Project.load(path)
        except (OSError, ValueError, SyntaxError) as error:
            self._warn_cannot_open(path, error)
            return False
        relinker = self._relink_missing_images(project)
        if relinker is None:
            return False
        try:
            project.remap_image_paths(relinker.substitute)
            stages = StageSequence()
            apply_stage_params(stages, project.stage_params)
        except ValueError as error:
            self._warn_cannot_open(path, error)
            return False
        dpi_fixed = self._fix_dpi_of_opened_project(project)
        if dpi_fixed is None:
            return False
        if not self._close_project_interactive():
            return False
        self._install_project(
            project,
            stages,
            project_file=path,
            modified=dpi_fixed or not relinker.is_empty(),
        )
        self._remember_project(path)
        return True

    def _fix_dpi_of_opened_project(self, project: Project) -> bool | None:
        """Ask for the DPI of stored images that have none, as on opening in C++.

        Args:
            project (Project): A freshly loaded project; updated in place.

        Returns:
            bool | None: Whether any DPI was changed, or None if the user cancelled.
        """
        if all(info.metadata.is_dpi_ok() for info in project.images):
            return False
        dialog = self.fix_dpi_dialog_factory(list(project.images), self.ui)
        if not self._run_dialog(dialog):
            return None
        changes = dialog.get_dpi_changes()
        for info in project.images:
            if info.id in changes:
                info.metadata = info.metadata.model_copy(
                    update={"dpi": changes[info.id]}
                )
        return bool(changes)

    def _warn_cannot_open(self, path: pathlib.Path, error: Exception) -> None:
        """Tell the user a project file could not be read."""
        _LOG.warning(f"Cannot open project {path}: {error}")
        QMessageBox.warning(
            self.ui,
            self.ui.tr("Error"),
            self.ui.tr("Unable to open the project file.") + f"\n{error}",
        )

    def _relink_missing_images(self, project: Project) -> Relinker | None:
        """Let the user relink the folders of image files that are missing.

        Args:
            project (Project): A freshly loaded project.

        Returns:
            Relinker | None: The mapping to apply (empty when nothing is missing),
                or None when the user cancelled.
        """
        missing = image_directories(
            [info.id.file_path for info in project.images], only_missing=True
        )
        if not missing:
            return Relinker()
        return self._ask_relinker(missing)

    def _ask_relinker(self, directories: list[pathlib.Path]) -> Relinker | None:
        """Show the relinking dialog.

        Args:
            directories (list[pathlib.Path]): Folders to offer for relinking.

        Returns:
            Relinker | None: What the user chose, or None if they cancelled.
        """
        dialog = self.relinking_dialog_factory(directories, self.ui)
        if not self._run_dialog(dialog):
            return None
        return dialog.relinker()

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

    def _display_position(self) -> int:
        """Return the row of the current page in the thumbnail list, or -1."""
        current = self._current_page
        return next(
            (
                row
                for row, page in enumerate(self.thumbnails.pages())
                if page == current
            ),
            -1,
        )

    def _go_to_adjacent_selected_page(self, step: int) -> None:
        """Make the nearest selected page before or after the current one current.

        Args:
            step (int): 1 for the next selected page in display order, -1 for the
                previous one.
        """
        displayed = self.thumbnails.pages()
        selected = set(self.thumbnails.selected_pages())
        index = self._display_position()
        while 0 <= (index := index + step) < len(displayed):
            if displayed[index] in selected:
                # NoUpdate keeps the multi-selection that the user is stepping through.
                self.thumbnails.view.selectionModel().setCurrentIndex(
                    self.thumbnails.model.index(
                        self.thumbnails.model.row_of(displayed[index])
                    ),
                    QtCore.QItemSelectionModel.SelectionFlag.NoUpdate,
                )
                return

    def goNextSelectedPage(self) -> None:
        """Go to next selected page."""
        self._go_to_adjacent_selected_page(1)

    def goPrevSelectedPage(self) -> None:
        """Go to previous selected page."""
        self._go_to_adjacent_selected_page(-1)

    def execGotoPageDialog(self) -> None:
        """Execute go to page dialog."""
        _LOG.info("Execute go to page dialog")

    def showAboutDialog(self) -> None:
        """Show about dialog."""
        _LOG.info("Show About Dialog")
        AboutDialog(self.ui).show()

    def reloadCurrentPage(self) -> None:
        """Render the current page again, e.g. after its source file changed."""
        if self._current_page is None:
            return
        self._invalidate_thumbnails(self._current_page)
        self._request_render()

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

    def fixDpiDialogRequested(self) -> None:
        """Let the user correct the DPI of the project's images."""
        project = self.project
        if project is None or self.batch_runner.is_running:
            return
        dialog = self.fix_dpi_dialog_factory(list(project.images), self.ui)
        if not self._run_dialog(dialog):
            return
        changes = dialog.get_dpi_changes()
        if not changes:
            return
        for info in project.images:
            if info.id in changes:
                info.metadata = info.metadata.model_copy(
                    update={"dpi": changes[info.id]}
                )
        self.modified = True
        self._invalidate_thumbnails()
        self._request_render()

    def showRelinkingDialog(self) -> None:
        """Let the user point the folders of the project's images elsewhere."""
        project = self.project
        if project is None or self.batch_runner.is_running:
            return
        directories = image_directories(
            [info.id.file_path for info in project.images], only_missing=False
        )
        relinker = self._ask_relinker(directories)
        if relinker is None or relinker.is_empty():
            return
        project.stage_params = collect_stage_params(self.stages, project.get_pages())
        project.selected_page = self.current_page
        project.remap_image_paths(relinker.substitute)
        stages = StageSequence()
        apply_stage_params(stages, project.stage_params)
        project_file = pathlib.Path(self.projectFile) if self.projectFile else None
        self._install_project(project, stages, project_file=project_file, modified=True)

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
        if pages is not None:
            pages = self._fix_dpi_of_new_pages(pages)
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
        if self.project is None or self.batch_runner.is_running:
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
        if pages:
            pages = self._fix_dpi_of_new_pages(pages)
        if not pages:
            return
        for image_id, metadata in pages:
            self.project.add_image(image_id, metadata)
        if self.current_page is None:
            self.current_page = self.pages()[0]
        self.modified = True

    @staticmethod
    def _run_dialog(dialog: QtWidgets.QDialog) -> bool:
        """Run a modal dialog and release it, as it is parented to the window.

        Args:
            dialog (QtWidgets.QDialog): The dialog to run.

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

    def _fix_dpi_of_new_pages(self, pages: PageMetadata) -> PageMetadata | None:
        """Ask for the DPI of new images when any has an undefined or odd one.

        Args:
            pages (PageMetadata): Metadata read from the image files.

        Returns:
            PageMetadata | None: The pages with the chosen DPI applied, or None
                when the user cancelled.
        """
        if all(metadata.has_plausible_dpi() for _, metadata in pages):
            return pages
        infos = [ImageInfo(id=image_id, metadata=meta) for image_id, meta in pages]
        dialog = self.fix_dpi_dialog_factory(infos, self.ui)
        if not self._run_dialog(dialog):
            return None
        changes = dialog.get_dpi_changes()
        return [
            (image_id, meta.model_copy(update={"dpi": changes.get(image_id, meta.dpi)}))
            for image_id, meta in pages
        ]

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
        self._discard_panels()
        self._thumbnail_epoch += 1
        self.project = project
        self.stages = stages
        self.thumbnails.set_project(project)
        self.projectFile = str(project_file) if project_file else ""
        self._modified = modified
        self._saved_state = None if modified else self._state_snapshot()
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
        self._remember_project(path)
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
        self._saved_state = self._state_snapshot()
        self.modified = False
        return True

    def _state_snapshot(self) -> str:
        """Serialise what a project file holds, apart from the selected page.

        Returns:
            str: JSON text that is equal for equal project content.
        """
        project = self.project
        if project is None:
            return ""
        stage_params = collect_stage_params(self.stages, project.get_pages())
        return project.model_copy(
            update={"stage_params": stage_params, "selected_page": None}
        ).model_dump_json()

    def has_unsaved_changes(self) -> bool:
        """Whether the open project differs from what is on disk.

        Edits that were undone again do not count.

        Returns:
            bool: True if closing the project would lose changes.
        """
        if self.project is None or not self._modified:
            return False
        if self._saved_state is None:
            return True
        if self._state_snapshot() == self._saved_state:
            self.modified = False
            return False
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
        self._abort_batch()
        if self.has_unsaved_changes():
            choice = self.save_prompt()
            if choice is SaveChoice.CANCEL:
                return False
            if choice is SaveChoice.SAVE and not self.save_project():
                return False
        self._close_project_without_saving()
        return True

    def _close_project_without_saving(self) -> None:
        """Drop the project and reset the window to its empty state."""
        self._discard_panels()
        self._thumbnail_epoch += 1
        self.project = None
        self.stages = StageSequence()
        self.thumbnails.set_project(None)
        self.projectFile = ""
        self.current_page = None
        self._modified = False
        self._saved_state = None
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
