"""Main Window."""

import contextlib
import enum
import pathlib
from collections.abc import Callable, Collection, Hashable, Iterable
from typing import Any, cast

from loguru import logger as _LOG
from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QGraphicsView,
    QInputDialog,
    QMessageBox,
    QScrollArea,
    QStackedLayout,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from scantailor.__about__ import APP_NAME, __version__
from scantailor.app.ui import UI_FOLDER, load_ui_widget
from scantailor.app.ui.batch_runner import BatchRunner
from scantailor.app.ui.dialogs.about import AboutDialog
from scantailor.app.ui.dialogs.apply_to import (
    DESKEW_OPTIONS,
    DESKEW_TITLE,
    MARGIN_SIDE_ICONS,
    MARGIN_SIDE_OPTIONS,
    MARGINS_OPTIONS_TITLE,
    OUTPUT_GROUP_TITLES,
    ApplyChangeDewarpingDialog,
    ApplyChangeDpiDialog,
    ApplyColorsDialog,
    ApplyDeskewDialog,
    ApplyFixOrientationDialog,
    ApplyPageLayoutDialog,
    ApplySelectContentDialog,
    ApplyToDialog,
)
from scantailor.app.ui.dialogs.batch_progress import BatchProgressDialog
from scantailor.app.ui.dialogs.default_params import DefaultParamsDialog
from scantailor.app.ui.dialogs.fix_dpi import FixDpiDialog
from scantailor.app.ui.dialogs.load_files_status import LoadFilesStatusDialog
from scantailor.app.ui.dialogs.pdf_export import PdfExportDialog, PdfExportOptions
from scantailor.app.ui.dialogs.pdf_import import DEFAULT_IMPORT_DPI
from scantailor.app.ui.dialogs.pdf_progress import PdfProgressDialog
from scantailor.app.ui.dialogs.project_files import ProjectFilesDialog
from scantailor.app.ui.dialogs.relinking import RelinkingDialog
from scantailor.app.ui.dialogs.remove_pages import RemovePagesDialog
from scantailor.app.ui.dialogs.settings import SettingsDialog
from scantailor.app.ui.dialogs.split_mode import SplitModeDialog
from scantailor.app.ui.icons import icon
from scantailor.app.ui.page_renderer import (
    PageRenderer,
    release_thumbnail_memory,
    render_thumbnail,
    shrink_to_thumbnail,
)
from scantailor.app.ui.panels.new_open_project import NewOpenProjectPanel
from scantailor.app.ui.pdf_worker import (
    PagePlan,
    PdfExportRequest,
    PdfExportResult,
    PdfJob,
    PdfSource,
    StoredTextRequest,
    TextRequest,
    export_pdf_pages,
)
from scantailor.app.ui.retranslation import LanguageChangeWatcher, retranslate_ui
from scantailor.app.ui.stage_panels import StagePanels
from scantailor.app.ui.thumbnails import ThumbnailListWidget
from scantailor.app.ui.thumbnails.stage_orders import is_page_incomplete
from scantailor.app.ui.units_provider import UnitsProvider
from scantailor.app.ui.utils import get_cwidget
from scantailor.app.ui.widgets.error_widget import ErrorWidget, load_failure_message
from scantailor.app.ui.widgets.processing_indicator import ProcessingIndicator
from scantailor.app.ui.widgets.stage_list_view import StageListView
from scantailor.app.ui.widgets.status_bar_panel import StatusBarPanel
from scantailor.core import diagnostics, propagation
from scantailor.core.batch import (
    BatchResult,
    PageDetected,
    ProgressEvent,
    RenderInfo,
    erase_output_files,
    output_file_names,
    rename_split_outputs,
)
from scantailor.core.default_params import apply_default_params
from scantailor.core.default_params_profiles import (
    CUSTOM_PROFILE,
    DEFAULT_PROFILE,
    SOURCE_PROFILE,
    ProfileError,
    ProfileStore,
    active_default_params,
)
from scantailor.core.filename_ordering import sort_smart
from scantailor.core.image_io import ImageLoadError, list_image_pages
from scantailor.core.models import Dpi, ImageId, PageId
from scantailor.core.page_source import DpiOverrides, dpi_overrides_of
from scantailor.core.pipeline import OutputExtras
from scantailor.core.project import ImageInfo, ImageMetadata, LayoutDirection, Project
from scantailor.core.recent_projects import RecentProjects
from scantailor.core.relinker import Relinker, image_directories
from scantailor.core.settings import ApplicationSettings, Units
from scantailor.core.stage_params import (
    apply_stage_defaults,
    apply_stage_params,
    collect_stage_defaults,
    collect_stage_params,
)
from scantailor.core.stage_sequence import FilterStage, StageSequence
from scantailor.core.text_layer import TextGeometry
from scantailor.core.xml_compat import import_xml_project_file
from scantailor.filters.output.params import ApplyGroup, DistortionModelSpec
from scantailor.filters.page_layout.params import ContentSize
from scantailor.filters.page_layout.settings import MARGIN_SIDES
from scantailor.filters.page_split.layout_type import LayoutType

PROJECT_SUFFIX = ".ScanTailor"
APP_TITLE = f"{APP_NAME} - {__version__}"
_MAIN_WINDOW_UI = UI_FOLDER / "MainWindow.ui"
USER_GUIDE_URL = "https://py-scantailor.readthedocs.io/"
_LAST_DIR_KEY = "project/lastDir"
_LAST_INPUT_DIR_KEY = "project/lastInputDir"
_LAST_PDF_DIR_KEY = "project/lastPdfDir"
_LAST_PDF_DPI_KEY = "project/lastPdfDpi"
THUMBNAIL_SIZE = QtCore.QSize(160, 160)
_AUTO_SAVE_INTERVAL_MS = 60_000
# Thumbnail zoom: one step and the limits of the largest thumbnail, in pixels.
_THUMBNAIL_STEP = (25.0, 16.0)
_MIN_THUMBNAIL_SIZE = (100.0, 64.0)
_MAX_THUMBNAIL_SIZE = (1000.0, 640.0)
# Repeated zoom steps rebuild the thumbnails once.
_THUMBNAIL_RESIZE_DELAY_MS = 350
# Alt+wheel zooms the thumbnails once it travels this far.
_WHEEL_ZOOM_THRESHOLD = 30
# A page that is half or more out of the thumbnail view stops being followed.
_VISIBLE_FRACTION = 0.5
_SELECT_CONTENT_OPTIONS = {
    "content_box": cast(
        str, QtCore.QT_TRANSLATE_NOOP("ApplyToDialog", "Apply content box")
    ),
    "page_box": cast(str, QtCore.QT_TRANSLATE_NOOP("ApplyToDialog", "Apply page box")),
}
_APPLY_OPTIONS: dict[FilterStage, dict[str, str]] = {
    FilterStage.SELECT_CONTENT: _SELECT_CONTENT_OPTIONS,
    FilterStage.DESKEW: DESKEW_OPTIONS,
}
_APPLY_DIALOGS: dict[FilterStage, type[ApplyToDialog]] = {
    FilterStage.FIX_ORIENTATION: ApplyFixOrientationDialog,
    FilterStage.DESKEW: ApplyDeskewDialog,
    FilterStage.SELECT_CONTENT: ApplySelectContentDialog,
    FilterStage.PAGE_LAYOUT: ApplyPageLayoutDialog,
    FilterStage.OUTPUT: ApplyColorsDialog,
}
# Output groups without an entry here use the colour dialog, as upstream does.
_OUTPUT_GROUP_DIALOGS: dict[ApplyGroup, type[ApplyToDialog]] = {
    ApplyGroup.DPI: ApplyChangeDpiDialog,
    ApplyGroup.DEWARPING: ApplyChangeDewarpingDialog,
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
PdfExportDialogFactory = Callable[[pathlib.Path, bool, QWidget], PdfExportDialog]
# Arguments: pages without output, pages in the project.
RunBatchPrompt = Callable[[int, int], bool]
# Arguments: pages to remove, parent window. Returns None when the user cancels,
# else whether the output files of the pages are to be erased too.
RemovePagesPrompt = Callable[[int, QWidget], bool | None]
PageMetadata = list[tuple[ImageId, ImageMetadata]]


def _add_pdf_sources(project: Project, sources: list[PdfSource]) -> None:
    """Record extracted PDFs on a project, merging pages of a PDF already there.

    Args:
        project (Project): The project to add the sources to.
        sources (list[PdfSource]): Sources of freshly extracted pages.
    """
    for source in sources:
        for position, existing in enumerate(project.pdf_sources):
            if (
                existing.pdf_path == source.pdf_path
                and existing.pages_dir == source.pages_dir
            ):
                project.pdf_sources[position] = existing.model_copy(
                    update={"pages": {**existing.pages, **source.pages}}
                )
                break
        else:
            project.pdf_sources.append(source)


def _create_directory(directory: pathlib.Path) -> None:
    """Create the output folder if possible; processing reports it when it cannot."""
    try:
        directory.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        _LOG.warning(f"Cannot create output folder {directory}: {error}")


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


class _AltWheelFilter(QtCore.QObject):
    """Event filter that turns Alt+wheel into a zoom step instead of scrolling."""

    def __init__(
        self, zoom: Callable[[int], None], parent: QtCore.QObject | None = None
    ) -> None:
        """Create the filter.

        Args:
            zoom (Callable[[int], None]): Called with 1 to zoom in, -1 to zoom out.
            parent (QtCore.QObject | None): Owner of the filter.
        """
        super().__init__(parent)
        self._zoom = zoom

    def eventFilter(self, watched: QtCore.QObject, event: QtCore.QEvent) -> bool:
        """Consume wheel events made with Alt held.

        Args:
            watched (QtCore.QObject): The filtered object.
            event (QtCore.QEvent): The event.

        Returns:
            bool: True when the event was consumed.
        """
        if event.type() != QtCore.QEvent.Type.Wheel:
            return False
        wheel = cast(QtGui.QWheelEvent, event)
        if wheel.modifiers() != QtCore.Qt.KeyboardModifier.AltModifier:
            return False
        # Qt reports an Alt wheel along either axis depending on the platform.
        distance = wheel.angleDelta().x() + wheel.angleDelta().y()
        if abs(distance) >= _WHEEL_ZOOM_THRESHOLD:
            self._zoom(1 if distance > 0 else -1)
        wheel.accept()
        return True


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
        self.ui = load_ui_widget(_MAIN_WINDOW_UI)

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
        self.pdf_export_dialog_factory: PdfExportDialogFactory = (
            self._default_pdf_export_dialog
        )
        self.run_batch_prompt: RunBatchPrompt = self._ask_run_batch
        self.remove_pages_prompt: RemovePagesPrompt = RemovePagesDialog.ask_removal

        self.project: Project | None = None
        self.stages = StageSequence()
        self._current_page: PageId | None = None
        self._panels: StagePanels | None = None
        self._failed_image: ImageId | None = None
        self._thumbnail_widget: QWidget | None = None
        self._welcome_widget: QWidget | None = None
        self.renderer = PageRenderer(self.ui)
        self.renderer.render_done.connect(self._on_render_done)
        self.renderer.distortion_model_found.connect(self._on_distortion_model_found)
        self.renderer.output_extras_found.connect(self._on_output_extras_found)
        self.renderer.render_failed.connect(self._on_render_failed)
        self._render_pending = False
        self._thumbnail_epoch = 0
        self._dpi_overrides: DpiOverrides = {}
        self._thumbnail_versions: dict[PageId, int] = {}
        self.batch_runner = BatchRunner(self.ui)
        self.batch_runner.detected.connect(self._on_batch_detected)
        self.batch_runner.progress.connect(self._on_batch_progress)
        self.batch_runner.finished.connect(self._on_batch_finished)
        self.batch_runner.failed.connect(self._on_batch_failed)
        self.batch_dialog: BatchProgressDialog | None = None
        self._batch_renders = True
        self._beep_when_finished = False
        self.last_batch_result: BatchResult | None = None
        self.pdf_job = PdfJob(self.ui)
        self.pdf_job.progress.connect(self._on_pdf_progress)
        self.pdf_job.finished.connect(self._on_pdf_finished)
        self.pdf_job.cancelled.connect(self._on_pdf_cancelled)
        self.pdf_job.failed.connect(self._on_pdf_failed)
        self._pdf_progress: PdfProgressDialog | None = None
        self._pdf_on_done: Callable[[Any], None] | None = None
        self._render_info: dict[PageId, RenderInfo] = {}
        self.last_pdf_export: PdfExportResult | None = None
        self._modified = False
        self._auto_save_timer = QtCore.QTimer(self.ui)
        self._auto_save_timer.setSingleShot(True)
        self._auto_save_timer.setInterval(_AUTO_SAVE_INTERVAL_MS)
        self._auto_save_timer.timeout.connect(self._auto_save_project)
        self._auto_save_deferred = False
        self._aggregate_size: ContentSize | None = None
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

        self.settings = QtCore.QSettings(self.ui)
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
        self.actionMagnifyThumbnails = get_cwidget(
            self.ui, QtGui.QAction, "actionMagnifyThumbnails"
        )
        self.actionDiminishThumbnails = get_cwidget(
            self.ui, QtGui.QAction, "actionDiminishThumbnails"
        )
        self.actionPixels = get_cwidget(self.ui, QtGui.QAction, "actionPixels")
        self.actionMilimeters = get_cwidget(self.ui, QtGui.QAction, "actionMilimeters")
        self.actionCentimetres = get_cwidget(
            self.ui, QtGui.QAction, "actionCentimetres"
        )
        self.actionInches = get_cwidget(self.ui, QtGui.QAction, "actionInches")
        self.actionAbout = get_cwidget(self.ui, QtGui.QAction, "actionAbout")
        self.actionUserGuide = get_cwidget(self.ui, QtGui.QAction, "actionUserGuide")
        self.actionOpenLogFolder = get_cwidget(
            self.ui, QtGui.QAction, "actionOpenLogFolder"
        )
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
        self.actionReverseTwoPageOrder = get_cwidget(
            self.ui, QtGui.QAction, "actionReverseTwoPageOrder"
        )
        self.actionReverseTwoPageOrder.setIcon(icon("swap-pages"))
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
        self.diminishThumbnailsBtn = get_cwidget(
            self.ui, QToolButton, "diminishThumbnailsBtn"
        )
        self.prevPageBtn = get_cwidget(self.ui, QToolButton, "prevPageBtn")
        self.nextPageBtn = get_cwidget(self.ui, QToolButton, "nextPageBtn")
        self.filterSelectedBtn = get_cwidget(self.ui, QToolButton, "filterSelectedBtn")
        self.gotoPageBtn = get_cwidget(self.ui, QToolButton, "gotoPageBtn")
        self.thumbColumnViewBtn = get_cwidget(
            self.ui, QToolButton, "thumbColumnViewBtn"
        )
        self.deviationHighlightingBtn = get_cwidget(
            self.ui, QToolButton, "deviationHighlightingBtn"
        )
        self.sortingOrderBtn = get_cwidget(self.ui, QToolButton, "sortingOrderBtn")
        self.pauseSortingBtn = get_cwidget(self.ui, QToolButton, "pauseSortingBtn")
        self.pauseSortingBtn.setIcon(icon("pause", "play"))
        self.filterList = get_cwidget(self.ui, StageListView, "filterList")
        self.actionAddFiles = QtGui.QAction(self.ui)
        self._insert_add_files_action()
        self.actionOpenPdf = QtGui.QAction(self.ui)
        self.actionExportPdf = QtGui.QAction(self.ui)
        self._insert_pdf_actions()
        self.menuOpenRecent = QtWidgets.QMenu(self.ui)
        self._insert_recent_menu()

        # Layouts
        self.imageFrameLayout = QStackedLayout(self.imageViewFrame)
        self.not_ready_widget = _message_widget("")
        self.imageFrameLayout.addWidget(self.not_ready_widget)
        self.load_error_widget = ErrorWidget()
        self.load_error_widget.link_activated.connect(
            lambda _link: self.showRelinkingDialog()
        )
        self.imageFrameLayout.addWidget(self.load_error_widget)
        self._translate_own_text()
        self.processing_indicator = ProcessingIndicator()
        self.units_provider = UnitsProvider.get_instance()
        self.status_panel = StatusBarPanel(self.units_provider)
        status_bar = get_cwidget(self.ui, QtWidgets.QStatusBar, "statusBar")
        status_bar.addPermanentWidget(self.status_panel)
        status_bar.addPermanentWidget(self.processing_indicator)
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
        self.actionUserGuide.triggered.connect(self.openUserGuide)
        self.actionOpenLogFolder.triggered.connect(self.openLogFolder)
        self.actionReloadPage.triggered.connect(self.reloadCurrentPage)
        self.actionSwitchFilter1.triggered.connect(self.switchFilter1)
        self.actionSwitchFilter2.triggered.connect(self.switchFilter2)
        self.actionSwitchFilter3.triggered.connect(self.switchFilter3)
        self.actionSwitchFilter4.triggered.connect(self.switchFilter4)
        self.actionSwitchFilter5.triggered.connect(self.switchFilter5)
        self.actionSwitchFilter6.triggered.connect(self.switchFilter6)
        self.actionFixDpi.triggered.connect(self.fixDpiDialogRequested)
        self.actionRelinking.triggered.connect(self.showRelinkingDialog)
        self.actionReverseTwoPageOrder.triggered.connect(
            self.toggleTwoPageSpreadReadingOrder
        )
        self.actionSettings.triggered.connect(self.openSettingsDialog)
        self.actionDefaults.triggered.connect(self.openDefaultParamsDialog)
        self.actionNewProject.triggered.connect(self.newProject)
        self.actionOpenProject.triggered.connect(self.openProject)
        self.actionSaveProject.triggered.connect(self.saveProjectTriggered)
        self.actionSaveProjectAs.triggered.connect(self.saveProjectAsTriggered)
        self.actionCloseProject.triggered.connect(self.closeProject)
        self.actionQuit.triggered.connect(self.quit)
        self.actionAddFiles.triggered.connect(self.addFiles)
        self.actionOpenPdf.triggered.connect(self.openPdf)
        self.actionExportPdf.triggered.connect(self.exportPdf)
        self.actionMagnifyThumbnails.triggered.connect(lambda: self.scale_thumbnails(1))
        self.actionDiminishThumbnails.triggered.connect(
            lambda: self.scale_thumbnails(-1)
        )
        self.magnifyThumbnailsBtn.clicked.connect(lambda: self.scale_thumbnails(1))
        self.diminishThumbnailsBtn.clicked.connect(lambda: self.scale_thumbnails(-1))
        self.prevPageBtn.clicked.connect(self._on_prev_page_button)
        self.nextPageBtn.clicked.connect(self._on_next_page_button)
        self.gotoPageBtn.clicked.connect(self.execGotoPageDialog)

        # Shortcuts of actions that are not in any menu only fire when attached
        # to a widget.
        self.ui.addActions(self._shortcut_actions())

        self._close_guard = _CloseGuard(self._close_project_interactive, self.ui)
        self.ui.installEventFilter(self._close_guard)
        self._language_watcher = LanguageChangeWatcher(self._retranslate, self.ui)
        self.ui.installEventFilter(self._language_watcher)

        # if self.settings.value("mainWindow/maximized") == False:
        #    if not self.restoreGeometry(geom.toByteArray()):
        #        pass

        # Additional setup (from C++ constructor)
        # self.setupThumbView()
        # self.updateThumbViewMinWidth()
        # self.updateSortOptions()
        # self.setupIcons()
        self.thumbnails = ThumbnailListWidget(
            self._render_page_thumbnail,
            params_hash=self._thumbnail_digest,
            thumbnail_size=THUMBNAIL_SIZE,
        )
        self.thumbnails.apply_settings(ApplicationSettings.get_instance())
        self.thumbnails.page_selected.connect(self.page_selected)
        self.thumbnails.insert_before_requested.connect(self.insert_images_before)
        self.thumbnails.insert_after_requested.connect(self.insert_images_after)
        self.thumbnails.insert_here_requested.connect(self.insert_images_at_end)
        self.thumbnails.remove_pages_requested.connect(self.remove_pages)
        self.set_thumbnail_widget(self.thumbnails)
        self._wire_thumbnail_controls()
        self._wire_units_menu()
        self.filterList.launch_batch_processing.connect(self.startBatchProcessing)
        self._update_welcome_panel()
        self._attach_stages()
        self._update_project_actions()
        self._update_window_title()
        self._rebuild_recent_menu()

    def _render_page_thumbnail(self, page: PageId, stage: str) -> QtGui.QImage:
        """Render a thumbnail on a worker thread; settings are only read.

        The render the user is waiting for goes first. When that render is of
        this page and stage with the same parameters, its image is shrunk
        instead of running the pipeline again.
        """
        self.renderer.wait_until_idle()
        end_stage = FilterStage(int(stage))
        shown = self.renderer.rendered(
            page, end_stage, self._thumbnail_digest(page, stage)
        )
        if shown is not None:
            return shrink_to_thumbnail(shown, self.thumbnails.render_size)
        return render_thumbnail(
            page,
            self.stages,
            end_stage,
            self.thumbnails.render_size,
            self._dpi_overrides,
            self.renderer.sources,
        )

    def _refresh_dpi_overrides(self) -> None:
        """Snapshot the project's DPIs on the GUI thread for the worker threads."""
        self._dpi_overrides = (
            dpi_overrides_of(self.project) if self.project is not None else {}
        )
        if self._panels is not None:
            self._panels.set_dpi_overrides(self._dpi_overrides)
            self._show_source_image_info()
        self._show_source_dpi()

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
        self.sortingOrderBtn.toggled.connect(self.thumbnails.set_sort_reversed)
        self.thumbnails.orders_changed.connect(self._refill_sort_options)
        self._refill_sort_options()
        self.sortOptions.currentIndexChanged.connect(self._update_status_bar)
        self.pauseSortingBtn.toggled.connect(self._on_pause_sorting_toggled)
        self.focusButton.toggled.connect(self._on_focus_toggled)
        self.selectionModeBtn.toggled.connect(
            self.thumbnails.set_multi_selection_enabled
        )
        self.thumbnails.set_follow_current(self.focusButton.isChecked())
        self.thumbnails.set_multi_selection_enabled(self.selectionModeBtn.isChecked())

        settings = ApplicationSettings.get_instance()
        self.thumbColumnViewBtn.setChecked(settings.single_column_thumbnail_display)
        self.deviationHighlightingBtn.setChecked(settings.highlight_deviation)
        self.thumbColumnViewBtn.clicked.connect(self._on_column_view_clicked)
        self.deviationHighlightingBtn.clicked.connect(self._on_deviation_clicked)

        self.thumbnails.view.verticalScrollBar().valueChanged.connect(
            self._on_thumbnails_scrolled
        )
        self._thumbnail_resize_timer = QtCore.QTimer(self.ui)
        self._thumbnail_resize_timer.setSingleShot(True)
        self._thumbnail_resize_timer.setInterval(_THUMBNAIL_RESIZE_DELAY_MS)
        self._thumbnail_resize_timer.timeout.connect(self._apply_thumbnail_size)
        self._wheel_zoom = _AltWheelFilter(self.scale_thumbnails, self.ui)
        view = self.thumbnails.view
        for target in (view, view.viewport(), view.verticalScrollBar()):
            target.installEventFilter(self._wheel_zoom)

    def _bind_thumbnail_stages(self) -> None:
        """Offer the current stages' sort options and incomplete marks."""
        self.thumbnails.bind_stage_sequence(self.stages)
        self.thumbnails.set_incompleteness_provider(self._page_incomplete)

    def _page_incomplete(self, page: PageId, stage_key: str) -> bool:
        """Tell whether a stage has no result for a page.

        Output keeps no parameters for a page that a batch rendered with the
        default settings, so a page whose written geometry is on record counts as
        done.
        """
        stage = FilterStage(int(stage_key))
        if not is_page_incomplete(self.stages, stage, page):
            return False
        project = self.project
        written = project is not None and project.text_geometry_of(page) is not None
        return not (stage == FilterStage.OUTPUT and written)

    def _refill_sort_options(self) -> None:
        """List the shown stage's sort options; hide the controls if it has none."""
        names = self.thumbnails.order_names()
        self.sortOptions.blockSignals(True)
        self.sortOptions.clear()
        self.sortOptions.addItems(names)
        self.sortOptions.setCurrentIndex(0 if names else -1)
        self.sortOptions.blockSignals(False)
        self.sortOptions.setVisible(bool(names))
        self.sortingOrderBtn.setVisible(bool(names))
        self.pauseSortingBtn.setEnabled(bool(names) and self.project is not None)
        if not names:
            self.pauseSortingBtn.setChecked(False)
        self._update_status_bar()

    def _on_pause_sorting_toggled(self, paused: bool) -> None:
        """Show the pages in the project's order while the sort order is paused."""
        self.thumbnails.set_sorting_paused(paused)
        self._update_status_bar()

    def _on_focus_toggled(self, checked: bool) -> None:
        """Follow the current page in the thumbnail list, scrolling to it now."""
        self.thumbnails.set_follow_current(checked)
        if checked:
            self.thumbnails.scroll_to_current()

    def _on_thumbnails_scrolled(self) -> None:
        """Stop following the current page once the user scrolled it out of view."""
        page = self._current_page
        if page is None or self.thumbnails.model.row_of(page) < 0:
            return
        item = self.thumbnails.item_rect(page)
        view = QtCore.QRectF(self.thumbnails.view.viewport().rect())
        margin = item.height() * _VISIBLE_FRACTION
        top_visible = item.top() >= view.top() and item.top() + margin <= view.bottom()
        bottom_visible = (
            item.bottom() <= view.bottom() and item.bottom() - margin >= view.top()
        )
        if not (top_visible or bottom_visible):
            self.focusButton.setChecked(False)

    def _on_column_view_clicked(self, single_column: bool) -> None:
        settings = ApplicationSettings.get_instance()
        settings.single_column_thumbnail_display = single_column
        settings.save()
        self.thumbnails.set_single_column(single_column)

    def _on_deviation_clicked(self, highlighted: bool) -> None:
        settings = ApplicationSettings.get_instance()
        settings.highlight_deviation = highlighted
        settings.save()
        self._invalidate_thumbnails()

    def scale_thumbnails(self, direction: int) -> None:
        """Make the thumbnails larger or smaller by one step.

        The size is stored at once; the thumbnails are rebuilt after a short
        delay so that repeated steps rebuild them once.

        Args:
            direction (int): 1 to magnify, -1 to diminish.
        """
        settings = ApplicationSettings.get_instance()
        step_x, step_y = _THUMBNAIL_STEP
        (min_x, min_y), (max_x, max_y) = _MIN_THUMBNAIL_SIZE, _MAX_THUMBNAIL_SIZE
        width = settings.max_thumbnail_width + step_x * direction
        height = settings.max_thumbnail_height + step_y * direction
        settings.max_thumbnail_width = min(max(width, min_x), max_x)
        settings.max_thumbnail_height = min(max(height, min_y), max_y)
        if not self._thumbnail_resize_timer.isActive():
            self._thumbnail_resize_timer.start()

    def _apply_thumbnail_size(self) -> None:
        settings = ApplicationSettings.get_instance()
        settings.save()
        self.thumbnails.apply_settings(settings)

    def _apply_thumbnail_settings(self, settings: ApplicationSettings) -> None:
        """Take the thumbnail options of ``settings`` into the list and its buttons."""
        self.thumbnails.apply_settings(settings)
        self.thumbColumnViewBtn.setChecked(settings.single_column_thumbnail_display)

    def _wire_units_menu(self) -> None:
        """Bind the Units menu to the shared unit and the stored setting."""
        group = QtGui.QActionGroup(self.ui)
        group.setExclusive(True)
        for action in self._units_actions().values():
            group.addAction(action)
        self._show_units_choice(self.units_provider.units)
        for units, action in self._units_actions().items():
            action.toggled.connect(
                lambda checked, chosen=units: checked and self._choose_units(chosen)
            )
        self.units_provider.units_changed.connect(self._show_units_choice)

    def _units_actions(self) -> dict[Units, QtGui.QAction]:
        return {
            Units.PIXELS: self.actionPixels,
            Units.MILLIMETERS: self.actionMilimeters,
            Units.CENTIMETERS: self.actionCentimetres,
            Units.INCHES: self.actionInches,
        }

    def _show_units_choice(self, units: Units) -> None:
        """Check the menu entry of ``units``, whoever chose the unit."""
        self._units_actions()[units].setChecked(True)

    def _choose_units(self, units: Units) -> None:
        """Make ``units`` the unit of every length shown, and remember it."""
        settings = ApplicationSettings.get_instance()
        if settings.units is not units:
            settings.units = units
            settings.save()
        self.units_provider.set_units(units)

    def _insert_add_files_action(self) -> None:
        """Place the add-files action in the File menu, above the save entries."""
        file_menu = get_cwidget(self.ui, QtWidgets.QMenu, "menuFile")
        actions = file_menu.actions()
        save_index = actions.index(self.actionSaveProject)
        file_menu.insertAction(actions[save_index - 1], self.actionAddFiles)

    def _insert_pdf_actions(self) -> None:
        """Place the PDF actions in the File menu, below the open entries."""
        file_menu = get_cwidget(self.ui, QtWidgets.QMenu, "menuFile")
        actions = file_menu.actions()
        save_index = actions.index(self.actionSaveProject)
        file_menu.insertAction(actions[save_index - 1], self.actionOpenPdf)
        file_menu.insertAction(actions[save_index - 1], self.actionExportPdf)

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

    def _update_welcome_panel(self) -> None:
        """Show the New/Open panel in the image area while no project is open."""
        if self._welcome_widget is not None:
            self.imageFrameLayout.removeWidget(self._welcome_widget)
            self._welcome_widget.deleteLater()
            self._welcome_widget = None
        if self.project is not None:
            return
        recent = RecentProjects.read(self.settings)
        if not recent.validate():
            recent.write(self.settings)
        panel = NewOpenProjectPanel([pathlib.Path(p) for p in recent.enumerate()])
        # Handlers run on the next event loop turn because they replace this panel.
        panel.new_project_requested.connect(
            lambda: QtCore.QTimer.singleShot(0, self.newProject)
        )
        panel.open_project_requested.connect(
            lambda: QtCore.QTimer.singleShot(0, self.openProject)
        )
        panel.recent_project_selected.connect(
            lambda chosen: QtCore.QTimer.singleShot(
                0, lambda: self._open_recent_from_welcome(chosen)
            )
        )
        centered = QWidget()
        grid = QtWidgets.QGridLayout(centered)
        grid.addWidget(panel, 1, 1)
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(2, 1)
        grid.setRowStretch(0, 1)
        grid.setRowStretch(2, 1)
        self.imageFrameLayout.addWidget(centered)
        self.imageFrameLayout.setCurrentWidget(centered)
        self._welcome_widget = centered

    def _open_recent_from_welcome(self, path: pathlib.Path) -> None:
        """Open a recent project; the list is refreshed if it turns out to be gone."""
        if not self.open_project_file(path) and self.project is None:
            self._update_welcome_panel()

    def _shortcut_actions(self) -> list[QtGui.QAction]:
        """Return the actions that have a shortcut but sit in no menu."""
        return [
            self.actionFirstPage,
            self.actionLastPage,
            self.actionNextPage,
            self.actionPrevPage,
            self.actionPrevPageQ,
            self.actionNextPageW,
            self.actionNextSelectedPage,
            self.actionPrevSelectedPage,
            self.actionNextSelectedPageW,
            self.actionPrevSelectedPageQ,
            self.actionGotoPage,
            self.actionMagnifyThumbnails,
            self.actionDiminishThumbnails,
            self.actionReloadPage,
            *self._switch_filter_actions(),
        ]

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
    def render_pending(self) -> bool:
        """Whether the shown page is still being rendered."""
        return self._render_pending

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
        self._update_auto_save_timer()
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
        if stage > FilterStage.PAGE_SPLIT and self.project is not None:
            self._adopt_stored_page_counts([info.id for info in self.project.images])
        self.thumbnails.set_stage(str(int(stage)))
        panels = StagePanels(stage, self.stages, self.ui)
        panels.set_source_cache(self.renderer.sources)
        panels.set_dpi_overrides(self._dpi_overrides)
        panels.set_removed_pages_provider(self._removed_halves_of)
        panels.unremove_page_requested.connect(self._on_unremove_page_requested)
        panels.source_failed.connect(self._on_source_failed)
        panels.settings_changed.connect(self._on_settings_edited)
        panels.page_size_changed.connect(self.status_panel.set_page_size)
        panels.cursor_moved.connect(self.status_panel.set_cursor_position)
        self._connect_apply_requests(panels)
        if panels.stage == FilterStage.PAGE_SPLIT:
            cast(Any, panels.options).page_count_requested.connect(
                self._on_page_count_requested
            )
        if panels.stage == FilterStage.DESKEW:
            cast(Any, panels.options).auto_oblique_default_changed.connect(
                self._remember_auto_oblique_default
            )
        if panels.stage == FilterStage.PAGE_LAYOUT:
            self._connect_margins_panel(cast(Any, panels.options))
        self.optionsFrameLayout.addWidget(panels.options)
        self.imageFrameLayout.addWidget(panels.view)
        self.imageFrameLayout.setCurrentWidget(panels.view)
        self._panels = panels
        if self._welcome_widget is not None:
            self.imageFrameLayout.setCurrentWidget(self._welcome_widget)
        self._show_current_page()

    def _removed_halves_of(self, image_id: ImageId) -> tuple[bool, bool]:
        """Return whether the left and the right half of an image were removed."""
        project = self.project
        info = (
            next((i for i in project.images if i.id == image_id), None)
            if project
            else None
        )
        if info is None:
            return (False, False)
        return (info.left_half_removed, info.right_half_removed)

    def _on_unremove_page_requested(self, page: PageId) -> None:
        """Bring back the removed half of the image the page belongs to."""
        project = self.project
        if project is None or self.batch_runner.is_running:
            return
        if not project.restore_removed_half(page.image_id):
            return
        self._refresh_dpi_overrides()
        self.modified = True
        self._refresh_page_list(page)
        self._invalidate_thumbnails()
        self._show_current_page()

    def _connect_margins_panel(self, options: Any) -> None:
        """Route the Margins panel's size matching and source DPI signals.

        Args:
            options (Any): The Margins options widget.
        """
        options.match_size_to_all_requested.connect(
            lambda: self.apply_to_pages(
                FilterStage.PAGE_LAYOUT, self.pages(), part="alignment"
            )
        )
        options.aggregate_size_changed.connect(self._on_aggregate_size_changed)
        options.fix_dpi_requested.connect(self.fixDpiDialogRequested)
        options.source_dpi_changed.connect(self._on_source_dpi_edited)

    def _on_aggregate_size_changed(self) -> None:
        """Redraw every page after the size the pages are matched to changed."""
        self.modified = True
        self._invalidate_thumbnails()
        self._request_render()

    def _on_source_dpi_edited(self, horizontal: int, vertical: int) -> None:
        """Store the resolution typed for the current page's image.

        Args:
            horizontal (int): Dots per inch along x.
            vertical (int): Dots per inch along y.
        """
        info = self._current_image_info()
        if info is None or self.batch_runner.is_running:
            return
        dpi = Dpi(horizontal=horizontal, vertical=vertical)
        if info.metadata.dpi == dpi:
            return
        info.metadata = info.metadata.model_copy(update={"dpi": dpi})
        self.modified = True
        self._refresh_dpi_overrides()
        self._invalidate_thumbnails()
        self._request_render()

    def _current_image_info(self) -> ImageInfo | None:
        """Return the project's record of the current page's image, if any."""
        page = self._current_page
        if page is None or self.project is None:
            return None
        return next(
            (info for info in self.project.images if info.id == page.image_id), None
        )

    def _show_source_image_info(self) -> None:
        """Tell the Margins panel the size and resolution of the current image."""
        panels = self._panels
        if panels is None or panels.stage != FilterStage.PAGE_LAYOUT:
            return
        info = self._current_image_info()
        if info is not None:
            metadata = info.metadata
            cast(Any, panels.options).set_source_image(
                metadata.width, metadata.height, metadata.dpi
            )

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

    def _refresh_page_list(self, preferred: PageId | None = None) -> None:
        """List the project's pages again after the pages changed.

        Args:
            preferred (PageId | None): The page to make current if it still exists.
        """
        pages = self.pages()
        previous = self._current_page
        if preferred in pages:
            target = preferred
        elif previous in pages or previous is None:
            target = previous
        else:
            same_image = [page for page in pages if page.image_id == previous.image_id]
            target = (same_image or pages or [None])[0]
        self.thumbnails.set_project(self.project)
        self._current_page = target
        if target is not None:
            self.thumbnails.set_current_page(target)
        if target != previous:
            self._update_auto_save_timer()
            self._show_current_page()
        self._update_status_bar()

    def _adopt_stored_page_counts(self, images: Iterable[ImageId]) -> bool:
        """Show the page count of the stored Page Split layouts in the page list.

        Args:
            images (Iterable[ImageId]): The images to check.

        Returns:
            bool: True if a page count changed and the page list was rebuilt.
        """
        project = self.project
        if project is None:
            return False
        if not propagation.adopt_stored_page_counts(project, self.stages, images):
            return False
        self.modified = True
        self._refresh_page_list()
        return True

    def _update_auto_save_timer(self) -> None:
        """Start the auto-save countdown unless it is already running."""
        if not self._auto_save_timer.isActive():
            self._auto_save_timer.start()

    def _auto_save_project(self) -> None:
        """Save the project to its file when auto-save is on and it has one."""
        if self.project is None or not self.projectFile:
            return
        if self.batch_runner.is_running:
            # The worker owns the settings that would be written; the save
            # runs when the batch ends.
            self._auto_save_deferred = True
            return
        self._auto_save_deferred = False
        if not ApplicationSettings.get_instance().auto_save_project:
            return
        self._write_project(pathlib.Path(self.projectFile))

    def _run_deferred_auto_save(self) -> None:
        """Save now if an auto-save came due while a batch was running."""
        if self._auto_save_deferred:
            self._auto_save_project()

    def _on_page_count_requested(self, image_id: ImageId, count: int) -> None:
        """Give an image the page count its chosen layout type implies.

        Args:
            image_id (ImageId): The image whose layout type was chosen.
            count (int): 1 or 2.
        """
        project = self.project
        if project is not None and propagation.set_page_count(project, image_id, count):
            self.modified = True
            self._refresh_page_list()

    def _remember_auto_oblique_default(self, enabled: bool) -> None:
        """Keep the oblique default of new pages in the default parameters in use.

        The built-in Default and Source profiles cannot be edited, so a change
        moves the choice to the Custom profile, which starts from their values.

        Args:
            enabled (bool): Whether new pages detect their oblique correction.
        """
        settings = ApplicationSettings.get_instance()
        store = ProfileStore()
        params = active_default_params(settings, store)
        if params.deskew.auto_oblique == enabled:
            return
        params = params.model_copy(
            update={
                "deskew": params.deskew.model_copy(update={"auto_oblique": enabled})
            }
        )
        profile = settings.current_profile
        if profile.strip().casefold() in (
            DEFAULT_PROFILE.casefold(),
            SOURCE_PROFILE.casefold(),
        ):
            profile = CUSTOM_PROFILE
        try:
            store.set_active(params, profile)
            if profile != settings.current_profile:
                settings.current_profile = profile
                settings.save()
        except ProfileError, OSError:
            _LOG.exception("The oblique default could not be stored")

    def _discard_panels(self) -> None:
        """Remove the widgets of the stage that is being left."""
        panels, self._panels = self._panels, None
        self._failed_image = None
        if panels is None:
            return
        self._cancel_render()
        self.status_panel.set_page_size(None, 0.0)
        self.status_panel.set_cursor_position(None)
        for widget, layout in (
            (panels.options, self.optionsFrameLayout),
            (panels.view, self.imageFrameLayout),
        ):
            layout.removeWidget(widget)
            widget.hide()
            widget.setParent(None)
            widget.deleteLater()
        panels.deleteLater()

    def _show_current_page(self) -> None:
        """Point the stage panels at the current page and render it."""
        self._update_status_bar()
        panels = self._panels
        if panels is None:
            return
        if self._failed_image is not None:
            self._failed_image = None
            self._show_not_ready(False)
        panels.set_page(self._current_page)
        self._show_source_image_info()
        self._request_render()

    def _update_status_bar(self) -> None:
        """Show the position, name and source DPI of the current page."""
        page = self._current_page
        displayed = self.thumbnails.pages()
        if page is None or page not in displayed:
            self.status_panel.clear()
            return
        self.status_panel.update_page(displayed.index(page) + 1, len(displayed), page)
        self._show_source_dpi()

    def _show_source_dpi(self) -> None:
        info = self._current_image_info()
        project = self.project
        self.status_panel.set_source_dpi(
            None if info is None else info.metadata.dpi,
            None if project is None else project.most_common_dpi(),
        )

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
        at_output = panels.stage == FilterStage.OUTPUT
        self.renderer.request(
            self._current_page,
            self.stages,
            panels.stage,
            self._dpi_overrides,
            pixels=at_output,
            extras=at_output,
            tag=(
                self._thumbnail_digest(self._current_page, str(int(panels.stage)))
                if at_output
                else None
            ),
        )

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
        """Swap the stage's options and view for the not-ready message, or undo it.

        A page whose source image failed to load keeps its error panel either way.
        """
        panels = self._panels
        if panels is None:
            return
        failed = self._failed_image is not None
        panels.options.setVisible(not (not_ready or failed))
        if failed:
            widget: QWidget = self.load_error_widget
        elif not_ready:
            widget = self.not_ready_widget
        else:
            widget = panels.view
        self.imageFrameLayout.setCurrentWidget(widget)

    def _on_source_failed(self, image_id: ImageId) -> None:
        """Show the load error panel in place of the current page's view."""
        page = self._current_page
        if page is None or page.image_id != image_id:
            return
        self._failed_image = image_id
        self._show_load_error_message()
        self._show_not_ready(False)

    def _show_load_error_message(self) -> None:
        if self._failed_image is not None:
            self.load_error_widget.set_message(
                *load_failure_message(self._failed_image.file_path)
            )

    def _on_settings_edited(self) -> None:
        """React to an edit made in the options widget."""
        self.modified = True
        self._invalidate_thumbnails(self._current_page)
        self._request_render()

    def _remember_aggregate_size(self) -> bool:
        """Note the size the pages are matched to; tell whether it moved.

        Page Layout records a page's content size when the page is processed,
        and that can change the largest aligned size, which every other page is
        sized against. Their thumbnails are then stale.

        Returns:
            bool: True if the size differs from the one noted last.
        """
        size = self.stages.page_layout_filter.get_aggregate_hard_size()
        changed = self._aggregate_size is not None and size != self._aggregate_size
        self._aggregate_size = size
        return changed

    def _on_render_done(self, page_id: PageId, image: object) -> None:
        self._set_render_pending(False)
        if self._remember_aggregate_size():
            self._invalidate_thumbnails()
        if self._adopt_stored_page_counts([page_id.image_id]):
            if page_id not in self.pages():
                return
        if self._panels is not None and page_id == self._current_page:
            self._panels.show_image(image)  # type: ignore[arg-type]

    def _on_output_extras_found(self, page_id: PageId, extras: OutputExtras) -> None:
        if self._panels is not None and page_id == self._current_page:
            self._panels.set_output_extras(extras)

    def _on_distortion_model_found(
        self, page_id: PageId, model: DistortionModelSpec
    ) -> None:
        if self.stages.output_filter.store_distortion_model(page_id, model):
            self.modified = True

    def _on_render_failed(self, page_id: PageId, message: str) -> None:
        self._set_render_pending(False)

    def set_thumbnail_widget(self, widget: QWidget) -> None:
        """Show ``widget`` in the thumbnails dock instead of the placeholder view.

        Args:
            widget (QWidget): The thumbnail list. Replacing it later hides it.
        """
        previous = self._thumbnail_widget or self.thumbView
        container = self.thumbView.parentWidget()
        layout = (
            container.findChild(QVBoxLayout, "verticalLayout") if container else None
        )
        assert layout is not None
        # The replaced widget keeps its parent so it is destroyed with the window.
        # Unparented, it would belong to Python and, being in a reference cycle
        # with this window, be destroyed by whichever thread next runs the garbage
        # collector, such as a render pool thread, which crashes Qt.
        layout.replaceWidget(previous, widget)
        previous.hide()
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
            self.actionExportPdf,
            self.actionFixDpi,
            self.actionRelinking,
            self.actionReverseTwoPageOrder,
        ):
            action.setEnabled(loaded)
        self.pauseSortingBtn.setEnabled(loaded)
        if not loaded:
            self.pauseSortingBtn.setChecked(False)

    def tr(self, source_text: str, disambiguation: str | None = None) -> str:
        """Translate text of this window in the context lupdate files it under.

        lupdate files ``self.tr`` calls under the enclosing class name, but this
        class is not a QObject, so the context is passed explicitly.

        Args:
            source_text (str): The English text.
            disambiguation (str | None): Tells apart equal texts with other meanings.

        Returns:
            str: The text in the installed language.
        """
        return QtCore.QCoreApplication.translate(
            "MainWindowManager", source_text, disambiguation
        )

    def _translate_own_text(self) -> None:
        """Give the actions and labels this class builds in code their text."""
        self.actionAddFiles.setText(self.tr("Add Files..."))
        self.actionOpenPdf.setText(self.tr("Open PDF..."))
        self.actionExportPdf.setText(self.tr("Export PDF..."))
        self.menuOpenRecent.setTitle(self.tr("Open Recent"))
        self.not_ready_widget.setText(
            self.tr(
                "Output is not yet possible, as the final size of pages is not "
                "yet known.\nTo determine it, run batch processing at "
                '"Select Content" or "Margins".'
            )
        )

    def _retranslate(self) -> None:
        """Show the window's lasting text in the language just installed.

        Panels, views and dialogs that are rebuilt as the user works pick the
        language up when next built, as upstream's do.
        """
        retranslate_ui(self.ui, _MAIN_WINDOW_UI)
        self._translate_own_text()
        self._update_window_title()
        self._show_load_error_message()
        if self._welcome_widget is not None:
            self._update_welcome_panel()

    def _update_window_title(self) -> None:
        """Show the project name and, via ``[*]``, the unsaved-changes marker."""
        if not self.is_project_loaded:
            self.ui.setWindowTitle(APP_TITLE)
            self.ui.setWindowModified(False)
            return
        name = (
            pathlib.Path(self.projectFile).stem
            if self.projectFile
            else self.tr("Unnamed")
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
            FilterStage.PAGE_SPLIT: [("change_requested", "")],
            FilterStage.DESKEW: [("apply_to_requested", "")],
            FilterStage.SELECT_CONTENT: [("apply_to_requested", "")],
            FilterStage.PAGE_LAYOUT: [
                ("apply_margins_requested", "margins"),
                ("apply_alignment_requested", "alignment"),
            ],
        }
        if panels.stage == FilterStage.OUTPUT:
            options.apply_group_requested.connect(
                lambda group: self.show_apply_dialog(
                    FilterStage.OUTPUT, part=group.value
                )
            )
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
        if stage == FilterStage.PAGE_SPLIT:
            return self._show_split_mode_dialog(page)
        margins = stage == FilterStage.PAGE_LAYOUT and part == "margins"
        options = MARGIN_SIDE_OPTIONS if margins else _APPLY_OPTIONS.get(stage, {})
        dialog_class = self._apply_dialog_class(stage, part)
        dialog = dialog_class(
            page,
            self.pages(),
            self.selectedPages(),
            self.ui,
            title=self._apply_dialog_title(stage, part),
            options=options,
            option_icons=MARGIN_SIDE_ICONS if margins else None,
            options_title=MARGINS_OPTIONS_TITLE if margins else None,
        )
        accepted = dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted
        targets = dialog.pages() if accepted else []
        flags = {key: dialog.option_checked(key) for key in options}
        dialog.deleteLater()
        others = [target for target in targets if target != page]
        sides = tuple(side for side in MARGIN_SIDE_OPTIONS if flags.get(side))
        if not others or (margins and not sides):
            return False
        self.apply_to_pages(
            stage,
            others,
            part=part,
            content_box=flags.get("content_box", True),
            page_box=flags.get("page_box", True),
            deskew=flags.get("deskew", True),
            oblique=flags.get("oblique", True),
            sides=sides if margins else MARGIN_SIDES,
        )
        return True

    @staticmethod
    def _apply_dialog_class(stage: FilterStage, part: str) -> type[ApplyToDialog]:
        """Return the apply dialog class for a stage and settings part.

        Args:
            stage (FilterStage): The stage whose settings are copied.
            part (str): The output group value, or the Page Layout part; empty
                when the whole stage is copied.

        Returns:
            type[ApplyToDialog]: The dialog class that owns the stage's labels.
        """
        if stage == FilterStage.OUTPUT and part:
            return _OUTPUT_GROUP_DIALOGS.get(ApplyGroup(part), ApplyColorsDialog)
        return _APPLY_DIALOGS.get(stage, ApplyToDialog)

    def _show_split_mode_dialog(self, page: PageId) -> bool:
        """Ask how pages are split and which pages get that, then apply it.

        Unlike the other apply dialogs the current page is a target too, because
        this is also where its own layout type is chosen.

        Args:
            page (PageId): The current page.

        Returns:
            bool: True if the choice was applied to at least one page.
        """
        params = self.stages.page_split_filter.get_params(page)
        if params.page_layout is None:
            return False
        dialog = SplitModeDialog(
            self.ui,
            page,
            self.pages(),
            self.selectedPages(),
            params.layout_type,
            params.page_layout.to_layout_type(),
        )
        chosen: list[tuple[set[PageId], LayoutType, bool]] = []
        dialog.settings_accepted.connect(
            lambda pages, layout_type, apply_cut: chosen.append(
                (pages, layout_type, apply_cut)
            )
        )
        dialog.exec()
        dialog.deleteLater()
        if not chosen or not chosen[0][0]:
            return False
        targets, layout_type, apply_cut = chosen[0]
        self.apply_to_pages(
            FilterStage.PAGE_SPLIT,
            [candidate for candidate in self.pages() if candidate in targets],
            layout_type=layout_type,
            apply_cut=apply_cut,
        )
        return True

    def _apply_dialog_title(self, stage: FilterStage, part: str) -> str:
        """Return the title of the apply dialog for a stage and settings part."""
        if stage == FilterStage.OUTPUT and part:
            return self.tr(OUTPUT_GROUP_TITLES[ApplyGroup(part)])
        if stage == FilterStage.DESKEW:
            return self.tr(DESKEW_TITLE)
        titles = {
            "margins": self.tr("Apply Margins"),
            "alignment": self.tr("Apply Alignment"),
        }
        return titles.get(part, self.tr("Apply to"))

    def apply_to_pages(
        self,
        stage: FilterStage,
        targets: list[PageId],
        *,
        part: str = "",
        content_box: bool = True,
        page_box: bool = True,
        deskew: bool = True,
        oblique: bool = True,
        layout_type: LayoutType = LayoutType.AUTO_LAYOUT_TYPE,
        apply_cut: bool = False,
        sides: Collection[str] = MARGIN_SIDES,
    ) -> None:
        """Copy the current page's settings of ``stage`` to other pages.

        Args:
            stage (FilterStage): The stage whose settings are copied.
            targets (list[PageId]): Pages that receive the settings.
            part (str): For Page Layout, ``"margins"`` or ``"alignment"``.
            content_box (bool): Select Content: copy a manual content box.
            page_box (bool): Select Content: copy a manual page box.
            deskew (bool): Deskew: copy the rotation angle and mode.
            oblique (bool): Deskew: copy the oblique angle and mode.
            layout_type (LayoutType): Page Split: the layout type to request.
            apply_cut (bool): Page Split: also copy the cutter lines, adapted to
                each target.
            sides (Collection[str]): Margins: the sides to copy.
        """
        source = self._current_page
        if source is None or not targets:
            return
        stages = self.stages
        match stage:
            case FilterStage.FIX_ORIENTATION:
                propagation.apply_orientation(stages, source, targets)
            case FilterStage.PAGE_SPLIT:
                propagation.apply_page_split(
                    stages, source, targets, layout_type, apply_cut=apply_cut
                )
                images = list(dict.fromkeys(target.image_id for target in targets))
                if self.project is not None and propagation.adopt_requested_page_counts(
                    self.project, stages, images
                ):
                    self._refresh_page_list()
            case FilterStage.DESKEW:
                propagation.apply_deskew(
                    stages, source, targets, apply_deskew=deskew, apply_oblique=oblique
                )
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
                propagation.apply_margins(stages, source, targets, sides)
            case FilterStage.OUTPUT:
                propagation.apply_output(stages, source, targets, ApplyGroup(part))
            case _:
                return
        self.modified = True
        self._invalidate_thumbnails(None if len(targets) > 1 else targets[0])
        if stage == FilterStage.PAGE_SPLIT:
            self._show_current_page()
        else:
            self._request_render()

    def _batch_pages(self) -> list[PageId]:
        """Return every page in display order, whichever order is chosen."""
        return self.thumbnails.pages()

    def startBatchProcessing(self) -> None:
        """Run every page, in display order, up to the selected stage.

        Only a batch from Output renders and writes files, as upstream does.
        """
        project = self.project
        if project is None or self.batch_runner.is_running:
            return
        self.focusButton.setChecked(True)
        self._start_batch(self._batch_pages(), FilterStage(self.curFilter))

    def _start_batch(
        self, pages: list[PageId], end_stage: FilterStage = FilterStage.OUTPUT
    ) -> None:
        """Process ``pages`` up to ``end_stage`` in the background.

        Args:
            pages (list[PageId]): Pages to process, in order.
            end_stage (FilterStage): Last stage to run; only Output writes files.
        """
        project = self.project
        if project is None or self.batch_runner.is_running or not pages:
            return
        self._cancel_render()
        self._set_batch_ui(True)
        if self.batch_dialog is not None:
            self.batch_dialog.deleteLater()
        renders = end_stage == FilterStage.OUTPUT
        self._batch_renders = renders
        dialog = BatchProgressDialog(len(pages), self.ui, renders=renders)
        dialog.beep_when_finished = self._beep_when_finished
        dialog.stop_requested.connect(self.stopBatchProcessing)
        dialog.load_changed.connect(self._on_batch_load_changed)
        self.batch_dialog = dialog
        dialog.show()
        self.batch_runner.worker_limit = dialog.load
        self.batch_runner.start(
            self.stages,
            pages,
            project.output_directory,
            layout_direction=project.layout_direction,
            dpi_overrides=self._dpi_overrides,
            file_labels=project.output_file_label_map(),
            end_stage=end_stage,
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
        """Lock the stage controls and page view while the worker owns the settings.

        Args:
            in_progress (bool): True when a batch starts, False when it ends.
        """
        self.filterList.set_batch_processing_in_progress(in_progress)
        self.thumbnails.set_context_menu_enabled(not in_progress)
        self.filterList.setEnabled(not in_progress)
        self.filterOptions.setEnabled(not in_progress)
        self.imageViewFrame.setEnabled(not in_progress)
        self.processing_indicator.set_active(in_progress)

    def _on_batch_load_changed(self, count: int) -> None:
        self.batch_runner.worker_limit = count

    def _announce_batch_completed(self) -> None:
        """Flash the taskbar entry if the user is elsewhere, and beep if asked."""
        if QtWidgets.QApplication.activeWindow() is None:
            QtWidgets.QApplication.alert(self.ui)
        if self._beep_when_finished:
            QtWidgets.QApplication.beep()

    def _on_batch_detected(self, event: PageDetected) -> None:
        if self.batch_dialog is not None:
            self.batch_dialog.add_detected(event)
        if not self._batch_renders:
            self._invalidate_thumbnails(event.page)

    def _on_batch_progress(self, event: ProgressEvent) -> None:
        if self.batch_dialog is not None:
            self.batch_dialog.add_result(event)
        self._invalidate_thumbnails(event.page)

    def _on_batch_finished(self, result: BatchResult) -> None:
        self.last_batch_result = result
        self._render_info.update(result.render_info)
        if self.project is not None:
            for page, info in result.render_info.items():
                self.project.set_text_geometry(
                    page,
                    TextGeometry.of(
                        info.transformation,
                        info.output_dpi,
                        dewarped=info.was_dewarped,
                        dewarp=info.dewarp,
                    ).model_dump(mode="json"),
                )
        self._set_batch_ui(False)
        if self.batch_dialog is not None:
            self._beep_when_finished = self.batch_dialog.beep_when_finished
            self.batch_dialog.finish(result)
        if self.project is None:
            return
        if result.detected or result.written or result.failed:
            self.modified = True
        self._adopt_stored_page_counts(
            dict.fromkeys(page.image_id for page in (*result.written, *result.failed))
        )
        self._invalidate_thumbnails()
        self._remember_aggregate_size()
        self._request_render()
        self._run_deferred_auto_save()
        if not result.cancelled:
            self._announce_batch_completed()
            self.goFirstPage()

    def _on_batch_failed(self, message: str) -> None:
        self._set_batch_ui(False)
        self._run_deferred_auto_save()
        _LOG.warning(f"Batch processing failed: {message}")
        if self.batch_dialog is not None:
            self.batch_dialog.deleteLater()
            self.batch_dialog = None
        QMessageBox.warning(
            self.ui,
            self.tr("Batch processing"),
            self.tr("Batch processing failed.") + f"\n{message}",
        )

    def _default_pdf_export_dialog(
        self, default_path: pathlib.Path, can_carry_text: bool, parent: QWidget
    ) -> PdfExportDialog:
        """Build the export options dialog."""
        return PdfExportDialog(
            default_path, can_carry_text=can_carry_text, parent=parent
        )

    def _ask_run_batch(self, unprocessed: int, total: int) -> bool:
        """Ask whether to process the pages that have no output yet."""
        answer = QMessageBox.question(
            self.ui,
            self.tr("Export PDF"),
            self.tr(
                "{unprocessed} of {total} pages have not been processed yet.\n"
                "Process them now? Export the PDF again once processing is done."
            ).format(unprocessed=unprocessed, total=total),
        )
        return answer == QMessageBox.StandardButton.Yes

    def openPdf(self, _checked: bool = False) -> None:
        """Ask for a PDF, extract its pages, then open New Project with them added.

        The pages go to ``<stem>_pages`` beside the PDF, which is also the output
        folder's parent. Cancelling at any step leaves the open project alone.
        """
        if self.pdf_job.is_running or self.batch_runner.is_running:
            return
        if not self._close_project_interactive():
            return
        start_dir = str(
            self.settings.value(_LAST_PDF_DIR_KEY, self._last_project_dir(), type=str)
        )
        chosen, _selected_filter = QFileDialog.getOpenFileName(
            self.ui, self.tr("Open PDF"), start_dir, "PDF (*.pdf)"
        )
        if not chosen:
            return
        pdf_path = pathlib.Path(chosen)
        self.settings.setValue(_LAST_PDF_DIR_KEY, str(pdf_path.parent))
        dialog = self.project_files_dialog_factory(self.ui)
        pages_dir = pdf_path.with_name(f"{pdf_path.stem}_pages")
        dialog.set_input_directory(pdf_path.parent)
        dialog.set_output_directory(pages_dir / "out")
        if not dialog.extract_pdf(pdf_path, self.ui):
            dialog.deleteLater()
            return
        if self._run_dialog(dialog):
            self._open_project_from_dialog(dialog)

    def exportPdf(self, _checked: bool = False) -> None:
        """Write the processed pages of the project to one PDF in the background."""
        project = self.project
        if project is None or self.batch_runner.is_running or self.pdf_job.is_running:
            return
        pages = project.get_pages()
        if not pages:
            return
        names = output_file_names(
            pages, project.layout_direction, project.output_file_label_map()
        )
        files = {page: project.output_directory / names[page] for page in pages}
        unprocessed = [page for page in pages if not files[page].is_file()]
        if unprocessed:
            if self.run_batch_prompt(len(unprocessed), len(pages)):
                self._start_batch(unprocessed)
            return
        can_carry_text = any(source.is_available for source in self.pdf_sources)
        dialog = self.pdf_export_dialog_factory(
            self._default_pdf_path(), can_carry_text, self.ui
        )
        accepted = dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted
        options = dialog.options()
        dialog.deleteLater()
        if not accepted:
            return
        request = self._pdf_export_request(files, options)
        self._run_pdf_job(
            self.tr("Export PDF"),
            lambda report, cancel: export_pdf_pages(request, report, cancel),
            self._on_pdf_exported,
        )

    def _default_pdf_path(self) -> pathlib.Path:
        """Return where the exported PDF goes unless the user picks another place."""
        sources = self.pdf_sources
        if len(sources) == 1:
            pdf_path = sources[0].pdf_path
            return pdf_path.with_name(f"{pdf_path.stem}_processed.pdf")
        output_directory = self.project.output_directory if self.project else None
        return (output_directory or pathlib.Path.cwd()) / "output.pdf"

    def _pdf_export_request(
        self, files: dict[PageId, pathlib.Path], options: PdfExportOptions
    ) -> PdfExportRequest:
        """Capture on the GUI thread everything the export thread needs.

        Args:
            files (dict[PageId, pathlib.Path]): Processed output image per page.
            options (PdfExportOptions): The user's choices.

        Returns:
            PdfExportRequest: Plain data; a page carries text only when its
            geometry is known from a batch run, in this session or a saved one.
        """
        project = self.project
        carries_text = bool(self.pdf_sources) and options.carry_text
        output_settings = self.stages.output_filter.settings
        plans: list[PagePlan] = []
        for page, file in files.items():
            info = self._render_info.get(page)
            dpi = (
                info.output_dpi if info else output_settings.get_params(page).output_dpi
            )
            text = None
            source = (
                project.source_of(page.image_id.file_path)
                if project is not None and carries_text
                else None
            )
            index = (
                source.page_index(page.image_id.file_path)
                if source is not None
                else None
            )
            if index is not None and info is not None:
                text = TextRequest(
                    index,
                    info.transformation,
                    info.output_dpi,
                    info.was_dewarped,
                    info.dewarp,
                )
            elif index is not None and project is not None:
                stored = project.text_geometry_of(page)
                if stored is not None:
                    text = StoredTextRequest(index, TextGeometry.model_validate(stored))
            plans.append(
                PagePlan(
                    file,
                    (float(dpi.horizontal), float(dpi.vertical)),
                    text,
                    source if text is not None else None,
                )
            )
        return PdfExportRequest(
            plans,
            options.out_path,
            options.bilevel_codec,
            options.jpeg_quality,
            list(self.pdf_sources),
            carry_text=options.carry_text,
        )

    def _on_pdf_exported(self, result: PdfExportResult) -> None:
        self.last_pdf_export = result
        message = self.tr("Exported {pages} pages to {path}").format(
            pages=result.pages, path=result.out_path
        )
        if result.text_pages:
            message += self.tr(" ({count} with text)").format(count=result.text_pages)
        get_cwidget(self.ui, QtWidgets.QStatusBar, "statusBar").showMessage(
            message, 10000
        )

    def _run_pdf_job(
        self,
        title: str,
        work: Callable[[Any, Any], object],
        on_done: Callable[[Any], None],
    ) -> None:
        """Run ``work`` on the PDF thread behind a modal progress window.

        Args:
            title (str): Title of the progress window.
            work (Callable): Called as ``work(report, cancel)`` on the thread.
            on_done (Callable[[Any], None]): Receives the result on the GUI thread.
        """
        dialog = PdfProgressDialog(title, self.ui)
        dialog.cancel_requested.connect(self.pdf_job.cancel)
        self._pdf_progress = dialog
        self._pdf_on_done = on_done
        dialog.show()
        self.pdf_job.start(work)

    def _end_pdf_job(self) -> Callable[[Any], None] | None:
        dialog, self._pdf_progress = self._pdf_progress, None
        on_done, self._pdf_on_done = self._pdf_on_done, None
        if dialog is not None:
            dialog.close_finished()
            dialog.deleteLater()
        return on_done

    def _on_pdf_progress(self, done: int, total: int, text: str) -> None:
        if self._pdf_progress is not None:
            self._pdf_progress.set_progress(done, total, text)

    def _on_pdf_finished(self, result: object) -> None:
        on_done = self._end_pdf_job()
        if on_done is not None:
            on_done(result)

    def _on_pdf_cancelled(self) -> None:
        self._end_pdf_job()
        get_cwidget(self.ui, QtWidgets.QStatusBar, "statusBar").showMessage(
            self.tr("Cancelled"), 5000
        )

    def _on_pdf_failed(self, message: str) -> None:
        self._end_pdf_job()
        _LOG.warning(f"PDF job failed: {message}")
        QMessageBox.warning(self.ui, self.tr("PDF"), message)

    def openProject(self, _checked: bool = False) -> None:
        """Ask for a project file and open it, replacing the current project."""
        project_file, _project_filter = QFileDialog.getOpenFileName(
            self.ui,
            self.tr("Open Project"),
            self._last_project_dir(),
            f"{self.tr('Scan Tailor Projects')} (*{PROJECT_SUFFIX})",
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
            xml_import = import_xml_project_file(path)
            project = xml_import.project if xml_import else Project.load(path)
        except (OSError, ValueError, SyntaxError) as error:
            self._warn_cannot_open(path, error)
            return False
        relinker = self._relink_missing_images(project)
        if relinker is None:
            return False
        try:
            self._apply_relinker(project, relinker)
            stages = StageSequence()
            apply_stage_params(stages, project.stage_params)
            apply_stage_defaults(stages, project.stage_defaults)
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
            self.tr("Error"),
            self.tr("Unable to open the project file.") + f"\n{error}",
        )

    @staticmethod
    def _relinkable_directories(
        project: Project, *, only_missing: bool
    ) -> list[pathlib.Path]:
        """List the image folders and the output folder of a project.

        Args:
            project (Project): The project.
            only_missing (bool): Keep only folders that do not exist; the output
                folder is listed when it is one of them.

        Returns:
            list[pathlib.Path]: Distinct folders in sorted order.
        """
        directories = image_directories(
            [info.id.file_path for info in project.images], only_missing=only_missing
        )
        output = project.output_directory
        if output not in directories and not (only_missing and output.is_dir()):
            directories = sorted([*directories, output])
        return directories

    @staticmethod
    def _apply_relinker(project: Project, relinker: Relinker) -> None:
        """Point the project's images and output folder at their relinked paths.

        Args:
            project (Project): The project, updated in place.
            relinker (Relinker): The chosen mapping.
        """
        project.remap_image_paths(relinker.substitute)
        project.output_directory = relinker.substitute(project.output_directory)
        _create_directory(project.output_directory)

    def _relink_missing_images(self, project: Project) -> Relinker | None:
        """Let the user relink the image folders and output folder that are missing.

        Args:
            project (Project): A freshly loaded project.

        Returns:
            Relinker | None: The mapping to apply (empty when nothing is missing),
                or None when the user cancelled.
        """
        missing = self._relinkable_directories(project, only_missing=True)
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

    def _can_navigate(self) -> bool:
        """Whether the page may change: a project is open and no batch runs."""
        return self.project is not None and not self.batch_runner.is_running

    def _go_to_page(self, page: PageId) -> None:
        """Make ``page`` current, scrolling the list to it only while Focus is on."""
        self.current_page = page

    def _go_to_page_index(self, index: int) -> None:
        """Make the displayed page at ``index`` current, clamping to the valid range.

        Args:
            index (int): Zero-based position in the thumbnail list.
        """
        pages = self.thumbnails.pages()
        if pages and self._can_navigate():
            self._go_to_page(pages[max(0, min(index, len(pages) - 1))])

    def goFirstPage(self) -> None:
        """Go to first page."""
        self._go_to_page_index(0)

    def goLastPage(self) -> None:
        """Go to last page."""
        self._go_to_page_index(len(self.thumbnails.pages()) - 1)

    def goNextPage(self) -> None:
        """Go to next page."""
        self._go_to_page_index(self._display_position() + 1)

    def goPrevPage(self) -> None:
        """Go to previous page."""
        self._go_to_page_index(max(self._display_position() - 1, 0))

    def _on_prev_page_button(self) -> None:
        if self.filterSelectedBtn.isChecked():
            self.goPrevSelectedPage()
        else:
            self.goPrevPage()

    def _on_next_page_button(self) -> None:
        if self.filterSelectedBtn.isChecked():
            self.goNextSelectedPage()
        else:
            self.goNextPage()

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
        if not self._can_navigate():
            return
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
        """Ask for a page number and go to that page."""
        pages = self.thumbnails.pages()
        if not pages or not self._can_navigate():
            return
        number, accepted = QInputDialog.getInt(
            self.ui,
            self.tr("Go To Page"),
            self.tr("Enter the page number:"),
            max(self._display_position(), 0) + 1,
            1,
            len(pages),
            1,
        )
        if accepted and pages[number - 1] != self._current_page:
            self._go_to_page(pages[number - 1])

    def openUserGuide(self) -> None:
        """Open the online user guide in the system web browser."""
        if not QtGui.QDesktopServices.openUrl(QtCore.QUrl(USER_GUIDE_URL)):
            _LOG.warning("Could not open the user guide at {}", USER_GUIDE_URL)

    def openLogFolder(self) -> None:
        """Show the log folder in the file manager, to attach logs to a bug report."""
        folder = diagnostics.log_folder()
        # A folder that cannot be created makes openUrl fail, which is logged.
        with contextlib.suppress(OSError):
            folder.mkdir(parents=True, exist_ok=True)
        if not QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(folder))):
            _LOG.warning("Could not open the log folder {}", folder)

    def showAboutDialog(self) -> None:
        """Show about dialog."""
        AboutDialog(self.ui).show()

    def reloadCurrentPage(self) -> None:
        """Render the current page again, e.g. after its source file changed."""
        if self._current_page is None or not self._can_navigate():
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
        self._refresh_dpi_overrides()
        self._invalidate_thumbnails()
        self._request_render()

    def showRelinkingDialog(self) -> None:
        """Let the user point the folders of the project's images elsewhere."""
        project = self.project
        if project is None or self.batch_runner.is_running:
            return
        relinker = self._ask_relinker(
            self._relinkable_directories(project, only_missing=False)
        )
        if relinker is None or relinker.is_empty():
            return
        project.stage_params = collect_stage_params(self.stages, project.get_pages())
        project.stage_defaults = collect_stage_defaults(self.stages)
        project.selected_page = self.current_page
        self._apply_relinker(project, relinker)
        stages = StageSequence()
        apply_stage_params(stages, project.stage_params)
        apply_stage_defaults(stages, project.stage_defaults)
        project_file = pathlib.Path(self.projectFile) if self.projectFile else None
        self._install_project(project, stages, project_file=project_file, modified=True)

    def toggleTwoPageSpreadReadingOrder(self, _checked: bool = False) -> None:
        """Swap which half of a two-page image comes first, in the list and names."""
        project = self.project
        if project is None or self.batch_runner.is_running:
            return
        selected = self.thumbnails.selected_pages()
        previous = project.layout_direction
        project.layout_direction = "RTL" if previous == "LTR" else "LTR"
        rename_split_outputs(
            project.output_directory,
            project.get_pages(),
            previous,
            project.layout_direction,
            project.output_file_label_map(),
        )
        self.modified = True
        self._invalidate_thumbnails()
        self._refresh_page_list()
        self.thumbnails.select_pages(selected)

    def openSettingsDialog(self) -> None:
        """Open settings dialog."""
        settings = ApplicationSettings.get_instance()
        dialog = SettingsDialog(settings, self.ui)
        dialog.settings_changed.connect(
            lambda: self._apply_thumbnail_settings(settings)
        )
        self._run_dialog(dialog)

    def openDefaultParamsDialog(self) -> None:
        """Open default params dialog."""
        self._run_dialog(
            DefaultParamsDialog(
                units=ApplicationSettings.get_instance().units, parent=self.ui
            )
        )

    def newProject(self) -> None:
        """Create a project from files chosen in the project files dialog."""
        if not self._close_project_interactive():
            return
        dialog = self.project_files_dialog_factory(self.ui)
        if not self._run_dialog(dialog):
            return
        self._open_project_from_dialog(dialog)

    def _open_project_from_dialog(self, dialog: ProjectFilesDialog) -> None:
        """Make a project from the pages an accepted files dialog holds.

        Args:
            dialog (ProjectFilesDialog): The accepted dialog. Everything is read
                here, since the dialog is deleted once the event loop runs again.
        """
        input_dir = dialog.get_input_directory()
        files = sort_smart(dialog.get_project_paths())
        output_dir = dialog.get_output_directory()
        layout_direction: LayoutDirection = "RTL" if dialog.is_rtl_layout() else "LTR"
        imported = dialog.get_imported_pdfs()
        if input_dir is None or not files:
            QMessageBox.warning(
                self.ui,
                self.tr("Error"),
                self.tr("No files were added to the project."),
            )
            return
        pages = self._read_pages(files)
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
        _add_pdf_sources(project, [item.source for item in imported])
        _create_directory(project.output_directory)
        stages = StageSequence()
        apply_default_params(
            stages, [image_id for image_id, _ in pages], active_default_params()
        )
        if imported:
            pdf_dpi = imported[-1].source.dpi
            stages.output_filter.settings.set_default_dpi(Dpi.uniform(pdf_dpi))
            self.settings.setValue(_LAST_PDF_DPI_KEY, pdf_dpi)
        self._install_project(project, stages, project_file=None, modified=True)
        self.settings.setValue(_LAST_INPUT_DIR_KEY, str(input_dir))

    def addFiles(self) -> None:
        """Append images chosen in the project files dialog to the open project."""
        self._insert_chosen_images(None, before=True)

    def insert_images_before(self, page: PageId) -> None:
        """Insert images chosen in the files dialog before the image of ``page``.

        Args:
            page (PageId): A page of the open project.
        """
        self._insert_chosen_images(page.image_id, before=True)

    def insert_images_after(self, page: PageId) -> None:
        """Insert images chosen in the files dialog after the image of ``page``.

        Args:
            page (PageId): A page of the open project.
        """
        self._insert_chosen_images(page.image_id, before=False)

    def insert_images_at_end(self) -> None:
        """Append images chosen in the files dialog after the last page."""
        self._insert_chosen_images(None, before=True)

    def _insert_chosen_images(self, existing: ImageId | None, *, before: bool) -> None:
        """Ask for image files and insert those not yet in the project.

        Args:
            existing (ImageId | None): The image to insert beside; None appends.
            before (bool): Insert before ``existing`` rather than after it.
        """
        project = self.project
        if project is None or self.batch_runner.is_running:
            return
        dialog = self.project_files_dialog_factory(self.ui)
        if not self._run_dialog(dialog):
            return
        imported = dialog.get_imported_pdfs()
        present = {info.id.file_path for info in project.images}
        paths = [
            path
            for path in sort_smart(dialog.get_project_paths())
            if path not in present
        ]
        pages = self._read_pages(paths)
        if pages:
            pages = self._fix_dpi_of_new_pages(pages)
        if not pages:
            return
        infos = [ImageInfo(id=image_id, metadata=meta) for image_id, meta in pages]
        project.register_output_file_labels()
        if not project.insert_images(infos, existing, before=before):
            return
        _add_pdf_sources(project, [item.source for item in imported])
        propagation.seed_new_pages(
            self.stages, [info.id for info in infos], active_default_params()
        )
        self._refresh_dpi_overrides()
        self.modified = True
        current = self._current_page
        self._refresh_page_list(current if current is not None else self.pages()[0])

    def remove_pages(self, pages: Iterable[PageId]) -> None:
        """Remove pages from the project after confirmation.

        Args:
            pages (Iterable[PageId]): The pages to remove.
        """
        project = self.project
        doomed = set(pages)
        if project is None or not doomed or self.batch_runner.is_running:
            return
        erase_output = self.remove_pages_prompt(len(doomed), self.ui)
        if erase_output is None:
            return
        survivor = self._page_to_select_after_removal(doomed)
        pages_before = self.pages()
        sources = [info.id.file_path for info in project.images]
        project.register_output_file_labels()
        removed = project.remove_pages(doomed)
        if not removed:
            return
        if erase_output:
            erase_output_files(
                removed,
                project.output_directory,
                pages_before,
                project.layout_direction,
                keep=sources,
                file_labels=project.output_file_label_map(),
            )
        self._refresh_dpi_overrides()
        self.modified = True
        self._refresh_page_list(survivor)

    def _page_to_select_after_removal(self, doomed: set[PageId]) -> PageId | None:
        """Pick the page to show once ``doomed`` are gone.

        Returns:
            PageId | None: The first page that stays if the list starts with a
                removed one, else the page before the first removed one.
        """
        displayed = self.thumbnails.pages()
        survivors = [page for page in displayed if page not in doomed]
        if not survivors:
            return None
        if displayed[0] in doomed:
            return survivors[0]
        before: PageId | None = None
        for page in displayed:
            if page in doomed:
                return before
            before = page
        return None

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
        """Ask for the DPI of new images when any has an undefined one.

        Args:
            pages (PageMetadata): Metadata read from the image files.

        Returns:
            PageMetadata | None: The pages with the chosen DPI applied, or None
                when the user cancelled.
        """
        if all(metadata.is_dpi_ok() for _, metadata in pages):
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
        self.renderer.sources.clear()
        self._thumbnail_epoch += 1
        self.project = project
        self._refresh_dpi_overrides()
        self.stages = stages
        self._bind_thumbnail_stages()
        self._remember_aggregate_size()
        self.thumbnails.set_project(project)
        self.projectFile = str(project_file) if project_file else ""
        if not project.pdf_sources:
            found = PdfSource.find(info.id.file_path for info in project.images)
            project.pdf_sources = found
            modified = modified or bool(found)
        self._render_info = {}
        self._modified = modified
        self._saved_state = None if modified else self._state_snapshot()
        pages = project.get_pages()
        if project.selected_page in pages:
            self.current_page = project.selected_page
        else:
            self.current_page = pages[0] if pages else None
        self._update_welcome_panel()
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
        pdf_dpi = cast(
            int, self.settings.value(_LAST_PDF_DPI_KEY, DEFAULT_IMPORT_DPI, type=int)
        )
        return ProjectFilesDialog(
            input_dir=input_dir,
            output_dir=output_dir,
            parent=parent,
            pdf_dpi=pdf_dpi,
        )

    @property
    def pdf_sources(self) -> list[PdfSource]:
        """The PDFs the open project pages were rendered from; empty when none."""
        return self.project.pdf_sources if self.project is not None else []

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
            f"{self.tr('Scan Tailor Projects')} (*{PROJECT_SUFFIX})",
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
        project.stage_defaults = collect_stage_defaults(self.stages)
        project.selected_page = self.current_page
        try:
            project.save(path)
        except OSError as error:
            _LOG.warning(f"Cannot save project {path}: {error}")
            QMessageBox.warning(
                self.ui,
                self.tr("Error"),
                self.tr("Error saving the project file!") + f"\n{error}",
            )
            return False
        self._saved_state = self._state_snapshot()
        self.modified = False
        if project.pdf_sources:
            PdfSource.discard_sidecars(info.id.file_path for info in project.images)
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
            update={
                "stage_params": stage_params,
                "stage_defaults": collect_stage_defaults(self.stages),
                "selected_page": None,
            }
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
        self.renderer.sources.clear()
        release_thumbnail_memory()
        self._thumbnail_epoch += 1
        self.project = None
        self._refresh_dpi_overrides()
        self.stages = StageSequence()
        self._bind_thumbnail_stages()
        self.thumbnails.set_project(None)
        self.projectFile = ""
        self._render_info = {}
        self.current_page = None
        self._modified = False
        self._saved_state = None
        self._update_welcome_panel()
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
            self.tr("Save Project"),
            self.tr("Save this project?"),
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
