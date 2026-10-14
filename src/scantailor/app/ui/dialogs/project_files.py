"""Project files dialog for adding/removing files from a project."""

import threading
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path

from loguru import logger
from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import QStringListModel, Signal

from scantailor.app.ui import embed_ui, load_ui_widget
from scantailor.app.ui.dialogs.pdf_import import (
    DEFAULT_IMPORT_DPI,
    ExistingPages,
    PdfExtractDialog,
    ask_existing_pages,
)
from scantailor.app.ui.dialogs.pdf_progress import PdfProgressDialog
from scantailor.app.ui.icons import icon
from scantailor.app.ui.pdf_worker import ImportedPdf, PdfJob, import_pdf_pages
from scantailor.app.ui.utils import get_cwidget
from scantailor.core.filename_ordering import sort_smart
from scantailor.core.image_io import ImageLoadError, list_image_pages
from scantailor.core.pdf_import import page_count, rendered_pages
from scantailor.core.pdf_source import PdfSource

_UI_FOLDER = Path(__file__).parent
_MIN_VISIBLE_ROWS = 12
_IMAGE_EXTENSIONS = frozenset(
    {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".gif"}
)

ExtractPdf = Callable[[Path, QtWidgets.QWidget], ImportedPdf | None]
"""Renders the pages of one PDF, asking the user first. None when abandoned."""

Report = Callable[[int, int, str], None]


def _is_pdf(path: Path) -> bool:
    return path.suffix.lower() == ".pdf"


def pages_folder_of(pdf_path: Path) -> Path:
    """Return the folder that holds the pages extracted from a PDF.

    Args:
        pdf_path (Path): The PDF.

    Returns:
        Path: ``<stem>_pages`` beside the PDF.
    """
    return pdf_path.with_name(f"{pdf_path.stem}_pages")


def _page_number(image: Path) -> int:
    return int(image.stem.removeprefix("page_"))


def _adopt_rendered_pages(
    pdf_path: Path, destination: Path, existing: list[Path]
) -> ImportedPdf:
    """Use the pages an earlier extraction left, at the resolution they carry.

    Args:
        pdf_path (Path): The PDF the pages were rendered from.
        destination (Path): Folder holding the rendered pages.
        existing (list[Path]): The rendered page images in page order.

    Returns:
        ImportedPdf: The pages and their mapping to the PDF.
    """
    dpi = DEFAULT_IMPORT_DPI
    try:
        horizontal = list_image_pages(existing[0])[0][1].dpi.horizontal
    except (ImageLoadError, OSError, IndexError):
        logger.warning("Could not read the resolution of {}", existing[0])
    else:
        dpi = round(horizontal) if horizontal > 0 else DEFAULT_IMPORT_DPI
    source = PdfSource(
        pdf_path=pdf_path,
        dpi=dpi,
        pages_dir=destination,
        pages={image.name: _page_number(image) - 1 for image in existing},
    )
    return ImportedPdf(source, existing)


def _run_pdf_job(
    title: str,
    work: Callable[[Report, threading.Event], object],
    parent: QtWidgets.QWidget,
) -> ImportedPdf | None:
    """Run a PDF job on a worker thread behind a progress window until it ends.

    The progress window is modal, so the nested event loop is the only way the
    user can reach the dialog while the job runs.

    Args:
        title (str): Title of the progress window.
        work (Callable): Called as ``work(report, cancel)`` on the worker thread.
        parent (QtWidgets.QWidget): Owner of the progress window and error box.

    Returns:
        ImportedPdf | None: The job's result, or None when it was cancelled or failed.
    """
    job = PdfJob(parent)
    progress = PdfProgressDialog(title, parent)
    outcome: dict[str, object] = {}
    loop = QtCore.QEventLoop()

    def on_finished(result: object) -> None:
        outcome["result"] = result
        loop.quit()

    def on_failed(message: str) -> None:
        outcome["error"] = message
        loop.quit()

    job.progress.connect(progress.set_progress)
    progress.cancel_requested.connect(job.cancel)
    job.finished.connect(on_finished)
    job.failed.connect(on_failed)
    job.cancelled.connect(loop.quit)
    progress.show()
    job.start(work)
    loop.exec()
    job.wait()
    progress.close_finished()
    progress.deleteLater()
    job.deleteLater()
    if "error" in outcome:
        QtWidgets.QMessageBox.warning(parent, "PDF", str(outcome["error"]))
        return None
    result = outcome.get("result")
    return result if isinstance(result, ImportedPdf) else None


def extract_pdf_pages(
    pdf_path: Path, parent: QtWidgets.QWidget, default_dpi: int = DEFAULT_IMPORT_DPI
) -> ImportedPdf | None:
    """Ask which pages of a PDF to render, then render them beside the PDF.

    Pages go to ``<stem>_pages`` next to the PDF. When that folder already holds
    rendered pages, the user chooses between using them and rendering again.
    Rendering again replaces the images only once every page is done.

    Args:
        pdf_path (Path): The PDF to extract.
        parent (QtWidgets.QWidget): Owner of the prompts and the progress window.
        default_dpi (int): Resolution offered first in the prompt.

    Returns:
        ImportedPdf | None: The rendered pages, or None when the user cancelled
            or rendering failed.
    """
    try:
        total = page_count(pdf_path)
    except ValueError as error:
        QtWidgets.QMessageBox.warning(parent, "PDF", str(error))
        return None
    destination = pages_folder_of(pdf_path)
    prompt = PdfExtractDialog(pdf_path, total, destination, default_dpi, parent)
    accepted = prompt.exec() == QtWidgets.QDialog.DialogCode.Accepted
    options = prompt.options()
    prompt.deleteLater()
    if not accepted:
        return None
    existing = rendered_pages(destination)
    replace = False
    if existing:
        choice = ask_existing_pages(destination, len(existing), parent)
        if choice is ExistingPages.CANCEL:
            return None
        if choice is ExistingPages.USE:
            return _adopt_rendered_pages(pdf_path, destination, existing)
        replace = True

    def work(report: Report, cancel: threading.Event) -> object:
        return import_pdf_pages(
            pdf_path,
            destination,
            options.dpi,
            report,
            cancel,
            first_page=options.first_page,
            last_page=options.last_page,
            clear_existing=replace,
        )

    return _run_pdf_job("Extract pages", work, parent)


class ProjectFilesDialog(QtWidgets.QDialog):
    """Dialog for managing files in a project.

    Allows users to:
    - Set input and output directories
    - Add images from the input directory to the project
    - Extract the pages of a PDF, in the input directory or elsewhere, and add them
    - Drop folders, PDFs and images onto the dialog
    - Remove files from the project
    - Configure RTL layout and fix DPI options

    Signals:
        files_changed: Emitted when the file list changes.
    """

    files_changed = Signal()

    def __init__(
        self,
        input_dir: Path | None = None,
        output_dir: Path | None = None,
        project_files: Sequence[str | Path] | None = None,
        parent: QtWidgets.QWidget | None = None,
        *,
        extract_pdf: ExtractPdf | None = None,
        pdf_dpi: int = DEFAULT_IMPORT_DPI,
    ) -> None:
        """Initialize the dialog.

        Args:
            input_dir: Initial input directory path.
            output_dir: Initial output directory path.
            project_files: Files already in the project. A bare name is taken to
                be in ``input_dir``.
            parent: Parent widget.
            extract_pdf: Renders the pages of a PDF. Defaults to asking the user
                and rendering on a worker thread; tests pass a fake.
            pdf_dpi: Resolution offered first when extracting a PDF.
        """
        super().__init__(parent)
        self._input_dir = input_dir
        self._output_dir = output_dir
        self._project_files: set[Path] = {
            self._resolve(entry, input_dir) for entry in project_files or []
        }
        self._available_files: set[Path] = set()
        self._extra_pdfs: set[Path] = set()
        self._pdf_pages: dict[Path, int] = {}
        self._imported: dict[Path, ImportedPdf] = {}
        self._extract_pdf: ExtractPdf = extract_pdf or (
            lambda path, owner: extract_pdf_pages(path, owner, pdf_dpi)
        )

        # Load UI
        self.ui = load_ui_widget(_UI_FOLDER / "ProjectFilesDialog.ui", self)
        embed_ui(self, self.ui)
        self._setup_widgets()
        self._reserve_list_rows()
        self._populate_lists()
        self._connect_signals()
        self.setAcceptDrops(True)

    def _setup_widgets(self) -> None:
        """Set up widget references."""
        # Directory inputs
        self._input_dir_line = get_cwidget(self.ui, QtWidgets.QLineEdit, "inpDirLine")
        self._input_dir_browse_btn = get_cwidget(
            self.ui, QtWidgets.QPushButton, "inpDirBrowseBtn"
        )
        self._output_dir_line = get_cwidget(self.ui, QtWidgets.QLineEdit, "outDirLine")
        self._output_dir_browse_btn = get_cwidget(
            self.ui, QtWidgets.QPushButton, "outDirBrowseBtn"
        )

        # File lists
        self._off_project_list = get_cwidget(
            self.ui, QtWidgets.QListView, "offProjectList"
        )
        self._in_project_list = get_cwidget(
            self.ui, QtWidgets.QListView, "inProjectList"
        )
        self._off_project_select_all = get_cwidget(
            self.ui, QtWidgets.QPushButton, "offProjectSelectAllBtn"
        )
        self._in_project_select_all = get_cwidget(
            self.ui, QtWidgets.QPushButton, "inProjectSelectAllBtn"
        )
        self._add_pdf_btn = get_cwidget(self.ui, QtWidgets.QPushButton, "addPdfBtn")

        # Transfer buttons
        self._add_to_project_btn = get_cwidget(
            self.ui, QtWidgets.QToolButton, "addToProjectBtn"
        )
        self._remove_from_project_btn = get_cwidget(
            self.ui, QtWidgets.QToolButton, "removeFromProjectBtn"
        )

        # Options
        self._rtl_layout_cb = get_cwidget(self.ui, QtWidgets.QCheckBox, "rtlLayoutCB")
        self._force_fix_dpi_cb = get_cwidget(
            self.ui, QtWidgets.QCheckBox, "forceFixDpi"
        )

        # Progress bar
        self._progress_bar = get_cwidget(self.ui, QtWidgets.QProgressBar, "progressBar")
        self._progress_bar.setVisible(False)

        # Button box
        self._button_box = get_cwidget(self.ui, QtWidgets.QDialogButtonBox, "buttonBox")

        # Set initial directory paths
        if self._input_dir:
            self._input_dir_line.setText(str(self._input_dir))
        if self._output_dir:
            self._output_dir_line.setText(str(self._output_dir))

        # Create models for lists
        self._off_project_paths: list[Path] = []
        self._in_project_paths: list[Path] = []
        self._off_project_model = QtGui.QStandardItemModel(self)
        self._in_project_model = QStringListModel(self)
        self._off_project_list.setModel(self._off_project_model)
        self._in_project_list.setModel(self._in_project_model)

        self._add_to_project_btn.setIcon(icon("circle-arrow-right"))
        self._remove_from_project_btn.setIcon(icon("circle-arrow-left"))

    def _reserve_list_rows(self) -> None:
        """Make both lists show ``_MIN_VISIBLE_ROWS`` rows at the default size."""
        for view in (self._off_project_list, self._in_project_list):
            row_height = view.fontMetrics().height() + 2 * view.frameWidth() + 4
            view.setMinimumHeight(
                _MIN_VISIBLE_ROWS * row_height + 2 * view.frameWidth()
            )
        self.resize(self.sizeHint().expandedTo(self.size()))

    def _populate_lists(self) -> None:
        """Populate the file lists."""
        self._scan_input_directory()
        self._update_list_models()

    def _scan_input_directory(self) -> None:
        """Scan the input directory for image files and PDFs."""
        self._available_files.clear()

        input_path = self._input_dir_line.text()
        if not input_path:
            return

        input_dir = Path(input_path)
        if not input_dir.is_dir():
            return

        for file_path in input_dir.iterdir():
            if not file_path.is_file():
                continue
            if _is_pdf(file_path) or file_path.suffix.lower() in _IMAGE_EXTENSIONS:
                self._available_files.add(file_path)

    def _page_count_of(self, pdf_path: Path) -> int:
        """Return the pages of a PDF, 0 when it cannot be read. Cached per file."""
        if pdf_path not in self._pdf_pages:
            try:
                self._pdf_pages[pdf_path] = page_count(pdf_path)
            except ValueError:
                logger.warning("Not listing {}: it cannot be read as a PDF", pdf_path)
                self._pdf_pages[pdf_path] = 0
        return self._pdf_pages[pdf_path]

    def _update_list_models(self) -> None:
        """Update the list models with current file sets.

        PDFs come first, then images, each group in natural order.
        """
        available = (self._available_files | self._extra_pdfs) - self._project_files
        pdfs = sort_smart(
            path for path in available if _is_pdf(path) and self._page_count_of(path)
        )
        images = sort_smart(path for path in available if not _is_pdf(path))
        self._off_project_paths = [*pdfs, *images]
        self._off_project_model.clear()
        names = self._display_names(self._off_project_paths)
        for path, name in zip(self._off_project_paths, names, strict=True):
            item = QtGui.QStandardItem(
                self._pdf_label(path, name) if _is_pdf(path) else name
            )
            item.setEditable(False)
            item.setIcon(icon("file") if _is_pdf(path) else icon("image"))
            item.setToolTip(str(path))
            self._off_project_model.appendRow(item)

        self._in_project_paths = sort_smart(self._project_files)
        self._in_project_model.setStringList(
            self._display_names(self._in_project_paths)
        )

    def _pdf_label(self, pdf_path: Path, name: str) -> str:
        """Name a PDF with its page count, and whether its pages are extracted."""
        pages = self._page_count_of(pdf_path)
        label = f"{name} ({pages} {'page' if pages == 1 else 'pages'}"
        if rendered_pages(pages_folder_of(pdf_path)):
            label += ", extracted"
        return label + ")"

    @staticmethod
    def _resolve(entry: str | Path, input_dir: Path | None) -> Path:
        """Turn a project file entry into a path, anchoring bare names."""
        path = Path(entry)
        if input_dir is not None and path.parent == Path():
            return input_dir / path
        return path

    @staticmethod
    def _display_names(paths: list[Path]) -> list[str]:
        """Name each file by its file name, adding the folder where names repeat."""
        counts = Counter(path.name for path in paths)
        return [
            path.name if counts[path.name] == 1 else f"{path.name} ({path.parent})"
            for path in paths
        ]

    def _connect_signals(self) -> None:
        """Connect widget signals."""
        self._input_dir_browse_btn.clicked.connect(self._browse_input_dir)
        self._output_dir_browse_btn.clicked.connect(self._browse_output_dir)
        self._add_pdf_btn.clicked.connect(self._browse_pdf)
        self._add_to_project_btn.clicked.connect(self._add_selected_to_project)
        self._remove_from_project_btn.clicked.connect(
            self._remove_selected_from_project
        )
        self._input_dir_line.textChanged.connect(self._on_input_dir_changed)
        self._button_box.accepted.connect(self.accept)
        self._button_box.rejected.connect(self.reject)

    def _browse_input_dir(self) -> None:
        """Browse for input directory."""
        current = self._input_dir_line.text() or str(Path.home())
        directory = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            "Select Input Directory",
            current,
        )
        if directory:
            self._input_dir_line.setText(directory)

    def _browse_output_dir(self) -> None:
        """Browse for output directory."""
        current = self._output_dir_line.text() or str(Path.home())
        directory = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            "Select Output Directory",
            current,
        )
        if directory:
            self._output_dir_line.setText(directory)

    def _browse_pdf(self) -> None:
        """Add a PDF from any folder to the left list; its pages are extracted later."""
        current = self._input_dir_line.text() or str(Path.home())
        chosen, _selected_filter = QtWidgets.QFileDialog.getOpenFileName(
            self, "Add PDF", current, "PDF (*.pdf)"
        )
        if chosen:
            self._extra_pdfs.add(Path(chosen))
            self._update_list_models()

    def _on_input_dir_changed(self, text: str) -> None:
        """Handle input directory change."""
        self._scan_input_directory()
        self._update_list_models()

    def _add_selected_to_project(self) -> None:
        """Add selected images to the project and extract selected PDFs.

        PDFs are extracted one after the other, each with its own prompt. A PDF
        that is not extracted stops the rest.
        """
        selection = self._off_project_list.selectionModel()
        rows = sorted({index.row() for index in selection.selectedIndexes()})
        selected = [self._off_project_paths[row] for row in rows]
        self._project_files.update(path for path in selected if not _is_pdf(path))
        self._update_list_models()
        for path in selected:
            if _is_pdf(path) and not self.extract_pdf(path):
                break

        self._update_list_models()
        self.files_changed.emit()

    def extract_pdf(
        self, pdf_path: Path, owner: QtWidgets.QWidget | None = None
    ) -> bool:
        """Extract the pages of a PDF and add them to the project files.

        Args:
            pdf_path (Path): The PDF to extract.
            owner (QtWidgets.QWidget | None): Parent of the prompts and progress
                window; this dialog when None.

        Returns:
            bool: True when pages were rendered and added.
        """
        imported = self._extract_pdf(pdf_path, owner or self)
        if imported is None:
            return False
        self._record(imported)
        self._project_files.update(imported.images)
        self._update_list_models()
        self.files_changed.emit()
        return True

    def _record(self, imported: ImportedPdf) -> None:
        """Remember an extraction, merging it with an earlier one of the same PDF."""
        key = imported.source.pdf_path
        earlier = self._imported.get(key)
        if earlier is None:
            self._imported[key] = imported
            return
        pages = {**earlier.source.pages, **imported.source.pages}
        images = sort_smart({*earlier.images, *imported.images})
        source = imported.source.model_copy(update={"pages": pages})
        self._imported[key] = ImportedPdf(source, images)

    def get_imported_pdfs(self) -> list[ImportedPdf]:
        """Get the extracted PDFs whose pages are still in the project files.

        Returns:
            list[ImportedPdf]: One per PDF, holding only the pages still in the
                project. PDFs with no pages left are omitted.
        """
        kept: list[ImportedPdf] = []
        for imported in self._imported.values():
            images = [path for path in imported.images if path in self._project_files]
            if not images:
                continue
            pages = {
                image.name: imported.source.pages[image.name]
                for image in images
                if image.name in imported.source.pages
            }
            source = imported.source.model_copy(update={"pages": pages})
            kept.append(ImportedPdf(source, images))
        return kept

    def set_input_directory(self, directory: Path) -> None:
        """Show the files of another input folder.

        Args:
            directory (Path): The folder to scan.
        """
        self._input_dir_line.setText(str(directory))

    def set_output_directory(self, directory: Path) -> None:
        """Set the output folder shown in the dialog.

        Args:
            directory (Path): The output folder.
        """
        self._output_dir_line.setText(str(directory))

    def dragEnterEvent(self, event: QtGui.QDragEnterEvent) -> None:
        """Accept drags that carry files or folders."""
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            super().dragEnterEvent(event)

    def dropEvent(self, event: QtGui.QDropEvent) -> None:
        """Take dropped folders, PDFs and images as if they were chosen here."""
        paths = _local_paths(event.mimeData().urls())
        event.acceptProposedAction()
        for path in paths:
            self._accept_dropped(path)
        self._update_list_models()
        self.files_changed.emit()

    def _accept_dropped(self, path: Path) -> None:
        if path.is_dir():
            self.set_input_directory(path)
        elif _is_pdf(path) and path.is_file():
            self._extra_pdfs.add(path)
            self.extract_pdf(path)
        elif path.suffix.lower() in _IMAGE_EXTENSIONS and path.is_file():
            self._project_files.add(path)

    def _remove_selected_from_project(self) -> None:
        """Remove selected files from project."""
        selection = self._in_project_list.selectionModel()
        for index in selection.selectedIndexes():
            self._project_files.discard(self._in_project_paths[index.row()])

        self._update_list_models()
        self.files_changed.emit()

    def get_input_directory(self) -> Path | None:
        """Get the selected input directory.

        Returns:
            Input directory path, or None if empty.
        """
        text = self._input_dir_line.text()
        return Path(text) if text else None

    def get_output_directory(self) -> Path | None:
        """Get the selected output directory.

        Returns:
            Output directory path, or None if empty.
        """
        text = self._output_dir_line.text()
        return Path(text) if text else None

    def get_project_paths(self) -> list[Path]:
        """Get the files in the project, which may come from several folders.

        Returns:
            Full paths in natural filename order.
        """
        return sort_smart(self._project_files)

    def get_project_files(self) -> list[str]:
        """Get the file names of the files in the project.

        Returns:
            File names in the order of ``get_project_paths``.
        """
        return [path.name for path in self.get_project_paths()]

    def is_rtl_layout(self) -> bool:
        """Check if RTL layout is enabled.

        Returns:
            True if RTL layout is enabled.
        """
        return self._rtl_layout_cb.isChecked()

    def is_force_fix_dpi(self) -> bool:
        """Check if force fix DPI is enabled.

        Returns:
            True if force fix DPI is enabled.
        """
        return self._force_fix_dpi_cb.isChecked()


def _local_paths(urls: Iterable[QtCore.QUrl]) -> list[Path]:
    return [Path(url.toLocalFile()) for url in urls if url.isLocalFile()]
