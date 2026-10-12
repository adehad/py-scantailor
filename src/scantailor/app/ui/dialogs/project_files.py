"""Project files dialog for adding/removing files from a project."""

from collections import Counter
from collections.abc import Sequence
from pathlib import Path

from PySide6 import QtWidgets
from PySide6.QtCore import QStringListModel, Signal

from scantailor.app.ui import embed_ui, load_ui_widget
from scantailor.app.ui.icons import icon
from scantailor.app.ui.utils import get_cwidget
from scantailor.core.filename_ordering import sort_smart

_UI_FOLDER = Path(__file__).parent


class ProjectFilesDialog(QtWidgets.QDialog):
    """Dialog for managing files in a project.

    Allows users to:
    - Set input and output directories
    - Add files from the input directory to the project
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
    ) -> None:
        """Initialize the dialog.

        Args:
            input_dir: Initial input directory path.
            output_dir: Initial output directory path.
            project_files: Files already in the project. A bare name is taken to
                be in ``input_dir``.
            parent: Parent widget.
        """
        super().__init__(parent)
        self._input_dir = input_dir
        self._output_dir = output_dir
        self._project_files: set[Path] = {
            self._resolve(entry, input_dir) for entry in project_files or []
        }
        self._available_files: set[Path] = set()

        # Load UI
        self.ui = load_ui_widget(_UI_FOLDER / "ProjectFilesDialog.ui", self)
        embed_ui(self, self.ui)
        self._setup_widgets()
        self._populate_lists()
        self._connect_signals()

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
        self._off_project_model = QStringListModel()
        self._in_project_model = QStringListModel()
        self._off_project_list.setModel(self._off_project_model)
        self._in_project_list.setModel(self._in_project_model)

        self._add_to_project_btn.setIcon(icon("circle-arrow-right"))
        self._remove_from_project_btn.setIcon(icon("circle-arrow-left"))

    def _populate_lists(self) -> None:
        """Populate the file lists."""
        self._scan_input_directory()
        self._update_list_models()

    def _scan_input_directory(self) -> None:
        """Scan the input directory for image files."""
        self._available_files.clear()

        input_path = self._input_dir_line.text()
        if not input_path:
            return

        input_dir = Path(input_path)
        if not input_dir.is_dir():
            return

        # Supported image extensions
        extensions = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".gif"}

        for file_path in input_dir.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in extensions:
                self._available_files.add(file_path)

    def _update_list_models(self) -> None:
        """Update the list models with current file sets."""
        self._off_project_paths = sort_smart(
            self._available_files - self._project_files
        )
        self._off_project_model.setStringList(
            self._display_names(self._off_project_paths)
        )

        self._in_project_paths = sort_smart(self._project_files)
        self._in_project_model.setStringList(
            self._display_names(self._in_project_paths)
        )

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

    def _on_input_dir_changed(self, text: str) -> None:
        """Handle input directory change."""
        self._scan_input_directory()
        self._update_list_models()

    def _add_selected_to_project(self) -> None:
        """Add selected files to the project."""
        selection = self._off_project_list.selectionModel()
        for index in selection.selectedIndexes():
            self._project_files.add(self._off_project_paths[index.row()])

        self._update_list_models()
        self.files_changed.emit()

    def _remove_selected_from_project(self) -> None:
        """Remove selected files from the project."""
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
