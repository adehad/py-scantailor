"""Fix DPI dialog for images with undefined or incorrect DPI."""

from pathlib import Path
from typing import ClassVar

from PySide6 import QtGui, QtWidgets
from PySide6.QtCore import Qt
from PySide6.QtGui import QIntValidator

from scantailor.app.ui import load_ui_widget
from scantailor.app.ui.utils import get_cwidget
from scantailor.core import Dpi
from scantailor.core.project import ImageInfo

_UI_FOLDER = Path(__file__).parent


class FixDpiDialog(QtWidgets.QDialog):
    """Dialog for fixing DPI values of images.

    Displays two tabs:
    - Images with undefined DPI
    - All images (for bulk editing)

    Users can select images and apply new DPI values.
    """

    # Common DPI presets
    DPI_PRESETS: ClassVar[list[tuple[str, int, int]]] = [
        ("Custom", 0, 0),
        ("300", 300, 300),
        ("400", 400, 400),
        ("600", 600, 600),
        ("1200", 1200, 1200),
    ]

    def __init__(
        self,
        images: list[ImageInfo],
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        """Initialize the fix DPI dialog.

        Args:
            images: List of images to display.
            parent: Parent widget.
        """
        super().__init__(parent)
        self._images = images
        self._dpi_changes: dict[str, Dpi] = {}  # path -> new DPI

        # Load UI
        self.ui = load_ui_widget(_UI_FOLDER / "FixDpiDialog.ui", self)
        self._setup_widgets()
        self._populate_views()
        self._connect_signals()

    def _setup_widgets(self) -> None:
        """Set up widget references."""
        self._tab_widget = get_cwidget(self.ui, QtWidgets.QTabWidget, "tabWidget")

        # Tree views
        self._undefined_view = get_cwidget(
            self.ui, QtWidgets.QTreeView, "undefinedDpiView"
        )
        self._all_pages_view = get_cwidget(self.ui, QtWidgets.QTreeView, "allPagesView")

        # DPI controls
        self._dpi_combo = get_cwidget(self.ui, QtWidgets.QComboBox, "dpiCombo")
        self._x_dpi = get_cwidget(self.ui, QtWidgets.QLineEdit, "xDpi")
        self._y_dpi = get_cwidget(self.ui, QtWidgets.QLineEdit, "yDpi")
        self._apply_btn = get_cwidget(self.ui, QtWidgets.QPushButton, "applyBtn")

        # Button box
        self._button_box = get_cwidget(self.ui, QtWidgets.QDialogButtonBox, "buttonBox")

        # Set up DPI combo
        self._dpi_combo.clear()
        for name, _, _ in self.DPI_PRESETS:
            self._dpi_combo.addItem(name)

        # Set up validators for DPI input
        validator = QIntValidator(1, 9999, self)
        self._x_dpi.setValidator(validator)
        self._y_dpi.setValidator(validator)

        # Set tab titles
        self._tab_widget.setTabText(0, "Need Fixing")
        self._tab_widget.setTabText(1, "All Pages")

    def _populate_views(self) -> None:
        """Populate the tree views with image data."""
        # Create models for both views
        self._undefined_model = QtGui.QStandardItemModel()
        self._undefined_model.setHorizontalHeaderLabels(["File", "DPI"])

        self._all_model = QtGui.QStandardItemModel()
        self._all_model.setHorizontalHeaderLabels(["File", "DPI"])

        # Populate models
        for img in self._images:
            path = str(img.id.file_path.name)
            dpi = img.metadata.dpi

            # Format DPI string
            if dpi.horizontal == 0 or dpi.vertical == 0:
                dpi_str = "Undefined"
                needs_fixing = True
            elif dpi.horizontal == dpi.vertical:
                dpi_str = str(dpi.horizontal)
                needs_fixing = False
            else:
                dpi_str = f"{dpi.horizontal}x{dpi.vertical}"
                needs_fixing = False

            # Create items
            path_item = QtGui.QStandardItem(path)
            path_item.setData(str(img.id.file_path), Qt.ItemDataRole.UserRole)
            path_item.setEditable(False)

            dpi_item = QtGui.QStandardItem(dpi_str)
            dpi_item.setEditable(False)

            # Add to all pages model
            self._all_model.appendRow([path_item.clone(), dpi_item.clone()])

            # Add to undefined model if needed
            if needs_fixing:
                self._undefined_model.appendRow([path_item, dpi_item])

        # Set models
        self._undefined_view.setModel(self._undefined_model)
        self._all_pages_view.setModel(self._all_model)

        # Configure views
        for view in (self._undefined_view, self._all_pages_view):
            view.setSelectionMode(
                QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection
            )
            view.setRootIsDecorated(False)
            view.header().setStretchLastSection(True)

    def _connect_signals(self) -> None:
        """Connect widget signals."""
        self._button_box.accepted.connect(self.accept)
        self._button_box.rejected.connect(self.reject)

        # Enable/disable controls based on selection
        self._undefined_view.selectionModel().selectionChanged.connect(
            self._on_selection_changed
        )
        self._all_pages_view.selectionModel().selectionChanged.connect(
            self._on_selection_changed
        )
        self._tab_widget.currentChanged.connect(self._on_tab_changed)

        # DPI combo changes
        self._dpi_combo.currentIndexChanged.connect(self._on_dpi_preset_changed)

        # Apply button
        self._apply_btn.clicked.connect(self._apply_dpi)

    def _on_selection_changed(self) -> None:
        """Handle selection change in either view."""
        current_view = self._get_current_view()
        has_selection = current_view.selectionModel().hasSelection()

        self._dpi_combo.setEnabled(has_selection)
        self._x_dpi.setEnabled(has_selection)
        self._y_dpi.setEnabled(has_selection)
        self._apply_btn.setEnabled(has_selection)

    def _on_tab_changed(self, index: int) -> None:
        """Handle tab change."""
        self._on_selection_changed()

    def _get_current_view(self) -> QtWidgets.QTreeView:
        """Get the currently active tree view."""
        if self._tab_widget.currentIndex() == 0:
            return self._undefined_view
        return self._all_pages_view

    def _on_dpi_preset_changed(self, index: int) -> None:
        """Handle DPI preset selection change."""
        if index < 0 or index >= len(self.DPI_PRESETS):
            return

        _, x_dpi, y_dpi = self.DPI_PRESETS[index]
        if x_dpi > 0:
            self._x_dpi.setText(str(x_dpi))
            self._y_dpi.setText(str(y_dpi))

    def _apply_dpi(self) -> None:
        """Apply DPI to selected images."""
        try:
            x_dpi = int(self._x_dpi.text())
            y_dpi = int(self._y_dpi.text())
        except ValueError:
            QtWidgets.QMessageBox.warning(
                self,
                "Invalid DPI",
                "Please enter valid DPI values.",
            )
            return

        if x_dpi < 1 or y_dpi < 1:
            QtWidgets.QMessageBox.warning(
                self,
                "Invalid DPI",
                "DPI values must be positive.",
            )
            return

        # Get selected items
        current_view = self._get_current_view()
        model = current_view.model()
        if not isinstance(model, QtGui.QStandardItemModel):
            return
        selection = current_view.selectionModel().selectedRows()

        for index in selection:
            path_item = model.item(index.row(), 0)
            if path_item:
                file_path = path_item.data(Qt.ItemDataRole.UserRole)
                self._dpi_changes[file_path] = Dpi(horizontal=x_dpi, vertical=y_dpi)

                # Update display
                dpi_item = model.item(index.row(), 1)
                if dpi_item:
                    if x_dpi == y_dpi:
                        dpi_item.setText(str(x_dpi))
                    else:
                        dpi_item.setText(f"{x_dpi}x{y_dpi}")

    def get_dpi_changes(self) -> dict[str, Dpi]:
        """Get the DPI changes made by the user.

        Returns:
            Dictionary mapping file paths to new DPI values.
        """
        return self._dpi_changes

    def show(self) -> None:
        """Show the dialog."""
        self.ui.show()
